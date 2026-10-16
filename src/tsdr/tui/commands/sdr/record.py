from __future__ import annotations

import logging
from argparse import Namespace
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from tsdr.core.sdr import flac
from tsdr.core.sdr.config import PipelineConfig, StageType
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.sdr.exceptions import ConfigurationError, SDRException
from tsdr.core.units import parse_hz
from tsdr.tui.commands._format import device_id, freq_mhz, safe, success
from tsdr.tui.commands.base import Command, CommandParser, Completion
from tsdr.tui.commands.sdr._utils import device_id_completions, get_focused_device_id

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path("samples")
PIPELINE_NAME = "recording"
_MIN_SR_FOR_AUTOCOMPLETE = 30_000
_RATIONAL_DENOM_LIMIT = 10_000


class SDRRecordCommand(Command):
    """Record IQ from the focused device to samples/, as FLAC (16-bit I/Q).

    The whole capture by default; a lower ``--sr`` records the band around the dial
    (centered on it, fixed at the start). Three call patterns:
      * ``record for SECONDS`` - record for N seconds, then auto-stop.
      * ``record start [--sr H] [--device D] [--output P]`` - begin background
        recording. Runs until ``record stop`` is called.
      * ``record stop [--device D]`` - end a running recording.
    """

    @property
    def description(self) -> str:
        return "Record IQ to samples/ (record for <sec> | record start | record stop)"

    def configure(self, parser: CommandParser) -> None:
        sub = parser.add_subparsers(dest="action")

        for_p = sub.add_parser("for")
        for_p.add_argument("seconds", type=float, help="Duration in seconds")
        _add_common(for_p)

        start_p = sub.add_parser("start")
        _add_common(start_p)

        stop_p = sub.add_parser("stop")
        stop_p.add_argument("--device", dest="device_id", default=None)

    def run(self, args: Namespace) -> str:
        if args.action is None:
            return self.help_text()

        engine = get_engine()
        did = args.device_id or get_focused_device_id()

        if args.action == "stop":
            return self._stop(engine, did)
        if args.action == "start":
            return self._start(engine, did, args, duration=None)
        if args.action == "for":
            if args.seconds <= 0:
                raise ConfigurationError("Duration must be positive")
            return self._start(engine, did, args, duration=args.seconds)
        raise SDRException(f"Unknown record action: {args.action}")

    def _start(
        self,
        engine: SDREngine,
        did: str,
        args: Namespace,
        *,
        duration: float | None,
    ) -> str:
        device = engine.get_device(did)
        if PIPELINE_NAME in device.config.pipelines:
            raise SDRException(f"Already recording on {did} - stop it first with 'record stop'.")
        if not flac.available():
            raise SDRException(
                "Recording needs libFLAC: install it (Ubuntu: sudo apt install libflac12t64, "
                "macOS: brew install flac)"
            )

        device_rate = device.config.sample_rate
        target_rate = (
            parse_hz(args.sample_rate) if args.sample_rate is not None else int(device_rate)
        )
        if target_rate > device_rate:
            raise ConfigurationError(
                f"Target rate {target_rate} Hz exceeds device rate {int(device_rate)} Hz - "
                f"choose a lower --sr."
            )

        resample, out_rate = _compute_resample(device_rate, target_rate)
        max_samples = int(duration * out_rate) if duration is not None else None
        # A lower rate records the band around the dial, not the one around the LO.
        center = device.config.center_frequency
        record_center = device.config.tuned_frequency if resample is not None else None
        edge = ""
        if record_center is not None:
            if abs(record_center - center) + out_rate / 2 > device_rate / 2:
                edge = " [dim](partly past the capture's edge)[/]"
            center = record_center

        # An --output that is a directory gets an automatic name inside it.
        out_dir = Path(args.output) if args.output and Path(args.output).is_dir() else None
        explicit = args.output if args.output and out_dir is None else None
        if explicit and not explicit.lower().endswith(_EXT):
            explicit += _EXT

        output_path = (
            Path(explicit)
            if explicit
            else _autogenerate_path(
                center_frequency=center,
                out_rate=out_rate,
                duration=int(duration) if duration is not None else None,
                rf_gain=device.config.rf_gain,
                directory=out_dir,
            )
        )
        # A finished recording never replaces a file: an existing --output gets the next free name.
        taken = " [dim](exists: saved under the next free name)[/]" if output_path.exists() else ""

        pipeline_config = PipelineConfig(
            stages=(StageType.RECORD,),
            record_path=str(output_path),
            record_resample=resample,
            record_max_samples=max_samples,
            record_sample_format=device.device.get_sample_format(),
            record_center_hz=record_center,
        )

        engine.add_pipeline(did, PIPELINE_NAME, pipeline_config)

        path_md = safe(str(output_path))
        rate_md = f"[yellow]{int(out_rate)} Hz[/]"
        if record_center is not None:
            rate_md += f" around {freq_mhz(record_center, precision=4)}{edge}"
        if duration is not None:
            return success(
                f"Recording {duration:g}s on {device_id(did)} @ {rate_md} → {path_md}{taken}"
            )
        return success(
            f"Recording on {device_id(did)} @ {rate_md} → {path_md}{taken} "
            "[dim](stop with 'record stop')[/]"
        )

    def _stop(self, engine: SDREngine, did: str) -> str:
        device = engine.get_device(did)
        pipeline_config = device.config.pipelines.get(PIPELINE_NAME)
        if pipeline_config is None:
            raise SDRException(f"Not recording on '{did}'")
        engine.remove_pipeline(did, PIPELINE_NAME)
        # The file is finished on the pipeline thread after its last batch, and may take the
        # next free name: the "recording finished" notice reports where it went.
        return success(f"Stopped recording on {device_id(did)}")

    def complete(
        self,
        tokens: list[str],
        prefix: str,
        *,
        flag: str | None = None,
        subcommand: str | None = None,
    ) -> list[Completion]:
        if flag == "--device":
            return device_id_completions(prefix)
        if flag == "--sr":
            return _sr_divisor_completions(prefix)
        if flag == "--output":
            return []
        if not tokens:
            return [
                Completion("for", "record for N seconds"),
                Completion("start", "begin a background recording"),
                Completion("stop", "end a running recording"),
            ]
        return []


def _add_common(parser: CommandParser) -> None:
    parser.add_argument(
        "--sr",
        dest="sample_rate",
        default=None,
        help="Sample rate with SI suffix (e.g. 25k, 1.2M; default: the whole capture); "
        "a lower rate records the band centered on the dial",
    )
    parser.add_argument(
        "--device",
        dest="device_id",
        default=None,
        help="Device id (default: focused device)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output file or directory (default: samples/<auto>.flac; .flac is added)",
    )


def _compute_resample(device_rate: float, target_rate: int) -> tuple[tuple[int, int] | None, float]:
    """Return (resample_tuple, effective_output_rate).

    ``None`` means no resampling is needed. For integer divisors the returned
    tuple is ``(1, N)`` which the stage routes through StreamingDecimFilter.
    """
    device_int = int(device_rate)
    if target_rate == device_int:
        return None, device_rate
    if device_int % target_rate == 0:
        return (1, device_int // target_rate), float(target_rate)
    frac = Fraction(target_rate, device_int).limit_denominator(_RATIONAL_DENOM_LIMIT)
    if frac == 1:  # within a hair of the capture's rate: the capture as it is
        return None, device_rate
    out_rate = device_rate * frac.numerator / frac.denominator
    return (frac.numerator, frac.denominator), out_rate


_EXT = ".flac"


def _autogenerate_path(
    center_frequency: float,
    out_rate: float,
    duration: int | None,
    rf_gain: float,
    directory: Path | None = None,
) -> Path:
    folder = SAMPLES_DIR if directory is None else directory
    folder.mkdir(parents=True, exist_ok=True)
    fmhz = f"{center_frequency / 1e6:g}"
    if "." not in fmhz and "e" not in fmhz:
        fmhz += ".0"
    sr_khz = f"{out_rate / 1000:g}"  # 44.1, not 44: the iq-file device plays at this rate
    gain = int(round(rf_gain))
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    dur_segment = f"_dur={int(round(duration))}s" if duration is not None else ""
    stem = f"freq={fmhz}M_sr={sr_khz}k{dur_segment}_gain={gain}_{ts}"
    # A name taken by a finished recording is skipped. Two recordings may still pick one
    # name (both started in the same second): each writes its own file, and the later to
    # finish takes the next free -2, -3, ... (RecordStage), which its finished notice shows.
    path, n = folder / f"{stem}{_EXT}", 2
    while path.exists():
        path, n = folder / f"{stem}-{n}{_EXT}", n + 1
    return path


def _sr_divisor_completions(prefix: str) -> list[Completion]:
    try:
        engine = get_engine()
    except RuntimeError:
        return []
    if engine.focused_device is None:
        return []
    device = engine.devices.get(engine.focused_device)
    if device is None:
        return []
    device_rate = int(device.config.sample_rate)

    candidates: list[tuple[int, int]] = []
    n = 2
    while device_rate // n >= _MIN_SR_FOR_AUTOCOMPLETE:
        if device_rate % n == 0:
            candidates.append((device_rate // n, n))
        n += 1

    return [
        Completion(str(rate), f"÷{divisor}")
        for rate, divisor in candidates
        if str(rate).startswith(prefix)
    ]
