from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import IO, Any

import numpy as np
import zstandard as zstd

from tsdr.core.events.events import RecordingFinishedEvent
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler
from tsdr.radio.dsp.decimator import MultistageDecimator

logger = logging.getLogger(__name__)

_RESAMPLE_TAPS = 65
_DEFAULT_LEVEL = 3


class RecordStage:
    """Sink stage that writes IQ samples to a ``.cu8.zst`` / ``.cf32.zst`` file.

    ``sample_format`` selects the on-disk precision: ``UINT8_IQ`` (compact,
    lossless for 8-bit devices like RTL-SDR) or ``COMPLEX64`` (full float32,
    preserves the dynamic range of high-bit-depth devices like Airspy HF+).

    If ``max_samples`` is set the stage self-terminates once reached and
    publishes ``RecordingFinishedEvent``; otherwise it runs until the pipeline
    is removed from the device config.
    """

    def __init__(
        self,
        output_path: Path | str,
        pipeline_name: str = "recording",
        resample: tuple[int, int] | None = None,
        max_samples: int | None = None,
        sample_format: SampleFormat = SampleFormat.UINT8_IQ,
    ) -> None:
        self._path = Path(output_path)
        # This recording's own file while it is written (beside the final name, made at the
        # first batch); renamed to the final name, never replacing another file, at close.
        self._partial_path: Path | None = None
        self._pipeline_name = pipeline_name
        self._resample = resample
        self._max_samples = max_samples
        self._sample_format = sample_format

        self._decim: MultistageDecimator | None = None
        self._poly: StreamingPolyphaseResampler | None = None

        self._file: IO[bytes] | None = None
        self._zwriter: Any = None
        self._samples_written = 0
        self._closed = False
        # Where close() reports the saved file (the first batch's context).
        self._event_bus: Any = None
        self._device_id = ""

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if self._closed or data.iq_samples is None:
            return data
        if self._event_bus is None:
            self._event_bus = context.event_bus
            self._device_id = context.device_context.device_id

        self._ensure_open(data.sample_rate)
        samples = self._resample_chunk(data.iq_samples)

        if self._max_samples is not None:
            remaining = self._max_samples - self._samples_written
            if remaining <= 0:
                self._finalize_duration_mode(context)
                return data
            if len(samples) > remaining:
                samples = samples[:remaining]

        if len(samples) > 0:
            self._write_samples(samples)
            self._samples_written += len(samples)

        if self._max_samples is not None and self._samples_written >= self._max_samples:
            self._finalize_duration_mode(context)
        return data

    def _ensure_open(self, device_rate: float) -> None:
        if self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        while True:  # a name no other recording has, made exclusively (permissions as open())
            partial = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex[:8]}.partial")
            try:
                fd = os.open(
                    partial,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                    0o666,
                )
            except FileExistsError:
                continue
            break
        self._partial_path = partial
        self._file = os.fdopen(fd, "wb")  # owned by the stage, closed in close()
        self._zwriter = zstd.ZstdCompressor(level=_DEFAULT_LEVEL).stream_writer(self._file)

        if self._resample is None:
            return
        up, down = self._resample
        if up == down:
            self._resample = None
        elif up == 1 and down > 1:
            self._decim = MultistageDecimator(device_rate, down, (device_rate / down) * 0.45)
        else:
            self._poly = StreamingPolyphaseResampler(up, down, n_taps=_RESAMPLE_TAPS)

    def _resample_chunk(self, samples: np.ndarray) -> np.ndarray:
        if self._decim is not None:
            return self._decim.process(samples)
        if self._poly is not None:
            stacked = np.empty((len(samples), 2), dtype=np.float32)
            stacked[:, 0] = samples.real
            stacked[:, 1] = samples.imag
            out = self._poly.process(stacked)
            recombined = np.empty(len(out), dtype=np.complex64)
            recombined.real = out[:, 0]
            recombined.imag = out[:, 1]
            return recombined
        return samples

    def _write_samples(self, samples: np.ndarray) -> None:
        assert self._zwriter is not None
        if self._sample_format is SampleFormat.COMPLEX64:
            self._zwriter.write(memoryview(np.ascontiguousarray(samples, dtype=np.complex64)))
            return
        n = len(samples)
        cu8 = np.empty(n * 2, dtype=np.uint8)
        cu8[0::2] = np.clip(np.round(samples.real * 127.5 + 127.5), 0, 255).astype(np.uint8)
        cu8[1::2] = np.clip(np.round(samples.imag * 127.5 + 127.5), 0, 255).astype(np.uint8)
        self._zwriter.write(cu8.tobytes())

    def _finalize_duration_mode(self, context: PipelineContext) -> None:
        self.close()

    def close(self) -> None:
        """Finish the file and give it its name (the next free -2, -3, ... when the name is
        taken), then report it: `RecordingFinishedEvent` with where the data is, which also
        ends a `record for` pipeline. Reported even when finishing fails (a full disk, a
        refused rename: the data stays in the partial file, and the error goes on up). A
        recording that never got a batch wrote nothing."""
        if self._closed:
            return
        self._closed = True
        saved = self._partial_path  # where the data is, until it has its name
        try:
            try:
                if self._zwriter is not None:
                    zwriter, self._zwriter = self._zwriter, None
                    zwriter.close()
            finally:  # the file is closed even when the last flush fails (a full disk)
                if self._file is not None:
                    file, self._file = self._file, None
                    file.close()
            if saved is not None:
                saved = self._path = _promote(saved, self._path)
        finally:
            if saved is not None and self._event_bus is not None:
                self._event_bus.publish(
                    RecordingFinishedEvent(
                        source_id=f"record_{self._device_id}",
                        device_id=self._device_id,
                        pipeline_name=self._pipeline_name,
                        path=str(saved),
                        samples_written=self._samples_written,
                        recording=self,
                    )
                )

    def on_config_change(self, config: Any) -> None:
        pass

    def reset(self) -> None:
        pass

    @property
    def samples_written(self) -> int:
        return self._samples_written

    @property
    def path(self) -> Path:
        return self._path


# Extensions kept whole when a number goes before them ("x.cf32.zst" -> "x-2.cf32.zst").
_RECORDING_EXTENSIONS = (".cu8.zst", ".cf32.zst")


def _numbered(path: Path, n: int) -> Path:
    name = path.name
    ext = next((e for e in _RECORDING_EXTENSIONS if name.endswith(e)), path.suffix)
    return path.with_name(f"{name[: len(name) - len(ext)]}-{n}{ext}")


def _promote(partial: Path, final: Path) -> Path:
    """Give a finished recording its name without replacing a file already there (another
    recording, or one finished since the name was chosen): the next free -2, -3, ... then.
    The name is claimed with an empty file made exclusively, which every recording here
    respects, and the recording then replaces only that placeholder: atomic on any
    filesystem, hard links or not. Returns the name it got (the partial's, if the rename
    fails: the data is kept)."""
    candidate, n = final, 2
    while True:
        try:
            os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError:
            candidate, n = _numbered(final, n), n + 1
            continue
        break
    try:
        os.replace(partial, candidate)
    except OSError as e:
        logger.error("record_rename_failed partial=%s path=%s error=%r", partial, candidate, e)
        candidate.unlink(missing_ok=True)
        return partial
    if candidate != final:
        logger.warning("record_name_taken path=%s saved_as=%s", final, candidate)
    return candidate
