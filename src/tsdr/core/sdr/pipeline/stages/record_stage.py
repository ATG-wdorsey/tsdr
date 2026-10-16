from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from tsdr.core.events.events import RecordingFinishedEvent
from tsdr.core.sdr import flac
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.radio.dsp._kernels import (
    StreamingPolyphaseResampler,
    apply_freq_shift_c64,
    iq_to_int_pairs,
)
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation

logger = logging.getLogger(__name__)

# The rational step's prototype filter, per output phase (as `make_rational_resampler`).
_TAPS_PER_PHASE = 10
# Samples waiting for the writer thread, at most (0.5 s at 16 MS/s); past that a batch waits.
_QUEUE_BYTES = 64 << 20
_FULL_SCALE = 32768.0
# An 8-bit radio's values ((u - 127.5) / 127.5, or s / 127) land on multiples of 128 and 256,
# which FLAC stores in 9 and 8 bits; played back at 1/32768 they are 0.4 and 0.8 % weaker.
_EXACT_SCALE = {SampleFormat.UINT8_IQ: 127.5 * 256, SampleFormat.SINT8_IQ: 127.0 * 256}


class RecordStage:
    """Sink stage that writes IQ samples to a FLAC file (`core/sdr/flac.py`): 16-bit I/Q.

    A writer thread of its own encodes (libFLAC, at the SDRConfig's ``flac_level``);
    ``process`` resamples and converts, and waits only when the writer is half a second
    behind. ``sample_format`` is the radio's: an 8-bit radio's samples are kept exactly.

    With ``resample``, ``center_hz`` is the frequency the recording is centered on (the
    dial when it started): the capture is shifted by it less each batch's center before
    the low-pass, so a channel off the LO is recorded, not the band around the LO.

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
        sample_format: SampleFormat = SampleFormat.COMPLEX64,
        center_hz: float | None = None,
    ) -> None:
        self._path = Path(output_path)
        # This recording's own file while it is written (beside the final name, made at the
        # first batch); renamed to the final name, never replacing another file, at close.
        self._partial_path: Path | None = None
        self._pipeline_name = pipeline_name
        if resample is not None and resample[0] == resample[1]:
            resample = None  # the capture's own rate (a near-equal --sr)
        self._resample = resample
        self._max_samples = max_samples
        self._center_hz = center_hz if resample is not None else None
        self._scale = (
            _EXACT_SCALE.get(sample_format, _FULL_SCALE) if resample is None else _FULL_SCALE
        )

        self._decim: MultistageDecimator | None = None
        self._poly: StreamingPolyphaseResampler | None = None
        self._shift_hz = 0.0
        self._shift_phase = 0.0
        self._outside_logged = False

        self._queue: _WriterQueue | None = None
        self._writer: threading.Thread | None = None
        # Set by the writer thread when libFLAC fails; it then drops what it is given.
        self._writer_error: BaseException | None = None
        self._behind_logged = False
        self._opened = False
        self._samples_written = 0  # given to the writer
        self._encoded = 0  # written to the file (by the writer thread)
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

        if not self._opened:
            self._open(data, context)
        if self._writer_error is not None:
            self.close()  # reports what was saved, and raises the writer's error
            return data
        samples = self._resample_chunk(self._centered(data))

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

    def _open(self, data: SamplesBatch, context: PipelineContext) -> None:
        """Once: a recording that can't start ends (one error, not one a batch), and reports
        that it saved nothing, which removes its pipeline (`record start` works again)."""
        self._opened = True
        try:
            self._open_file(data, context)
        except BaseException:
            self._closed = True
            self._report("")
            raise

    def _open_file(self, data: SamplesBatch, context: PipelineContext) -> None:
        device_rate = out_rate = data.sample_rate
        if self._resample is not None:  # first: nothing to clean up if it fails
            up, down = self._resample
            out_rate = device_rate * up / down
            # A whole factor first, by the decimator's cascade (it keeps what folds onto the
            # band out), then the rational rest at the low rate, its filter sized to it.
            decimation, rest = (down, Fraction(1)) if up == 1 else _split_rate_change(up, down)
            if decimation > 1:
                self._decim = MultistageDecimator(device_rate, decimation, out_rate * 0.45)
            if rest != 1:
                p, q = rest.numerator, rest.denominator
                self._poly = StreamingPolyphaseResampler(p, q, 2 * _TAPS_PER_PHASE * max(p, q) + 1)
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
        os.close(fd)  # libFLAC opens it by name
        center = data.center_frequency if self._center_hz is None else self._center_hz
        tags = {
            flac.TAG_CENTER_FREQUENCY: flac.format_tag_number(center),
            flac.TAG_GAIN: f"{data.rf_gain:g}",
            flac.TAG_DEVICE: self._device_id,
        }
        writer = None
        try:
            writer = flac.FlacWriter(
                partial, sample_rate=out_rate, level=context.config.flac_level, tags=tags
            )
            pending = _WriterQueue(_QUEUE_BYTES)
            thread = threading.Thread(
                target=self._write_loop,
                args=(writer, pending),
                name=f"record_{self._device_id}",
                daemon=True,
            )
            thread.start()
        except BaseException:
            # Nothing recorded yet: the encoder (and its file) let go, the file removed.
            try:
                if writer is not None:
                    writer.finish()
            except OSError:
                pass
            finally:
                partial.unlink(missing_ok=True)
            raise
        self._partial_path, self._queue, self._writer = partial, pending, thread

    def _write_loop(self, writer: flac.FlacWriter, pending: _WriterQueue) -> None:
        """The writer thread: encodes until close() sends None, then finishes the file.
        After a failure it keeps taking (and dropping) batches, so none waits for it."""
        try:
            while (pairs := pending.get()) is not None:
                if self._writer_error is None:
                    try:
                        writer.write(pairs)
                        self._encoded += len(pairs) // 2
                    except Exception as e:  # noqa: BLE001 - any failure: this thread must keep draining
                        logger.error("record_write_failed path=%s error=%r", self._partial_path, e)
                        self._writer_error = e
        finally:
            try:
                writer.finish()
            except Exception as e:  # noqa: BLE001 - reported by close()
                if self._writer_error is None:
                    self._writer_error = e

    def _centered(self, data: SamplesBatch) -> np.ndarray:
        assert data.iq_samples is not None
        if self._center_hz is None:
            return data.iq_samples
        # Shifted by the recorded frequency less this batch's center (it moves when the LO is
        # retuned); the phase starts over when that changes.
        shift = self._center_hz - data.center_frequency
        if shift != self._shift_hz:
            self._shift_hz, self._shift_phase = shift, 0.0
        if shift == 0.0:
            return data.iq_samples
        if abs(shift) >= data.sample_rate / 2:  # not in this capture: silence keeps the timing
            if not self._outside_logged:
                self._outside_logged = True
                logger.warning(
                    "record_channel_outside_capture path=%s center=%.0f capture_center=%.0f",
                    self._path,
                    self._center_hz,
                    data.center_frequency,
                )
            return np.zeros(len(data.iq_samples), dtype=np.complex64)
        shifted: np.ndarray
        shifted, self._shift_phase = apply_freq_shift_c64(
            data.iq_samples, shift, data.sample_rate, self._shift_phase
        )
        return shifted

    def _resample_chunk(self, samples: np.ndarray) -> np.ndarray:
        if self._decim is not None:
            samples = self._decim.process(samples)
        if self._poly is not None:
            samples = self._poly.process_iq(samples)
        return samples

    def _write_samples(self, samples: np.ndarray) -> None:
        """Converts on this thread (a copy the batch's later users can't change) and hands
        the result to the writer thread."""
        assert self._queue is not None
        pairs = iq_to_int_pairs(
            np.ascontiguousarray(samples, dtype=np.complex64), self._scale, -32768, 32767
        )
        if self._queue.put(pairs) and not self._behind_logged:
            self._behind_logged = True
            logger.warning("record_writer_behind path=%s", self._path)

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
            if self._writer is not None:
                assert self._queue is not None
                self._queue.put(None)
                self._writer.join()  # the file is finished (and closed) when it returns
            if self._writer_error is not None:
                raise self._writer_error
            if saved is not None:
                saved = self._path = _promote(saved, self._path)
        finally:
            if saved is not None:
                self._report(str(saved))

    def _report(self, path: str) -> None:
        """`RecordingFinishedEvent`: where the data is ("" when nothing was saved)."""
        if self._event_bus is not None:
            self._event_bus.publish(
                RecordingFinishedEvent(
                    source_id=f"record_{self._device_id}",
                    device_id=self._device_id,
                    pipeline_name=self._pipeline_name,
                    path=path,
                    samples_written=self._encoded,
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


def _split_rate_change(up: int, down: int) -> tuple[int, Fraction]:
    """The rate change `up / down` as a whole decimation (its factors at most 7, for the
    decimator's cascade) to a rate up to 8x the target, and the exact rational rest:
    the decimation whose rest has the shortest filter (16 MS/s to 44.1 kS/s: 320, then
    441/500; the largest, 360, would leave 3969/4000)."""
    ratio = down / up
    best: tuple[int, int, Fraction] | None = None
    for d in range(max(1, int(ratio)), max(1, int(ratio / 8)) - 1, -1):
        if smooth_decimation(d) != d:
            continue
        rest = Fraction(up * d, down)
        size = max(rest.numerator, rest.denominator)
        if best is None or size < best[0]:
            best = (size, d, rest)
    assert best is not None  # d = 1 is always a candidate
    return best[1], best[2]


class _WriterQueue:
    """The batches waiting for the writer thread, bounded by their bytes: a put waits while
    those it holds reach `limit` (any one batch gets into an empty queue). A batch counts
    until the writer takes it."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._items: deque[np.ndarray | None] = deque()
        self._bytes = 0
        self._changed = threading.Condition()

    def put(self, item: np.ndarray | None) -> bool:
        """Adds `item` (None: the end); returns whether it had to wait."""
        size = 0 if item is None else item.nbytes
        waited = False
        with self._changed:
            while self._bytes and self._bytes + size > self._limit:
                waited = True
                self._changed.wait()
            self._items.append(item)
            self._bytes += size
            self._changed.notify_all()
        return waited

    def get(self) -> np.ndarray | None:
        with self._changed:
            while not self._items:
                self._changed.wait()
            item = self._items.popleft()
            if item is not None:
                self._bytes -= item.nbytes
            self._changed.notify_all()
        return item


def _numbered(path: Path, n: int) -> Path:
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


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
