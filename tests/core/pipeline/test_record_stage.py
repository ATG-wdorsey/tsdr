from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from tsdr.core.events.bus import EventBus
from tsdr.core.events.events import RecordingFinishedEvent
from tsdr.core.sdr import flac
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.io import load_iq
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages import record_stage
from tsdr.core.sdr.pipeline.stages.record_stage import RecordStage
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.core.sdr.workers.pipeline_worker import PipelineWorker
from tsdr.devices import MockParams
from tsdr.radio.dsp._kernels import iq_to_int_pairs


@dataclass
class _FakeDeviceContext:
    device_id: str = "test"


def _make_context() -> tuple[PipelineContext, EventBus, list[RecordingFinishedEvent]]:
    bus = EventBus()
    received: list[RecordingFinishedEvent] = []

    def handler(ev: RecordingFinishedEvent) -> None:
        received.append(ev)

    bus.subscribe(RecordingFinishedEvent, handler)
    ctx = PipelineContext(device_context=_FakeDeviceContext(), event_bus=bus, config=SDRConfig())  # type: ignore[arg-type]
    return ctx, bus, received


def _batch(iq: np.ndarray, sample_rate: float) -> SamplesBatch:
    return SamplesBatch(
        iq_samples=iq.astype(np.complex64),
        sample_rate=sample_rate,
        center_frequency=100e6,
        rf_gain=20.0,
    )


def test_a_recording_is_16_bit_flac_with_its_rate_frequency_and_gain(tmp_path: Path) -> None:
    """Recordings were zstd-compressed float32 (cf32) or 8-bit (cu8): zstd took more than
    a core at 16 MS/s on the pipeline thread and saved 8 %. They are 16-bit FLAC now,
    encoded on a thread of their own, with the true rate, frequency and gain as tags."""
    rng = np.random.default_rng(0)
    n = 10_000
    iq = (rng.uniform(-0.9, 0.9, n) + 1j * rng.uniform(-0.9, 0.9, n)).astype(np.complex64)
    out = tmp_path / "test.flac"

    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()

    stage.process(_batch(iq[: n // 2], 250_000.0), ctx)
    stage.process(_batch(iq[n // 2 :], 250_000.0), ctx)
    stage.close()

    loaded = load_iq(out)
    assert len(loaded) == n
    assert np.max(np.abs(loaded.real - iq.real)) <= 0.5 / 32768 + 1e-7  # 16-bit rounding
    info = flac.read_info(out)
    assert (info.bits_per_sample, info.channels, info.total_samples) == (16, 2, n)
    assert info.tags[flac.TAG_SAMPLE_RATE] == "250000"
    assert info.tags[flac.TAG_CENTER_FREQUENCY] == "100000000"
    assert info.tags[flac.TAG_GAIN] == "20"


def test_an_8_bit_radios_samples_are_kept_exactly_and_compactly(tmp_path: Path) -> None:
    """An 8-bit radio's values ((u - 127.5) / 127.5) land on multiples of 128 at 16 bits,
    which FLAC stores in 9 bits: no larger than the old cu8 files, and nothing lost."""
    rng = np.random.default_rng(0)
    n = 50_000
    u = rng.integers(0, 256, size=(n, 2))
    iq = ((u[:, 0] - 127.5) / 127.5 + 1j * (u[:, 1] - 127.5) / 127.5).astype(np.complex64)
    out = tmp_path / "rtl.flac"
    stage = RecordStage(output_path=out, sample_format=SampleFormat.UINT8_IQ)
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, 250_000.0), ctx)
    stage.close()
    loaded = load_iq(out) * np.float32(32768 / 32640)
    np.testing.assert_allclose(loaded, iq, atol=1e-6)
    assert out.stat().st_size < 2.5 * n  # random values: ~2.25 bytes a sample (at 32768: ~4)


def test_a_narrower_rate_is_centered_on_the_dial(tmp_path: Path) -> None:
    """`record --sr` resampled around the LO: with the dial off the LO (an LO offset, free
    tuning) the channel was left out. The capture is shifted to the dial first."""
    sr, offset = 2_400_000.0, 300_000.0
    n = int(sr)
    t = np.arange(n) / sr
    iq = (0.3 * np.exp(2j * np.pi * (offset + 1_000.0) * t)).astype(np.complex64)
    out = tmp_path / "narrow.flac"
    stage = RecordStage(output_path=out, resample=(1, 48), center_hz=100e6 + offset)
    ctx, _, _ = _make_context()
    for start in range(0, n, 96_000):
        stage.process(_batch(iq[start : start + 96_000], sr), ctx)
    stage.close()
    loaded = load_iq(out)[1000:]  # past the filters' start
    assert np.sqrt(np.mean(np.abs(loaded) ** 2)) > 0.25  # the tone, not what's left of it
    peak_hz = np.argmax(np.abs(np.fft.fft(loaded))) * 50_000 / len(loaded)
    assert abs(peak_hz - 1_000.0) < 50
    assert flac.read_info(out).tags[flac.TAG_CENTER_FREQUENCY] == "100300000"


def test_a_failed_write_ends_the_recording_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """libFLAC failing (a full disk) on the writer thread: the recording reports what it
    saved and the error, once, and takes no more batches; nothing waits on the writer."""
    real_write = flac.FlacWriter.write
    calls: list[int] = []

    def failing_write(writer: flac.FlacWriter, pairs: np.ndarray) -> None:
        calls.append(len(pairs))
        if len(calls) > 1:
            raise OSError(28, "No space left on device")
        real_write(writer, pairs)

    monkeypatch.setattr(flac.FlacWriter, "write", failing_write)
    ctx, _, received = _make_context()
    stage = RecordStage(tmp_path / "full.flac")
    batch = _batch(np.full(5_000, 0.5 + 0.25j), 250e3)
    with pytest.raises(OSError):
        for _ in range(500):
            stage.process(batch, ctx)
            time.sleep(0.002)
    assert len(received) == 1
    assert received[0].path.endswith(".partial")
    # What the file holds, not what was handed to the writer before it failed.
    assert received[0].samples_written == len(flac.load(received[0].path)[0]) == 5_000
    stage.process(batch, ctx)  # closed: no second error
    assert len(received) == 1


def test_tone_preserved(tmp_path: Path) -> None:
    """A pure tone round-trips with its peak bin in the right place."""
    sr = 250_000.0
    n = 4096
    t = np.arange(n) / sr
    iq = 0.5 * np.exp(2j * np.pi * 12_500 * t).astype(np.complex64)
    out = tmp_path / "tone.flac"

    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, sr), ctx)
    stage.close()

    loaded = load_iq(out)
    assert len(loaded) == n
    spectrum = np.abs(np.fft.fft(loaded))
    peak_bin = int(np.argmax(spectrum))
    expected_bin = int(round(12_500 * n / sr))
    assert abs(peak_bin - expected_bin) <= 1


def test_integer_decimation_fast_path(tmp_path: Path) -> None:
    """2.4 MHz → 240 kHz: (1, 10), must use StreamingDecimFilter and produce ~n/10 samples."""
    sr = 2_400_000.0
    decim = 10
    n = sr_int = int(sr)  # 1 s of data
    # DC + small offset tone at 10 kHz (well inside post-decim Nyquist of 120 kHz).
    t = np.arange(n) / sr
    iq = (0.3 * np.exp(2j * np.pi * 10_000 * t)).astype(np.complex64)
    out = tmp_path / "decim.flac"

    stage = RecordStage(output_path=out, resample=(1, decim))
    ctx, _, _ = _make_context()
    # Feed in ~40 ms chunks to exercise cross-chunk filter state.
    chunk = 96_000
    for start in range(0, n, chunk):
        stage.process(_batch(iq[start : start + chunk], sr), ctx)
    stage.close()

    loaded = load_iq(out)
    # Allow small slack for filter warm-up tail handling.
    assert abs(len(loaded) - sr_int // decim) <= decim
    # The 10 kHz tone at device rate stays at 10 kHz at decimated rate.
    post_sr = sr / decim
    spectrum = np.abs(np.fft.fft(loaded))
    peak_bin = int(np.argmax(spectrum))
    expected_bin = int(round(10_000 * len(loaded) / post_sr))
    assert abs(peak_bin - expected_bin) <= 2


def test_rational_resample(tmp_path: Path) -> None:
    """2.4 MHz → 64 kHz (up=2, down=75). Must produce ~n * 2/75 samples."""
    sr = 2_400_000.0
    up, down = 2, 75
    n = int(sr)  # 1 s
    t = np.arange(n) / sr
    iq = (0.3 * np.exp(2j * np.pi * 5_000 * t)).astype(np.complex64)
    out = tmp_path / "rational.flac"

    stage = RecordStage(output_path=out, resample=(up, down))
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, sr), ctx)
    stage.close()

    loaded = load_iq(out)
    expected_n = n * up // down
    # Polyphase warm-up plus internal timing → small slack.
    assert abs(len(loaded) - expected_n) < 200


def test_max_samples_finalizes_and_publishes_event(tmp_path: Path) -> None:
    """Duration mode: stage closes once max_samples is reached and publishes an event."""
    sr = 250_000.0
    n = 50_000
    rng = np.random.default_rng(42)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.2
    out = tmp_path / "duration.flac"

    max_samples = 30_000
    stage = RecordStage(output_path=out, max_samples=max_samples)
    ctx, _, events = _make_context()

    # Feed more than max_samples worth of data in two chunks.
    stage.process(_batch(iq[:20_000], sr), ctx)
    stage.process(_batch(iq[20_000:], sr), ctx)

    loaded = load_iq(out)
    assert len(loaded) == max_samples
    assert len(events) == 1
    assert events[0].samples_written == max_samples
    assert events[0].path == str(out)


def test_max_samples_in_output_units_with_decimation(tmp_path: Path) -> None:
    """max_samples counts OUTPUT samples so the cap works under downsampling.

    Regression guard: earlier the command computed max_samples from the device
    rate while the stage counted post-resample samples - the cap was never hit
    and the file kept growing. max_samples must be in output units.
    """
    sr = 2_400_000.0
    decim = 10
    out_rate = sr / decim
    # Feed ~5 seconds of device-rate data; cap the stage at 2 seconds of OUTPUT.
    n = int(sr) * 5
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.2
    out = tmp_path / "capped.flac"
    max_output_samples = int(out_rate * 2)  # 2 seconds of output

    stage = RecordStage(output_path=out, resample=(1, decim), max_samples=max_output_samples)
    ctx, _, events = _make_context()
    chunk = 96_000
    for start in range(0, n, chunk):
        stage.process(_batch(iq[start : start + chunk], sr), ctx)

    assert len(events) == 1
    loaded = load_iq(out)
    # Small slack for the cut-at-boundary step in the stage.
    assert abs(len(loaded) - max_output_samples) < decim


def test_close_is_idempotent(tmp_path: Path) -> None:
    """close() can be called twice without raising."""
    stage = RecordStage(output_path=tmp_path / "idem.flac")
    stage.close()  # never opened - should no-op
    stage.close()  # second close - must not raise


def test_no_iq_samples_is_noop(tmp_path: Path) -> None:
    """Batches without iq_samples pass through untouched and don't open the file."""
    out = tmp_path / "none.flac"
    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()
    stage.process(SamplesBatch(sample_rate=250_000.0), ctx)
    stage.close()
    # Nothing should have been created at all.
    assert list(tmp_path.iterdir()) == []


def test_engine_removes_pipeline_on_recording_finished_event(tmp_path: Path) -> None:
    """Layering check: the engine itself tears down the recording pipeline on auto-stop.

    Proves that pipeline lifecycle lives in core - no TUI needed. This is the
    scenario a headless/core-only caller relies on, and it would have been
    broken if the remove_pipeline call lived only in the TUI event handler.
    """
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.add_pipeline(
            "rec",
            "recording",
            PipelineConfig(
                stages=(StageType.RECORD,),
                record_path=str(tmp_path / "headless.flac"),
                record_max_samples=1000,
            ),
        )
        assert "recording" in engine.get_device("rec").config.pipelines

        engine.event_bus.publish(
            RecordingFinishedEvent(
                source_id="record_rec",
                device_id="rec",
                pipeline_name="recording",
                path=str(tmp_path / "headless.flac"),
                samples_written=1000,
            )
        )

        assert "recording" not in engine.get_device("rec").config.pipelines
    finally:
        engine.shutdown(timeout=2.0)


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class _HeldWrites:
    """Holds RecordStage's writes until released, and notes each close: its thread and
    whether a write was under way."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.in_write = threading.Event()
        self.release = threading.Event()
        self.writing = threading.Event()
        self.closes: list[tuple[str, bool]] = []
        real_write = RecordStage._write_samples
        real_close = RecordStage.close

        def held_write(stage: RecordStage, samples: np.ndarray) -> None:
            self.writing.set()
            self.in_write.set()
            self.release.wait(5.0)
            try:
                real_write(stage, samples)
            finally:
                self.writing.clear()

        def watched_close(stage: RecordStage) -> None:
            self.closes.append((threading.current_thread().name, self.writing.is_set()))
            real_close(stage)

        monkeypatch.setattr(RecordStage, "_write_samples", held_write)
        monkeypatch.setattr(RecordStage, "close", watched_close)


def _recording(path: Path, **kwargs: object) -> PipelineConfig:
    return PipelineConfig(stages=(StageType.RECORD,), record_path=str(path), **kwargs)  # type: ignore[arg-type]


def test_a_recording_stopped_mid_write_is_closed_by_the_pipeline_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`record stop` runs on the console's thread. It closed the recording's compressor
    and file right there, while the pipeline worker could be inside a write to them: two
    threads in one zstd context, and tsdr segfaulted (12 MS/s, 2026-10-16). The worker now
    closes a stage the config dropped, between batches."""
    held = _HeldWrites(monkeypatch)
    path = tmp_path / "stopped.flac"
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.start_device("rec")
        engine.add_pipeline("rec", "recording", _recording(path))
        assert held.in_write.wait(5.0)
        engine.remove_pipeline("rec", "recording")  # the console's thread, mid-write
        held.release.set()
        assert _wait(lambda: bool(held.closes)), "the dropped recording was never closed"
        thread, during_write = held.closes[0]
        assert not during_write
        assert thread != threading.current_thread().name
    finally:
        held.release.set()
        engine.shutdown(timeout=2.0)
    assert path.exists()
    assert len(load_iq(str(path))) > 0


def test_a_stop_that_gives_up_on_the_worker_leaves_the_recording_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop whose join times out (a long write) abandons the pipeline worker mid-batch.
    The stop then closed the stages retired meanwhile on its own thread, under that
    write. The worker closes them at its exit now, and the device can't start again
    beside it until then."""
    held = _HeldWrites(monkeypatch)
    path = tmp_path / "abandoned.flac"
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.start_device("rec")
        engine.add_pipeline("rec", "recording", _recording(path))
        assert held.in_write.wait(5.0)
        engine.remove_pipeline("rec", "recording")
        context = engine.devices["rec"]
        context.stop(engine.worker_runner, timeout=0.2)  # gives up on the held worker
        assert held.closes == []
        assert context.state is DeviceState.STOPPED
        assert context.worker_owns_stages
        with pytest.raises(SDRException):
            engine.start_device("rec")
        held.release.set()
        assert _wait(lambda: bool(held.closes)), "the abandoned worker never closed it"
        thread, during_write = held.closes[0]
        assert not during_write
        assert thread != threading.current_thread().name
        assert _wait(lambda: not context.worker_owns_stages)
    finally:
        held.release.set()
        engine.shutdown(timeout=2.0)
    assert len(load_iq(str(path))) > 0


def _finish(engine: SDREngine, stage: RecordStage) -> None:
    """A `record for` stage's finished event, naming that stage, as its close publishes it."""
    engine.event_bus.publish(
        RecordingFinishedEvent(
            source_id="record_rec",
            device_id="rec",
            pipeline_name="recording",
            path=str(stage.path),
            samples_written=1000,
            recording=stage,
        )
    )


def _recording_stage(engine: SDREngine) -> RecordStage:
    stage = engine.devices["rec"].pipelines["recording"].stages[0]
    assert isinstance(stage, RecordStage)
    return stage


def test_a_finished_recording_leaves_the_one_started_after_it(tmp_path: Path) -> None:
    """A stopped `record for` can finish its last batch, and publish its finished event,
    after a new recording has started under the same pipeline name. The engine removed
    the pipeline by name, so the new recording went."""
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.add_pipeline(
            "rec", "recording", _recording(tmp_path / "a.flac", record_max_samples=1000)
        )
        first = _recording_stage(engine)
        engine.remove_pipeline("rec", "recording")
        engine.add_pipeline("rec", "recording", _recording(tmp_path / "b.flac"))
        second = _recording_stage(engine)
        _finish(engine, first)
        assert _recording_stage(engine) is second
        _finish(engine, second)
        assert "recording" not in engine.get_device("rec").config.pipelines
    finally:
        engine.shutdown(timeout=2.0)


def test_a_recording_kept_through_a_pipeline_edit_still_ends(tmp_path: Path) -> None:
    """An edit of the recording pipeline (`pipeline add agc --name recording`) makes a new
    config but keeps the RecordStage at its place. A finished event naming the old config
    was taken for a replaced recording's, and the finished recording stayed configured."""
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        config = _recording(tmp_path / "kept.flac", record_max_samples=1000)
        engine.add_pipeline("rec", "recording", config)
        stage = _recording_stage(engine)
        pipelines = dict(engine.get_device("rec").config.pipelines)
        pipelines["recording"] = replace(config, stages=(StageType.RECORD, StageType.AGC))
        engine.update_device_config("rec", pipelines=MappingProxyType(pipelines))
        assert _recording_stage(engine) is stage
        _finish(engine, stage)
        assert "recording" not in engine.get_device("rec").config.pipelines
    finally:
        engine.shutdown(timeout=2.0)


@pytest.mark.parametrize("how", ["shutdown", "remove"])
def test_a_recording_still_on_is_finalized_when_its_device_goes(tmp_path: Path, how: str) -> None:
    """Quitting, or removing the device, left a running recording as a `.partial` that
    never got its name. A plain stop keeps it (it carries on at the next start)."""
    path = tmp_path / "left.flac"

    def partials() -> list[Path]:
        return list(tmp_path.glob("left.flac.*.partial"))

    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.start_device("rec")
        engine.add_pipeline("rec", "recording", _recording(path))
        assert _wait(lambda: _recording_stage(engine).samples_written > 0)
        if how == "remove":
            engine.stop_device("rec")
            assert partials()  # a stop keeps the recording
            engine.remove_device("rec")
    finally:
        engine.shutdown(timeout=2.0)
    assert partials() == []
    assert len(load_iq(str(path))) > 0


def test_two_recordings_to_one_name_both_survive(tmp_path: Path) -> None:
    """Two recordings with one name (one `--output`, or two started in the same second)
    wrote one `.partial`: each truncated and overwrote the other, and the first finished
    file was rewritten through the second's open file. Each recording writes its own file
    now, and the later to finish takes the next free name."""
    path = tmp_path / "same.flac"
    context, _bus, _received = _make_context()
    first, second = RecordStage(path), RecordStage(path)
    first.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    second.process(_batch(np.full(200, -0.5 - 0.25j), 250e3), context)
    first.close()
    second.close()
    assert first.path == path
    assert second.path == tmp_path / "same-2.flac"
    assert len(load_iq(str(first.path))) == 100
    assert len(load_iq(str(second.path))) == 200
    assert list(tmp_path.glob("*.partial")) == []


def test_a_stopped_recording_reports_the_name_it_got(tmp_path: Path) -> None:
    """`record stop` reported the name the recording was started with, though it may be
    saved under the next free one. Every recording reports its file when it closes."""
    path = tmp_path / "reported.flac"
    path.write_bytes(b"another recording")
    context, _bus, received = _make_context()
    stage = RecordStage(path)
    stage.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    stage.close()  # as the pipeline worker does after `record stop`
    assert [e.path for e in received] == [str(tmp_path / "reported-2.flac")]


def test_a_recording_that_fails_to_finish_still_reports_where_its_data_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `record for` whose finishing failed (a full disk, a refused rename) published no
    finished event: its pipeline stayed configured ("Already recording") and nobody heard
    where the data was. The close reports it anyway, and the error goes on up."""
    context, _bus, received = _make_context()
    stage = RecordStage(tmp_path / "full.flac", max_samples=100)

    def no_room(partial: Path, final: Path) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(record_stage, "_promote", no_room)
    with pytest.raises(OSError):
        stage.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    assert len(received) == 1
    assert received[0].path.endswith(".partial")
    assert Path(received[0].path).exists()


def test_a_worker_slow_to_set_up_still_owns_the_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline worker's thread can be slow to reach its setup. Claimed there, the
    stages looked unowned to a stop that timed out meanwhile: a restart ran a second
    worker, and the first one's teardown then gave up the second's claim. A worker
    claims them when it is made, and only it gives them up."""
    gate = threading.Event()
    real_setup = PipelineWorker.setup

    def held_setup(self: PipelineWorker, context: object) -> None:
        gate.wait(5.0)
        real_setup(self, context)  # type: ignore[arg-type]

    monkeypatch.setattr(PipelineWorker, "setup", held_setup)
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.start_device("rec")
        context = engine.devices["rec"]
        context.stop(engine.worker_runner, timeout=0.2)  # the worker is still before its setup
        assert context.state is DeviceState.STOPPED
        assert context.worker_owns_stages
        with pytest.raises(SDRException):
            engine.start_device("rec")
        gate.set()
        assert _wait(lambda: not context.worker_owns_stages)
        engine.start_device("rec")
        assert context.worker_owns_stages
    finally:
        gate.set()
        engine.shutdown(timeout=2.0)


def test_a_finished_recording_never_replaces_a_file_already_there(tmp_path: Path) -> None:
    """A finished recording's rename replaced whatever had its name: a recording that
    finished into it between the next command's name check and its reservation, or an
    existing `--output`. It takes the next free name instead."""
    path = tmp_path / "taken.flac"
    path.write_bytes(b"another recording")
    context, _bus, _received = _make_context()
    stage = RecordStage(path)
    stage.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    stage.close()
    assert path.read_bytes() == b"another recording"
    assert stage.path == tmp_path / "taken-2.flac"
    assert len(load_iq(str(stage.path))) == 100


def test_the_engine_starts_no_device_after_its_shutdown() -> None:
    """Shutdown closes every stage (finalizing recordings); a device started after it ran
    without its pipelines. Shutdown is final."""
    engine = SDREngine()
    engine.add_device("rec", "mock", MockParams(), DeviceConfig())
    engine.shutdown(timeout=2.0)
    context = engine.devices["rec"]
    try:
        with pytest.raises(SDRException):
            engine.start_device("rec")
    finally:  # a device started anyway would keep its threads, and the test run, alive
        if context.state is not DeviceState.STOPPED:
            context.stop(engine.worker_runner, timeout=2.0)


def test_the_writer_queue_is_bounded_by_bytes() -> None:
    """The queue to the writer thread held a number of batches set by the first batch's
    size: a later, larger batch size (`config --fps`) let it hold 5x its 64 MB. It counts
    bytes: a put waits while the queued batches reach the limit; any one batch gets in."""
    pending = record_stage._WriterQueue(100)
    assert not pending.put(np.zeros(25, dtype=np.int32))  # 100 bytes: over the limit alone
    second = threading.Thread(target=pending.put, args=(np.zeros(5, dtype=np.int32),))
    second.start()
    second.join(0.2)
    assert second.is_alive()  # waits for the writer
    assert pending.get() is not None
    second.join(2.0)
    assert not second.is_alive()


def test_a_recording_that_cant_start_leaves_no_file_and_no_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording whose writer thread couldn't start kept libFLAC's encoder and its open
    `.partial`; one that couldn't open stayed configured, refusing `record start` ("Already
    recording") until `record stop`. Either now lets go of the file and ends the recording."""

    class NoThread(threading.Thread):
        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(record_stage.threading, "Thread", NoThread)
    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.add_pipeline("rec", "recording", _recording(tmp_path / "x.flac"))
        context = PipelineContext(
            device_context=engine.devices["rec"], event_bus=engine.event_bus, config=SDRConfig()
        )
        with pytest.raises(RuntimeError):
            _recording_stage(engine).process(_batch(np.zeros(100), 250e3), context)
        assert "recording" not in engine.get_device("rec").config.pipelines
    finally:
        engine.shutdown(timeout=2.0)
    assert list(tmp_path.iterdir()) == []


def test_conversion_saturates_huge_values() -> None:
    """A huge float or an infinity became an integer before the clip and wrapped to -32768."""
    x = np.array([1e20, np.inf, -np.inf, np.nan, 2.0, -2.0, 0.5], dtype=np.float32).astype(
        np.complex64
    )
    pairs = iq_to_int_pairs(x, 32768.0, -32768, 32767).reshape(-1, 2)[:, 0]
    assert pairs.tolist() == [32767, 32767, -32768, -32768, 32767, -32768, 16384]


def test_a_rational_rate_keeps_only_the_band_around_the_dial(tmp_path: Path) -> None:
    """A rate that doesn't divide the capture's (2.4 MS/s to 44.1 kS/s) went through one
    65-tap polyphase filter: a signal 100 kHz off the dial came through at full strength,
    and half the samples were zeros. A whole decimation comes first now."""
    sr = 2_400_000.0
    n = int(sr)
    t = np.arange(n) / sr
    channel = 0.3 * np.exp(2j * np.pi * 1_000.0 * t)
    beside = 0.3 * np.exp(2j * np.pi * 100_000.0 * t)
    levels = []
    for iq in (channel, beside):
        out = tmp_path / f"r{len(levels)}.flac"
        stage = RecordStage(output_path=out, resample=(147, 8000), center_hz=100e6)
        ctx, _, _ = _make_context()
        for start in range(0, n, 96_000):
            stage.process(_batch(iq[start : start + 96_000], sr), ctx)
        stage.close()
        loaded = load_iq(out)[2000:]
        levels.append(np.sqrt(np.mean(np.abs(loaded) ** 2)))
    assert levels[0] > 0.25
    assert levels[1] < 0.3 * 10 ** (-50 / 20)
