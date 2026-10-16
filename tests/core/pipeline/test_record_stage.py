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


def test_passthrough_round_trip(tmp_path: Path) -> None:
    """No resample: cu8 round-trip preserves samples within quantization error."""
    rng = np.random.default_rng(0)
    n = 10_000
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.3
    out = tmp_path / "test.cu8.zst"

    stage = RecordStage(output_path=out)
    ctx, _, _ = _make_context()

    stage.process(_batch(iq[: n // 2], 250_000.0), ctx)
    stage.process(_batch(iq[n // 2 :], 250_000.0), ctx)
    stage.close()

    assert out.exists()
    loaded = load_iq(out)
    assert len(loaded) == n
    # cu8 has ~1/127.5 quantization on each component → RMS error ≲ 0.005 for unit-scale.
    assert np.mean(np.abs(loaded - iq)) < 0.01


def test_complex64_preserves_subcu8_lsb(tmp_path: Path) -> None:
    """cf32 format preserves low-amplitude signal that cu8 would crush to 127/128.

    Mirrors the Airspy HF+ / SpyServer case: a real but tiny-amplitude signal is
    below one cu8 LSB, so 8-bit recording flatlines it. float32 is lossless.
    """
    rng = np.random.default_rng(0)
    n = 8_000
    # Far below one cu8 LSB (1/255): every sample rounds to the 127/128 boundary.
    iq = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.0005).astype(np.complex64)

    cf32_out = tmp_path / "hi.cf32.zst"
    stage = RecordStage(output_path=cf32_out, sample_format=SampleFormat.COMPLEX64)
    ctx, _, _ = _make_context()
    stage.process(_batch(iq, 250_000.0), ctx)
    stage.close()
    loaded = load_iq(cf32_out)
    assert len(loaded) == n
    np.testing.assert_array_equal(loaded, iq)  # float32 -> float32: bit-exact

    # The same data through cu8 collapses to the two center codes (information lost).
    cu8_out = tmp_path / "lo.cu8.zst"
    stage2 = RecordStage(output_path=cu8_out, sample_format=SampleFormat.UINT8_IQ)
    stage2.process(_batch(iq, 250_000.0), ctx)
    stage2.close()
    crushed = load_iq(cu8_out)
    # Collapsed to the two center codes (127/128): only one bit of amplitude survives.
    assert len(np.unique(crushed.real)) <= 2
    assert len(np.unique(crushed.imag)) <= 2


def test_tone_preserved(tmp_path: Path) -> None:
    """A pure tone round-trips with its peak bin in the right place."""
    sr = 250_000.0
    n = 4096
    t = np.arange(n) / sr
    iq = 0.5 * np.exp(2j * np.pi * 12_500 * t).astype(np.complex64)
    out = tmp_path / "tone.cu8.zst"

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
    out = tmp_path / "decim.cu8.zst"

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
    out = tmp_path / "rational.cu8.zst"

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
    out = tmp_path / "duration.cu8.zst"

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
    out = tmp_path / "capped.cu8.zst"
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
    stage = RecordStage(output_path=tmp_path / "idem.cu8.zst")
    stage.close()  # never opened - should no-op
    stage.close()  # second close - must not raise


def test_no_iq_samples_is_noop(tmp_path: Path) -> None:
    """Batches without iq_samples pass through untouched and don't open the file."""
    out = tmp_path / "none.cu8.zst"
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
                record_path=str(tmp_path / "headless.cu8.zst"),
                record_max_samples=1000,
            ),
        )
        assert "recording" in engine.get_device("rec").config.pipelines

        engine.event_bus.publish(
            RecordingFinishedEvent(
                source_id="record_rec",
                device_id="rec",
                pipeline_name="recording",
                path=str(tmp_path / "headless.cu8.zst"),
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
    path = tmp_path / "stopped.cu8.zst"
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
    path = tmp_path / "abandoned.cu8.zst"
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
            "rec", "recording", _recording(tmp_path / "a.cu8.zst", record_max_samples=1000)
        )
        first = _recording_stage(engine)
        engine.remove_pipeline("rec", "recording")
        engine.add_pipeline("rec", "recording", _recording(tmp_path / "b.cu8.zst"))
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
        config = _recording(tmp_path / "kept.cu8.zst", record_max_samples=1000)
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
    path = tmp_path / "left.cu8.zst"

    def partials() -> list[Path]:
        return list(tmp_path.glob("left.cu8.zst.*.partial"))

    engine = SDREngine()
    try:
        engine.add_device("rec", "mock", MockParams(), DeviceConfig())
        engine.start_device("rec")
        engine.add_pipeline("rec", "recording", _recording(path))
        assert _wait(lambda: any(p.stat().st_size > 0 for p in partials()))
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
    path = tmp_path / "same.cu8.zst"
    context, _bus, _received = _make_context()
    first, second = RecordStage(path), RecordStage(path)
    first.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    second.process(_batch(np.full(200, -0.5 - 0.25j), 250e3), context)
    first.close()
    second.close()
    assert first.path == path
    assert second.path == tmp_path / "same-2.cu8.zst"
    assert len(load_iq(str(first.path))) == 100
    assert len(load_iq(str(second.path))) == 200
    assert list(tmp_path.glob("*.partial")) == []


def test_a_stopped_recording_reports_the_name_it_got(tmp_path: Path) -> None:
    """`record stop` reported the name the recording was started with, though it may be
    saved under the next free one. Every recording reports its file when it closes."""
    path = tmp_path / "reported.cu8.zst"
    path.write_bytes(b"another recording")
    context, _bus, received = _make_context()
    stage = RecordStage(path)
    stage.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    stage.close()  # as the pipeline worker does after `record stop`
    assert [e.path for e in received] == [str(tmp_path / "reported-2.cu8.zst")]


def test_a_recording_that_fails_to_finish_still_reports_where_its_data_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `record for` whose finishing failed (a full disk, a refused rename) published no
    finished event: its pipeline stayed configured ("Already recording") and nobody heard
    where the data was. The close reports it anyway, and the error goes on up."""
    context, _bus, received = _make_context()
    stage = RecordStage(tmp_path / "full.cu8.zst", max_samples=100)

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
    path = tmp_path / "taken.cu8.zst"
    path.write_bytes(b"another recording")
    context, _bus, _received = _make_context()
    stage = RecordStage(path)
    stage.process(_batch(np.full(100, 0.5 + 0.25j), 250e3), context)
    stage.close()
    assert path.read_bytes() == b"another recording"
    assert stage.path == tmp_path / "taken-2.cu8.zst"
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
