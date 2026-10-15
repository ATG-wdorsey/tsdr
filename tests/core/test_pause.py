"""Pause (Space): the radio keeps streaming and its samples are dropped; SEARCH stays on,
frozen; a resume carries on without a restart."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tsdr.core import search_control
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import DeviceStateChangedEvent, Event, FFTUpdateEvent
from tsdr.core.memories import init_memory_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.search_stage import SearchStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import ATTACK_S
from tsdr.core.search_control import start_search
from tsdr.core.tuning_state import init_tuning_state
from tsdr.devices import MockParams, mock
from tsdr.devices.iq_file import IQFileParams

RATE = 10e6
DIAL = 146.52e6


def _noise_file(tmp_path: Path, rate: float = 250e3, seconds: float = 4.0) -> Path:
    rng = np.random.default_rng(0)
    n = int(rate * seconds)
    iq = ((rng.normal(size=n) + 1j * rng.normal(size=n)) * 1e-3).astype(np.complex64)
    path = tmp_path / "noise.cf32"
    iq.tofile(path)
    return path


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_paused_the_radio_streams_on_and_nothing_is_processed(tmp_path: Path) -> None:
    engine = SDREngine()
    engine.add_device(
        "file",
        "iq-file",
        IQFileParams(path=str(_noise_file(tmp_path))),
        DeviceConfig(sample_rate=250e3),
    )
    frames: list[float] = []
    states: list[DeviceStateChangedEvent] = []
    engine.event_bus.subscribe(FFTUpdateEvent, lambda _e: frames.append(time.monotonic()))

    def on_state(event: Event) -> None:
        if isinstance(event, DeviceStateChangedEvent):
            states.append(event)

    engine.event_bus.subscribe(DeviceStateChangedEvent, on_state)
    context = engine.devices["file"]
    try:
        engine.start_device("file")
        _wait(lambda: context.state == DeviceState.RUNNING and len(frames) > 3)
        assert frames

        engine.set_paused("file", True)
        time.sleep(0.3)  # what was in flight at the pause
        read_before, frames_before = context.total_samples_read, len(frames)
        time.sleep(0.6)
        assert len(frames) == frames_before  # nothing shown
        assert context.total_samples_read > read_before  # the radio is still read
        assert context.state == DeviceState.RUNNING

        engine.set_paused("file", False)
        _wait(lambda: len(frames) > frames_before + 3, timeout=2.0)
        assert len(frames) > frames_before + 3
        assert context.paused_seconds == pytest.approx(0.9, abs=0.3)
    finally:
        engine.stop_device("file")
        engine.shutdown(timeout=5.0)
    assert [(s.running, s.paused) for s in states][-3:] == [
        (True, True),
        (True, False),
        (False, False),
    ]
    assert not context.paused  # a stop ends a pause


@pytest.fixture
def engine() -> SDREngine:
    engine = SDREngine()
    engine.add_device("m", "mock", MockParams(), DeviceConfig())
    yield engine
    engine.devices["m"].state = DeviceState.STOPPED
    engine._audio_workers.clear()
    engine.shutdown(timeout=2.0)


def test_batches_captured_during_a_pause_are_dropped_after_the_resume(engine: SDREngine) -> None:
    context = engine.devices["m"]
    context.resumed_at_utc = 1000.0
    assert context.drops(999.9)
    assert not context.drops(1000.1)
    assert not context.drops(0.0)  # not stamped (tests)
    context.paused = True
    assert context.drops(1000.1)


def test_only_a_running_radio_pauses(engine: SDREngine) -> None:
    with pytest.raises(SDRException, match="not running"):
        engine.set_paused("m", True)


def test_a_pause_silences_at_once_and_a_resume_drops_what_was_queued(engine: SDREngine) -> None:
    engine.devices["m"].state = DeviceState.RUNNING
    flushes: list[str] = []

    class _Audio:
        def request_flush(self) -> None:
            flushes.append("flush")

    engine._audio_workers["m"] = _Audio()  # type: ignore[assignment]
    engine.set_paused("m", True)
    engine.set_paused("m", True)  # no change, no flush
    engine.set_paused("m", False)
    assert flushes == ["flush", "flush"]


# --- SEARCH's clock -----------------------------------------------------------


@pytest.fixture
def search_engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    caps = replace(mock._MOCK_CAPABILITIES, zero_if=True, frequency_range=(70e6, 6e9))
    monkeypatch.setattr(mock, "_MOCK_CAPABILITIES", caps)
    init_memory_store()
    init_tuning_state()
    monkeypatch.setattr(search_control, "_settings", {})
    engine = SDREngine()
    engine.add_device(
        "pluto",
        "mock",
        MockParams(),
        DeviceConfig(
            tuned_frequency=DIAL, center_frequency=DIAL + 1e6, sample_rate=RATE, fft_size=8192
        ),
    )
    # Set running after the demodulator, so no audio output starts (as test_search_engine).
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.devices["pluto"].state = DeviceState.RUNNING
    yield engine
    engine.devices["pluto"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _batch(config: DeviceConfig, t: float, tones: dict[float, float]) -> SamplesBatch:
    n = config.fft_size
    spectrum = np.full(n, -120.0, dtype=np.float32)
    for f, level in tones.items():
        spectrum[int(round((f - config.center_frequency) / RATE * n + n / 2))] = level
    return SamplesBatch(
        instant_spectrum=spectrum,
        spectrum=spectrum,
        center_frequency=config.center_frequency,
        sample_rate=RATE,
        capture_utc_s=t,
    )


def test_a_pause_freezes_searchs_hold(
    search_engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hold running out when the radio was paused runs out after the resume, not at it."""
    engine = search_engine
    start_search("pluto", trigger_db=-40.0, hold_s=0.5)
    context = engine.devices["pluto"]
    first, second = 147.0e6, 145.0e6
    first_on = [True]  # the squelch opens on the first channel while it transmits
    monkeypatch.setattr(
        type(context),
        "demod_status",
        property(
            lambda c: DemodStatus(squelch_open=first_on[0] and c.config.tuned_frequency == first)
        ),
    )
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    t = [1000.0]

    def feed(seconds: float, tones: dict[float, float]) -> None:
        for _ in range(round(seconds / 0.05)):
            stage.on_config_change(context.config)
            stage.process(_batch(context.config, t[0], tones), pipeline_context)
            t[0] += 0.05

    feed(ATTACK_S + 0.15, {first: -30.0})
    assert context.config.tuned_frequency == first
    first_on[0] = False
    feed(0.25, {second: -30.0})  # the first has gone quiet; its hold is running
    assert context.config.tuned_frequency == first

    # Ten seconds paused: the capture clock moves on, SEARCH's does not.
    t[0] += 10.0
    context.paused_seconds += 10.0
    feed(0.1, {second: -30.0})
    assert context.config.tuned_frequency == first  # 0.35 s of 0.5 s held
    feed(0.4, {second: -30.0})
    assert context.config.tuned_frequency == second
