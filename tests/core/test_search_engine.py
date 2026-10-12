"""SEARCH through the engine: starting it, its jumps, what stops it, lockouts,
and the pipeline stage that measures the channels."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from tsdr.core import search_control
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import (
    ConfigChangedEvent,
    Event,
    MemoriesChangedEvent,
    SearchStatusEvent,
    SearchTuneEvent,
)
from tsdr.core.memories import get_memory_store, init_memory_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.search_stage import SearchStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import LOCKOUT_TAG
from tsdr.core.sdr.tune_policy import fitting_offset
from tsdr.core.search_control import (
    clear_session_lockouts,
    lock_out_channel,
    set_search_edge,
    set_search_step,
    start_search,
    stop_search,
)
from tsdr.core.tuning_state import init_tuning_state
from tsdr.devices import MockParams, mock
from tsdr.devices.iq_file import IQFileParams

RATE = 10e6
DIAL = 146.52e6


@pytest.fixture(autouse=True)
def _stores(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = replace(mock._MOCK_CAPABILITIES, zero_if=True, frequency_range=(70e6, 6e9))
    monkeypatch.setattr(mock, "_MOCK_CAPABILITIES", caps)
    init_memory_store()
    init_tuning_state()
    monkeypatch.setattr(search_control, "_settings", {})


@pytest.fixture
def engine() -> SDREngine:
    engine = SDREngine()
    config = DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=RATE)
    engine.add_device("pluto", "mock", MockParams(), config)
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.devices["pluto"].state = DeviceState.RUNNING
    yield engine
    engine.devices["pluto"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _config(engine: SDREngine) -> DeviceConfig:
    return engine.devices["pluto"].config


def test_start_switches_to_free_tuning_and_turns_the_squelch_on(engine: SDREngine) -> None:
    center = _config(engine).center_frequency
    search = start_search("pluto")
    config = _config(engine)
    assert config.search == search
    assert config.tuning_mode == "free"
    assert config.center_frequency == center
    assert config.pipelines["audio"].squelch_enabled
    half = fitting_offset(RATE, 12_500.0)
    assert (search.low_hz, search.high_hz) == (center - half, center + half)
    assert search.trigger_db is None  # follows the squelch
    assert search.step_hz == 12_500.0


def test_start_refuses_what_search_cannot_do(engine: SDREngine) -> None:
    with pytest.raises(SDRException, match="above the squelch"):
        start_search("pluto", trigger_db=-60.0)  # squelch default -50
    with pytest.raises(SDRException, match="within"):
        start_search("pluto", low_hz=DIAL - 6e6, high_hz=DIAL)
    engine.set_audio_demod("pluto", DemodSpec(mode="WFM"))
    with pytest.raises(SDRException, match="narrowband"):
        start_search("pluto")
    engine.remove_pipeline("pluto", "audio")
    with pytest.raises(SDRException, match="demodulator"):
        start_search("pluto")
    assert _config(engine).search is None


def test_start_needs_a_running_device(engine: SDREngine) -> None:
    engine.devices["pluto"].state = DeviceState.STOPPED
    with pytest.raises(SDRException, match="Start"):
        start_search("pluto")


def test_a_jump_moves_only_the_dial(engine: SDREngine) -> None:
    search = start_search("pluto")
    center = _config(engine).center_frequency
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=147.0e6, run=search.run)
    )
    config = _config(engine)
    assert config.tuned_frequency == 147.0e6
    assert config.center_frequency == center
    assert config.search == search


def test_a_jump_that_would_move_the_hardware_is_dropped(engine: SDREngine) -> None:
    search = start_search("pluto")
    center = _config(engine).center_frequency
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=center + 10e3, run=search.run)
    )
    assert _config(engine).tuned_frequency == DIAL
    assert _config(engine).center_frequency == center


def test_a_jump_from_an_earlier_run_is_dropped(engine: SDREngine) -> None:
    old = start_search("pluto")
    start_search("pluto")
    engine.event_bus.publish(SearchTuneEvent(device_id="pluto", frequency_hz=147.0e6, run=old.run))
    assert _config(engine).tuned_frequency == DIAL


@pytest.mark.parametrize(
    "change",
    [
        {"tuned_frequency": 147.0e6},
        {"sample_rate": 5e6},
        {"tuning_mode": "center"},
        {"lo_offset": "above"},
    ],
    ids=["manual-tune", "sample-rate", "center-tuning", "lo-moved"],
)
def test_changes_that_end_search(engine: SDREngine, change: dict) -> None:
    start_search("pluto")
    engine.update_device_config("pluto", **change)
    assert _config(engine).search is None


def test_switching_to_a_wideband_or_no_demodulator_ends_search(engine: SDREngine) -> None:
    start_search("pluto")
    engine.set_audio_demod("pluto", DemodSpec(mode="WFM"))
    assert _config(engine).search is None
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    start_search("pluto")
    engine.remove_pipeline("pluto", "audio")
    assert _config(engine).search is None


def test_squelch_and_bandwidth_changes_keep_searching(engine: SDREngine) -> None:
    search = start_search("pluto")
    engine.update_squelch("pluto", "audio", threshold_db=-55.0)
    engine.update_device_config("pluto", channel_bandwidth=10_000.0)
    assert _config(engine).search == search


def test_stop_leaves_the_dial_and_free_tuning(engine: SDREngine) -> None:
    search = start_search("pluto")
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=147.0e6, run=search.run)
    )
    assert stop_search("pluto")
    assert not stop_search("pluto")
    config = _config(engine)
    assert config.search is None
    assert config.tuned_frequency == 147.0e6
    assert config.tuning_mode == "free"


def test_restart_keeps_the_last_settings(engine: SDREngine) -> None:
    start_search(
        "pluto", low_hz=145e6, high_hz=148e6, trigger_db=-40.0, hold_s=3.0, max_dwell_s=20.0
    )
    stop_search("pluto")
    search = start_search("pluto")
    assert (search.low_hz, search.high_hz, search.trigger_db) == (145e6, 148e6, -40.0)
    assert (search.hold_s, search.max_dwell_s) == (3.0, 20.0)
    assert start_search("pluto", max_dwell_s=0).max_dwell_s is None


def test_session_lockout(engine: SDREngine) -> None:
    start_search("pluto")
    assert lock_out_channel("pluto") == DIAL
    assert _config(engine).search.lockouts == frozenset({DIAL})
    stop_search("pluto")
    assert start_search("pluto").lockouts == frozenset({DIAL})
    assert clear_session_lockouts("pluto") == 1
    assert _config(engine).search.lockouts == frozenset()


def test_permanent_lockout_is_a_tagged_memory(engine: SDREngine) -> None:
    start_search("pluto")
    lock_out_channel("pluto", permanent=True)
    lock_out_channel("pluto", permanent=True)
    memories = get_memory_store().find_by_tag(LOCKOUT_TAG)
    assert [m.frequency for m in memories] == [int(DIAL)]
    assert memories[0].audio_spec.mode == "NFM"
    assert _config(engine).search.memory_lockouts == frozenset({DIAL})

    store = get_memory_store()
    store.remove(memories[0].id)
    engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
    assert _config(engine).search.memory_lockouts == frozenset()


def test_a_step_change_follows_into_search(engine: SDREngine) -> None:
    start_search("pluto")
    set_search_step("pluto", 5_000.0)
    assert _config(engine).search.step_hz == 5_000.0


# --- the stage --------------------------------------------------------------


def _batch(
    config: DeviceConfig, t: float, tones: dict[float, float], rate: float = RATE
) -> SamplesBatch:
    """An unaveraged spectrum with a noise floor and carriers at given dBFS;
    `rate` is the rate the device reports."""
    n = config.fft_size
    spectrum = np.full(n, -120.0, dtype=np.float32)
    for f, level in tones.items():
        k = int(round((f - config.center_frequency) / rate * n + n / 2))
        spectrum[k] = level
    return SamplesBatch(
        instant_spectrum=spectrum,
        spectrum=spectrum,
        center_frequency=config.center_frequency,
        sample_rate=rate,
        capture_utc_s=t,
    )


@pytest.mark.parametrize("reported_rate", [RATE, RATE - 2.0], ids=["exact", "pluto-9999998"])
def test_the_stage_finds_a_carrier_and_the_engine_tunes_to_it(
    engine: SDREngine, reported_rate: float
) -> None:
    """A PlutoSDR set to 10 MS/s reports 9999998: batches carry that, and SEARCH
    once skipped every one of them."""
    search = start_search("pluto", trigger_db=-40.0)
    context = engine.devices["pluto"]
    events: list[Event] = []
    engine.event_bus.subscribe(SearchStatusEvent, events.append)
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    carrier = 147.0e6
    for i in range(4):
        stage.on_config_change(context.config)
        batch = _batch(context.config, 1000.0 + 0.05 * i, {carrier: -30.0}, reported_rate)
        stage.process(batch, pipeline_context)
    config = _config(engine)
    assert config.tuned_frequency == carrier
    assert config.search == search
    assert events[-1].status.channel_hz == carrier
    assert events[-1].status.state == "holding"


def test_the_stage_is_idle_without_search(engine: SDREngine) -> None:
    context = engine.devices["pluto"]
    events: list[Event] = []
    engine.event_bus.subscribe(SearchStatusEvent, events.append)
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    for i in range(4):
        stage.process(_batch(context.config, 1000.0 + 0.05 * i, {147.0e6: -30.0}), pipeline_context)
    assert events == []
    assert _config(engine).tuned_frequency == DIAL


# --- end to end ---------------------------------------------------------------


def _nfm(rate: float, seconds: float, offset_hz: float, start: float, stop: float) -> np.ndarray:
    """An FM carrier (2.5 kHz deviation, 1 kHz tone) at -20 dBFS, on between start and stop."""
    t = np.arange(int(rate * seconds)) / rate
    phase = 2 * np.pi * offset_hz * t + 2.5 * np.sin(2 * np.pi * 1_000.0 * t)
    on = (t >= start) & (t < stop)
    return (0.1 * np.exp(1j * phase) * on).astype(np.complex64)


def test_search_follows_transmissions_through_the_running_pipeline(tmp_path) -> None:
    """A recording with two transmissions, one after the other: SEARCH goes to
    the first, holds it, and moves to the second once the first has been
    quiet for the hold time."""
    rate, seconds = 1e6, 2.0
    rng = np.random.default_rng(0)
    n = int(rate * seconds)
    iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64) * np.float32(
        1e-3 / np.sqrt(2)
    )
    iq += _nfm(rate, seconds, 200e3, 0.3, 1.3)
    iq += _nfm(rate, seconds, -250e3, 1.0, 2.0)
    path = tmp_path / "two.cf32"
    iq.tofile(path)

    engine = SDREngine()
    engine.add_device(
        "file", "iq-file", IQFileParams(path=str(path)), DeviceConfig(sample_rate=rate)
    )
    tuned: list[float] = []

    def on_config(_event: Event) -> None:
        f = engine.devices["file"].config.tuned_frequency
        if not tuned or tuned[-1] != f:
            tuned.append(f)

    engine.event_bus.subscribe(ConfigChangedEvent, on_config)
    try:
        engine.start_device("file")
        engine.set_audio_demod("file", DemodSpec(mode="NFM"))
        deadline = time.monotonic() + 5.0
        while engine.devices["file"].state != DeviceState.RUNNING and time.monotonic() < deadline:
            time.sleep(0.01)
        start_search("file", trigger_db=-45.0, hold_s=0.2)
        time.sleep(1.8)
    finally:
        engine.stop_device("file")
        engine.shutdown(timeout=5.0)
    center = DeviceConfig().center_frequency
    jumps = [f - center for f in tuned if f != center]
    assert jumps[:2] == [200e3, -250e3]


def test_a_default_trigger_follows_the_squelch(engine: SDREngine) -> None:
    search = start_search("pluto")
    assert search.trigger_for(-50.0) == -44.0
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    assert _config(engine).search == search  # a squelch change keeps searching
    assert search.trigger_for(-80.0) == -74.0


def test_a_trigger_set_by_hand_is_kept(engine: SDREngine) -> None:
    start_search("pluto", trigger_db=-40.0)
    stop_search("pluto")
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    search = start_search("pluto")
    assert search.trigger_db == -40.0
    assert search.trigger_for(-80.0) == -40.0


def test_edges_at_the_dial_move_a_running_search(engine: SDREngine) -> None:
    search = start_search("pluto")
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=145.0e6, run=search.run)
    )
    assert set_search_edge("pluto", "low") == 145.0e6
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=148.0e6, run=search.run)
    )
    set_search_edge("pluto", "high")
    running = _config(engine).search
    assert (running.low_hz, running.high_hz) == (145.0e6, 148.0e6)
    assert running.run == search.run


def test_edges_set_while_stopped_apply_at_the_next_start(engine: SDREngine) -> None:
    engine.update_device_config("pluto", tuning_mode="free")
    engine.update_device_config("pluto", tuned_frequency=145.0e6)
    set_search_edge("pluto", "low")
    search = start_search("pluto")
    assert search.low_hz == 145.0e6
    assert search.high_hz == _config(engine).center_frequency + fitting_offset(RATE, 12_500.0)


def test_an_edge_on_the_wrong_side_drops_the_other(engine: SDREngine) -> None:
    engine.update_device_config("pluto", tuning_mode="free")
    engine.update_device_config("pluto", tuned_frequency=145.0e6)
    set_search_edge("pluto", "high")
    engine.update_device_config("pluto", tuned_frequency=147.0e6)
    set_search_edge("pluto", "low")
    search = start_search("pluto")
    assert search.low_hz == 147.0e6
    assert search.high_hz > 147.0e6


def test_full_band_forgets_the_edges(engine: SDREngine) -> None:
    start_search("pluto", low_hz=145e6, high_hz=148e6)
    search = start_search("pluto", full_band=True)
    half = fitting_offset(RATE, 12_500.0)
    center = _config(engine).center_frequency
    assert (search.low_hz, search.high_hz) == (center - half, center + half)


def test_edges_that_no_longer_fit_give_way_to_the_band(engine: SDREngine) -> None:
    start_search("pluto", low_hz=145e6, high_hz=148e6)
    stop_search("pluto")
    engine.update_device_config("pluto", tuning_mode="center")
    engine.update_device_config("pluto", tuned_frequency=440e6)
    search = start_search("pluto")
    center = _config(engine).center_frequency
    assert search.low_hz == center - fitting_offset(RATE, 12_500.0)


def test_stopping_the_radio_stops_search(engine: SDREngine) -> None:
    start_search("pluto")
    engine.stop_device("pluto")
    assert _config(engine).search is None
