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
    FFTUpdateEvent,
    MemoriesChangedEvent,
    SearchStatusEvent,
    SearchTuneEvent,
)
from tsdr.core.landmarks import next_target
from tsdr.core.memories import get_memory_store, init_memory_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.search_stage import SearchStage, audio_channel
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import ATTACK_S, LOCKOUT_TAG
from tsdr.core.sdr.tune_policy import fitting_offset
from tsdr.core.search_control import (
    clear_session_lockouts,
    jump_targets,
    lock_out_channel,
    search_jump,
    set_search_edge,
    set_search_step,
    start_search,
    stop_search,
    unlock,
)
from tsdr.core.tuning_state import get_tuning_state, init_tuning_state
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


def test_a_fixed_width_decoder_is_measured_at_its_own_width(engine: SDREngine) -> None:
    """ADS-B is 2 MHz wide whatever the bandwidth setting says: SEARCH took a
    12.5 kHz setting left from NFM as its width and searched it."""
    engine.update_device_config("pluto", channel_bandwidth=12_500.0)
    engine.set_audio_demod("pluto", DemodSpec(mode="ADSB"))
    assert audio_channel(_config(engine)).bandwidth_hz == 2_000_000
    with pytest.raises(SDRException, match="narrowband"):
        start_search("pluto")


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


def test_the_grid_is_never_finer_than_1_khz_or_an_fft_bin(engine: SDREngine) -> None:
    get_tuning_state().set_step("NFM", 10.0)
    assert start_search("pluto").step_hz == 1_000.0
    assert start_search("pluto", step_hz=100.0).step_hz == 1_000.0
    engine.update_device_config("pluto", fft_size=4096)  # 2.4 kHz bins at 10 MS/s
    assert start_search("pluto").step_hz == 5_000.0  # the next ladder step


def test_session_lockout(engine: SDREngine) -> None:
    start_search("pluto")
    assert lock_out_channel("pluto") == DIAL
    assert _config(engine).search.lockouts == frozenset({DIAL})
    stop_search("pluto")
    assert start_search("pluto").lockouts == frozenset({DIAL})
    assert clear_session_lockouts("pluto") == 1
    assert _config(engine).search.lockouts == frozenset()


def test_unlock_at_the_dial_removes_its_lockouts_only(engine: SDREngine) -> None:
    search_control.search_settings("pluto").lockouts.add(147.0e6)
    start_search("pluto")
    lock_out_channel("pluto")
    lock_out_channel("pluto", permanent=True)
    result = unlock("pluto")
    assert (result.frequency_hz, result.session, result.saved) == (DIAL, 1, 1)
    assert get_memory_store().find_by_tag(LOCKOUT_TAG) == []
    search = _config(engine).search
    assert search.lockouts == frozenset({147.0e6}) and search.memory_lockouts == frozenset()


def test_unlock_elsewhere_drops_every_session_lockout(engine: SDREngine) -> None:
    settings = search_control.search_settings("pluto")
    settings.lockouts |= {147.0e6, 147.5e6}
    get_memory_store().add(146_000_000, "saved", DemodSpec(mode="NFM"), 12_500, tags=(LOCKOUT_TAG,))
    start_search("pluto")
    result = unlock("pluto")
    assert (result.frequency_hz, result.session, result.saved) == (None, 2, 0)
    assert _config(engine).search.lockouts == frozenset()
    assert len(get_memory_store().find_by_tag(LOCKOUT_TAG)) == 1  # saved lockouts stay


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


# Batches (at 20 a second) for SEARCH to find a carrier, jump to it and report it.
FIND_BATCHES = round(ATTACK_S / 0.05) + 3


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
    for i in range(FIND_BATCHES):
        stage.on_config_change(context.config)
        batch = _batch(context.config, 1000.0 + 0.05 * i, {carrier: -30.0}, reported_rate)
        stage.process(batch, pipeline_context)
    config = _config(engine)
    assert config.tuned_frequency == carrier
    assert config.search == search
    assert events[-1].status.channel_hz == carrier
    assert events[-1].status.state == "holding"


def test_the_stage_holds_while_the_demodulators_squelch_is_open(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hold follows the squelch the listener hears: frames of the channel
    that read above the threshold do not hold it once the squelch has shut."""
    start_search("pluto", trigger_db=-40.0, hold_s=0.5)
    context = engine.devices["pluto"]
    squelch = {"open": False}
    monkeypatch.setattr(
        type(context), "demod_status", property(lambda _: DemodStatus(squelch_open=squelch["open"]))
    )
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    first, second = 147.0e6, 145.0e6
    t = [1000.0]

    def feed(seconds: float, tones: dict[float, float]) -> None:
        for _ in range(round(seconds / 0.05)):
            stage.on_config_change(context.config)
            stage.process(_batch(context.config, t[0], tones), pipeline_context)
            t[0] += 0.05

    feed(ATTACK_S + 0.15, {first: -30.0})
    assert _config(engine).tuned_frequency == first
    squelch["open"] = True
    above_squelch = audio_channel(context.config).squelch_db + 3.0
    feed(1.0, {second: -30.0})  # the frames miss the first channel
    assert _config(engine).tuned_frequency == first  # its squelch is open
    squelch["open"] = False
    feed(1.0, {first: above_squelch, second: -30.0})
    assert _config(engine).tuned_frequency == second


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
    """A recording with two transmissions, one after the other: SEARCH, running in the
    engine's pipeline, goes to each. In either order: the file plays in real time, and on a
    cold numba cache the pipeline's first batches wait seconds for its kernels to compile,
    so what it can't keep up with is dropped (the order and the hold are the controller
    tests' business)."""
    rate, seconds = 1e6, 3.0
    rng = np.random.default_rng(0)
    n = int(rate * seconds)
    iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64) * np.float32(
        1e-3 / np.sqrt(2)
    )
    iq += _nfm(rate, seconds, 200e3, 0.3, 1.3)
    iq += _nfm(rate, seconds, -250e3, 1.0, 3.0)
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
        # Polled: the second jump comes at about 1.6 s, later on a busy machine or a cold cache.
        deadline = time.monotonic() + 10.0
        while len(set(tuned) - {DeviceConfig().center_frequency}) < 2:
            if time.monotonic() > deadline:
                break
            time.sleep(0.02)
    finally:
        engine.stop_device("file")
        engine.shutdown(timeout=5.0)
    center = DeviceConfig().center_frequency
    jumps = [f - center for f in tuned if f != center]
    assert sorted(jumps[:2]) == [-250e3, 200e3]


def test_a_jump_decided_before_a_manual_tune_does_not_land(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline thread checks the run, then applies its jump; a manual tune
    (which stops SEARCH) can come in between and was then overridden."""
    search = start_search("pluto")
    real = engine.update_device_config

    def tune_meanwhile(device_id: str, **kw: object) -> bool:
        if kw.get("from_search"):
            real(device_id, tuned_frequency=146.6e6)
        return real(device_id, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(engine, "update_device_config", tune_meanwhile)
    engine.event_bus.publish(
        SearchTuneEvent(device_id="pluto", frequency_hz=147.0e6, run=search.run)
    )
    assert (_config(engine).tuned_frequency, _config(engine).search) == (146.6e6, None)


def test_the_run_is_checked_with_the_change_itself(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same race inside the update: SEARCH stops between the engine's
    checks and the config swap."""
    search = start_search("pluto")
    context = engine.devices["pluto"]
    real = context.update_config

    def stop_meanwhile(**kw: object) -> bool:
        if kw.get("expect") is not None:
            real(search=None)
        return real(**kw)  # type: ignore[arg-type]

    monkeypatch.setattr(context, "update_config", stop_meanwhile)
    applied = engine.update_device_config(
        "pluto", from_search=True, while_search=search, tuned_frequency=147.0e6
    )
    assert not applied
    assert (_config(engine).tuned_frequency, _config(engine).search) == (DIAL, None)


def test_a_stale_lockout_update_does_not_restart_search(engine: SDREngine) -> None:
    search = start_search("pluto")
    stop_search("pluto")
    assert not engine.update_device_config(
        "pluto", while_search=search, search=replace(search, lockouts=frozenset({DIAL}))
    )
    assert _config(engine).search is None


def test_a_default_trigger_follows_the_squelch(engine: SDREngine) -> None:
    search = start_search("pluto")
    assert search.trigger_for(-50.0) == -50.0  # the SQ line itself
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    assert _config(engine).search == search  # a squelch change keeps searching
    assert search.trigger_for(-80.0) == -80.0


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


# --- [ / ] targets ------------------------------------------------------------


def _fft(config: DeviceConfig, tones: dict[float, float]) -> FFTUpdateEvent:
    batch = _batch(config, 0.0, tones)
    assert batch.spectrum is not None
    return FFTUpdateEvent(
        device_id="pluto",
        spectrum=batch.spectrum,
        frequencies=None,
        center_frequency=config.center_frequency,
        sample_rate=RATE,
    )


def test_jump_targets_are_signals_on_the_grid_and_memories_in_the_band(engine: SDREngine) -> None:
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    store = get_memory_store()
    store.add(146_000_000, "in band", DemodSpec(mode="NFM"), 12_500)
    store.add(160_000_000, "out of band", DemodSpec(mode="NFM"), 12_500)
    context = engine.devices["pluto"]
    # An off-grid carrier, one under the squelch, and the one the dial is on.
    fft = _fft(context.config, {147.0031e6: -30.0, 145.5e6: -112.0, DIAL: -30.0})
    targets, limits = jump_targets(context, fft)
    assert 147.0e6 in targets  # on the 12.5 kHz grid
    assert 146.0e6 in targets
    assert not any(abs(t - DIAL) < 6_250.0 for t in targets)
    assert not any(abs(t - 145.5e6) < 6_250.0 for t in targets)
    assert next_target(1, DIAL, targets, limits) == 147.0e6
    assert next_target(-1, DIAL, targets, limits) == 146.0e6
    assert next_target(1, 147.0e6, targets, limits) is None  # 160 MHz is beyond the band


def test_while_searching_jump_targets_are_signals_in_its_range(engine: SDREngine) -> None:
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    search = start_search("pluto", low_hz=146.0e6, high_hz=147.5e6)
    engine.update_device_config("pluto", search=replace(search, lockouts=frozenset({147.0e6})))
    get_memory_store().add(146_250_000, "in range", DemodSpec(mode="NFM"), 12_500)
    context = engine.devices["pluto"]
    fft = _fft(context.config, {145.5e6: -30.0, 147.0e6: -30.0, 147.25e6: -30.0})
    targets, limits = jump_targets(context, fft)
    assert limits == (146.0e6, 147.5e6)
    assert next_target(1, DIAL, targets, limits) == 147.25e6  # past the locked-out channel
    assert next_target(-1, DIAL, targets, limits) is None  # 145.5 MHz is outside; no memories


def test_a_jump_key_keeps_search_running(engine: SDREngine) -> None:
    search = start_search("pluto")
    assert search_jump("pluto", 147.0e6)
    assert _config(engine).tuned_frequency == 147.0e6
    assert _config(engine).search == search
    assert not search_jump("pluto", DIAL + 20e6)  # would retune the hardware: dropped
    assert _config(engine).tuned_frequency == 147.0e6


def test_jump_targets_without_a_demodulator_use_spectrum_peaks(engine: SDREngine) -> None:
    engine.remove_pipeline("pluto", "audio")
    context = engine.devices["pluto"]
    targets, _ = jump_targets(context, _fft(context.config, {147.0031e6: -30.0}))
    assert any(abs(t - 147.0031e6) < RATE / context.config.fft_size for t in targets)


def _stage_run(engine: SDREngine, tones: dict[float, float], batches: int = FIND_BATCHES) -> None:
    context = engine.devices["pluto"]
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    for i in range(batches):
        stage.on_config_change(context.config)
        stage.process(_batch(context.config, 2000.0 + 0.05 * i, tones), pipeline_context)


def test_search_lands_on_the_signal_not_a_step_beside_it(engine: SDREngine) -> None:
    """5 kHz steps, 12.5 kHz NFM channels: a carrier fills the channels a step
    either side as fully as its own, and SEARCH landed 5 kHz low."""
    start_search("pluto", trigger_db=-40.0, step_hz=5_000.0)
    _stage_run(engine, {147.0e6: -30.0})
    assert _config(engine).tuned_frequency == 147.0e6


def test_search_does_not_stop_on_its_edge_for_a_signal_beyond_it(engine: SDREngine) -> None:
    """A carrier 4 kHz below the lower edge fills the edge channel's 12.5 kHz
    window; the grid channel past the edge, measured too, shows whose it is."""
    start_search("pluto", low_hz=146.0e6, high_hz=147.0e6, trigger_db=-40.0, step_hz=5_000.0)
    _stage_run(engine, {145.996e6: -30.0})
    assert _config(engine).tuned_frequency == DIAL


def test_jump_targets_land_on_the_signal_with_a_fine_step(engine: SDREngine) -> None:
    get_tuning_state().set_step("NFM", 5_000.0)
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    context = engine.devices["pluto"]
    targets, limits = jump_targets(context, _fft(context.config, {147.0e6: -30.0}))
    assert next_target(1, DIAL, targets, limits) == 147.0e6
