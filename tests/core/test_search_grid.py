"""SEARCH's grid is never coarser than the channel: channels are measured one channel wide
at the grid points, so a coarser grid leaves gaps. A 100 kHz tuning step carried into a
2 m search left every station between 144.0, 144.1, ... unseen."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tsdr.core import search_control
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.memories import init_memory_store
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stages.search_stage import SearchStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import ATTACK_S
from tsdr.core.search_control import set_search_step, start_search
from tsdr.core.tuning_state import get_tuning_state, init_tuning_state
from tsdr.devices import MockParams, mock

RATE = 10e6
DIAL = 146.0e6


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
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
        DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=RATE),
    )
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.devices["pluto"].state = DeviceState.RUNNING
    yield engine
    engine.devices["pluto"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


@pytest.mark.parametrize(("step", "grid"), [(100_000, 12_500), (25_000, 12_500), (5_000, 5_000)])
def test_a_tuning_step_coarser_than_the_channel_gives_way(
    engine: SDREngine, step: float, grid: float
) -> None:
    get_tuning_state().set_step("NFM", step)
    assert start_search("pluto").step_hz == grid


def test_am_falls_back_to_its_own_automatic_step(engine: SDREngine) -> None:
    engine.set_audio_demod("pluto", DemodSpec(mode="AM"))
    get_tuning_state().set_step("AM", 25_000)  # AM is 10 kHz wide
    assert start_search("pluto").step_hz == 1_000


def test_a_coarse_step_chosen_during_search_gives_way_too(engine: SDREngine) -> None:
    start_search("pluto")
    assert set_search_step("pluto", 100_000) == 12_500
    assert engine.devices["pluto"].config.search.step_hz == 12_500


def test_another_mode_brings_its_own_step_into_search(engine: SDREngine) -> None:
    """Each mode keeps its tuning step. SEARCH started in DMR (auto: 12.5 kHz on 2 m),
    then NFM (step 5 kHz) picked: SEARCH and `[` / `]` stayed on the 12.5 kHz grid."""
    get_tuning_state().set_step("NFM", 5_000)
    engine.set_audio_demod("pluto", DemodSpec(mode="DMR"))
    follower = search_control.SearchStepFollower()
    assert start_search("pluto").step_hz == 12_500
    follower.check()
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    follower.check()
    assert engine.devices["pluto"].config.search.step_hz == 5_000
    follower.check()  # a run's own step (search --step) is kept until the channel changes
    set_search_step("pluto", 10_000)
    follower.check()
    assert engine.devices["pluto"].config.search.step_hz == 10_000


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


def test_a_station_between_100_khz_grid_points_is_found(engine: SDREngine) -> None:
    get_tuning_state().set_step("NFM", 100_000)
    start_search("pluto", trigger_db=-40.0)
    context = engine.devices["pluto"]
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    station = 146.950e6  # off the 100 kHz grid, on the 12.5 kHz one
    for i in range(round(ATTACK_S / 0.05) + 3):
        stage.on_config_change(context.config)
        stage.process(_batch(context.config, 1000.0 + 0.05 * i, {station: -30.0}), pipeline_context)
    assert context.config.tuned_frequency == station
