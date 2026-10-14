"""SEARCH's grid is never coarser than the channel: channels are measured one channel wide
at the grid points, so a coarser grid leaves gaps. A 100 kHz tuning step carried into a
2 m search left every station between 144.0, 144.1, ... unseen."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
from tsdr.core.search_control import set_search_step, start_search
from tsdr.core.tuning_state import get_tuning_state, init_tuning_state
from tsdr.devices import MockParams, mock
from tsdr.tui.tuning_mixin import TuningMixin

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
    get_tuning_state().step = step
    assert start_search("pluto").step_hz == grid


def test_am_falls_back_to_its_own_automatic_step(engine: SDREngine) -> None:
    engine.set_audio_demod("pluto", DemodSpec(mode="AM"))
    get_tuning_state().step = 25_000  # AM is 10 kHz wide
    assert start_search("pluto").step_hz == 1_000


def test_a_coarse_step_chosen_during_search_gives_way_too(engine: SDREngine) -> None:
    start_search("pluto")
    assert set_search_step("pluto", 100_000) == 12_500
    assert engine.devices["pluto"].config.search.step_hz == 12_500


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
    get_tuning_state().step = 100_000
    start_search("pluto", trigger_db=-40.0)
    context = engine.devices["pluto"]
    stage = SearchStage(context.config)
    pipeline_context = PipelineContext(
        device_context=context, event_bus=engine.event_bus, config=engine.config
    )
    station = 146.950e6  # off the 100 kHz grid, on the 12.5 kHz one
    for i in range(6):
        stage.on_config_change(context.config)
        stage.process(_batch(context.config, 1000.0 + 0.05 * i, {station: -30.0}), pipeline_context)
    assert context.config.tuned_frequency == station


def test_slash_puts_the_tuner_on_the_step_search_uses(engine: SDREngine) -> None:
    get_tuning_state().step = 100_000
    keys = SimpleNamespace(statuses=[], errors=[], published=[])
    keys._focused = lambda: engine.devices["pluto"]
    keys.show_status = keys.statuses.append
    keys._show_error = keys.errors.append
    keys._publish_tuning_state_changed = lambda: keys.published.append(get_tuning_state().step)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    assert keys.errors == []
    assert get_tuning_state().step is None  # auto: 12.5 kHz for NFM
    assert keys.published == [None]
    assert "would miss stations" in keys.statuses[-1]
