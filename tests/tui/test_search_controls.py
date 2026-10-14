"""SEARCH controls: `/` starts and stops it, `x`/`X` lock out the current
channel, `s` changes its step, the `search` command, and the tuner's status."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core import search_control
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.memories import get_memory_store, init_memory_store
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.tuning_state import get_tuning_state, init_tuning_state
from tsdr.devices import MockParams, mock
from tsdr.tui.keyboard import KeyboardMixin
from tsdr.tui.tuning_mixin import TuningMixin

DIAL = 146.52e6
TONE = 146.625e6  # on the 12.5 kHz grid


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
        DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=10e6),
    )
    engine.set_audio_demod("pluto", DemodSpec(mode="NFM"))
    engine.devices["pluto"].state = DeviceState.RUNNING
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    yield engine
    engine.devices["pluto"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _keys(engine: SDREngine) -> SimpleNamespace:
    keys = SimpleNamespace(statuses=[], errors=[])
    keys._focused = lambda: engine.devices["pluto"]
    keys.show_status = keys.statuses.append
    keys._show_error = keys.errors.append
    keys._publish_tuning_state_changed = lambda: None
    keys._apply_freq = lambda device, freq: engine.update_device_config(
        device.device_id, tuned_frequency=freq
    )
    return keys


def _search(engine: SDREngine):
    return engine.devices["pluto"].config.search


def test_the_step_key_changes_the_search_step_down_to_its_floor(engine: SDREngine) -> None:
    """A 10 Hz step over 9 MHz is 890,000 channels a batch and stalled the
    audio: SEARCH stays at 1 kHz or coarser, and says so."""
    engine.set_audio_demod("pluto", DemodSpec(mode="USB"))  # NFM's steps start at 2.5 kHz
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    TuningMixin._cycle_step(keys, True)  # type: ignore[arg-type]
    assert get_tuning_state().step_for("USB") == 10
    assert _search(engine).step_hz == 1_000.0
    assert keys.statuses[-1] == "Step: 10 Hz (SEARCH: 1 kHz)"
    for _ in range(5):  # 50 Hz, 100 Hz, 500 Hz, 1 kHz, 2.5 kHz
        TuningMixin._cycle_step(keys, True)  # type: ignore[arg-type]
    assert _search(engine).step_hz == get_tuning_state().step_for("USB") == 2_500


def _fft(config: DeviceConfig, tones: dict[float, float]) -> FFTUpdateEvent:
    n = config.fft_size
    spectrum = np.full(n, -130.0, dtype=np.float32)
    for f, level in tones.items():
        spectrum[round((f - config.center_frequency) / config.sample_rate * n) + n // 2] = level
    return FFTUpdateEvent(
        device_id="pluto",
        spectrum=spectrum,
        frequencies=None,
        center_frequency=config.center_frequency,
        sample_rate=config.sample_rate,
    )


def test_brackets_move_search_on_without_stopping_it(engine: SDREngine) -> None:
    """While SEARCH runs, `[` / `]` jump to the next signal in its range, not
    one at the LO, and SEARCH keeps running."""
    engine.update_squelch("pluto", "audio", threshold_db=-80.0)
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    config = engine.devices["pluto"].config
    lo = config.center_frequency  # the zero-IF radio's LO, beside the dial
    assert lo < DIAL
    keys._latest_fft_by_device = {"pluto": _fft(config, {lo: -30.0, TONE: -30.0})}
    TuningMixin._jump_target(keys, 1)  # type: ignore[arg-type]
    assert engine.devices["pluto"].config.tuned_frequency == TONE
    assert _search(engine) is not None
    assert keys.statuses[-1].startswith("Jump:")
    TuningMixin._jump_target(keys, -1)  # type: ignore[arg-type]
    assert engine.devices["pluto"].config.tuned_frequency == TONE  # not the LO
    assert keys.statuses[-1] == "No signal further in the search range"


def test_brackets_never_move_the_lo_in_free_tuning(engine: SDREngine) -> None:
    """SEARCH off, free tuning, zero-IF radio: a memory at the LO was a target,
    and tuning there moved the LO (146.470 -> 146.420 MHz)."""
    engine.update_device_config("pluto", tuning_mode="free")
    lo = engine.devices["pluto"].config.center_frequency
    assert lo < DIAL
    get_memory_store().add(int(lo), "at the LO", DemodSpec(mode="NFM"), 12_500)
    keys = _keys(engine)
    keys._latest_fft_by_device = {}
    TuningMixin._jump_target(keys, -1)  # type: ignore[arg-type]
    config = engine.devices["pluto"].config
    assert (config.center_frequency, config.tuned_frequency) == (lo, DIAL)


def test_turning_the_squelch_off_stops_search(engine: SDREngine) -> None:
    """SEARCH holds a channel by the squelch: with it off, SEARCH kept running
    with the noise open between transmissions."""
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    KeyboardMixin._disable_squelch(keys)  # type: ignore[arg-type]
    assert _search(engine) is None
    assert keys.statuses[-1] == "Squelch off (SEARCH stopped)"
