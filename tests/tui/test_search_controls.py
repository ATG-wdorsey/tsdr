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
from tsdr.core.sdr.search import LOCKOUT_TAG, SearchStatus
from tsdr.core.tuning_state import get_tuning_state, init_tuning_state
from tsdr.devices import MockParams, mock
from tsdr.tui.commands.sdr.search import SDRSearchCommand
from tsdr.tui.keybindings import Action, Context, build_lookup
from tsdr.tui.tuning_mixin import TuningMixin
from tsdr.tui.widgets.tuner_widget import _search_lines

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
    return keys


def _search(engine: SDREngine):
    return engine.devices["pluto"].config.search


def test_search_keys_are_bound() -> None:
    lookup = build_lookup()
    assert lookup[(Context.GLOBAL, "slash")] is Action.SEARCH_TOGGLE
    assert lookup[(Context.GLOBAL, "x")] is Action.SEARCH_LOCKOUT
    assert lookup[(Context.GLOBAL, "X")] is Action.SEARCH_LOCKOUT_SAVE
    assert lookup[(Context.GLOBAL, "ctrl+x")] is Action.SEARCH_UNLOCK
    assert lookup[(Context.GLOBAL, "left_curly_bracket")] is Action.SEARCH_EDGE_LOW
    assert lookup[(Context.GLOBAL, "right_curly_bracket")] is Action.SEARCH_EDGE_HIGH


def test_braces_set_the_edges_at_the_dial(engine: SDREngine) -> None:
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    TuningMixin._set_search_edge(keys, "low")  # type: ignore[arg-type]
    assert _search(engine).low_hz == DIAL
    assert keys.statuses[-1] == "SEARCH lower edge: 146.52 MHz"


def test_slash_starts_and_stops_search(engine: SDREngine) -> None:
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    assert _search(engine) is not None
    assert keys.statuses[-1].startswith("SEARCH 141.")
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    assert _search(engine) is None
    assert keys.statuses[-1] == "SEARCH off"


def test_slash_explains_why_search_cannot_start(engine: SDREngine) -> None:
    engine.set_audio_demod("pluto", DemodSpec(mode="WFM"))
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    assert _search(engine) is None
    assert "narrowband" in keys.errors[-1]


def test_x_locks_out_for_the_session_and_shift_x_for_good(engine: SDREngine) -> None:
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    TuningMixin._lock_out_channel(keys, permanent=False)  # type: ignore[arg-type]
    assert _search(engine).lockouts == frozenset({DIAL})
    assert "this session" in keys.statuses[-1]
    TuningMixin._lock_out_channel(keys, permanent=True)  # type: ignore[arg-type]
    assert [m.frequency for m in get_memory_store().find_by_tag(LOCKOUT_TAG)] == [int(DIAL)]
    assert _search(engine).memory_lockouts == frozenset({DIAL})


def test_the_step_key_changes_the_search_step_down_to_its_floor(engine: SDREngine) -> None:
    """A 10 Hz step over 9 MHz is 890,000 channels a batch and stalled the
    audio: SEARCH stays at 1 kHz or coarser, and says so."""
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    TuningMixin._cycle_step(keys, True)  # type: ignore[arg-type]
    assert get_tuning_state().step == 10
    assert _search(engine).step_hz == 1_000.0
    assert keys.statuses[-1] == "Step: 10 Hz (SEARCH: 1 kHz)"
    for _ in range(3):  # 100 Hz, 1 kHz, 5 kHz
        TuningMixin._cycle_step(keys, True)  # type: ignore[arg-type]
    assert _search(engine).step_hz == get_tuning_state().step == 5_000


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


def test_ctrl_x_unlocks_the_channel_or_all(engine: SDREngine) -> None:
    keys = _keys(engine)
    TuningMixin._toggle_search(keys)  # type: ignore[arg-type]
    TuningMixin._lock_out_channel(keys, permanent=False)  # type: ignore[arg-type]
    TuningMixin._unlock(keys)  # type: ignore[arg-type]
    assert keys.statuses[-1].startswith("Unlocked")
    assert _search(engine).lockouts == frozenset()
    TuningMixin._unlock(keys)  # type: ignore[arg-type]
    assert keys.statuses[-1] == "No session lockouts to drop"


def test_command_starts_reports_and_stops(engine: SDREngine) -> None:
    cmd = SDRSearchCommand()
    out = cmd.execute(
        ["on", "--device", "pluto", "--low", "144M", "--high", "148M", "--threshold", "-40"]
    )
    assert "SEARCH" in out and "144.0000 MHz" in out
    search = _search(engine)
    assert (search.low_hz, search.high_hz, search.trigger_db) == (144e6, 148e6, -40.0)
    assert "trigger" in cmd.execute(["--device", "pluto"])
    assert "stopped" in cmd.execute(["off", "--device", "pluto"])
    assert _search(engine) is None
    assert "off" in cmd.execute(["--device", "pluto"])


def test_command_settings_alone_restart_search(engine: SDREngine) -> None:
    cmd = SDRSearchCommand()
    cmd.execute(["on", "--device", "pluto"])
    cmd.execute(["--device", "pluto", "--hold", "4", "--dwell", "30"])
    search = _search(engine)
    assert (search.hold_s, search.max_dwell_s) == (4.0, 30.0)


def test_command_rejects_a_trigger_under_the_squelch(engine: SDREngine) -> None:
    out = SDRSearchCommand().execute(["on", "--device", "pluto", "--threshold", "-70"])
    assert "above the squelch" in out
    assert _search(engine) is None


def test_command_lockouts(engine: SDREngine) -> None:
    cmd = SDRSearchCommand()
    cmd.execute(["on", "--device", "pluto"])
    assert "this session" in cmd.execute(["lockout", "--device", "pluto"])
    assert "1 session lockout" in cmd.execute(["unlock", "--device", "pluto"])
    assert "memory" in cmd.execute(["lockout", "--permanent", "--device", "pluto"])
    assert _search(engine).memory_lockouts == frozenset({DIAL})
    assert "146.5200 MHz (saved)" in cmd.execute(["--device", "pluto"])
    cmd.execute(["off", "--device", "pluto"])
    assert "146.5200 MHz (saved)" in cmd.execute(["--device", "pluto"])


def test_tuner_shows_search_state_and_level() -> None:
    assert "SEARCH" in _search_lines(None)[0]
    holding = SearchStatus("holding", DIAL, -62.4, 1.5, 1)
    badge, level = _search_lines(holding)
    assert "HOLD" in badge and "1.5s" in badge
    assert "-62" in level
    assert "WAIT" in _search_lines(SearchStatus("waiting", DIAL, float("-inf"), 0.0, 0))[0]
    assert "2 L/O" in _search_lines(holding, lockouts=2)[1]
    assert "L/O" not in _search_lines(holding)[1]
    assert _search_lines(SearchStatus("waiting", DIAL, float("-inf"), 0.0, 0))[1] == ""


def test_command_full_forgets_the_edges(engine: SDREngine) -> None:
    cmd = SDRSearchCommand()
    cmd.execute(["on", "--device", "pluto", "--low", "144M", "--high", "148M"])
    cmd.execute(["--full", "--device", "pluto"])
    assert _search(engine).low_hz < 144e6
