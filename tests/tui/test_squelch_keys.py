"""`u` / `U` move the squelch threshold, and with it SEARCH's trigger, 1 dB a press."""

from types import SimpleNamespace

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.tui.keyboard import KeyboardMixin

DIAL = 146.52e6


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    engine = SDREngine()
    engine.add_device(
        "radio",
        "mock",
        MockParams(),
        DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=2.4e6),
    )
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    return engine


def _threshold(engine: SDREngine) -> float:
    return engine.devices["radio"].config.pipelines["audio"].squelch_threshold_db


def test_each_press_moves_the_threshold_one_db(engine: SDREngine) -> None:
    keys = SimpleNamespace(statuses=[], errors=[])
    keys.show_status = keys.statuses.append
    keys._show_error = keys.errors.append
    start = _threshold(engine)

    KeyboardMixin._adjust_squelch(keys, 1)  # type: ignore[arg-type]
    assert _threshold(engine) == pytest.approx(start + 1.0)
    KeyboardMixin._adjust_squelch(keys, 1)  # type: ignore[arg-type]
    KeyboardMixin._adjust_squelch(keys, -1)  # type: ignore[arg-type]
    KeyboardMixin._adjust_squelch(keys, -1)  # type: ignore[arg-type]
    KeyboardMixin._adjust_squelch(keys, -1)  # type: ignore[arg-type]
    assert _threshold(engine) == pytest.approx(start - 1.0)
    assert keys.statuses[-1] == f"Squelch on, threshold: {start - 1.0:.1f} dB"
    assert keys.errors == []
