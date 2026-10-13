"""The mouse wheel on the spectrum tunes one tuning step, like the arrow keys,
and scrolling up tunes up. It moved the dial two display columns per notch
(about 163 kHz at 10 MS/s on a 123-column spectrum), downward for scroll-up,
and rounded to that width, off the step grid."""

from types import SimpleNamespace

import pytest

from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.devices import MockParams
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget

DIAL = 146.52e6


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    engine = SDREngine()
    engine.add_device(
        "pluto",
        "mock",
        MockParams(),
        DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, sample_rate=10e6),
    )
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    return engine


def _widget(
    monkeypatch: pytest.MonkeyPatch, tunes: list[int], errors: list[str] | None = None
) -> SpectrumWidget:
    app = SimpleNamespace(
        _tune=tunes.append, _show_error=(errors if errors is not None else []).append
    )
    monkeypatch.setattr(SpectrumWidget, "app", property(lambda _self: app))
    return SpectrumWidget()


def _event() -> SimpleNamespace:
    return SimpleNamespace(shift=False, stop=lambda: None)


@pytest.mark.usefixtures("engine")
def test_the_wheel_tunes_one_step_and_up_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    tunes: list[int] = []
    widget = _widget(monkeypatch, tunes)
    widget.on_mouse_scroll_up(_event())  # type: ignore[arg-type]
    widget.on_mouse_scroll_down(_event())  # type: ignore[arg-type]
    assert tunes == [1, -1]


def test_the_wheel_pans_a_zoomed_view_in_free_tuning(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.update_device_config("pluto", tuning_mode="free", spectrum_span=1e6)
    tunes: list[int] = []
    widget = _widget(monkeypatch, tunes)
    widget.on_mouse_scroll_up(_event())  # type: ignore[arg-type]
    config = engine.devices["pluto"].config
    assert tunes == []
    assert config.spectrum_center is not None and config.spectrum_center > DIAL
    assert config.tuned_frequency == DIAL


@pytest.mark.usefixtures("engine")
def test_a_failed_wheel_tune_shows_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wheel goes through the tuning helper, which can raise; the keys catch
    that, and the wheel must too, or the app ends."""
    errors: list[str] = []

    def fail(_direction: int) -> None:
        raise SDRException("out of range")

    widget = _widget(monkeypatch, [], errors)
    monkeypatch.setattr(widget.app, "_tune", fail)
    widget.on_mouse_scroll_up(_event())  # type: ignore[arg-type]
    assert errors == ["out of range"]
