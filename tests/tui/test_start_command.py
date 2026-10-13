"""`start` brings the started device into view when the focused one is stopped:
the spectrum, waterfall and keys follow the focused device, so starting a second
radio after stopping the first used to leave a dead display that tuned nothing."""

import pytest

from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams
from tsdr.tui.commands.sdr.start import SDRStartCommand


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    engine = SDREngine()
    engine.add_device("airspy", "mock", MockParams())
    engine.add_device("pluto", "mock", MockParams())
    engine.set_focused_device("airspy")

    def start(device_id: str) -> None:
        engine.devices[device_id].state = DeviceState.RUNNING

    monkeypatch.setattr(engine, "start_device", start)
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    yield engine
    for context in engine.devices.values():
        context.state = DeviceState.STOPPED


def test_starting_another_device_brings_it_into_view(engine: SDREngine) -> None:
    out = SDRStartCommand().execute(["pluto"])
    assert engine.focused_device == "pluto"
    assert "now in view" in out


def test_a_running_device_keeps_the_view(engine: SDREngine) -> None:
    engine.devices["airspy"].state = DeviceState.RUNNING
    out = SDRStartCommand().execute(["pluto"])
    assert engine.focused_device == "airspy"
    assert "now in view" not in out


def test_starting_the_focused_device_changes_nothing(engine: SDREngine) -> None:
    out = SDRStartCommand().execute([])
    assert engine.devices["airspy"].state == DeviceState.RUNNING
    assert engine.focused_device == "airspy"
    assert "now in view" not in out
