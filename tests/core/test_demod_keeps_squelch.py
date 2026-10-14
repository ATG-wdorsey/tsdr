"""Changing (or re-selecting) the demodulator keeps the squelch: it used to come
back off at -50 dBFS, so a squelch set just above the noise was silently lost."""

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams


@pytest.fixture
def engine() -> SDREngine:
    engine = SDREngine()
    engine.add_device("radio", "mock", MockParams(), DeviceConfig(sample_rate=2.4e6))
    yield engine
    engine.devices["radio"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _squelch(engine: SDREngine) -> tuple[bool, float, float]:
    audio = engine.devices["radio"].config.pipelines["audio"]
    return audio.squelch_enabled, audio.squelch_threshold_db, audio.squelch_hang_ms


@pytest.mark.parametrize("running", [False, True], ids=["rebuilt", "swapped-in-place"])
def test_squelch_survives_demod_changes(engine: SDREngine, running: bool) -> None:
    if running:
        engine.devices["radio"].state = DeviceState.RUNNING  # audio output starts: swap path
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-82.0, hang_ms=250.0)
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    assert _squelch(engine) == (True, -82.0, 250.0)
    engine.set_audio_demod("radio", DemodSpec(mode="AM"))
    assert _squelch(engine) == (True, -82.0, 250.0)


def test_first_demodulator_gets_the_default_squelch(engine: SDREngine) -> None:
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    assert _squelch(engine) == (False, -50.0, 100.0)


def test_squelch_survives_turning_demodulation_off_and_on(engine: SDREngine) -> None:
    # Live 2026-10-14: `D o` then `d n` brought the squelch back off at -50; -78 was lost.
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-78.0, hang_ms=100.0)
    engine.remove_pipeline("radio", "audio")
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    assert _squelch(engine) == (True, -78.0, 100.0)
