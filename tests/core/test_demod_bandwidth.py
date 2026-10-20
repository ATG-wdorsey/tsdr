"""A demodulator change starts the new demodulator at its own bandwidth: the
previous mode's used to carry over whenever it fit, so NFM's 12.5 kHz into WFM
(or CW's 200 Hz into NFM) left the new mode nearly silent until widened."""

import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.devices import MockParams


@pytest.fixture(params=[False, True], ids=["rebuilt", "swapped-in-place"])
def engine(request: pytest.FixtureRequest) -> SDREngine:
    engine = SDREngine()
    engine.add_device("radio", "mock", MockParams(), DeviceConfig(sample_rate=2.4e6))
    if request.param:
        engine.devices["radio"].state = DeviceState.RUNNING  # audio output starts: swap path
    yield engine
    engine.devices["radio"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _bandwidth(engine: SDREngine) -> float | None:
    return engine.devices["radio"].config.channel_bandwidth


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [("NFM", "WFM", 200_000), ("CW", "NFM", 12_500), ("WFM", "AM", 10_000)],
)
def test_another_demodulator_starts_at_its_own_bandwidth(
    engine: SDREngine, old: str, new: str, expected: float
) -> None:
    engine.set_audio_demod("radio", DemodSpec(mode=old))
    engine.update_device_config("radio", channel_bandwidth=15_000.0)
    engine.set_audio_demod("radio", DemodSpec(mode=new))
    assert _bandwidth(engine) == expected


@pytest.mark.parametrize(("old", "new"), [("NFM", "NFM"), ("LSB", "USB"), ("USB", "LSB")])
def test_the_same_demodulator_keeps_the_users_bandwidth(
    engine: SDREngine, old: str, new: str
) -> None:
    engine.set_audio_demod("radio", DemodSpec(mode=old))
    engine.update_device_config("radio", channel_bandwidth=2_400.0)
    engine.set_audio_demod("radio", DemodSpec(mode=new))
    assert _bandwidth(engine) == 2_400.0


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [("NFM", "WFM", 200_000), ("P25", "WFM", 200_000), ("NFM", "NFM", 15_000.0)],
)
def test_a_mode_picked_after_demodulation_off_is_a_change_from_the_last(
    engine: SDREngine, old: str, new: str, expected: float
) -> None:
    """`d o` then `d w` after NFM (or P25 on NFM's bandwidth) left WFM at 12.5 kHz:
    with no demodulator, the engine took it for a restore at startup."""
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    engine.update_device_config("radio", channel_bandwidth=15_000.0)
    engine.set_audio_demod("radio", DemodSpec(mode=old))
    engine.stop_audio_output("radio")
    engine.remove_pipeline("radio", "audio")
    engine.set_audio_demod("radio", DemodSpec(mode=new))
    assert _bandwidth(engine) == expected


def test_a_restored_demodulator_keeps_the_configured_bandwidth(engine: SDREngine) -> None:
    engine.update_device_config("radio", channel_bandwidth=20_000.0)
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    assert _bandwidth(engine) == 20_000.0


def test_a_fixed_width_decoder_leaves_the_setting_alone(engine: SDREngine) -> None:
    """ADS-B and DAB measure their own fixed width; setting their (inherited)
    default left 12.5 kHz configured, which other code then took as theirs."""
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    engine.update_device_config("radio", channel_bandwidth=15_000.0)
    engine.set_audio_demod("radio", DemodSpec(mode="ADSB"))
    assert _bandwidth(engine) == 15_000.0
    assert engine.devices["radio"].demod_profile.channel_bandwidth == 2_000_000
