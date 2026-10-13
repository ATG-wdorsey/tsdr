"""WFM de-emphasis: 50 µs or 75 µs, a global setting (`config --deemphasis`)
that reaches every WFM demodulator and is remembered across restarts."""

from types import MappingProxyType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.preferences import load_preferences, restore_engine_config, save_engine_config
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.engine import SDREngine, get_engine
from tsdr.core.sdr.pipeline.stage_factory import create_stage
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator
from tsdr.tui.commands.sdr.config import SDRConfigCommand

RATE = 250_000


def _tone_level(demod: WidebandFMDemodulator, tone_hz: float) -> float:
    """Output level (dB) of a mono FM tone at 20 % deviation."""
    t = np.arange(int(RATE * 0.5)) / RATE
    iq = np.exp(1j * 2 * np.pi * 75_000 * np.cumsum(0.2 * np.sin(2 * np.pi * tone_hz * t)) / RATE)
    out = []
    for i in range(0, len(iq), 5000):
        demod.demodulate(iq[i : i + 5000].astype(np.complex64), 0.0)
        out.extend(b.samples[:, 0] for b in demod.get_audio())
    audio = np.concatenate(out)[len(out[0]) * 10 :]
    ref = np.exp(-2j * np.pi * tone_hz * np.arange(len(audio)) / demod.output_sample_rate)
    return 20 * np.log10(2 * abs(np.mean(audio * ref)))


def test_75_us_cuts_10_khz_about_3_db_more_than_50_us():
    def treble(tc: float) -> float:
        demod = WidebandFMDemodulator(sample_rate=RATE, rds_enabled=False)
        demod.set_de_emphasis(tc)
        return _tone_level(demod, 10_000) - _tone_level(demod, 1_000)

    # Analog: 1 / sqrt(1 + (2 pi f tau)^2) at 10 kHz is -13.6 dB (75 µs), -10.4 dB (50 µs)
    assert treble(75e-6) - treble(50e-6) == pytest.approx(-3.2, abs=0.5)


def _audio_pipeline(mode: str) -> MappingProxyType:
    stages = (StageType.FREQUENCY_SHIFT, StageType.DEMODULATOR, StageType.DENOISER)
    return MappingProxyType(
        {"audio": PipelineConfig(stages=stages, audio_spec=DemodSpec(mode=mode))}
    )


def test_the_setting_reaches_the_demodulator_and_survives_a_mode_change():
    device = DeviceConfig(sample_rate=RATE, pipelines=_audio_pipeline("WFM"))
    pipeline = device.pipelines["audio"]
    stage = create_stage(
        StageType.DEMODULATOR, pipeline, SDRConfig(wfm_de_emphasis_us=75), device, "d", "audio"
    )
    assert stage.demodulator.de_emphasis_tc == 75e-6

    stage.on_config_change(SDRConfig(wfm_de_emphasis_us=50))
    assert stage.demodulator.de_emphasis_tc == 50e-6

    stage.on_config_change(SDRConfig(wfm_de_emphasis_us=75))
    stage.on_config_change(device.with_changes(pipelines=_audio_pipeline("NFM")))
    stage.on_config_change(device)  # back to WFM: a new demodulator
    assert isinstance(stage.demodulator, WidebandFMDemodulator)
    assert stage.demodulator.de_emphasis_tc == 75e-6


def test_the_setting_is_remembered():
    engine = SDREngine()
    engine.update_global_config(wfm_de_emphasis_us=75)
    save_engine_config(engine)

    SDREngine()  # the next session starts at the default
    assert get_engine().config.wfm_de_emphasis_us == 50
    restore_engine_config(load_preferences())
    assert get_engine().config.wfm_de_emphasis_us == 75


def test_a_bad_saved_value_is_ignored():
    SDREngine()
    restore_engine_config({"engine": {"wfm_de_emphasis_us": 60}})
    assert get_engine().config.wfm_de_emphasis_us == 50
    with pytest.raises(ValueError, match="50 or 75"):
        SDRConfig(wfm_de_emphasis_us=60).validate()


def test_config_deemphasis_sets_the_global_setting():
    engine = MagicMock()
    engine.devices = {}
    with patch("tsdr.tui.commands.sdr.config.get_engine", return_value=engine):
        result = SDRConfigCommand().execute(["--device", "d", "--deemphasis", "75"])
    engine.update_global_config.assert_called_once_with(wfm_de_emphasis_us=75)
    engine.update_device_config.assert_not_called()
    assert "75 µs" in result
