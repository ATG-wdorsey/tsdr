"""IQRateConverter: a radio's capture rate to the exact rate a decoder is written for."""

import numpy as np
import pytest

from tsdr.radio.dsp.rate_converter import IQRateConverter


def _tone(fs: float, f: float, seconds: float = 0.1) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return (0.5 * np.exp(2j * np.pi * f * t)).astype(np.complex64)


def _convert(conv: IQRateConverter, x: np.ndarray, batch: int) -> np.ndarray:
    return np.concatenate([conv.process(x[i : i + batch]) for i in range(0, len(x), batch)])


@pytest.mark.parametrize("source", [2_500_000, 5_000_000, 9_999_998, 10_000_000, 20_000_000])
def test_a_tone_in_the_passband_arrives_at_its_frequency_and_level(source: float) -> None:
    conv = IQRateConverter(source, 2_400_000, passband_hz=1.0e6)
    assert not conv.passthrough
    y = _convert(conv, _tone(source, 700e3), batch=int(source * 0.02))
    assert len(y) == pytest.approx(0.1 * 2_400_000, abs=50)
    y = y[5_000:]
    freq = np.fft.fftfreq(len(y), 1 / 2_400_000)[np.argmax(np.abs(np.fft.fft(y)))]
    assert freq == pytest.approx(700e3, abs=2_400_000 / len(y))
    assert np.abs(y).mean() == pytest.approx(0.5, rel=0.05)


def test_what_would_fold_onto_the_passband_is_stopped() -> None:
    # From 10 MS/s, 1.5 MHz would fold into the +-1 MHz ADS-B is decoded from.
    conv = IQRateConverter(10e6, 2_400_000, passband_hz=1.0e6)
    y = _convert(conv, _tone(10e6, 1.5e6), batch=200_000)[5_000:]
    assert 20 * np.log10(np.abs(y).mean() / 0.5) < -45.0


def test_the_same_rate_passes_through_and_a_slower_one_is_flagged() -> None:
    same = IQRateConverter(2_399_999, 2_400_000, passband_hz=1.0e6)
    assert same.passthrough and not same.too_slow
    x = _tone(2.4e6, 100e3)
    np.testing.assert_array_equal(same.process(x), x)
    slow = IQRateConverter(2_000_000, 2_400_000, passband_hz=1.0e6)
    assert slow.too_slow and slow.passthrough
