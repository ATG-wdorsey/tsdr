"""The rational resampler (audio to 48 kHz, RNNoise, VSQ, FT8, recordings, IQ for ADS-B/DAB).

Each polyphase branch must meet the newest input sample with its first tap. It was
applied in time-reversed order, so every output was taken at the mirrored fraction of
a sample: up to one input sample of jitter. A 3 kHz tone through 50 -> 48 kHz came
out with THD+N of -13 dB, and through 12 -> 48 kHz the products outweighed the tone.
"""

import numpy as np
import pytest

from tsdr.radio.dsp._kernels import make_rational_resampler


def _thdn_db(y: np.ndarray, fs: float, f: float) -> float:
    w = np.blackman(len(y))
    spec = np.abs(np.fft.rfft(y * w)) ** 2
    k = int(round(f * len(y) / fs))
    tone = spec[max(0, k - 8) : k + 9].sum()
    return float(10 * np.log10((spec.sum() - tone) / tone))


@pytest.mark.parametrize(
    ("source", "tone"),
    [(50_000, 1000), (50_000, 3000), (44_100, 3000), (12_000, 3000), (51_200, 1000)],
)
def test_a_tone_comes_out_clean(source: int, tone: float) -> None:
    n = source * 2
    x = (0.5 * np.sin(2 * np.pi * tone * np.arange(n) / source)).astype(np.float32)
    rs = make_rational_resampler(48_000, source)
    # In the audio worker's batch sizes, so the state carried between calls counts too.
    y = np.concatenate([rs.process(x[i : i + 2500, None]) for i in range(0, n, 2500)])[:, 0]
    y = y[9_600:86_400]
    assert _thdn_db(y, 48_000, tone) < -60.0


def test_iq_matches_two_real_channels_and_keeps_the_frequency() -> None:
    fs_in, fs_out, f = 2_500_000, 2_400_000, -700_000.0
    t = np.arange(250_000) / fs_in
    iq = (0.5 * np.exp(2j * np.pi * f * t)).astype(np.complex64)

    complex_rs = make_rational_resampler(fs_out, fs_in)
    real_rs = make_rational_resampler(fs_out, fs_in)
    out = np.concatenate(
        [complex_rs.process_iq(iq[i : i + 50_000]) for i in range(0, len(iq), 50_000)]
    )
    pairs = iq.view(np.float32).reshape(-1, 2)
    ref = np.concatenate(
        [real_rs.process(pairs[i : i + 50_000]) for i in range(0, len(iq), 50_000)]
    )
    np.testing.assert_allclose(out.real, ref[:, 0], atol=1e-6)
    np.testing.assert_allclose(out.imag, ref[:, 1], atol=1e-6)

    y = out[2_000:]
    peak = np.fft.fftfreq(len(y), 1 / fs_out)[np.argmax(np.abs(np.fft.fft(y * np.hanning(len(y)))))]
    assert peak == pytest.approx(f, abs=fs_out / len(y))
    # A clean tone: everything but the tone at least 50 dB down (window leakage aside).
    spec = np.abs(np.fft.fft(y * np.blackman(len(y)))) ** 2
    k = int(np.argmax(spec))
    tone = spec[k - 8 : k + 9].sum()
    assert 10 * np.log10((spec.sum() - tone) / tone) < -50.0
