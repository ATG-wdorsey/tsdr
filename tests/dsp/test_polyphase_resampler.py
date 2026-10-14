"""The rational resampler (audio to 48 kHz, RNNoise, VSQ, FT8, recordings).

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
