"""ADS-B and DAB at the radio's own capture rate (a Pluto at 10 MS/s, an Airspy at 2.5).

Both are written for one rate (2.4 and 2.048 MS/s) and ignored any other: the tuner
said "Needs 2.4 MS/s" and nothing decoded. The recordings here are taken to other
rates by FFT, independently of tsdr's resamplers, and must decode as at their own.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.adsb import ADSBDecoder
from tsdr.radio.decoders.dab.decoder import DABDecoder

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
ADSB_FILE = SAMPLES / "freq=1090.0M_sr=2400k_dur=30s_gain=23_20260328T1217.cu8.zst"
DAB_FILE = SAMPLES / "freq=227.36M_sr=2048k_dur=10s_gain=10_20260308T2248.cu8.zst"


def fft_resample(
    x: np.ndarray, up: int, down: int, core_blocks: int = 2048
) -> Iterator[np.ndarray]:
    """Band-limited resampling by `up`/`down` with FFTs, in overlapping blocks."""
    core, margin, n = down * core_blocks, down * 64, len(x)
    for start in range(0, n, core):
        a, b = max(0, start - margin), min(n, start + core + margin)
        seg = x[a:b].astype(np.complex128)
        seg = np.concatenate([seg, np.zeros((-len(seg)) % down, np.complex128)])
        n_out = len(seg) * up // down
        spec = np.fft.fft(seg)
        out = np.zeros(n_out, np.complex128)
        keep = min(len(seg), n_out)
        half = keep // 2
        out[:half] = spec[:half]
        out[n_out - (keep - half) :] = spec[len(seg) - (keep - half) :]
        y = np.fft.ifft(out) * (n_out / len(seg))
        first = (start - a) * up // down
        yield y[first : first + min(core, n - start) * up // down].astype(np.complex64)


def _need(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Sample file not found: {path}")


def _adsb_counts(blocks: Iterator[np.ndarray] | list[np.ndarray], rate: float) -> tuple[int, int]:
    decoder = ADSBDecoder(sample_rate=rate)
    for block in blocks:
        for i in range(0, len(block), 200_000):
            decoder.demodulate(block[i : i + 200_000], 0.0)
    return decoder._messages_decoded, decoder._df17_count


@pytest.fixture(scope="module")
def adsb_12s() -> np.ndarray:
    _need(ADSB_FILE)
    return load_iq(ADSB_FILE)[: 12 * 2_400_000]


@pytest.fixture(scope="module")
def adsb_native(adsb_12s: np.ndarray) -> tuple[int, int]:
    return _adsb_counts([adsb_12s], 2_400_000)


@pytest.mark.parametrize(("rate", "up", "down"), [(2_500_000, 25, 24), (10_000_000, 25, 6)])
def test_adsb_decodes_at_the_radios_rate(
    adsb_12s: np.ndarray, adsb_native: tuple[int, int], rate: int, up: int, down: int
) -> None:
    messages, df17 = _adsb_counts(fft_resample(adsb_12s, up, down), rate)
    native_messages, native_df17 = adsb_native
    assert native_df17 >= 40
    assert df17 >= 0.9 * native_df17
    assert messages >= 0.9 * native_messages


def test_dab_decodes_at_the_radios_rate() -> None:
    _need(DAB_FILE)
    rate = 5_000_000
    decoder = DABDecoder(sample_rate=rate)
    audio_s = 0.0
    selected = False
    for block in fft_resample(load_iq(DAB_FILE), 625, 256):
        for i in range(0, len(block), 250_000):
            decoder.demodulate(block[i : i + 250_000], 0.0)
            decoder.get_messages()
            audio_s += sum(len(b.samples) / b.sample_rate for b in decoder.get_audio())
            if not selected and decoder.stats.frames_processed > 10:
                result = decoder.select_service()
                selected = bool(result) and not result.startswith("Error")
    stats = decoder.stats
    # As at 2.048 MS/s (test_decoder.py): every frame, every FIC CRC, the audio.
    assert stats.frames_processed >= 100
    assert stats.fibs_decoded > 0 and stats.fibs_crc_ok == stats.fibs_decoded
    assert audio_s > 6.0
