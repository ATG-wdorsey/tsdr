"""Unit tests for find_peaks on synthetic spectra, and the meter's C/N on synthetic IQ."""

from __future__ import annotations

import numpy as np

from tsdr.core.sdr.processing import BandFloor, channel_cn_db, compute_fft, find_peaks


def _make_spectrum(
    n_bins: int,
    *,
    noise_db: float = -80.0,
    peaks: dict[int, float] | None = None,
) -> np.ndarray:
    """Flat noise floor with optional peaks injected at specific bin indices."""
    spectrum = np.full(n_bins, noise_db, dtype=np.float32)
    if peaks:
        for bin_idx, db in peaks.items():
            spectrum[bin_idx] = db
    return spectrum


def test_find_peaks_empty_below_threshold() -> None:
    spectrum = _make_spectrum(1024)
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert peaks == ()


def test_find_peaks_returns_strong_signal() -> None:
    n = 1024
    spectrum = _make_spectrum(n, peaks={300: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert len(peaks) == 1
    freq_res = 2.4e6 / n
    expected_freq = 100e6 + (300 - n // 2) * freq_res
    assert abs(peaks[0][0] - expected_freq) < 1e-6
    assert peaks[0][1] == -30.0


def test_find_peaks_threshold_rejects() -> None:
    n = 1024
    # Median = noise_db = -80. Threshold = -70. -75 < threshold → rejected.
    spectrum = _make_spectrum(n, peaks={300: -75.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6, threshold_db_above_median=10.0)
    assert peaks == ()


def test_find_peaks_multiple_distinct() -> None:
    n = 1024
    spectrum = _make_spectrum(n, peaks={200: -40.0, 600: -35.0, 900: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6)
    assert len(peaks) == 3
    # Ascending by freq.
    freqs = [p[0] for p in peaks]
    assert freqs == sorted(freqs)


def test_find_peaks_cluster_suppression() -> None:
    n = 1024
    # Three closely-spaced peaks; cluster_bins=5 should keep only the strongest.
    spectrum = _make_spectrum(n, peaks={500: -40.0, 502: -35.0, 504: -30.0})
    peaks = find_peaks(spectrum, 100e6, 2.4e6, cluster_bins=5)
    assert len(peaks) == 1
    assert peaks[0][1] == -30.0


def test_find_peaks_top_n_cap() -> None:
    n = 4096
    # 50 peaks 50 bins apart, all above threshold.
    injected = {100 + i * 50: -40.0 + i * 0.1 for i in range(50)}
    spectrum = _make_spectrum(n, peaks=injected)
    peaks = find_peaks(spectrum, 100e6, 2.4e6, max_peaks=10)
    assert len(peaks) == 10


def test_find_peaks_tiny_spectrum() -> None:
    # Spectrum too small to have interior bins → return empty.
    assert find_peaks(np.array([1.0, 2.0], dtype=np.float32), 100e6, 2.4e6) == ()
    assert find_peaks(np.array([], dtype=np.float32), 100e6, 2.4e6) == ()


def _band_noise(n: int, fs: float, center: float, width: float, power: float, rng) -> np.ndarray:
    """Complex noise of `power` filling `width` Hz around `center` (a stand-in for a signal)."""
    spectrum = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spectrum[np.abs(np.fft.fftfreq(n, 1 / fs) - center) > width / 2] = 0
    x = np.fft.ifft(spectrum)
    return x * np.sqrt(power / np.mean(np.abs(x) ** 2))


def test_the_meter_reads_a_channels_c_n_whatever_its_neighbors() -> None:
    """A 150 kHz signal at 30 dB C/N in its 200 kHz channel, a station 20 dB stronger 220 kHz
    away. The meter read the strongest bin over the channel either side, where the neighbor,
    a strong signal's own skirts or an FM station's HD Radio sidebands sit. Its noise is now
    the band's floor (its quietest tenth), its signal the channel's whole power."""
    rng = np.random.default_rng(1)
    fs, n, frames = 2.4e6, 8192, 80
    noise = 1e-6
    signal = noise * 200e3 / fs * 10**3.0
    x = (rng.standard_normal(n * frames) + 1j * rng.standard_normal(n * frames)) * np.sqrt(
        noise / 2
    )
    x += _band_noise(n * frames, fs, 300e3, 150e3, signal, rng)
    x += _band_noise(n * frames, fs, 520e3, 150e3, 100 * signal, rng)
    window = np.hanning(n).astype(np.float32)
    inv_norm_sq = 1.0 / (float(window.mean()) * n) ** 2
    floor = BandFloor()
    readings = []
    for k in range(frames):
        spectrum = compute_fft(x[k * n : (k + 1) * n].astype(np.complex64), window, inv_norm_sq)
        floor.update(spectrum, 1, 0.0, fs, 0.05)
        cn = channel_cn_db(spectrum, fs, 300e3, 200e3, floor)
        if k >= 40 and cn is not None:  # the floor averaged over 2 s
            readings.append(cn)
    assert len(readings) == frames - 40
    assert abs(float(np.median(readings)) - 30.0) < 1.0
