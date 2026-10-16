from __future__ import annotations

# TODO: should these be moved?
import math
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from tsdr.core.sdr.datatypes import SignalStatistics

if TYPE_CHECKING:
    from tsdr.radio.dsp.fft import FFTPlan


def fft_window(window_type: str, size: int) -> np.ndarray:
    """The spectrum's window function ("hanning", "hamming", "blackman"; else rectangular)."""
    if window_type == "hanning":
        return np.hanning(size).astype(np.float32)
    if window_type == "hamming":
        return np.hamming(size).astype(np.float32)
    if window_type == "blackman":
        return np.blackman(size).astype(np.float32)
    return np.ones(size, dtype=np.float32)


@cache
def window_enbw_bins(window_type: str, size: int) -> float:
    """Equivalent noise bandwidth of the window, in bins (1.5 for Hann).

    compute_fft scales a tone to its amplitude, so summed bins overstate
    noise power by this factor.
    """
    window = fft_window(window_type, size).astype(np.float64)
    return float(size * np.sum(window**2) / np.sum(window) ** 2)


def compute_fft(
    iq_samples: np.ndarray,
    window: np.ndarray,
    inv_norm_sq: float,
    plan: FFTPlan | None = None,
) -> np.ndarray:
    """Compute FFT power spectrum from IQ samples.

    Args:
        iq_samples: Complex IQ samples (numpy array, dtype=complex64/128)
        window: Pre-computed window function array
        inv_norm_sq: Pre-computed 1/(coherent_gain * N)^2 normalization factor
        plan: Pre-configured FFTPlan for FFTW acceleration.
              Falls back to numpy when None.

    Returns:
        Power spectrum in dB (numpy array, float32)
        - Zero frequency at center (fftshift applied)
        - Values in dBFS scale: 10 * log10(power)
    """
    windowed = iq_samples * window
    if plan is not None:
        fft_result = np.fft.fftshift(plan.execute(windowed))
    else:
        fft_result = np.fft.fftshift(np.fft.fft(windowed))

    # |X|^2 directly avoids the sqrt inside np.abs.
    power: np.ndarray = (fft_result.real**2 + fft_result.imag**2).astype(np.float32, copy=False)
    power *= inv_norm_sq
    # log10(0) guard; must stay below the noise floor of high-dynamic-range
    # receivers (1e-20 = -200 dBFS), else it clamps their spectrum flat.
    power += 1e-20
    # 10 log10 as log times a constant: numpy's float32 log10 takes 15x longer than its log.
    np.log(power, out=power)
    power *= np.float32(10.0 / np.log(10.0))
    return power


# The noise floor (the owner's 10 % rule): the 10th percentile of the band's bins, leaving
# out the outer tenth of each side (the anti-alias filter's roll-off) and the channel
# measured, in the spectrum's power averaged over FLOOR_TAU_S: the quietest tenth of the
# band. In one frame each noise bin spreads over 20 dB, and where the floor is not flat
# (106.7 MHz on a Pluto: -105 dBFS at the band's edges, -92 among the FM stations) its
# 10th percentile came out 4 dB above that of the averaged spectrum. A band with fewer bins
# left has no floor.
FLOOR_PERCENTILE = 10.0
FLOOR_TAU_S = 1.0
_FLOOR_BAND = 0.8
_FLOOR_MIN_BINS = 64
_FLOOR_MAX_FRAMES = 256
_CN_MIN_DB = -20.0


@cache
def _noise_percentile_ratio(frames: int) -> float:
    """A noise bin's FLOOR_PERCENTILE-th percentile over its mean, its power averaged over
    `frames` FFT frames: gamma distributed, shape `frames` (one frame: exponential, 9.8 dB
    under the mean; eight: 2.4 dB)."""
    target = FLOOR_PERCENTILE / 100.0

    def cdf(x: float) -> float:  # gamma(frames, 1), integer shape
        term, total = 1.0, 1.0
        for i in range(1, frames):
            term *= x / i
            total += term
        return 1.0 - math.exp(-x) * total

    lo, hi = 0.0, float(frames)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < target else (lo, mid)
    return (lo + hi) / 2 / frames


def channel_bins(n: int, sample_rate: float, offset_hz: float, bandwidth: float) -> tuple[int, int]:
    """The bins [lo, hi) a channel covers (their centers) in an fftshifted n-point spectrum."""
    bin_hz = sample_rate / n
    position = offset_hz / bin_hz + n / 2
    half = bandwidth / 2 / bin_hz
    return max(int(np.ceil(position - half)), 0), min(int(np.ceil(position + half)), n)


class BandFloor:
    """The band's noise floor by the 10 % rule, from its power averaged over FLOOR_TAU_S (a
    leaky integrator); a retune or a new rate or FFT size starts it over."""

    def __init__(self) -> None:
        self._power: np.ndarray | None = None
        self._key: tuple[float, float, int] | None = None
        # The average's spread as a share of the mean, squared: 1/frames for independent
        # frames, kept exact through the integrator.
        self._spread = 1.0

    def update(
        self, spectrum_db: np.ndarray, frames: int, center_hz: float, sample_rate: float, dt: float
    ) -> None:
        """Take in a spectrum (compute_fft's, its power averaged over `frames` frames), `dt`
        seconds after the last."""
        power = np.exp(spectrum_db.astype(np.float32) * np.float32(np.log(10.0) / 10.0))
        key = (center_hz, sample_rate, len(spectrum_db))
        if self._power is None or key != self._key:
            self._power, self._key, self._spread = power, key, 1.0 / max(frames, 1)
            return
        a = 1.0 - math.exp(-max(dt, 0.0) / FLOOR_TAU_S)
        self._power += np.float32(a) * (power - self._power)
        self._spread = (1.0 - a) ** 2 * self._spread + a**2 / max(frames, 1)

    def noise_per_bin(self, exclude: tuple[int, int]) -> float | None:
        """Mean noise power per bin (compute_fft's linear units), leaving out the bins
        `exclude` (a channel measured); None without enough band."""
        power = self._power
        if power is None:
            return None
        n = len(power)
        edge = int(n * (1 - _FLOOR_BAND) / 2)
        lo, hi = exclude
        band = np.concatenate([power[edge : min(lo, n - edge)], power[max(hi, edge) : n - edge]])
        if len(band) < _FLOOR_MIN_BINS:
            return None
        frames = min(max(round(1.0 / self._spread), 1), _FLOOR_MAX_FRAMES)
        return float(np.percentile(band, FLOOR_PERCENTILE)) / _noise_percentile_ratio(frames)


def channel_cn_db(
    spectrum_db: np.ndarray,
    sample_rate: float,
    channel_offset_hz: float,
    channel_bandwidth: float,
    floor: BandFloor,
) -> float | None:
    """C/N of a channel in dB: the power in the channel (the signal's area) over the noise
    in its bandwidth, the band's floor per bin times its bins. `spectrum_db` is compute_fft's
    (fftshifted); `floor` has taken it in."""
    n = len(spectrum_db)
    if n == 0 or channel_bandwidth <= 0:
        return None
    lo, hi = channel_bins(n, sample_rate, channel_offset_hz, channel_bandwidth)
    if hi <= lo:
        return None
    noise_per_bin = floor.noise_per_bin((lo, hi))
    if noise_per_bin is None or noise_per_bin <= 0:
        return None
    total = float(np.sum(np.power(10.0, spectrum_db[lo:hi].astype(np.float64) / 10.0)))
    noise = noise_per_bin * (hi - lo)
    return max(10.0 * math.log10(max(total - noise, 0.0) / noise + 1e-30), _CN_MIN_DB)


def compute_statistics(
    spectrum: np.ndarray,
    center_frequency: float,
    sample_rate: float,
) -> SignalStatistics:
    """Compute statistics from power spectrum.

    Args:
        spectrum: Power spectrum in dB (fftshifted, center bin = center freq)
        center_frequency: Center frequency in Hz
        sample_rate: Sample rate in Hz
    """
    n_bins = len(spectrum)
    center_bin = n_bins // 2

    peak_bin = int(np.argmax(spectrum))
    peak_power = float(spectrum[peak_bin])

    average_power = float(np.mean(spectrum))

    # Estimate noise floor (lower 20th percentile via partial sort)
    k = max(1, n_bins // 5)
    noise_floor = float(np.partition(spectrum, k)[k])
    bin_offset = peak_bin - center_bin

    # Frequency resolution
    freq_resolution = sample_rate / n_bins

    # Peak frequency = center_frequency + (bin_offset * freq_resolution)
    peak_frequency = center_frequency + (bin_offset * freq_resolution)

    return SignalStatistics(
        peak_power=peak_power,
        average_power=average_power,
        peak_frequency=peak_frequency,
        peak_bin=peak_bin,
        noise_floor=noise_floor,
        dynamic_range=peak_power - noise_floor,
    )


def find_peaks(
    spectrum_db: np.ndarray,
    center_frequency: float,
    sample_rate: float,
    *,
    threshold_db_above_median: float = 10.0,
    cluster_bins: int = 5,
    max_peaks: int = 32,
) -> tuple[tuple[float, float], ...]:
    """Return spectrum peaks as ((freq_hz, power_db), ...) ascending by freq.

    A bin qualifies as a peak when it is strictly greater than its two
    neighbours and at least `threshold_db_above_median` above the median power
    floor. Closely-spaced peaks (within `cluster_bins` of a stronger one) are
    suppressed. Only the strongest `max_peaks` survive.
    """
    n_bins = len(spectrum_db)
    if n_bins < 3:
        return ()

    threshold = float(np.median(spectrum_db)) + threshold_db_above_median

    interior = spectrum_db[1:-1]
    left = spectrum_db[:-2]
    right = spectrum_db[2:]
    mask = (interior > left) & (interior > right) & (interior > threshold)
    candidate_bins = np.flatnonzero(mask) + 1
    if candidate_bins.size == 0:
        return ()

    candidate_powers = spectrum_db[candidate_bins]

    # Cluster suppression: walk strongest-first, drop any candidate within
    # cluster_bins of an already-accepted one.
    order = np.argsort(candidate_powers)[::-1]
    accepted_bins: list[int] = []
    for idx in order:
        b = int(candidate_bins[idx])
        if all(abs(b - kept) > cluster_bins for kept in accepted_bins):
            accepted_bins.append(b)
            if len(accepted_bins) >= max_peaks:
                break

    center_bin = n_bins // 2
    freq_resolution = sample_rate / n_bins

    peaks = [
        (
            center_frequency + (b - center_bin) * freq_resolution,
            float(spectrum_db[b]),
        )
        for b in accepted_bins
    ]
    peaks.sort(key=lambda p: p[0])
    return tuple(peaks)


def apply_dc_offset_correction(iq_samples: np.ndarray) -> np.ndarray:
    """Remove DC offset from IQ samples.

    Many SDR devices have a DC spike at the center frequency due to
    imperfections in the mixer. Subtracting the mean removes it.
    """
    mean: np.ndarray = np.mean(iq_samples)  # type: ignore[assignment]
    result: np.ndarray = iq_samples - mean  # type: ignore[assignment]
    return result


def apply_iq_imbalance_correction(iq_samples: np.ndarray) -> np.ndarray:
    """Correct IQ imbalance.

    IQ imbalance occurs when I and Q channels have different gains or phases.
    This applies a simple correction by normalizing the I and Q channels.

    Args:
        iq_samples: Complex IQ samples

    Returns:
        IQ-corrected samples

    Example:
        >>> iq_corrected = apply_iq_imbalance_correction(iq_samples)
    """
    # Normalize I and Q to have the same variance: scale Q to match I's std.
    i_samples = iq_samples.real
    q_samples = iq_samples.imag

    i_std = np.std(i_samples)
    q_std = np.std(q_samples)

    if i_std > 0 and q_std > 0:
        scale_factor = i_std / q_std
        q_normalized: np.ndarray = q_samples * scale_factor  # type: ignore[assignment]
        result: np.ndarray = i_samples + 1j * q_normalized  # type: ignore[assignment]
        return result
    else:
        return iq_samples
