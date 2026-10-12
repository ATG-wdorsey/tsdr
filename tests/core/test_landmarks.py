from __future__ import annotations

import numpy as np

from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.landmarks import next_target, peak_frequencies


def _fft_with_peak_at(peak_freq: float) -> FFTUpdateEvent:
    """Synthesize a spectrum with one discoverable peak at (approximately) peak_freq."""
    n_bins = 128
    sample_rate = 2_400_000.0
    center_freq = 14_000_000.0
    freq_resolution = sample_rate / n_bins
    center_bin = n_bins // 2
    spectrum = np.full(n_bins, -80.0, dtype=np.float32)
    peak_bin = int(round((peak_freq - center_freq) / freq_resolution)) + center_bin
    if 0 <= peak_bin < n_bins:
        spectrum[peak_bin] = -30.0  # well above noise floor + threshold
    return FFTUpdateEvent(
        device_id="d",
        spectrum=spectrum,
        frequencies=np.empty(n_bins, dtype=np.float32),
        center_frequency=center_freq,
        sample_rate=sample_rate,
    )


def test_next_target_picks_nearest_in_direction() -> None:
    targets = [14_100_000.0, 14_200_000.0, 13_900_000.0]
    assert next_target(1, 14_000_000, targets) == 14_100_000
    assert next_target(-1, 14_000_000, targets) == 13_900_000


def test_next_target_stays_within_the_limits() -> None:
    """Targets beyond the captured band would move the LO: `[` / `]` stop at its edge."""
    targets = [14_100_000.0, 15_000_000.0]
    assert next_target(1, 14_100_000, targets, (13_000_000.0, 14_500_000.0)) is None
    assert next_target(1, 14_000_000, targets, (13_000_000.0, 14_500_000.0)) == 14_100_000


def test_next_target_empty_returns_none() -> None:
    assert next_target(1, 14_000_000, []) is None


def test_next_target_zero_direction() -> None:
    assert next_target(0, 14_000_000, [14_100_000.0]) is None


def test_next_target_skips_current_freq() -> None:
    # A target exactly at current_freq is *not* in either direction.
    assert next_target(1, 14_000_000, [14_000_000.0, 14_100_000.0]) == 14_100_000


def test_peak_frequencies() -> None:
    peaks = peak_frequencies(_fft_with_peak_at(14_050_000))
    assert len(peaks) == 1 and abs(peaks[0] - 14_050_000) < 20_000
