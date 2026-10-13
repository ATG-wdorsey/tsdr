"""Targets for `[` / `]`: the nearest one in a direction, within the captured band."""

from __future__ import annotations

from collections.abc import Iterable

from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.sdr.processing import find_peaks


def next_target(
    direction: int,
    current_freq: float,
    targets: Iterable[float],
    limits: tuple[float, float] | None = None,
) -> float | None:
    """Nearest target strictly in `direction` and within `limits`. None if none."""
    if direction == 0:
        return None
    lo, hi = limits if limits is not None else (float("-inf"), float("inf"))
    in_dir = (t for t in targets if lo <= t <= hi and (t - current_freq) * direction > 0)
    return min(in_dir, key=lambda t: abs(t - current_freq), default=None)


def peak_frequencies(fft: FFTUpdateEvent) -> list[float]:
    """Spectrum peaks, for when there is no demodulated channel to measure."""
    return [freq for freq, _ in find_peaks(fft.spectrum, fft.center_frequency, fft.sample_rate)]
