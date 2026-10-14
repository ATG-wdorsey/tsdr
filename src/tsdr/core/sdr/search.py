"""SEARCH: find activity within the capture and put the dial on it.

The dial jumps between channels on the tuning-step grid inside the captured
band while the hardware stays put (free tuning). Levels are in-channel power
in dBFS, the squelch's units, summed from the unaveraged spectrum. A channel
that stays above the trigger for ATTACK_S is eligible, and the strongest
eligible channel wins. The squelch threshold is the hold threshold: SEARCH
leaves the current channel only after it has stayed below the squelch for
longer than the hold time, or after the maximum dwell time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tsdr.core.sdr.voice_squelch import Verdict

LOCKOUT_TAG = "lockout"
DEFAULT_HOLD_S = 2.0
# Without a trigger set, SEARCH stops on channels this far above the squelch.
TRIGGER_ABOVE_SQUELCH_DB = 3.0
# A channel must stay above the trigger this long before SEARCH jumps to it.
ATTACK_S = 0.1
# SEARCH is for narrowband channels: not broadcast FM, DAB or ADS-B.
MAX_CHANNEL_BANDWIDTH_HZ = 50e3
# A jump not yet seen in the config after this long may be requested again.
_PENDING_TIMEOUT_S = 0.5

SearchState = Literal["searching", "holding", "waiting"]


@dataclass(frozen=True)
class SearchConfig:
    """An active SEARCH (`DeviceConfig.search`; None = off)."""

    low_hz: float
    high_hz: float
    step_hz: float
    trigger_db: float | None = None  # None = TRIGGER_ABOVE_SQUELCH_DB over the squelch, live
    hold_s: float = DEFAULT_HOLD_S
    max_dwell_s: float | None = None  # None = unlimited
    voice_squelch: bool = False  # VSQ: transmissions without speech are let go
    lockouts: frozenset[float] = frozenset()  # this session's, Hz
    memory_lockouts: frozenset[float] = frozenset()  # memories tagged LOCKOUT_TAG, Hz
    run: int = 0  # bumped by each start, so a jump requested by an earlier run is dropped

    def validate(self) -> None:
        numbers = (self.low_hz, self.high_hz, self.step_hz, self.hold_s)
        optional = (self.trigger_db, self.max_dwell_s)
        if not all(math.isfinite(v) for v in numbers + tuple(v for v in optional if v is not None)):
            raise ValueError(f"search settings must be finite numbers, got {self}")
        if not 0 < self.low_hz < self.high_hz:
            raise ValueError(f"search range must be increasing, got {self.low_hz}-{self.high_hz}")
        if self.step_hz <= 0:
            raise ValueError(f"search step must be positive, got {self.step_hz}")
        if self.hold_s < 0:
            raise ValueError(f"search hold must not be negative, got {self.hold_s}")
        if self.max_dwell_s is not None and self.max_dwell_s <= 0:
            raise ValueError(f"search max dwell must be positive, got {self.max_dwell_s}")

    def trigger_for(self, squelch_db: float) -> float:
        if self.trigger_db is None:
            return squelch_db + TRIGGER_ABOVE_SQUELCH_DB
        return self.trigger_db

    def channels(self) -> np.ndarray:
        """Channel centers: the step grid within [low_hz, high_hz]."""
        return step_grid(self.low_hz, self.high_hz, self.step_hz)

    def is_locked_out(self, frequency_hz: float, bandwidth_hz: float) -> bool:
        half = max(self.step_hz, bandwidth_hz) / 2
        return any(abs(frequency_hz - f) < half for f in self.lockouts | self.memory_lockouts)


def step_grid(low_hz: float, high_hz: float, step_hz: float) -> np.ndarray:
    """Multiples of `step_hz` within [low_hz, high_hz]."""
    first = math.ceil(low_hz / step_hz - 1e-9)
    last = math.floor(high_hz / step_hz + 1e-9)
    return np.arange(first, last + 1, dtype=np.float64) * step_hz


def locating_bandwidth(step_hz: float, bandwidth_hz: float) -> float:
    """Measuring width for placing a signal on the step grid.

    A channel wider than the step (12.5 kHz NFM on a 5 kHz grid) holds a
    narrow signal whole at several grid points, which then read alike, and
    the strongest of them was as likely one step off as on the signal.
    Windows one step wide put it at a single grid point.
    """
    return min(step_hz, bandwidth_hz)


def locate_levels(
    spectrum_db: np.ndarray,
    center_hz: float,
    sample_rate: float,
    channels_hz: np.ndarray,
    step_hz: float,
    bandwidth_hz: float,
    enbw_bins: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Levels over locating_bandwidth for a step grid of channels, and for
    the grid channels just past each end (for local_peaks)."""
    if len(channels_hz) == 0:
        return np.empty(0), (-np.inf, -np.inf)
    grid = np.concatenate(([channels_hz[0] - step_hz], channels_hz, [channels_hz[-1] + step_hz]))
    levels = channel_powers_db(
        spectrum_db,
        center_hz,
        sample_rate,
        grid,
        locating_bandwidth(step_hz, bandwidth_hz),
        enbw_bins,
    )
    return levels[1:-1], (float(levels[0]), float(levels[-1]))


def local_peaks(
    locate_db: np.ndarray, outer_db: tuple[float, float] = (-np.inf, -np.inf)
) -> np.ndarray:
    """Channels at a signal's center: stronger than the channels on either side
    (`outer_db`: the grid channels just past each end, so a signal beyond an
    edge doesn't make the edge a peak). A strong signal leaks into the channels
    beside it, 13-15 dB down 10 kHz away for NFM; those slope down from its
    center and are never peaks. Neighbors that read exactly alike measured the
    same FFT bins (a grid finer than the bins): such a run is one peak, at its
    middle."""
    n = len(locate_db)
    peaks = np.zeros(n, dtype=bool)
    if n == 0:
        return peaks
    edges = np.flatnonzero(np.diff(locate_db) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [n])) - 1
    level = locate_db[starts]
    below = np.where(starts > 0, locate_db[starts - 1], outer_db[0])
    above = np.where(ends < n - 1, locate_db[np.minimum(ends + 1, n - 1)], outer_db[1])
    is_peak = (level > below) & (level > above)
    peaks[(starts[is_peak] + ends[is_peak]) // 2] = True
    return peaks


def signal_peaks(
    channels_hz: np.ndarray,
    levels_db: np.ndarray,
    threshold_db: float,
    *,
    locate_db: np.ndarray | None = None,
    outer_db: tuple[float, float] = (-np.inf, -np.inf),
    around_hz: float | None = None,
    half_width_hz: float = 0.0,
) -> list[float]:
    """One frequency per signal: channels at or above `threshold_db` that are
    local peaks of `locate_db` (levels over locating_bandwidth; default
    `levels_db`), leaving out a peak within `half_width_hz` of `around_hz`
    (the signal the dial is on)."""
    locate = levels_db if locate_db is None else locate_db
    found = (levels_db >= threshold_db) & local_peaks(locate, outer_db)
    if around_hz is not None:
        found &= np.abs(channels_hz - around_hz) >= half_width_hz
    return [float(f) for f in channels_hz[found]]


@dataclass(frozen=True)
class SearchStatus:
    state: SearchState
    channel_hz: float
    level_db: float
    hold_left_s: float
    active: int  # channels above the trigger
    voice: Verdict | None = None  # VSQ's verdict on the dial's transmission; None = VSQ off


def channel_powers_db(
    spectrum_db: np.ndarray,
    center_hz: float,
    sample_rate: float,
    channels_hz: np.ndarray,
    bandwidth_hz: float,
    enbw_bins: float,
) -> np.ndarray:
    """In-channel power of each channel in dBFS, the squelch's units.

    `spectrum_db` is compute_fft's output: fftshifted, scaled so a tone reads
    its amplitude. The bins a channel covers, summed and divided by the
    window's noise bandwidth, give the power a `bandwidth_hz` channel filter
    passes. A channel narrower than a bin, covering no bin center, takes the
    bin nearest its center. Channels outside the spectrum read -inf.
    """
    n = len(spectrum_db)
    out = np.full(len(channels_hz), -np.inf)
    if n == 0 or len(channels_hz) == 0:
        return out
    bin_hz = sample_rate / n
    # Bin k is centered at center_hz + (k - n/2) * bin_hz; a channel takes the
    # bins whose centers it covers, [lo, hi).
    position = (channels_hz - center_hz) / bin_hz + n / 2
    half = bandwidth_hz / 2 / bin_hz
    lo = np.ceil(position - half)
    hi = np.ceil(position + half)
    narrow = hi <= lo
    nearest = np.floor(position + 0.5)
    lo = np.clip(np.where(narrow, nearest, lo), 0, n).astype(np.intp)
    hi = np.clip(np.where(narrow, nearest + 1, hi), 0, n).astype(np.intp)
    inside = hi > lo
    if not inside.any():
        return out
    first, last = int(lo[inside].min()), int(hi[inside].max())
    power = np.power(10.0, spectrum_db[first:last].astype(np.float64) / 10.0)
    sums = np.concatenate(([0.0], np.cumsum(power)))
    total = sums[hi[inside] - first] - sums[lo[inside] - first]
    out[inside] = 10.0 * np.log10(total / enbw_bins + 1e-20)
    return out


class SearchController:
    """SEARCH decisions from channel levels, for one run."""

    def __init__(self) -> None:
        self._channels = np.empty(0)
        self._above_since = np.empty(0)
        self._current: float | None = None
        self._arrived_at = 0.0
        self._last_open = -math.inf
        self._heard = False
        # Channels left at the maximum dwell or as not voice (VSQ): not eligible
        # until they drop below the hold.
        self._disarmed: set[float] = set()
        self._pending: float | None = None
        self._pending_at = 0.0

    def update(
        self,
        now: float,
        config: SearchConfig,
        channels_hz: np.ndarray,
        levels_db: np.ndarray,
        *,
        tuned_hz: float,
        current_db: float,
        hold_db: float,  # the squelch threshold
        bandwidth_hz: float,
        lo_hz: float,
        lo_exclusion_hz: float,
        locate_db: np.ndarray | None = None,
        outer_locate_db: tuple[float, float] = (-np.inf, -np.inf),
        voice: Verdict | None = None,
    ) -> tuple[float | None, SearchStatus]:
        """Returns the channel to jump to (or None) and the status. `locate_db`
        (levels over locating_bandwidth) picks among eligible channels, which
        must be local peaks of it (`outer_locate_db`: the grid channels just
        past each end): SEARCH stops on a signal's center, never on the edge of
        a strong signal, locked out or not. `voice` is VSQ's verdict on the
        dial's transmission (None = VSQ off): "not_voice" closes the squelch
        for the hold, and SEARCH leaves the channel disarmed."""
        if not np.array_equal(channels_hz, self._channels):
            self._channels = channels_hz
            self._above_since = np.full(len(channels_hz), np.nan)
        trigger_db = config.trigger_for(hold_db)
        hold_db = min(hold_db, trigger_db)
        is_open = current_db >= hold_db and voice != "not_voice"

        if tuned_hz != self._current:
            # Within a run the dial moves only by SEARCH's jumps and `[` / `]`
            # (any other move stops it): each new channel gets a full hold. The
            # channel the dial is on at the start is held only if open.
            arrived = self._current is not None
            self._current = tuned_hz
            self._arrived_at = now
            self._last_open = now if arrived else -math.inf
            self._pending = None
        if is_open:
            self._last_open = now
        self._heard = self._heard or self._last_open > -math.inf

        released = (
            config.is_locked_out(tuned_hz, bandwidth_hz) or now - self._last_open > config.hold_s
        )
        dwelt = config.max_dwell_s is not None and now - self._arrived_at >= config.max_dwell_s

        above = levels_db >= trigger_db
        self._above_since[above & np.isnan(self._above_since)] = now
        self._above_since[~above] = np.nan

        half = max(config.step_hz, bandwidth_hz) / 2
        self._disarmed = {
            f
            for f in self._disarmed
            if np.any(levels_db[np.abs(channels_hz - f) < half] >= hold_db)
        }
        pick = levels_db if locate_db is None else locate_db
        eligible = above & (now - self._above_since >= ATTACK_S)
        eligible &= local_peaks(pick, outer_locate_db)
        eligible &= np.abs(channels_hz - tuned_hz) >= half
        eligible &= np.abs(channels_hz - lo_hz) >= lo_exclusion_hz
        for f in config.lockouts | config.memory_lockouts | self._disarmed:
            eligible &= np.abs(channels_hz - f) >= half

        target: float | None = None
        waiting_on_jump = self._pending is not None and now - self._pending_at < _PENDING_TIMEOUT_S
        if (released or dwelt) and not waiting_on_jump and eligible.any():
            target = float(channels_hz[np.argmax(np.where(eligible, pick, -np.inf))])
            if (dwelt and not released) or voice == "not_voice":
                self._disarmed.add(tuned_hz)
            self._pending = target
            self._pending_at = now

        if not released:
            state: SearchState = "holding"
            hold_left = (
                max(0.0, config.hold_s - (now - self._last_open)) if not is_open else config.hold_s
            )
        else:
            state = "waiting" if self._heard else "searching"
            hold_left = 0.0
        status = SearchStatus(
            state, tuned_hz, current_db, hold_left, int(np.count_nonzero(above)), voice
        )
        return target, status
