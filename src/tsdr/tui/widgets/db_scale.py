"""The spectrum's dB scale: grid levels, the label column on the left, and
where the squelch and SEARCH trigger read on the trace.

The trace is power per FFT bin, the squelch power in the whole channel. Noise
spreads over the channel, so a bin holds a small share of it: noise the
squelch reads at -82 dBFS in 12.5 kHz shows about 14 dB lower on the trace (a
~460 Hz bin noise bandwidth at 10 MS/s and 32768 points), and averaging dB
values lowers it another 2.5 dB. The reference lines sit where noise at those
levels shows, so a squelch set just above the noise floor is drawn just above
it. A narrow carrier, all in one bin, reads its full channel power there: it
must rise that much further above the line to open the squelch.
"""

from __future__ import annotations

import math

from tsdr.core.sdr.processing import window_enbw_bins

GUTTER_COLS = 5
# Below this many plot columns the labels cost too much: no label column.
_MIN_PLOT_COLS = 20
# Mean of 10*log10(X) for exponentially distributed X (a noise bin's power)
# is 10*log10(E[X]) minus 10*log10(e) times Euler's gamma.
DB_MEAN_NOISE_BIAS = 10.0 * math.log10(math.e) * 0.5772156649015329


def gutter_cols(width: int) -> int:
    """Columns for the dB labels at the left of a `width`-column display."""
    return GUTTER_COLS if width >= GUTTER_COLS + _MIN_PLOT_COLS else 0


def grid_levels(db_min: float, db_max: float, rows: float | None = None) -> list[float]:
    """Grid lines strictly inside the dB window: every 10 dB, or 5 dB when
    the window is under 40 dB; on a plot `rows` text rows tall, the step
    doubles until the lines are at least two rows apart, room for a label each."""
    window = db_max - db_min
    step = 10.0 if window >= 40.0 else 5.0
    while rows is not None and window > 0 and step / window * rows < 2.0:
        step *= 2.0
    level = math.ceil(db_min / step) * step
    levels = []
    while level < db_max:
        if level > db_min:
            levels.append(level)
        level += step
    return levels


def grid_label(level: float) -> str:
    return f"{level:4.0f} "[:GUTTER_COLS]


def noise_equivalent_db(
    channel_db: float,
    channel_bandwidth_hz: float,
    sample_rate: float,
    fft_size: int,
    window: str,
) -> float:
    """Where noise that a channel measures at `channel_db` (squelch units)
    reads on the spectrum trace."""
    bin_noise_hz = sample_rate / fft_size * window_enbw_bins(window, fft_size)
    return channel_db - 10.0 * math.log10(channel_bandwidth_hz / bin_noise_hz) - DB_MEAN_NOISE_BIAS
