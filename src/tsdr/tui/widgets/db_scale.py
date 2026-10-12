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
# The waterfall's color key: this wide, centered in the label column.
KEY_COLS = 3
# Below this many plot columns the labels cost too much: no label column.
_MIN_PLOT_COLS = 20
# Mean of 10*log10(X) for exponentially distributed X (a noise bin's power)
# is 10*log10(E[X]) minus 10*log10(e) times Euler's gamma.
DB_MEAN_NOISE_BIAS = 10.0 * math.log10(math.e) * 0.5772156649015329
# The grid, and the dB labels the waterfall's colors don't suit.
GRID_GREY = "grey35"
# dB labels take the waterfall's color for their level, brightened to at
# least this luma (0-255; grey35 is 89) so the colormap's dark reds stay
# legible on a dark background...
LABEL_MIN_LUMA = 110.0
# ...but low in the window, colors darker than this are the noise floor's on
# the waterfall (Classic's navy): those labels stay grey.
LABEL_GREY_LUMA = 60.0


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


def _luma(rgb: tuple[float, float, float]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def label_color(rgb: tuple[int, int, int], position: float) -> str:
    """The color of a dB label whose level shows as `rgb` on the waterfall,
    `position` (0-1) of the way up the dB window."""
    if position < 0.5 and _luma(rgb) < LABEL_GREY_LUMA:
        return GRID_GREY
    r, g, b = legible(rgb)
    return f"rgb({r},{g},{b})"


def legible(rgb: tuple[int, int, int], min_luma: float = LABEL_MIN_LUMA) -> tuple[int, int, int]:
    """`rgb` brightened to at least `min_luma` for text on a dark background:
    scaled up, keeping its hue, then mixed with white where full brightness
    isn't enough (pure blue reaches only 29)."""
    r, g, b = (float(c) for c in rgb)
    luma = _luma((r, g, b))
    if luma >= min_luma:
        return rgb
    peak = max(r, g, b)
    if peak > 0:
        scale = min(min_luma / luma, 255.0 / peak)
        r, g, b, luma = r * scale, g * scale, b * scale, luma * scale
    if luma < min_luma:
        mix = (min_luma - luma) / (255.0 - luma)
        r, g, b = (c + mix * (255.0 - c) for c in (r, g, b))
    return round(r), round(g), round(b)


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
