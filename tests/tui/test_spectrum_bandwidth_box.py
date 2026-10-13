"""Bandwidth-box column clamps: the box tracks the dial while the view can
pan away from it entirely, so both columns must stay inside [0, width]."""

import numpy as np

from tsdr.tui.widgets.dsp_utils import shade_columns
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget


def _widget(tuned: float, bw: float | None, sideband: str | None = None) -> SpectrumWidget:
    w = SpectrumWidget()
    w._tuned_frequency = tuned
    w._channel_bandwidth = bw
    w._sideband = sideband
    return w


def test_box_inside_view() -> None:
    w = _widget(tuned=100e6, bw=20e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) == (40, 60)


def test_box_right_of_view_is_none() -> None:
    w = _widget(tuned=120e6, bw=12.5e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None


def test_box_left_of_view_is_none() -> None:
    w = _widget(tuned=90e6, bw=12.5e3)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None


def test_box_partially_off_view_clamps() -> None:
    w = _widget(tuned=100.05e6, bw=20e3)
    low, high = w._compute_bandwidth_range(100, 99.95e6, 100.05e6)
    assert (low, high) == (90, 100)


def test_no_bandwidth_is_none() -> None:
    w = _widget(tuned=100e6, bw=None)
    assert w._compute_bandwidth_range(100, 99.95e6, 100.05e6) is None


# --- SEARCH range shading ---------------------------------------------------


def test_search_columns_follow_the_view() -> None:
    w = _widget(tuned=100e6, bw=12.5e3)
    assert w._search_x(100, 99.95e6, 100.05e6) is None  # SEARCH off
    w._search_range = (99.97e6, 100.02e6)
    assert w._search_x(100, 99.95e6, 100.05e6) == (20, 70)
    w._search_range = (99.9e6, 100.2e6)
    assert w._search_x(100, 99.95e6, 100.05e6) == (-50, 250)  # edges off screen
    w._search_range = (100.1e6, 100.2e6)
    assert w._search_x(100, 99.95e6, 100.05e6) is None


def test_shading_tints_only_empty_pixels_and_marks_visible_edges() -> None:
    buf = np.zeros((4, 10, 4), dtype=np.uint8)
    buf[3, :] = [1, 2, 3, 255]  # the trace's fill
    tint = np.array([9, 9, 9, 28], dtype=np.uint8)
    edge = np.array([7, 7, 7, 120], dtype=np.uint8)
    shade_columns(buf, 2, 6, tint, edge)
    assert (buf[0, 3] == tint).all() and (buf[0, 1] == 0).all() and (buf[0, 6] == 0).all()
    assert (buf[0, 2] == edge).all() and (buf[0, 5] == edge).all()
    assert (buf[3, 3] == [1, 2, 3, 255]).all()
    clear = np.zeros((2, 10, 4), dtype=np.uint8)
    shade_columns(clear, -5, 20, tint, edge)  # both edges off the plot: no edge lines
    assert (clear[:, 0] == tint).all() and (clear[:, 9] == tint).all()


def test_text_rows_shade_the_search_columns() -> None:
    w = _widget(tuned=100e6, bw=12.5e3)
    strip = w._render_bar_row((0, 0, 0, 0, 0), 5, (1, 3))
    backgrounds = [seg.style.bgcolor for seg in strip for _ in seg.text]
    assert [bg is not None for bg in backgrounds] == [False, True, True, False, False]
