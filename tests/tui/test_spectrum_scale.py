"""The spectrum's dB scale: grid lines, the label column, the squelch and
SEARCH trigger lines, and the waterfall kept in line with the plot."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from rich.style import Style
from textual.geometry import Size

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.memories import init_memory_store
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.processing import compute_fft, fft_window
from tsdr.core.sdr.search import TRIGGER_ABOVE_SQUELCH_DB
from tsdr.core.search_control import start_search
from tsdr.core.tuning_state import init_tuning_state
from tsdr.devices import MockParams
from tsdr.tui.widgets import waterfall_widget as waterfall_module
from tsdr.tui.widgets.db_scale import (
    GRID_GREY,
    GUTTER_COLS,
    KEY_COLS,
    LABEL_MIN_LUMA,
    grid_label,
    grid_levels,
    gutter_cols,
    label_color,
    legible,
    noise_equivalent_db,
)
from tsdr.tui.widgets.dsp_utils import draw_hline, project_spectrum
from tsdr.tui.widgets.spectrum_widget import _TRACE_COLOR, SpectrumWidget
from tsdr.tui.widgets.waterfall_widget import WaterfallWidget, level_color, level_label_color

RATE = 10e6
CENTER = 146.0e6


def test_grid_every_10_db_or_5_in_a_narrow_window() -> None:
    assert grid_levels(-115.0, -25.0) == [-110.0 + 10 * i for i in range(9)]
    assert grid_levels(-100.0, -70.0) == [-95.0, -90.0, -85.0, -80.0, -75.0]


def test_grid_spreads_out_on_a_short_plot() -> None:
    levels = grid_levels(-115.0, -25.0, rows=8)
    step = levels[1] - levels[0]
    assert step / 90.0 * 8 >= 2.0
    assert grid_levels(-115.0, -25.0, rows=18)[1] - grid_levels(-115.0, -25.0, rows=18)[0] == 10.0


def test_labels_fit_the_column() -> None:
    assert grid_label(-110.0) == "-110 "
    assert len(grid_label(-5.0)) == GUTTER_COLS
    assert gutter_cols(100) == GUTTER_COLS and gutter_cols(20) == 0


def _luma(rgb: tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def test_label_colors_are_brightened_only_as_far_as_needed() -> None:
    navy = (0, 0, 0x2C)
    lifted = legible(navy)
    assert _luma(lifted) >= LABEL_MIN_LUMA - 0.5
    assert lifted[2] == 255 and lifted[0] == lifted[1] < lifted[2]  # still blue
    dark_red = legible((0x4A, 0, 0))
    assert dark_red[0] == 255 and dark_red[1] == dark_red[2] < 100  # still red
    assert legible((0, 0, 0)) == (110, 110, 110)
    assert legible((0xFF, 0xFF, 0x00)) == (0xFF, 0xFF, 0x00)  # already legible: unchanged


def _css(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"rgb({r},{g},{b})"


def test_the_noise_floors_dark_colors_label_in_grey() -> None:
    assert label_color((0, 0, 100), 0.17) == GRID_GREY  # Classic's navy, low in the window
    assert label_color((17, 82, 208), 0.25) == _css(legible((17, 82, 208)))
    # Dark red high in the window is a strong signal: brightened, never grey.
    assert label_color((0x4A, 0, 0), 1.0) == _css(legible((0x4A, 0, 0)))


def _rgb(style) -> tuple[int, int, int]:  # type: ignore[no-untyped-def]
    triplet = style.color.get_truecolor()
    return triplet.red, triplet.green, triplet.blue


def test_the_squelch_line_sits_on_noise_the_squelch_reads_at_that_level() -> None:
    """Through the display path (FFT, the FFT stage's dB averaging, column
    averaging), noise the squelch measures at X dBFS in the channel shows
    where the line for X is drawn."""
    n, sigma2 = 32768, 1e-6
    window = fft_window("hanning", n)
    norm = 1.0 / (float(np.sum(window)) / n * n) ** 2
    rng = np.random.default_rng(1)
    avg = None
    for _ in range(30):
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(sigma2 / 2)
        spec = compute_fft(iq.astype(np.complex64), window, norm)
        avg = spec.copy() if avg is None else avg + (spec - avg) / 3
    assert avg is not None
    trace = project_spectrum(avg, -RATE / 2, RATE / 2, -RATE / 2, RATE / 2, 600)
    channel_db = 10 * math.log10(sigma2 * 12_500 / RATE)
    assert float(np.mean(trace)) == pytest.approx(
        noise_equivalent_db(channel_db, 12_500, RATE, n, "hanning"), abs=0.3
    )


# --- the widget -----------------------------------------------------------------


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> SDREngine:
    init_memory_store()
    init_tuning_state()
    engine = SDREngine()
    config = DeviceConfig(tuned_frequency=CENTER, center_frequency=CENTER, sample_rate=RATE)
    engine.add_device("radio", "mock", MockParams(), config)
    engine.set_audio_demod("radio", DemodSpec(mode="NFM"))
    monkeypatch.setattr(engine_module, "_active_engine", engine)
    yield engine
    engine.devices["radio"].state = DeviceState.STOPPED
    engine.shutdown(timeout=2.0)


def _widget() -> SpectrumWidget:
    w = SpectrumWidget()
    w.set_reactive(SpectrumWidget.db_min, -115.0)
    w.set_reactive(SpectrumWidget.db_max, -25.0)
    w.set_reactive(SpectrumWidget.image_mode, False)
    return w


def _event() -> FFTUpdateEvent:
    spectrum = np.full(4096, -105.0, dtype=np.float32)
    return FFTUpdateEvent(
        device_id="radio",
        spectrum=spectrum,
        frequencies=None,
        center_frequency=CENTER,
        sample_rate=RATE,
    )


def _strips(w: SpectrumWidget, width: int = 100, height: int = 22) -> list:
    w._read_config()
    frame = w._prepare_frame(_event(), width, height, (CENTER - RATE / 2, CENTER + RATE / 2))
    return w._render_frame(frame)


def _text_rows(w: SpectrumWidget, width: int = 100, height: int = 22) -> list[str]:
    return ["".join(seg.text for seg in strip) for strip in _strips(w, width, height)]


def test_reference_lines_follow_the_squelch_and_search(engine: SDREngine) -> None:
    w = _widget()
    w._read_config()
    assert w._squelch_line_db is None and w._trigger_line_db is None  # squelch off
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-82.0)
    w._read_config()
    expected = noise_equivalent_db(-82.0, 12_500.0, RATE, 2**15, "hanning")
    assert w._squelch_line_db == pytest.approx(expected)
    assert w._trigger_line_db is None
    engine.devices["radio"].state = DeviceState.RUNNING
    start_search("radio")
    w._read_config()
    assert w._trigger_line_db == pytest.approx(expected + TRIGGER_ABOVE_SQUELCH_DB)


def test_text_mode_labels_the_grid_in_its_own_column(engine: SDREngine) -> None:
    rows = _text_rows(_widget())
    assert all(len(row) == 100 for row in rows)
    labeled = [row[:GUTTER_COLS] for row in rows[1:] if row[:GUTTER_COLS].strip()]
    assert labeled[0] == " -30 " and labeled[-1] == "-110 "
    assert not any("SQ" in row for row in rows)


def test_grid_labels_take_the_waterfalls_color_for_their_level(engine: SDREngine) -> None:
    labels = {}
    for strip in _strips(_widget())[1:]:  # past the header
        seg = list(strip)[0]
        if seg.text.strip():
            labels[seg.text] = _rgb(seg.style)
    assert len(labels) == 9
    for text, rgb in labels.items():
        assert rgb == _rgb(Style(color=level_label_color(float(text), -115.0, -25.0)))
    grey = _rgb(Style(color=GRID_GREY))
    assert labels["-110 "] == labels["-100 "] == grey  # the noise floor's navy
    assert labels[" -90 "] == legible(level_color(-90.0, -115.0, -25.0))  # azure
    assert labels[" -70 "][0] >= 250 and labels[" -70 "][2] < 60  # orange
    assert labels[" -30 "][0] == 255 > labels[" -30 "][1]  # dark red, as red


def test_text_mode_marks_the_squelch_and_trigger(engine: SDREngine) -> None:
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-82.0)
    engine.devices["radio"].state = DeviceState.RUNNING
    start_search("radio", low_hz=145.0e6, high_hz=147.0e6, trigger_db=-60.0)
    strips = _strips(_widget())
    labels = ["".join(seg.text for seg in strip)[:GUTTER_COLS] for strip in strips]
    assert "  SQ " in labels and "  TR " in labels
    # The squelch and trigger labels keep their lines' colors.
    assert list(strips[labels.index("  SQ ")])[0].style.color.name == "dark_orange"
    assert list(strips[labels.index("  TR ")])[0].style.color.name == "yellow"
    yellow = []
    col = 0
    for seg in strips[labels.index("  TR ")]:
        if (
            seg.style is not None
            and seg.style.color is not None
            and seg.style.color.name == "yellow"
        ):
            yellow.extend(range(col, col + len(seg.text)))
        col += len(seg.text)
    # Only over the SEARCH range: 145-147 MHz of the 141-151 MHz plot (95 columns).
    plot = [c - GUTTER_COLS for c in yellow if c >= GUTTER_COLS]
    assert (plot[0], plot[-1]) == (int(0.4 * 95), int(0.6 * 95) - 1)


def test_image_lines_keep_the_trace_and_dash_the_squelch() -> None:
    buf = np.zeros((10, 40, 4), dtype=np.uint8)
    buf[5, 10] = _TRACE_COLOR
    color = np.array([255, 140, 0, 230], dtype=np.uint8)
    draw_hline(buf, 5, 0, 40, color, keep=_TRACE_COLOR, dash=6)
    assert (buf[5, 10] == _TRACE_COLOR).all()
    assert (buf[5, 0] == color).all() and (buf[5, 6] == 0).all() and (buf[5, 12] == color).all()
    grid = np.array([110, 110, 110, 90], dtype=np.uint8)
    buf[2, 3] = [40, 40, 40, 255]  # the channel box stays over the grid
    draw_hline(buf, 2, 0, 40, grid, over_alpha=0x60)
    assert (buf[2, 3] == [40, 40, 40, 255]).all() and (buf[2, 4] == grid).all()


def test_a_click_on_the_labels_does_nothing_and_the_plot_maps_past_them(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SpectrumWidget, "size", property(lambda _self: Size(105, 20)))
    w = _widget()
    w._read_config()
    tuned = engine.devices["radio"].config.tuned_frequency
    stop = SimpleNamespace(stop=lambda: None)
    w.on_click(SimpleNamespace(x=2, **vars(stop)))  # type: ignore[arg-type]
    assert engine.devices["radio"].config.tuned_frequency == tuned
    w.on_click(SimpleNamespace(x=GUTTER_COLS + 50, **vars(stop)))  # type: ignore[arg-type]
    assert engine.devices["radio"].config.tuned_frequency == pytest.approx(CENTER, abs=12_500)


def test_the_waterfall_lines_up_with_the_plot(
    engine: SDREngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WaterfallWidget, "size", property(lambda _self: Size(100, 12)))
    w = WaterfallWidget()
    w.set_reactive(WaterfallWidget.db_min, -115.0)
    w.set_reactive(WaterfallWidget.db_max, -25.0)
    event = _event()
    w.current_event = event
    w._add_line(event)
    assert w._buffer is not None and w._buffer.shape[1] == 100 - GUTTER_COLS
    w._rebuild_strips()
    data_rows = w._strips[2:]
    assert all(strip.cell_length == 100 for strip in data_rows)
    keys = ["".join(s.text for s in strip)[:GUTTER_COLS] for strip in data_rows]
    # The color key: strongest (db_max) at the top, weakest (db_min) at the bottom,
    # KEY_COLS wide and centered in the label column.
    assert keys[0] == " -25 " and keys[-1] == "-115 "
    assert KEY_COLS == 3 and all(k == " ▀▀▀ " for k in keys[1:-1])


def test_the_color_key_runs_from_the_strongest_color_down() -> None:
    w = WaterfallWidget()
    w.set_reactive(WaterfallWidget.db_min, -115.0)
    w.set_reactive(WaterfallWidget.db_max, -25.0)
    colors = [
        next(s for s in w._color_key_strip(row, 10, GUTTER_COLS) if s.text.strip()).style.color
        for row in range(1, 9)
    ]
    lut = {f"rgb({r},{g},{b})": i for i, (r, g, b) in enumerate(waterfall_module._RGB_LUT)}
    indices = [lut[f"rgb({c.triplet.red},{c.triplet.green},{c.triplet.blue})"] for c in colors]
    assert indices == sorted(indices, reverse=True) and indices[0] > 200 and indices[-1] < 60


def test_the_color_keys_end_labels_are_in_their_colors() -> None:
    w = WaterfallWidget()
    w.set_reactive(WaterfallWidget.db_min, -115.0)
    w.set_reactive(WaterfallWidget.db_max, -25.0)
    for row, level, color in (
        (0, -25.0, _css(legible(level_color(-25.0, -115.0, -25.0)))),
        (9, -115.0, GRID_GREY),
    ):
        (seg,) = list(w._color_key_strip(row, 10, GUTTER_COLS))
        assert seg.text == grid_label(level)
        assert seg.style.color == Style(color=color).color
        assert seg.style.bgcolor is None


def test_a_reference_line_off_the_window_is_labeled_at_its_edge(engine: SDREngine) -> None:
    """A -75 dBFS squelch on a 30 Hz-bin trace sits near -102: below a -100
    floor. Its label waits at the bottom with an arrow, with no line drawn."""
    engine.update_squelch("radio", "audio", enabled=True, threshold_db=-75.0)
    w = _widget()
    w.set_reactive(SpectrumWidget.db_min, -80.0)
    rows = _text_rows(w)
    labels = [row[:GUTTER_COLS] for row in rows]
    bar_rows = [i for i, label in enumerate(labels) if label == " SQ↓ "]
    assert bar_rows and bar_rows[0] == len(rows) - 3  # the last plot row
    assert "  SQ " not in labels
