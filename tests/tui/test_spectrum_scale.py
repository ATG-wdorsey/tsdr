"""The spectrum's dB scale: grid lines, the label column, the squelch and
SEARCH trigger lines, and the waterfall kept in line with the plot."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from textual.geometry import Size

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import FFTUpdateEvent
from tsdr.core.memories import init_memory_store
from tsdr.core.sdr import engine as engine_module
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.sdr.processing import compute_fft, fft_window
from tsdr.core.tuning_state import init_tuning_state
from tsdr.devices import MockParams
from tsdr.tui.widgets.db_scale import (
    GUTTER_COLS,
    noise_equivalent_db,
)
from tsdr.tui.widgets.dsp_utils import project_spectrum
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget

RATE = 10e6
CENTER = 146.0e6


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
