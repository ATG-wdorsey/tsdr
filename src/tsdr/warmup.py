"""Compile tsdr's numba kernels before they are first used.

numba compiles a kernel on its first call and keeps the machine code in its disk cache
(``cache=True``). After an install the cache no longer matches the sources, so every kernel
compiles again on its first use: a stall of up to a second or so in the pipeline (the first
DMR call, the first SEARCH) or the display (the first spectrum), about 30 s of compiling in
all. ``python -m tsdr.warmup`` calls each kernel with the argument types tsdr gives it, in a
process of its own, which fills the cache without holding tsdr's GIL; tsdr then loads each
kernel from the cache. tsdr starts it in the background at startup (``start_if_needed``) the
first time after an install: about 26 s of one core at the lowest priority.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.processing import compute_fft
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.devices.soapy_reader import _published
from tsdr.radio.decoders.dab.constants import T_FRAME
from tsdr.radio.decoders.dab.fec import _check_fib_crc
from tsdr.radio.decoders.dab.ofdm import _ofdm_demod_frame
from tsdr.radio.decoders.dab.viterbi import _viterbi_decode
from tsdr.radio.decoders.tetra.bit_reader import BitReader
from tsdr.radio.decoders.tetra.channel import decode_block, rm3014_decode
from tsdr.radio.decoders.tetra.scramble import scramble_hard
from tsdr.radio.decoders.wsjt.decode import decode_slot
from tsdr.radio.dsp import CostasLoop
from tsdr.radio.dsp._kernels import (
    _iq_metrics_c64,
    add_power_c64,
    apply_freq_shift_c64,
    iq_sums_c64,
    iq_to_int_pairs,
    window_corrected_into,
)
from tsdr.radio.registry import DEMODULATOR_CLASSES, make_demodulator
from tsdr.radio.vocoder.acelp import TetraAcelpVocoder
from tsdr.radio.vocoder.ambe import AmbePlus2Decoder
from tsdr.tui.widgets import constellation_widget, spectrum_widget
from tsdr.tui.widgets.dsp_utils import render_spectrum_to_buf

_RATE = 1_000_000.0
_IQ_S = 0.1  # of noise per mode, twice: a stream's first and later chunks


def _noise(n: int, rng: np.random.Generator, scale: float = 0.1) -> np.ndarray:
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * scale
    return iq.astype(np.complex64)


def _forms(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`iq` in both forms the pipelines get it: converted from 8-bit samples (writable) and as a
    device's complex64 bytes (read-only; numba compiles a kernel for a read-only array apart)."""
    return iq, np.frombuffer(iq.tobytes(), dtype=np.complex64)


def _device_input() -> None:
    for sample_format in (SampleFormat.UINT8_IQ, SampleFormat.SINT8_IQ):
        SamplesBatch(raw_samples=bytes(128), sample_format=sample_format).to_iq_array()
    _published(np.zeros(8, dtype=np.int64))
    window = np.hanning(1024).astype(np.float32)
    out = np.zeros(1024, dtype=np.complex64)
    total = np.zeros(1024, dtype=np.float32)
    for segment in _forms(np.zeros(1024, dtype=np.complex64)):
        _iq_metrics_c64(segment)
        apply_freq_shift_c64(segment, 1000.0, _RATE, 0.0)  # tuned off the capture's center
        iq_to_int_pairs(segment, 32768.0, -32768, 32767)  # a recording's samples
        # SEARCH's frames (fft_stage._search_spectrum)
        iq_sums_c64(segment)
        window_corrected_into(segment, window, 0.0, 0.0, 1.0, out)
    add_power_c64(out, total)
    compute_fft(out, window, 1.0)


def _display() -> None:
    buf = np.zeros((20, 64, 4), dtype=np.uint8)
    render_spectrum_to_buf(
        buf,
        np.zeros(64, dtype=np.intp),
        spectrum_widget._FILL_COLOR,
        spectrum_widget._TRACE_COLOR,
        -1,
        -1,
        spectrum_widget._BW_BOX_COLOR,
        32,
        spectrum_widget._CENTER_LINE_COLOR,
    )
    square = np.zeros((32, 32, 4), dtype=np.uint8)
    constellation_widget._decay_alpha(square[:, :, 3], constellation_widget._DECAY)
    points = np.zeros(4, dtype=np.intp)
    constellation_widget._plot(square, points, points, constellation_widget._DOT_COLOR)


def _modes(rng: np.random.Generator) -> None:
    """Every mode's demodulator on noise, as the pipeline feeds it: shifted to the tuned
    frequency or converted from 8-bit samples, and a device's own complex64 when tuned to the
    capture's center (_forms); ADS-B and DAB also from a rate they convert from."""
    for mode, cls in sorted(DEMODULATOR_CLASSES.items()):
        rates = [cls.FIXED_SAMPLE_RATE or _RATE]
        if cls.FIXED_SAMPLE_RATE:
            rates.append(10_000_000.0)
        for rate in rates:
            n = int(rate * _IQ_S)
            for form in range(2):
                demod = make_demodulator(DemodSpec(mode=mode), rate)
                for _ in range(2):
                    demod.demodulate(_forms(_noise(n, rng))[form], 1.7e9)
                    demod.get_audio()


def _vocoders(rng: np.random.Generator) -> None:
    """DMR's AMBE+2 and TETRA's ACELP decode only frames that carry speech."""
    ambe = AmbePlus2Decoder()
    for _ in range(64):  # until frames decode as speech: the synthesis runs
        speech = ambe.decode_frame_float(rng.integers(0, 256, 9, dtype=np.uint8).tobytes())
        if np.any(speech):
            break
    acelp = TetraAcelpVocoder()
    for _ in range(4):
        acelp.decode_frame(rng.integers(0, 256, 18, dtype=np.uint8).tobytes())
    for mode in ("bpsk", "qpsk"):
        CostasLoop(mode=mode).process(np.zeros(64, dtype=np.complex64))


def _decoders(rng: np.random.Generator) -> None:
    """Decoders' kernels that run only on a signal with their structure."""
    decode_block(rng.standard_normal(120).astype(np.float32), "SB1", 0x0003)
    rm3014_decode(rng.standard_normal(30).astype(np.float32))
    scramble_hard(np.zeros(64, dtype=np.uint8), 0x0003)
    BitReader(np.zeros(64, dtype=np.uint8)).u(8)
    _ofdm_demod_frame(_noise(T_FRAME, rng))
    _viterbi_decode(rng.standard_normal(3096).astype(np.float32))
    _check_fib_crc(bytes(32))
    for is_ft4, seconds in ((False, 15.0), (True, 7.5)):
        audio = (rng.standard_normal(int(12_000 * seconds)) * 0.1).astype(np.float32)
        # Two candidates, whatever their score: their symbols are read and LDPC decoded.
        decode_slot(audio, is_ft4=is_ft4, num_candidates=2, min_score=-1e9)


def warm_up() -> None:
    """Call every kernel tsdr uses, most needed first."""
    rng = np.random.default_rng(0)
    _device_input()
    _display()
    _modes(rng)
    _vocoders(rng)
    _decoders(rng)


def _stamp() -> tuple[Path, str]:
    """Where a finished warm-up is noted, and what it notes: this module's file as installed
    (an install rewrites it, and the sources numba's cache matches with it), numba's version
    and the cache folder it was given (NUMBA_CACHE_DIR): a change of either leaves the cache
    tsdr reads without the kernels."""
    here = Path(__file__)
    info = here.stat()
    cache = os.environ.get("NUMBA_CACHE_DIR", "")
    stamp = f"{info.st_mtime_ns} {info.st_size} numba {numba.__version__} {cache}"
    return here.parent / "__pycache__" / "numba-warmup.stamp", stamp


def start_if_needed() -> subprocess.Popen[bytes] | None:
    """Start the warm-up unless it has run since this copy of tsdr was installed."""
    path, stamp = _stamp()
    try:
        if path.read_text() == stamp:
            return None
    except OSError:
        pass
    return start()


def start() -> subprocess.Popen[bytes]:
    """Run the warm-up in a background process (it lowers its own priority)."""
    return subprocess.Popen(
        # -P: not a tsdr in the current folder; an argument, as soundcard reads one at import
        [sys.executable, "-P", "-m", "tsdr.warmup", "warmup"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


if __name__ == "__main__":
    if hasattr(os, "nice"):
        os.nice(19)  # behind everything tsdr runs
    warm_up()
    path, stamp = _stamp()
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text(stamp)
    except OSError:
        pass  # a read-only install warms up at every start
