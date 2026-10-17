"""P25 Phase 1 symbol recovery for both modulations, C4FM and CQPSK (LSM, simulcast).

IQ -> decimate to ~48 kHz -> coarse frequency shift -> 3 kHz low-pass -> Gardner timing
   -> differential phase -> symbols

Both modulations move the carrier's phase by +-45 or +-135 degrees from one symbol instant to the
next (C4FM's +-600/+-1800 Hz deviation held for a 4800-baud symbol is exactly that), so one
differential detector serves both: the phase step between symbol samples, in units of 45 degrees, is
the symbol (+3, +1, -1, -3). A frequency offset adds a constant to every step; it is tracked by the
fourth-power method (each step's fourth power lands on -1) and taken off. That measures the offset
only modulo 1200 Hz (a quarter of the symbol rate); `coarse_hz`, set by the caller from whether
frames decode, takes off the whole multiples ahead of the channel filter. Gardner's timing detector
needs no carrier lock.

The channel filter is a plain 3 kHz low-pass, not CQPSK's root-raised-cosine matched filter: C4FM is
FM, wider than the RRC's +-2.9 kHz, and through the RRC its outer symbols came out at +-2.6 with
errors even without noise. Measured (2026-10-17, messages decoded): real C4FM with noise added to
5 dB 1,574 vs the RRC's 1,247; a real CQPSK simulcast 952 vs 874 at 9 dB; synthesized clean CQPSK
level with the RRC from 3 dB up (2.8-3.2 kHz all within a few percent; wider is worse on CQPSK).

Reference: TIA-102.BAAA (modulation), Gardner, "A BPSK/QPSK timing-error detector for sampled
receivers" (1986).
"""

import numba as nb
import numpy as np

from tsdr.radio.dsp._kernels import StreamingFilter
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation
from tsdr.radio.dsp.filters import firwin

SYMBOL_RATE = 4800.0
_TARGET_RATE = SYMBOL_RATE * 10
_CHANNEL_CUTOFF_HZ = 3_000.0
_CHANNEL_SYMBOLS = 8  # the channel filter's length
# Timing loop gain per symbol, on Gardner's error from unit-RMS samples.
_TIMING_GAIN = 0.05
# The frequency offset's average: this share of each new step (about 0.1 s), after a plain mean
# over the first 1 / _FREQ_WEIGHT steps (a quick start).
_FREQ_WEIGHT = 0.002
# The signal power's average, for unit RMS: its time constant.
_POWER_TC_S = 0.25
# The fourth-power method's range: offsets are measured modulo this.
OFFSET_AMBIGUITY_HZ = SYMBOL_RATE / 4


@nb.njit(cache=True)
def _gardner_diff(
    x: np.ndarray,
    sps: float,
    t_in: float,
    prev_re: float,
    prev_im: float,
    acc_re: float,
    acc_im: float,
    count: int,
    gain: float,
    weight: float,
) -> tuple[np.ndarray, float, float, float, float, float, int]:
    """Symbols (in units of 45 degrees) from channel-filtered samples `x`.

    `t_in` is the next symbol instant in samples of `x` (it may be negative: back into the previous
    call's tail, which the caller keeps at the front of `x`); prev_* is the last symbol sample,
    acc_* the fourth-power average over `count` steps so far. Returns (symbols, t for the next call
    relative to the end of `x`, prev_re, prev_im, acc_re, acc_im, count).
    """
    n = x.shape[0]
    # A step can be as short as (1 - gain) symbols (the error is clipped to +-1).
    out = np.empty(int(n / (sps * (1.0 - gain))) + 4, dtype=np.float32)
    k = 0
    t = t_in
    half = sps / 2.0
    while t + 1.0 < n:
        i = int(t)
        f = t - i
        y_re = x[i].real * (1.0 - f) + x[i + 1].real * f
        y_im = x[i].imag * (1.0 - f) + x[i + 1].imag * f
        tm = t - half
        im_ = int(tm)
        fm = tm - im_
        if im_ < 0:
            m_re = x[0].real
            m_im = x[0].imag
        else:
            m_re = x[im_].real * (1.0 - fm) + x[im_ + 1].real * fm
            m_im = x[im_].imag * (1.0 - fm) + x[im_ + 1].imag * fm
        # Gardner: Re{(y[k] - y[k-1]) conj(y[k-1/2])}; positive = sampling late.
        err = (y_re - prev_re) * m_re + (y_im - prev_im) * m_im
        # The phase step y[k] conj(y[k-1]).
        d_re = y_re * prev_re + y_im * prev_im
        d_im = y_im * prev_re - y_re * prev_im
        mag = np.sqrt(d_re * d_re + d_im * d_im)
        if mag > 0.0:
            u_re = d_re / mag
            u_im = d_im / mag
            # Fourth power: every symbol's step lands on -1 plus four times the offset.
            s_re = u_re * u_re - u_im * u_im
            s_im = 2.0 * u_re * u_im
            q_re = s_re * s_re - s_im * s_im
            q_im = 2.0 * s_re * s_im
            count += 1
            w = max(weight, 1.0 / count)
            acc_re += w * (q_re - acc_re)
            acc_im += w * (q_im - acc_im)
            offset = np.arctan2(-acc_im, -acc_re) / 4.0
            phase = np.arctan2(d_im, d_re) - offset
            if phase > np.pi:
                phase -= 2.0 * np.pi
            elif phase < -np.pi:
                phase += 2.0 * np.pi
            out[k] = phase / (np.pi / 4.0)
        else:
            out[k] = 0.0
        k += 1
        prev_re = y_re
        prev_im = y_im
        if err > 1.0:
            err = 1.0
        elif err < -1.0:
            err = -1.0
        t += sps - gain * sps * err
    return out[:k], t - n, prev_re, prev_im, acc_re, acc_im, count


class P25SymbolDemod:
    """Streaming: complex baseband at `sample_rate` (the channel at 0 Hz) in, symbols in units of 45
    degrees out (ideally +3, +1, -1, -3)."""

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        decimation = smooth_decimation(sample_rate / _TARGET_RATE)
        self.rate = sample_rate / decimation
        self.sps = self.rate / SYMBOL_RATE
        self._decimator = MultistageDecimator(
            sample_rate, decimation, self.rate * 0.45, dtype=np.complex64
        )
        taps = firwin(int(_CHANNEL_SYMBOLS * self.sps) | 1, _CHANNEL_CUTOFF_HZ, fs=self.rate)
        self._channel = StreamingFilter(taps / taps.sum(), [1.0], dtype=np.complex64)
        self._tail = np.zeros(0, dtype=np.complex64)
        self._power = 0.0
        self.coarse_hz = 0.0
        self._coarse_phase = 0.0
        self.reset_loops()

    def reset_loops(self) -> None:
        self._t = 0.0
        self._prev = (0.0, 0.0)
        self._acc = (-1.0, 0.0)
        self._count = 0

    @property
    def frequency_offset_hz(self) -> float:
        """The frequency offset (positive: the signal is above the channel center): the coarse
        shift and the tracked rest."""
        offset = np.arctan2(-self._acc[1], -self._acc[0]) / 4.0
        return self.coarse_hz + float(offset * SYMBOL_RATE / (2 * np.pi))

    def set_coarse(self, hz: float) -> None:
        """Take `hz` off ahead of the channel filter; the fine tracking starts over."""
        self.coarse_hz = hz
        self._acc = (-1.0, 0.0)
        self._count = 0

    def process(self, iq: np.ndarray) -> np.ndarray:
        z = self._decimator.process(iq)
        if len(z) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.coarse_hz:
            step = -2.0 * np.pi * self.coarse_hz / self.rate
            ramp = self._coarse_phase + step * np.arange(len(z))
            z = z * np.exp(1j * ramp).astype(np.complex64)
            self._coarse_phase = float((ramp[-1] + step) % (2.0 * np.pi))
        y = self._channel.process(z)
        # Unit RMS for the timing detector (a slow average, so a fade doesn't jump the loop gain).
        p = float(np.mean(y.real**2 + y.imag**2))
        if not np.isfinite(p):
            # A NaN or inf sample would stay in the filters and the loops for good.
            self.reset()
            return np.zeros(0, dtype=np.float32)
        w = 1.0 - np.exp(-len(y) / (_POWER_TC_S * self.rate))
        self._power = p if self._power == 0.0 else self._power + w * (p - self._power)
        if self._power > 0.0:
            y = (y / np.sqrt(self._power)).astype(np.complex64)
        # Keep half a symbol and a sample before the next instant, for the mid-sample interpolation.
        x = np.concatenate([self._tail, y])
        start = self._t + len(self._tail)
        symbols, t_end, pre, pim, are, aim, self._count = _gardner_diff(
            x,
            self.sps,
            start,
            self._prev[0],
            self._prev[1],
            self._acc[0],
            self._acc[1],
            self._count,
            _TIMING_GAIN,
            _FREQ_WEIGHT,
        )
        keep = int(np.ceil(self.sps / 2.0)) + 2
        self._tail = x[-keep:].copy()
        self._t = t_end
        self._prev = (pre, pim)
        self._acc = (are, aim)
        return symbols

    def reset(self) -> None:
        """Back to the state of a new demodulator, except the coarse shift (a property of the
        receiver's frequency error, which a retune doesn't change)."""
        self._decimator.reset()
        self._channel.reset()
        self._tail = np.zeros(0, dtype=np.complex64)
        self._power = 0.0
        self._coarse_phase = 0.0
        self.reset_loops()
