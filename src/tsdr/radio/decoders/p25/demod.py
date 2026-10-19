"""P25 Phase 1 symbol recovery for both modulations, C4FM and CQPSK (LSM, simulcast).

IQ -> decimate to ~48 kHz -> coarse frequency shift -> two channel filters (3 and 5 kHz)
   -> Gardner timing -> differential phase in half-symbol steps -> trained equalizer -> symbols

Both modulations move the carrier's phase by +-45 or +-135 degrees from one symbol instant to the
next (C4FM's +-600/+-1800 Hz deviation held for a 4800-baud symbol is exactly that), so one
differential detector serves both: the phase step between symbol samples, in units of 45 degrees, is
the symbol (+3, +1, -1, -3). For C4FM that step is the integral of the frequency over the symbol, the
standard's integrate-and-dump receiver. A frequency offset adds a constant to every step; it is
tracked by the fourth-power method (each step's fourth power lands on -1) and taken off. That measures
the offset only modulo 1200 Hz (a quarter of the symbol rate); `coarse_hz`, set by the caller from
whether frames decode, takes off the whole multiples ahead of the channel filters. Gardner's timing
detector needs no carrier lock.

**Two channel filters, one timing loop.** C4FM is FM, about +-4 kHz wide; a 3 kHz low-pass ahead of
the detector distorts its phase, so each symbol carries a few percent of its neighbors and the outer
levels come out near +-2.7 (a sheared eye), while a 5 kHz one passes the signal whole but lets in more
noise, which a differential detector can't average away. Measured on real control channels
(2026-10-17, `qa/work/p25/m3/filter/`): at 4 dB the 5 kHz filter decodes a tenth of the blocks the
3 kHz one does, while on strong signals it leaves the symbols far cleaner. Both run, sampled at the
same instants, and the output follows whichever has the smaller symbol spread (with hysteresis); the
timing follows the path in use, the frequency tracking the narrow path always. The timing loop is fast until frame syncs show up
in the decisions, then slow while they keep coming (`_TIMING_GAIN_SLOW`). (CQPSK's root-raised-cosine matched filter did
worse than either on both modulations.)

**Trained equalizer.** Each path's symbol is a weighted sum of the half-symbol phase steps around it
(two symbols either side) and a constant, starting at the plain detector (the symbol's own two halves
at weight 1). Each half step is wrapped on its own, so an FM click inside it still drops out. The
weights adapt by normalized LMS on the 24 known symbols of every frame sync (applied once the sync is
recognized, from the stored inputs) and on confident decisions, with a slow pull back toward the plain
detector. It takes out what is left of the filters' intersymbol interference and level compression.
The output lags the timing loop by two symbols.

All of it, against the 3 kHz filter and a fast timing loop alone, on the captures measured (RMS symbol
error; blocks): C4FM at 26 dB 0.27 -> 0.08, outer levels 2.72 -> 2.99, and with noise added to 4 dB
430/167 -> 546/90 good/bad blocks; an 800 MHz CQPSK simulcast at 30 dB 0.18 -> 0.10, bad blocks
2.9 % -> 1.2 %; a weak one 694/388 -> 1014/145.

Reference: TIA-102.BAAA (modulation), Gardner, "A BPSK/QPSK timing-error detector for sampled
receivers" (1986); Haykin, "Adaptive Filter Theory" (normalized LMS).
"""

import collections

import numba as nb
import numpy as np

from tsdr.radio.decoders.p25 import fec
from tsdr.radio.dsp._kernels import StreamingFilter
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation
from tsdr.radio.dsp.filters import firwin

SYMBOL_RATE = 4800.0
_TARGET_RATE = SYMBOL_RATE * 10
_NARROW_HZ = 3_000.0
_WIDE_HZ = 5_000.0
_CHANNEL_SYMBOLS = 8  # the channel filters' length (the same for both, so the same delay)
# Timing loop gain per symbol, on Gardner's error from unit-RMS samples: fast to find the timing, slow
# to hold it once frame syncs show it is right (within _SLOW_HOLD_SYMBOLS of the last). A simulcast's
# delay spread makes Gardner's error noisy, and a fast loop slips symbols mid-frame: on a weak 800 MHz
# simulcast voice channel 0.05 left 41 % of IMBE frames with 10 or more errors, 0.004 4 % (2026-10-17,
# qa/work/p25/m3/op25cmp/); C4FM at 4 dB 430 -> 536 good blocks.
_TIMING_GAIN_FAST = 0.05
_TIMING_GAIN_SLOW = 0.004
# About 1.5 LDU periods: the next transmission, perhaps from another transmitter with unrelated
# timing, finds the fast gain once the last one's frames have stopped (local QA round 2: a 1 s hold lost
# an HDU and 1-2 LDUs of a call 0.1-0.9 s after another).
_SLOW_HOLD_SYMBOLS = 1300.0
# The frequency offset's average: this share of each new step (about 0.1 s), after a plain mean
# over the first 1 / _FREQ_WEIGHT steps (a quick start).
_FREQ_WEIGHT = 0.002
# It is tracked only while a signal is present (noise alone would only blur it): the input holds a
# carrier (the two filters' powers), or the fourth powers of the latest steps agree, their average over
# about _COHERENT_STEPS steps at least _COHERENT long. A strong neighbor in the wide filter can hide the
# carrier but not the agreement, as its leakage doesn't agree (measured on the narrow path: a CQPSK
# neighbor's leakage through the wide one did). Measured (12,000-sample chunks, 10 s,
# `qa/work/p25/m3/r27fix/coh3.out`): white noise at most 0.26, also beside a C4FM neighbor 12.5 kHz
# away and 50-60 dB stronger or a CQPSK one 6.25-7.5 kHz away; C4FM at 10 dB C/N 0.39 (median), at
# 30 dB 0.60 (the narrow filter's ISI), beside the C4FM neighbor 0.48 / 0.42 / 0.32; CQPSK at 6 dB 0.41.
_COHERENT_STEPS = 48.0
_COHERENT = 0.3
# After this many steps without a signal the offset goes back to its value at the last frame sync:
# what a carrier without frames (an analog transmission, interference) moved it by is dropped.
_ABSENT_STEPS = 1300.0
# The signal power's average, for unit RMS (the timing loop's gain): a fast rise and a slow fall.
# A fall is a fade, and a simulcast's fades must not run the loop at full gain (0.25 s: 2.1 % of an
# 800 MHz simulcast's blocks lost, 1 s: 1.0 %); a rise is a signal arriving after noise or a quiet
# stretch, which must not find the gain too high. Until a fall's time constant of samples has come,
# a plain mean, so the start doesn't depend on the input's chunks.
_POWER_TC_S = 1.0
_POWER_RISE_S = 0.05
# The fourth-power method's range: offsets are measured modulo this.
OFFSET_AMBIGUITY_HZ = SYMBOL_RATE / 4
# A carrier is in the channel when the narrow filter's power is at least this share of the wide
# one's: white noise alone gives 3/5, a P25 signal near 0.9 (about 0.77 at 0 dB C/N).
_CARRIER_RATIO = 0.7
# The powers it compares are those of the latest _CARRIER_WINDOW_S of input at least (the latest chunk
# when chunks are longer): small chunks of noise don't read as a carrier now and then (review 26), and a
# strong signal's end shows within that time (review 27: an average decaying with the signal held the
# carrier 150-250 ms after a strong one).
_CARRIER_WINDOW_S = 0.02
# The signal's mean frequency in the narrow path (the phase of its lag-1 autocorrelation, averaged over
# about this long, a plain mean until then; started over at each coarse step): unlike the fourth-power
# tracking it has no ambiguity, so it tells a signal near the channel center from one a whole turn per
# symbol (4800 Hz) away, which decodes a block now and then (reviews 28, 29). Measured
# (`qa/work/p25/m3/r29fix/center.out`): within 600 Hz of the center it reads within 550 Hz, also beside
# strong neighbors; 4.2-5.1 kHz away, 1450-2850 Hz at 12-30 dB C/N (below that no block decodes there).
_CENTER_TC_S = 0.1
# The spectrum around the channel, for the decoder's search (`located_hz`). A transmission more than about 1 kHz off the
# channel filters' center is beyond what the timing loop follows, so no measure taken at the symbols sees it (the fourth
# powers don't agree through either filter), and on an idle channel a strong neighbor pulls the mean frequency anywhere
# (150 Hz to 12 kHz beside one 40-60 dB stronger). In the spectrum it is a bump, at least _BUMP_MIN_HZ wide at
# _BUMP_DB down (a P25 signal 4.1-5.3 kHz; a receiver's spur 1.5 kHz, the smoothing), standing _BUMP_DB above the
# lowest point on one side out to _BUMP_FIELD_HZ and _BUMP_DIP_DB above the one on the other. A neighbor 12.5 kHz
# away only slopes up toward its side, but a 50 dB one's skirt fills the valley beside a transmission to within 4-5 dB
# of its top (review 35), so that side is held to less, and the bump ends at the valley rather than climbing the skirt
# (`qa/work/p25/m3/r34eval/`: `presence.out`, `peak-cen.out`, `peak-width.out`; `r35eval/`). The usual coarse
# acquisition ahead of timing, a band-edge frequency-locked loop (harris), compares the powers at the band's two
# edges, which a neighbor's skirt tips; this takes the whole band's shape instead. The mean power of the
# _SPECTRUM_SIZE-point Hann-windowed segments of the latest _SPECTRUM_S (none before that much: a plain window, so a
# transmission's bump is gone that soon after it ends), smoothed over _BUMP_SMOOTH_HZ; bumps peaking within
# _BUMP_SPAN_HZ of the dial, the field out to _BUMP_FIELD_HZ from it, whatever the shift (review 36: after a
# transmitter at -2400 Hz a reply at +2400 was 4.8 kHz from the shift, outside a view centered there, and lost whole);
# a neighbor 12.5 kHz from a signal within +-3 kHz of the dial stays 9.5 kHz or more from it. Started over at each
# coarse step.
_SPECTRUM_SIZE = 256
_SPECTRUM_S = 0.1
_BUMP_SMOOTH_HZ = 1500.0
_BUMP_SPAN_HZ = 4500.0
_BUMP_FIELD_HZ = 7000.0
_BUMP_DB = 6.0
_BUMP_DIP_DB = 3.0
_BUMP_MIN_HZ = 3000.0
# The fourth-power estimate measures the offset only modulo OFFSET_AMBIGUITY_HZ: +900 Hz reads as -300.
# Read a whole OFFSET_AMBIGUITY_HZ off, every symbol turns by one level (a quarter turn). The reading is
# the estimate kept continuous from step to step (a branch: whole OFFSET_AMBIGUITY_HZ added), so an
# offset near +-600 Hz doesn't jump between its two readings on noise (review 29's lead 4), and the
# frame syncs pick the branch: the decisions are also made on the other three quarter turns, and a sync
# found there, and not on the current one, moves the reading to it and corrects that frame's symbols
# still in hand (reviews 31-33: a new transmitter read on the wrong branch was lost whole; review 34: one
# two branches away, -900 Hz after +900). A half turn is two branches up or down: the one nearer the mean
# frequency (`center_hz`; the two lie 4800 Hz apart). A sync found so moves the reading only once the NID
# after it decodes on that branch too (review 34: a TSBK whose coded bits matched a sync a branch away
# was lost, the move turning its symbols). Unlike the mean frequency, a sync doesn't depend on the data
# (a clear call's header, mostly zeros, is nearly a tone: it pulled the mean 400 Hz) or on a carrier
# reading (a neighbor hides it), and it comes with each frame. Nothing moves ahead of the filters, so a
# neighbor can't be brought closer (review 30's half steps did); the decoder moves the coarse shift a
# whole step when decodes read a branch away.
_READING_MAX = 2.5  # past this many OFFSET_AMBIGUITY_HZ the reading starts over from the estimate
# Each path's mean squared symbol error (equalized, averaged over _SPREAD_SYMBOLS); the output moves
# to the other path once its error is below this share of the one in use.
_SWITCH_RATIO = 0.9
_SPREAD_SYMBOLS = 2400.0
# The equalizer: half-symbol steps over two symbols either side, plus a constant.
_EQ_SPAN = 2
EQ_TAPS = 2 + 4 * _EQ_SPAN
_EQ_MU = 0.005
_EQ_CONFIDENT = 0.5  # a decision within this of the output trains the equalizer
_EQ_LEAK = 1e-4  # per symbol, toward the plain detector
# Frame sync 0x5575F5FF77FF in symbols (one polarity; either is recognized).
_SYNC = np.array(
    [3, 3, 3, 3, 3, -3, 3, 3, -3, -3, 3, 3, -3, -3, -3, -3, 3, -3, 3, -3, -3, -3, -3, -3],
    dtype=np.float64,
)
_SYNC_MISSES = 4
# The NID after a sync found on another branch: 32 dibits and the status symbol among them (from the
# frame's start every 36th dibit: the 12th after the sync). Within _NID_MAX_ERRORS bits of a NID with a
# defined DUID it confirms the branch; random bits come that close 8 times in a million.
_NID_SYMBOLS = 33
_NID_STATUS = 12
_NID_MAX_ERRORS = 8
_NID_WORDS = fec.nid_words()
# Output symbols held back one call: a sync's and its NID's, which a confirmed move corrects.
_HOLD = len(_SYNC) + _NID_SYMBOLS
# Indices into the loop-state vector.
_N_RE, _N_IM, _W_RE, _W_IM, _ACC_RE, _ACC_IM, _COUNT, _MSQ_N, _MSQ_W, _WIDE, _HALVES, _OUT = range(
    12
)
_SINCE_SYNC = 12  # output symbols since the last frame sync recognized
_ABSENT = 13  # steps without a signal
_SAVED_RE, _SAVED_IM = 14, 15  # the fourth-power average at the last frame sync (re > 1: none)
_COH_RE, _COH_IM = 16, 17  # the short average of the fourth powers
_BRANCH = 18  # whole OFFSET_AMBIGUITY_HZ added to the estimate (continuity, frame syncs)
_READING = 19  # the offset read at the latest step: the estimate on its branch
_SAVED_READING = 20  # the reading at the last frame sync
_STEADY = 21  # output symbols since the reading last jumped (not by a sync)
_PENDING = 22  # a sync found on another branch, NID awaited: quarter turns (1 up, 2 down, 3 half)
_PENDING_POL = 23  # its polarity
_PENDING_AGE = 24  # output symbols since it
_NID_HI, _NID_LO = 25, 26  # its NID's dibits so far, decided on its branch: 16 in each, as integers
_LOOP_LEN = 27
_NONE = 2.0


@nb.njit(cache=True)
def _wrap8(v: float) -> float:
    """Into (-4, 4]: phase in units of 45 degrees."""
    while v > 4.0:
        v -= 8.0
    while v <= -4.0:
        v += 8.0
    return v


@nb.njit(cache=True)
def _decide(v: float) -> float:
    d = np.round((v + 3.0) / 2.0) * 2.0 - 3.0
    return float(min(3.0, max(-3.0, d)))


@nb.njit(cache=True)
def _angle_step(a_re: float, a_im: float, b_re: float, b_im: float) -> float:
    """The phase from b to a, in units of 45 degrees."""
    return float(np.arctan2(a_im * b_re - a_re * b_im, a_re * b_re + a_im * b_im) / (np.pi / 4.0))


@nb.njit(cache=True)
def _is_nid(hi: float, lo: float, nids: np.ndarray) -> bool:
    """Whether the 64 bits whose halves are `hi` and `lo` lie within _NID_MAX_ERRORS of one of `nids`."""
    word = (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)
    for c in range(nids.shape[0]):
        v = word ^ nids[c]
        n = 0
        while v != np.uint64(0) and n <= _NID_MAX_ERRORS:
            v &= v - np.uint64(1)
            n += 1
        if n <= _NID_MAX_ERRORS:
            return True
    return False


@nb.njit(cache=True)
def _nlms(w: np.ndarray, p: int, x: np.ndarray, e: float, mu: float) -> None:
    norm = 1.0
    for j in range(x.shape[0]):
        norm += x[j] * x[j]
    g = mu * e / norm
    for j in range(x.shape[0]):
        w[p, j] += g * x[j]


@nb.njit(cache=True)
def _symbols(
    xn: np.ndarray,
    xw: np.ndarray,
    sps: float,
    t_in: float,
    loop: np.ndarray,
    w: np.ndarray,
    hist: np.ndarray,
    feats: np.ndarray,
    recent: np.ndarray,
    sync: np.ndarray,
    track: bool,
    held: np.ndarray,
    nids: np.ndarray,
    center: float,
) -> tuple[np.ndarray, float]:
    """Symbols (in units of 45 degrees) from the narrow- and wide-filtered samples `xn`, `xw`.

    `t_in` is the next symbol instant in samples (it may be negative: back into the previous call's
    tail, which the caller keeps at the front of both arrays). The state arrays are updated in place:
    `loop` (the last symbol samples, the fourth-power average, the spread, which path is used, counts),
    `w` (each path's equalizer weights), `hist` (each path's latest half steps, oldest first), `feats`
    and `recent` (each path's inputs and the decisions for the last 24 output symbols, a ring).
    `track`: the input holds a carrier (the frequency offset's tracking: _COHERENT). `held`: the
    previous call's last output symbols, not yet passed on, which a sync's correction may reach.
    `nids`: every NID (_NID_WORDS). `center`: the mean frequency in OFFSET_AMBIGUITY_HZ (a half turn's
    side).
    Returns (symbols, t for the next call relative to the end of the arrays).
    """
    n = xn.shape[0]
    taps = hist.shape[1]
    span = (taps - 2) // 4
    ring = recent.shape[1]
    # A step can be as short as (1 - gain) symbols (the error is clipped to +-1).
    out = np.empty(int(n / (sps * (1.0 - _TIMING_GAIN_FAST))) + 4, dtype=np.float32)
    x = np.empty(taps + 1)
    ys = np.empty(2)
    sy = np.empty((2, 6))
    k = 0
    t = t_in
    half = sps / 2.0
    a_spread = 1.0 / _SPREAD_SYMBOLS
    while t + 1.0 < n:
        i = int(t)
        f = t - i
        tm = t - half
        im_ = int(tm)
        fm = tm - im_
        if im_ < 0:
            im_, fm = 0, 0.0
        wide = loop[_WIDE] > 0.5
        # Both paths' samples at the symbol instant (y), half a symbol before it (m) and at the
        # previous instant (q).
        for p in range(2):
            src = xw if p == 1 else xn
            sy[p, 0] = src[i].real * (1.0 - f) + src[i + 1].real * f
            sy[p, 1] = src[i].imag * (1.0 - f) + src[i + 1].imag * f
            sy[p, 2] = src[im_].real * (1.0 - fm) + src[im_ + 1].real * fm
            sy[p, 3] = src[im_].imag * (1.0 - fm) + src[im_ + 1].imag * fm
            sy[p, 4] = loop[_W_RE] if p == 1 else loop[_N_RE]
            sy[p, 5] = loop[_W_IM] if p == 1 else loop[_N_IM]
        # Timing and frequency follow the path in use.
        a = 1 if wide else 0
        y_re, y_im, m_re, m_im, q_re, q_im = (
            sy[a, 0],
            sy[a, 1],
            sy[a, 2],
            sy[a, 3],
            sy[a, 4],
            sy[a, 5],
        )
        # Gardner: Re{(y[k] - y[k-1]) conj(y[k-1/2])}; positive = sampling late.
        err = (y_re - q_re) * m_re + (y_im - q_im) * m_im
        # Fourth power of the whole step: every symbol lands on -1 plus four times the offset. From the
        # narrow path whichever is in use: it keeps out more of a neighbor (review 27: a CQPSK signal 7.5
        # kHz away agreed through the wide one), and its ISI moves opposite steps oppositely, so the
        # offset stays unbiased.
        d_re = sy[0, 0] * sy[0, 4] + sy[0, 1] * sy[0, 5]
        d_im = sy[0, 1] * sy[0, 4] - sy[0, 0] * sy[0, 5]
        mag = np.sqrt(d_re * d_re + d_im * d_im)
        present = track
        if mag > 0.0:
            u_re = d_re / mag
            u_im = d_im / mag
            s_re = u_re * u_re - u_im * u_im
            s_im = 2.0 * u_re * u_im
            c_re = s_re * s_re - s_im * s_im
            c_im = 2.0 * s_re * s_im
            loop[_COH_RE] += (c_re - loop[_COH_RE]) / _COHERENT_STEPS
            loop[_COH_IM] += (c_im - loop[_COH_IM]) / _COHERENT_STEPS
            coh = loop[_COH_RE] * loop[_COH_RE] + loop[_COH_IM] * loop[_COH_IM]
            present = present or coh >= _COHERENT * _COHERENT
            if present:
                loop[_COUNT] += 1.0
                g = max(_FREQ_WEIGHT, 1.0 / loop[_COUNT])
                loop[_ACC_RE] += g * (c_re - loop[_ACC_RE])
                loop[_ACC_IM] += g * (c_im - loop[_ACC_IM])
        restored = False
        if present:
            loop[_ABSENT] = 0.0
        else:
            loop[_ABSENT] += 1.0
            if loop[_ABSENT] == _ABSENT_STEPS and loop[_SAVED_RE] < _NONE:
                loop[_ACC_RE] = loop[_SAVED_RE]
                loop[_ACC_IM] = loop[_SAVED_IM]
                loop[_READING] = loop[_SAVED_READING]
                loop[_STEADY] = 0.0
                restored = True
        # The offset's phase per symbol is atan2(...) / 4; in 45-degree units, halved for each
        # half step: atan2(...) / (2 pi). Kept continuous: a jump of more than half a branch is the
        # estimate wrapping, not the offset moving. Not while a new estimate is unsettled (the plain
        # mean of its first steps, over a transmission's first symbols while the timing pulls in): it
        # swings freely, and followed it carried a fresh start beside a strong neighbor two branches
        # off (+1820 Hz for -580 Hz), where no sync is looked for. It is read plainly then, and a
        # jump between the branches is a jump.
        offset = np.arctan2(-loop[_ACC_IM], -loop[_ACC_RE]) / (2.0 * np.pi) + loop[_BRANCH]
        jump = offset - loop[_READING]
        if abs(jump) > 0.5:
            if restored or loop[_COUNT] >= 1.0 / _FREQ_WEIGHT:
                loop[_BRANCH] -= np.floor(jump + 0.5)
                offset -= np.floor(jump + 0.5)
            else:
                loop[_STEADY] = 0.0
        if abs(offset) > _READING_MAX:
            offset -= loop[_BRANCH]
            loop[_BRANCH] = 0.0
            loop[_STEADY] = 0.0
        loop[_READING] = offset
        for p in range(2):
            # The two half steps, each wrapped on its own, the offset's share taken off each.
            for j in range(taps - 2):
                hist[p, j] = hist[p, j + 2]
            hist[p, taps - 2] = _wrap8(_angle_step(sy[p, 2], sy[p, 3], sy[p, 4], sy[p, 5]) - offset)
            hist[p, taps - 1] = _wrap8(_angle_step(sy[p, 0], sy[p, 1], sy[p, 2], sy[p, 3]) - offset)
        loop[_N_RE], loop[_N_IM] = sy[0, 0], sy[0, 1]
        loop[_W_RE], loop[_W_IM] = sy[1, 0], sy[1, 1]
        if err > 1.0:
            err = 1.0
        elif err < -1.0:
            err = -1.0
        # Slow also while a sync found on another branch awaits its NID: a new transmitter's first sync.
        slow = loop[_SINCE_SYNC] < _SLOW_HOLD_SYMBOLS or loop[_PENDING] > 0.0
        gain = _TIMING_GAIN_SLOW if slow else _TIMING_GAIN_FAST
        t += sps - gain * sps * err
        loop[_HALVES] += 2.0
        if loop[_HALVES] < taps:
            continue
        # The symbol `span` back, whose half steps sit in the middle of the history.
        slot = int(loop[_OUT]) % ring
        for p in range(2):
            for j in range(taps):
                x[j] = hist[p, j]
            x[taps] = 1.0
            y = 0.0
            for j in range(taps + 1):
                y += w[p, j] * x[j]
            y = _wrap8(y)
            ys[p] = y
            for j in range(taps + 1):
                feats[p, slot, j] = x[j]
            e = _decide(y) - y
            msq = _MSQ_W if p == 1 else _MSQ_N
            loop[msq] += a_spread * (e * e - loop[msq])
            if abs(e) < _EQ_CONFIDENT:
                _nlms(w, p, x, e, _EQ_MU)
            # The pull back toward the plain detector.
            for j in range(taps + 1):
                target = 1.0 if (j == 2 * span or j == 2 * span + 1) else 0.0
                w[p, j] += _EQ_LEAK * (target - w[p, j])
        y = ys[1] if wide else ys[0]
        # The path with the smaller error from here on (hysteresis: by a margin).
        if wide and loop[_MSQ_N] < _SWITCH_RATIO * loop[_MSQ_W]:
            loop[_WIDE] = 0.0
        elif not wide and loop[_MSQ_W] < _SWITCH_RATIO * loop[_MSQ_N]:
            loop[_WIDE] = 1.0
        out[k] = y
        k += 1
        # The decisions on this branch, one up (a level less), one down and two away.
        recent[0, slot] = _decide(y)
        recent[1, slot] = _decide(_wrap8(y - 2.0))
        recent[2, slot] = _decide(_wrap8(y + 2.0))
        recent[3, slot] = _decide(_wrap8(y + 4.0))
        loop[_OUT] += 1.0
        loop[_SINCE_SYNC] += 1.0
        loop[_STEADY] += 1.0
        if loop[_PENDING] > 0.0:
            # The NID after a sync found on another branch, decided on that branch. Once it is in and
            # close to a NID: move there, and correct the sync's and the NID's symbols still in hand.
            loop[_PENDING_AGE] += 1.0
            age = int(loop[_PENDING_AGE])
            turns = int(loop[_PENDING])
            if age != _NID_STATUS:
                sym = loop[_PENDING_POL] * recent[turns, slot]
                dibit = 1.0 if sym > 2.0 else (0.0 if sym > 0.0 else (2.0 if sym > -2.0 else 3.0))
                part = _NID_HI if (age - 1 if age < _NID_STATUS else age - 2) < 16 else _NID_LO
                loop[part] = loop[part] * 4.0 + dibit
            if age == _NID_SYMBOLS:
                loop[_PENDING] = 0.0
                if loop[_STEADY] >= ring + _NID_SYMBOLS and _is_nid(
                    loop[_NID_HI], loop[_NID_LO], nids
                ):
                    if turns == 1:
                        step = 1.0
                    elif turns == 2:
                        step = -1.0
                    else:
                        step = 2.0 if center > loop[_READING] else -2.0
                    loop[_BRANCH] += step
                    loop[_READING] += step
                    for j in range(ring + _NID_SYMBOLS):
                        if j < k:
                            out[k - 1 - j] = _wrap8(out[k - 1 - j] - 2.0 * step)
                        elif j - k < held.shape[0]:
                            h = held.shape[0] - 1 - (j - k)
                            held[h] = _wrap8(held[h] - 2.0 * step)
                    for j in range(ring):
                        recent[0, j] = recent[turns, j]
                        for r in range(1, 4):
                            recent[r, j] = 0.0
                    loop[_SINCE_SYNC] = 0.0
                    loop[_SAVED_RE] = loop[_ACC_RE]
                    loop[_SAVED_IM] = loop[_ACC_IM]
                    loop[_SAVED_READING] = loop[_READING]
        if loop[_OUT] < ring:
            continue
        # A frame sync among the last 24 decisions (either polarity): train on its known symbols. Not on
        # this branch but on another: await its NID (above). Only if all 24 were decided on one reading
        # (and the NID too): a sync straddling a jump of the reading (a new estimate's, a restore, a
        # coarse step) mixes two branches, and one so matched moved a fresh call at +720 Hz the wrong
        # way, to -1700 Hz.
        found = -1
        polarity = 1.0
        for r in range(4 if loop[_STEADY] >= ring else 1):
            for pol in (1.0, -1.0):
                misses = 0
                for j in range(ring):
                    if recent[r, (slot + 1 + j) % ring] != pol * sync[j]:
                        misses += 1
                        if misses > _SYNC_MISSES:
                            break
                if misses <= _SYNC_MISSES:
                    found = r
                    polarity = pol
                    break
            if found >= 0:
                break
        if found < 0:
            continue
        if found > 0:
            loop[_PENDING] = float(found)
            loop[_PENDING_POL] = polarity
            loop[_PENDING_AGE] = 0.0
            loop[_NID_HI] = 0.0
            loop[_NID_LO] = 0.0
            continue
        loop[_PENDING] = 0.0
        loop[_SINCE_SYNC] = 0.0
        loop[_SAVED_RE] = loop[_ACC_RE]
        loop[_SAVED_IM] = loop[_ACC_IM]
        loop[_SAVED_READING] = loop[_READING]
        for p in range(2):
            for j in range(ring):
                s = (slot + 1 + j) % ring
                for q in range(taps + 1):
                    x[q] = feats[p, s, q]
                y = 0.0
                for q in range(taps + 1):
                    y += w[p, q] * x[q]
                e = _wrap8(polarity * sync[j] - y)
                e = min(1.0, max(-1.0, e))
                _nlms(w, p, x, e, _EQ_MU)
    return out[:k], t - n


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
        length = int(_CHANNEL_SYMBOLS * self.sps) | 1
        self._channels = []
        for cutoff in (_NARROW_HZ, _WIDE_HZ):
            # Below Nyquist even at a rate too low for the channel (a narrow recording played back).
            taps = firwin(length, min(cutoff, 0.45 * self.rate), fs=self.rate)
            self._channels.append(StreamingFilter(taps / taps.sum(), [1.0], dtype=np.complex64))
        self._tails = (np.zeros(0, dtype=np.complex64), np.zeros(0, dtype=np.complex64))
        self._power = [0.0, 0.0]
        self._power_samples = [0, 0]
        self._chunk_power = [0.0, 0.0]
        # (samples, narrow power sum, wide power sum) of the latest chunks, _CARRIER_WINDOW_S at least.
        self._carrier_chunks: collections.deque[tuple[int, float, float]] = collections.deque()
        self._center = 0j  # the narrow path's lag-1 autocorrelation, averaged
        self._center_samples = 0
        self._spectrum_window = np.hanning(_SPECTRUM_SIZE)
        self._spectrum_hz = np.fft.fftshift(np.fft.fftfreq(_SPECTRUM_SIZE, 1.0 / self.rate))
        self._spectra = np.zeros(
            (max(1, round(_SPECTRUM_S * self.rate / _SPECTRUM_SIZE)), _SPECTRUM_SIZE)
        )
        self._reset_spectrum()
        self.coarse_hz = 0.0
        self._coarse_phase = 0.0
        self.reset_loops()

    def reset_loops(self) -> None:
        self._t = 0.0
        self._loop = np.zeros(_LOOP_LEN)
        self._loop[_ACC_RE] = -1.0
        self._loop[_MSQ_N] = 1.0
        self._loop[_MSQ_W] = 1.0
        self._loop[_SINCE_SYNC] = _SLOW_HOLD_SYMBOLS  # no sync yet: the fast gain
        self._loop[_SAVED_RE] = _NONE
        # The coarse shift of the last frame sync, and the average and the reading then.
        self._sync_coarse: float | None = None
        self._sync_saved = (_NONE, 0.0, 0.0)
        self._weights = np.zeros((2, EQ_TAPS + 1))
        self._weights[:, 2 * _EQ_SPAN] = 1.0
        self._weights[:, 2 * _EQ_SPAN + 1] = 1.0
        self._hist = np.zeros((2, EQ_TAPS))
        self._feats = np.zeros((2, len(_SYNC), EQ_TAPS + 1))
        self._recent = np.zeros((4, len(_SYNC)))  # decisions on this branch, up, down, two away
        self._held = np.zeros(0, dtype=np.float32)  # output symbols not yet passed on

    @property
    def center_hz(self) -> float:
        """The narrow path's mean frequency at the current coarse shift (see _CENTER_TC_S)."""
        return float(np.angle(self._center) * self.rate / (2 * np.pi))

    @property
    def located_hz(self) -> float | None:
        """Where a transmission shows in the spectrum around the channel, in Hz from the coarse shift (the
        center of its bump), or None (see _SPECTRUM_SIZE)."""
        if self._spectrum_count < len(self._spectra):
            return None
        k = max(1, round(_BUMP_SMOOTH_HZ * _SPECTRUM_SIZE / self.rate))
        mean = np.fft.fftshift(self._spectra.mean(axis=0))
        smooth = np.convolve(mean, np.ones(k) / k, mode="same")
        field = np.abs(self._spectrum_hz + self.coarse_hz) <= _BUMP_FIELD_HZ  # around the dial
        hz, power = self._spectrum_hz[field], smooth[field]
        db = 10.0 * np.log10(power + 1e-30)
        lowest_left = np.minimum.accumulate(db)
        lowest_right = np.minimum.accumulate(db[::-1])[::-1]
        j = np.arange(1, len(db) - 1)
        dial = np.abs(hz[j] + self.coarse_hz)
        j = j[(db[j] >= db[j - 1]) & (db[j] >= db[j + 1]) & (dial <= _BUMP_SPAN_HZ)]
        if len(j) == 0:
            return None
        higher = np.maximum(lowest_left[j - 1], lowest_right[j + 1])
        lower = np.minimum(lowest_left[j - 1], lowest_right[j + 1])
        prominence = np.where(db[j] - lower >= _BUMP_DB, db[j] - higher, -np.inf)
        top = int(j[np.argmax(prominence)])
        if prominence.max() < _BUMP_DIP_DB:
            return None
        # Out to _BUMP_DB down, but not past the lowest point on either side (up a neighbor's skirt).
        left, right = int(np.argmin(db[:top])), top + int(np.argmin(db[top:]))
        lo = hi = top
        while lo > left and db[lo - 1] >= db[top] - _BUMP_DB:
            lo -= 1
        while hi < right and db[hi + 1] >= db[top] - _BUMP_DB:
            hi += 1
        if hz[hi] - hz[lo] < _BUMP_MIN_HZ:
            return None
        return float(np.sum(power[lo : hi + 1] * hz[lo : hi + 1]) / np.sum(power[lo : hi + 1]))

    @property
    def frequency_offset_hz(self) -> float:
        """The frequency offset (positive: the signal is above the channel center): the coarse
        shift and the tracked rest, read on its branch (see _READING_MAX)."""
        return self.coarse_hz + float(self._loop[_READING]) * OFFSET_AMBIGUITY_HZ

    @property
    def wide(self) -> bool:
        """Whether the symbols come through the wide channel filter (a strong signal)."""
        return bool(self._loop[_WIDE] > 0.5)

    @property
    def carrier(self) -> bool:
        """Whether the latest input held a signal rather than noise alone (the two channel
        filters' powers)."""
        return self._chunk_power[1] > 0.0 and (
            self._chunk_power[0] >= _CARRIER_RATIO * self._chunk_power[1]
        )

    def set_coarse(self, hz: float) -> None:
        """Take `hz` off ahead of the channel filters. The fine tracking starts over, except back at
        the shift of the last frame sync, where it resumes from its value then."""
        self.coarse_hz = hz
        self._center = 0j
        self._center_samples = 0
        self._reset_spectrum()
        self._loop[_BRANCH] = 0.0
        self._loop[_STEADY] = 0.0
        self._loop[_PENDING] = 0.0
        if hz == self._sync_coarse:
            re_, im_, reading = self._sync_saved
            self._loop[_ACC_RE], self._loop[_ACC_IM] = re_, im_
            self._loop[_SAVED_RE], self._loop[_SAVED_IM] = re_, im_
            self._loop[_READING] = self._loop[_SAVED_READING] = reading
            self._loop[_COUNT] = 1.0 / _FREQ_WEIGHT
        else:
            self._loop[_ACC_RE] = -1.0
            self._loop[_ACC_IM] = 0.0
            self._loop[_COUNT] = 0.0
            self._loop[_READING] = 0.0
            self._loop[_SAVED_RE] = _NONE  # the last sync's average belongs to another shift

    def step_coarse(self, step: float) -> None:
        """Move the coarse shift by `step`, a whole multiple of OFFSET_AMBIGUITY_HZ, keeping the
        tracking: such a step leaves the fourth power as it was, and the reading moves the other way
        (the shift follows a signal read a branch away, so it stays near the filters' center)."""
        n = step / OFFSET_AMBIGUITY_HZ
        if self._sync_coarse == self.coarse_hz:
            re_, im_, reading = self._sync_saved
            self._sync_saved = (re_, im_, reading - n)
            self._sync_coarse = self.coarse_hz + step
        self.coarse_hz += step
        for i in (_BRANCH, _READING, _SAVED_READING):
            self._loop[i] -= n
        self._center = 0j
        self._center_samples = 0
        self._reset_spectrum()

    def _reset_spectrum(self) -> None:
        self._spectrum_count = 0  # segments taken (the latest len(self._spectra) are kept, a ring)
        self._spectrum_rest = np.zeros(0, dtype=np.complex64)

    def _add_spectrum(self, z: np.ndarray) -> None:
        z = np.concatenate([self._spectrum_rest, z])
        k = len(z) // _SPECTRUM_SIZE
        if k:
            segments = z[: k * _SPECTRUM_SIZE].reshape(k, _SPECTRUM_SIZE) * self._spectrum_window
            for power in (np.abs(np.fft.fft(segments, axis=1)) ** 2)[-len(self._spectra) :]:
                self._spectra[self._spectrum_count % len(self._spectra)] = power
                self._spectrum_count += 1
        self._spectrum_rest = z[k * _SPECTRUM_SIZE :]

    def process(self, iq: np.ndarray) -> np.ndarray:
        z = self._decimator.process(iq)
        if len(z) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.coarse_hz:
            step = -2.0 * np.pi * self.coarse_hz / self.rate
            ramp = self._coarse_phase + step * np.arange(len(z))
            z = z * np.exp(1j * ramp).astype(np.complex64)
            self._coarse_phase = float((ramp[-1] + step) % (2.0 * np.pi))
        self._add_spectrum(z)
        paths = []
        powers = []
        for p, channel in enumerate(self._channels):
            y = channel.process(z)
            # Unit RMS for the timing detector (a slow average, so a fade doesn't jump the loop gain).
            power = float(np.mean(y.real**2 + y.imag**2))
            if not np.isfinite(power):
                # A NaN or inf sample would stay in the filters and the loops for good.
                self.reset()
                return np.zeros(0, dtype=np.float32)
            powers.append(power)
            if p == 0 and len(y) > 1:
                lag = complex(np.vdot(y[:-1], y[1:]))
                self._center_samples += len(y)
                w = max(
                    1.0 - np.exp(-len(y) / (_CENTER_TC_S * self.rate)),
                    len(y) / self._center_samples,
                )
                self._center += min(w, 1.0) * (lag / len(y) - self._center)
            old = self._power[p]
            self._power_samples[p] += len(y)
            tc = _POWER_RISE_S if power > old else _POWER_TC_S
            w = max(1.0 - np.exp(-len(y) / (tc * self.rate)), len(y) / self._power_samples[p])
            self._power[p] = old + min(w, 1.0) * (power - old)
            if self._power[p] > 0.0:
                y = (y / np.sqrt(self._power[p])).astype(np.complex64)
            # Keep half a symbol and a sample before the next instant, for the mid-sample
            # interpolation.
            paths.append(np.concatenate([self._tails[p], y]))
        chunks = self._carrier_chunks
        chunks.append((len(z), powers[0] * len(z), powers[1] * len(z)))
        total = sum(c[0] for c in chunks)
        while len(chunks) > 1 and total - chunks[0][0] >= _CARRIER_WINDOW_S * self.rate:
            total -= chunks.popleft()[0]
        self._chunk_power = [sum(c[1] for c in chunks) / total, sum(c[2] for c in chunks) / total]
        start = self._t + len(self._tails[0])
        symbols: np.ndarray
        symbols, t_end = _symbols(
            paths[0],
            paths[1],
            self.sps,
            start,
            self._loop,
            self._weights,
            self._hist,
            self._feats,
            self._recent,
            _SYNC,
            self.carrier,
            self._held,
            _NID_WORDS,
            self.center_hz / OFFSET_AMBIGUITY_HZ,
        )
        if self._loop[_SINCE_SYNC] < len(symbols):
            self._sync_coarse = self.coarse_hz
            self._sync_saved = (
                float(self._loop[_SAVED_RE]),
                float(self._loop[_SAVED_IM]),
                float(self._loop[_SAVED_READING]),
            )
        keep = int(np.ceil(self.sps / 2.0)) + 2
        self._tails = (paths[0][-keep:].copy(), paths[1][-keep:].copy())
        self._t = t_end
        # A sync's and its NID's worth of symbols waits for the next call (12 ms), so that a sync found
        # on another branch is corrected whole whatever the chunks.
        symbols = np.concatenate([self._held, symbols])
        cut = max(len(symbols) - _HOLD, 0)
        self._held = symbols[cut:].copy()
        return symbols[:cut]

    def reset(self) -> None:
        """Back to the state of a new demodulator, except the coarse shift (a property of the
        receiver's frequency error, which a retune doesn't change)."""
        self._decimator.reset()
        for channel in self._channels:
            channel.reset()
        self._tails = (np.zeros(0, dtype=np.complex64), np.zeros(0, dtype=np.complex64))
        self._power = [0.0, 0.0]
        self._power_samples = [0, 0]
        self._chunk_power = [0.0, 0.0]
        self._carrier_chunks.clear()
        self._center = 0j
        self._center_samples = 0
        self._reset_spectrum()
        self._coarse_phase = 0.0
        self.reset_loops()
