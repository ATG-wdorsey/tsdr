# Derived from mbelib (https://github.com/szechyjs/mbelib), whose license follows:
#
# Copyright (C) 2010 mbelib Author
# GPG Key ID: 0xEA5EFE2C (9E7A 5527 9CDC EBF7 BF1B  D772 4F98 E863 EA5E FE2C)
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS.  IN NO EVENT SHALL ISC BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""IMBE 7200x4400 decoder: P25 Phase 1 voice, on the MBE synthesis shared with AMBE+2.

A port of mbelib's imbe7200x4400.c (its license above):

- the frame's error correction: Golay(23,12) on c0..c3, Hamming(15,11) on c4..c6, c7 sent as is;
  c1..c6 are sent under a pseudo-random sequence seeded by c0's twelve data bits;
- the 88-bit parameter vector and the model parameters (`_kernels.decode_parms`);
- mbelib's frame repeat rule: a bad frame (or more than five corrected errors) repeats the last
  parameters, and after three repeats the output mutes and the state starts over;
- mbelib's spectral enhancement and synthesis (`radio/vocoder/ambe`).

Reference: TIA-102.BABA (the IMBE vocoder).
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.vocoder.ambe._constants import (
    GOLAY_GENERATOR,
    GOLAY_MATRIX,
    HAMMING_GENERATOR,
    HAMMING_MATRIX,
)
from tsdr.radio.vocoder.ambe._rng import Xorshift32
from tsdr.radio.vocoder.ambe.enhance import spectral_amp_enhance
from tsdr.radio.vocoder.ambe.fec import golay_23_12, hamming_15_11
from tsdr.radio.vocoder.ambe.params import MbeParams
from tsdr.radio.vocoder.ambe.synthesis import (
    float_to_short,
    synthesize_silencef,
    synthesize_speechf,
)
from tsdr.radio.vocoder.imbe import _tables
from tsdr.radio.vocoder.imbe._kernels import OK, decode_parms
from tsdr.radio.vocoder.imbe._soft import soft_correct

FRAME_SHAPE = (8, 23)
# Corrected errors above this repeat the last frame's parameters (mbelib).
_MAX_ERRORS = 5
_MAX_REPEATS = 3
# Soft-decision correction (`_soft`): the bits flipped in every combination, and the soft distances
# (c0's, the frame's) up to which a corrected frame is trusted. A frame is played when mbelib's rule
# would play it (errors counted on the soft correction) or the correction is trusted. Measured on
# synthesized voice (qa/work/p25/m3/soft/), frames played with c0..c3 right / wrong: 5 dB 80.2 / 17.3 %
# (hard decoding and mbelib's rule: 65.5 / 18.1 %), 6 dB 99.9 / 0.0 % (93.7 / 0.7 %).
_CHASE_GOLAY = 4
_CHASE_HAMMING = 3
_SOFT_C0_MAX = 2.0
_SOFT_FRAME_MAX = 5.0


def correct_frame(imbe_fr: np.ndarray) -> tuple[np.ndarray, int]:
    """The (8, 23) codewords (mbelib's layout: c0..c3 in 23 bits, c4..c6 in 15, c7 in 7; element
    [i, 0] is codeword i's last bit) into the 88-bit parameter vector, and the errors corrected."""
    fr = np.asarray(imbe_fr, dtype=np.int8)
    if fr.shape != FRAME_SHAPE:
        raise ValueError(f"expected shape {FRAME_SHAPE}, got {fr.shape}")
    c0, errors = golay_23_12(fr[0].copy())
    # The pseudo-random sequence from c0's data bits (most significant first): x(n+1) = 173 x(n) +
    # 13849 mod 2^16; each later value's top bit is XORed onto c1..c6, first bit first.
    seed = 0
    for j in range(22, 10, -1):
        seed = (seed << 1) | int(c0[j])
    pr = 16 * seed
    out = np.empty(88, dtype=np.int8)
    out[:12] = c0[22:10:-1]
    pos = 12
    for i in range(1, 7):
        width = 23 if i < 4 else 15
        word = fr[i, :width].copy()
        for j in range(width - 1, -1, -1):
            pr = (173 * pr + 13849) & 0xFFFF
            word[j] ^= pr >> 15
        if i < 4:
            fixed, e = golay_23_12(word)
            out[pos : pos + 12] = fixed[22:10:-1]
            pos += 12
        else:
            fixed, e = hamming_15_11(word)
            out[pos : pos + 11] = fixed[14:3:-1]
            pos += 11
        errors += e
    out[81:88] = fr[7, 6::-1]
    return out, errors


def encode_frame(imbe_d: np.ndarray) -> np.ndarray:
    """The inverse of `correct_frame`: an 88-bit parameter vector as the (8, 23) codewords sent
    (for synthesized signals)."""
    bits = [int(b) for b in imbe_d]

    def value(lo: int, hi: int) -> int:
        out = 0
        for b in bits[lo:hi]:
            out = out << 1 | b
        return out

    fr = np.zeros(FRAME_SHAPE, dtype=np.int8)
    for i in range(4):
        word = value(12 * i, 12 * i + 12) << 11
        parity = 0
        for k in range(12):
            if word >> (22 - k) & 1:
                parity ^= int(GOLAY_GENERATOR[k])
        word |= parity
        fr[i] = [(word >> j) & 1 for j in range(23)]
    for i in range(3):
        data = value(48 + 11 * i, 59 + 11 * i) << 4
        word = next(
            data | p
            for p in range(16)
            if all(bin((data | p) & int(g)).count("1") % 2 == 0 for g in HAMMING_GENERATOR)
        )
        fr[4 + i, :15] = [(word >> j) & 1 for j in range(15)]
    fr[7, 6::-1] = bits[81:88]
    pr = 16 * value(0, 12)
    for i in range(1, 7):
        for j in range(22 if i < 4 else 14, -1, -1):
            pr = (173 * pr + 13849) & 0xFFFF
            fr[i, j] ^= pr >> 15
    return fr


class ImbeDecoder:
    """Stateful IMBE decoder for one voice stream (one P25 call at a time).

    Example::

        >>> dec = ImbeDecoder()
        >>> pcm = dec.decode_frame_float(codewords)  # float32[160]
    """

    def __init__(self, uvquality: int = 3, rng_seed: int = 1) -> None:
        self.uvquality = uvquality
        self.cur_mp = MbeParams()
        self.prev_mp = MbeParams()
        self.prev_mp_enh = MbeParams()
        self._rng = Xorshift32(rng_seed)
        self.last_errors = 0

    def decode_frame_float(
        self, imbe_fr: np.ndarray, reliability: np.ndarray | None = None
    ) -> np.ndarray:
        """One frame's (8, 23) codewords into float32[160] on mbelib's synthesis scale (speech
        peaks of hundreds to a few thousand); `last_errors` holds the errors corrected.

        With `reliability` (8, 23: each received bit's confidence), the codewords are corrected by
        soft decisions, and a frame mbelib's error count would repeat still plays when the correction
        is trusted (its soft distances); without it, mbelib's hard decoding and rule."""
        if reliability is None:
            imbe_d, errors = correct_frame(imbe_fr)
            self.last_errors = errors
            return self.decode_imbe_d_float(imbe_d, errors)
        fr = np.asarray(imbe_fr, dtype=np.int8)
        if fr.shape != FRAME_SHAPE:
            raise ValueError(f"expected shape {FRAME_SHAPE}, got {fr.shape}")
        imbe_d, errors, d0, dt = soft_correct(
            fr,
            np.asarray(reliability, dtype=np.float64),
            GOLAY_GENERATOR,
            GOLAY_MATRIX,
            HAMMING_GENERATOR,
            HAMMING_MATRIX,
            _CHASE_GOLAY,
            _CHASE_HAMMING,
        )
        self.last_errors = int(errors)
        trusted = d0 <= _SOFT_C0_MAX and dt <= _SOFT_FRAME_MAX
        return self.decode_imbe_d_float(imbe_d, 0 if trusted else int(errors))

    def decode_frame(self, imbe_fr: np.ndarray) -> np.ndarray:
        """`decode_frame_float` with mbelib's gain and int16 clip."""
        return float_to_short(self.decode_frame_float(imbe_fr))

    def decode_imbe_d_float(self, imbe_d: np.ndarray, errors: int = 0) -> np.ndarray:
        """An already corrected 88-bit parameter vector into float32[160]."""
        cur, prev = self.cur_mp, self.prev_mp
        status, w0, n = decode_parms(
            np.asarray(imbe_d, dtype=np.int8),
            _tables.BO,
            _tables.B2,
            _tables.BA,
            _tables.HOBA,
            _tables.JI,
            _tables.QUANTSTEP,
            _tables.STANDDEV,
            prev.L,
            prev.Ml,
            prev.log2_ml,
            cur.Vl,
            cur.Ml,
            cur.log2_ml,
        )
        if status != OK or errors > _MAX_ERRORS:
            repeat = prev.repeat + 1
            cur.use_last(prev)
            cur.repeat = repeat
        else:
            cur.w0, cur.L, cur.repeat = w0, n, 0
        if cur.repeat > _MAX_REPEATS:
            self.reset()
            return synthesize_silencef()
        prev.copy_from(cur)
        spectral_amp_enhance(cur)
        audio = synthesize_speechf(cur, self.prev_mp_enh, self.uvquality, self._rng)
        self.prev_mp_enh.copy_from(cur)
        return audio

    def reset(self) -> None:
        """A new call: mbelib's initial parameters."""
        self.cur_mp.reset()
        self.prev_mp.reset()
        self.prev_mp_enh.reset()
