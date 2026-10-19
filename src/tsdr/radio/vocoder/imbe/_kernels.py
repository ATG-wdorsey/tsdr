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

"""IMBE 7200x4400 parameter decoding (TIA-102.BABA), ported from mbelib's imbe7200x4400.c.

The arithmetic follows mbelib's C: float32, with the DCT angles and a few products in double where C
promotes to double.
"""

import math

import numba as nb
import numpy as np

_F32 = np.float32
_ZERO = _F32(0.0)
_ONE = _F32(1.0)
_TWO = _F32(2.0)
_HALF = _F32(0.5)
_FOUR_PI = _F32(4.0 * math.pi)

# decode_parms' status
OK = 0
SILENCE = 1  # b0 216-219: mbelib treats it as a bad frame (the last parameters are repeated)
INVALID = 2  # b0 208-215 or 220-255, or L out of range


@nb.njit(cache=True)
def decode_parms(
    imbe_d: np.ndarray,
    bo: np.ndarray,
    b2_table: np.ndarray,
    ba: np.ndarray,
    hoba: np.ndarray,
    ji: np.ndarray,
    quantstep: np.ndarray,
    standdev: np.ndarray,
    prev_n: int,
    prev_ml: np.ndarray,
    prev_log2ml: np.ndarray,
    cur_vl: np.ndarray,
    cur_ml: np.ndarray,
    cur_log2ml: np.ndarray,
) -> tuple[int, float, int]:
    """The 88-bit vector `imbe_d` (one bit per element) into the current frame's voicing, spectral
    amplitudes and their logs (index 1..L), predicted from the previous frame's (`prev_*`, which a
    larger L extends in place, as mbelib does). Returns (status, w0, L); on a status other than OK
    nothing of the current frame is meaningful."""
    b0 = 0
    for i in range(6):
        b0 = (b0 << 1) | imbe_d[i]
    b0 = (b0 << 1) | imbe_d[85]
    b0 = (b0 << 1) | imbe_d[86]
    if b0 > 207:
        return (SILENCE if 216 <= b0 <= 219 else INVALID), 0.0, 0
    w0 = _F32(_FOUR_PI / _F32(float(b0) + 39.5))
    n = int(0.9254 * int(math.pi / float(w0) + 0.25))
    if n > 56 or n < 9:
        return INVALID, 0.0, 0
    k_bands = int(_F32(n + 2) / _F32(3)) if n < 37 else 12
    row = n - 9

    # Bits 6..84 into the parameter words b0..b(L+1), most significant bit first.
    bb = np.zeros((58, 12), dtype=np.int8)
    for i in range(79):
        bb[bo[row, i, 0], bo[row, i, 1]] = imbe_d[6 + i]

    # Voicing: three harmonics to a band, the first band in the highest bit of b1.
    j = 1
    k = k_bands - 1
    for harm in range(1, n + 1):
        cur_vl[harm] = bb[1, k]
        if j == 3:
            j = 1
            if k > 0:
                k -= 1
        else:
            j += 1

    # The gain vector: G1 from b2's table, G2..G6 uniformly quantized (b3..b7).
    gm = np.zeros(7, dtype=np.float32)
    b2 = 0
    for bit in range(5, -1, -1):
        b2 = (b2 << 1) | bb[2, bit]
    gm[1] = b2_table[b2]
    for i in range(2, 7):
        bits = int(ba[row, i - 2, 0])
        step = ba[row, i - 2, 1]
        bm = 0
        for bit in range(bits - 1, -1, -1):
            bm = (bm << 1) | bb[i + 1, bit]
        gm[i] = step * (_F32(bm) - _F32(math.pow(2.0, float(bits - 1))) + _HALF)

    # Its inverse DCT: the first coefficient of each of the six blocks.
    ri = np.zeros(7, dtype=np.float32)
    for i in range(1, 7):
        s = _ZERO
        for m in range(1, 7):
            am = _ONE if m == 1 else _TWO
            angle = math.pi * float(m - 1) * (float(i) - 0.5) / 6.0
            s = s + am * gm[m] * _F32(math.cos(_F32(angle)))
        ri[i] = s

    # The higher-order coefficients (b8..b(L+1)), then each block's inverse DCT.
    cik = np.zeros((7, 11), dtype=np.float32)
    m = 8
    for i in range(1, 7):
        cik[i, 1] = ri[i]
        for kk in range(2, ji[row, i - 1] + 1):
            bits = hoba[row, m - 8]
            if bits == 0:
                cik[i, kk] = _ZERO
            else:
                bm = 0
                for bit in range(bits - 1, -1, -1):
                    bm = (bm << 1) | bb[m, bit]
                scale = quantstep[bits - 1] * standdev[kk - 2]
                level = float(_F32(bm) - _F32(math.pow(2.0, float(bits - 1)))) + 0.5
                cik[i, kk] = _F32(float(scale) * level)
            m += 1
    tl = np.zeros(57, dtype=np.float32)
    harm = 1
    for i in range(1, 7):
        size = ji[row, i - 1]
        for jj in range(1, size + 1):
            s = _ZERO
            for kk in range(1, size + 1):
                ak = _ONE if kk == 1 else _TWO
                angle = math.pi * float(kk - 1) * (float(jj) - 0.5) / float(size)
                s = s + ak * cik[i, kk] * _F32(math.cos(_F32(angle)))
            tl[harm] = s
            harm += 1

    # The log magnitudes: the residual plus the prediction from the previous frame's, interpolated
    # to this frame's harmonics, less the prediction's mean.
    if n <= 15:
        rho = _F32(0.4)
    elif n <= 24:
        rho = _F32(0.03 * float(n) - 0.05)
    else:
        rho = _F32(0.7)
    if n > prev_n:
        for harm in range(prev_n + 1, n + 1):
            prev_ml[harm] = prev_ml[prev_n]
            prev_log2ml[harm] = prev_log2ml[prev_n]
    ratio = _F32(prev_n) / _F32(n)
    intkl = np.zeros(57, dtype=np.int64)
    deltal = np.zeros(57, dtype=np.float32)
    sum77 = _ZERO
    for harm in range(1, n + 1):
        flokl = ratio * _F32(harm)
        intkl[harm] = int(flokl)
        deltal[harm] = flokl - _F32(intkl[harm])
        sum77 = sum77 + (
            (_ONE - deltal[harm]) * prev_log2ml[intkl[harm]]
            + deltal[harm] * prev_log2ml[intkl[harm] + 1]
        )
    sum77 = (rho / _F32(n)) * sum77
    for harm in range(1, n + 1):
        c1 = rho * (_ONE - deltal[harm]) * prev_log2ml[intkl[harm]]
        c2 = rho * deltal[harm] * prev_log2ml[intkl[harm] + 1]
        cur_log2ml[harm] = tl[harm] + c1 + c2 - sum77
        cur_ml[harm] = _F32(math.pow(_TWO, cur_log2ml[harm]))
    return OK, float(w0), n
