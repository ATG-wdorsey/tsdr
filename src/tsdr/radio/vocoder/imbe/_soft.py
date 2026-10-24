"""Soft-decision correction of an IMBE frame: Chase-II decoding of its codewords.

For each Golay(23,12) and Hamming(15,11) codeword, the `p` least reliable received bits are flipped
in every combination (2^p trials), each trial is decoded by the hard decoder, and the codeword whose
disagreement with the received bits weighs least (the sum of the reliabilities of the bits it
changes) is kept. c0 first: its data seeds the pseudo-random sequence that covers c1..c6.

Measured (2026-10-17): synthesized C4FM voice with noise, frames whose
88 bits come out right, hard -> soft: 6 dB 67.0 -> 76.8 %, 8 dB 91.3 -> 93.7 %; c0..c3 right at
5 dB 77.8 -> 82.4 %. On a weak simulcast it recovered a frame whose c0 the hard decoder miscorrected.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _golay_parity(data: int, generator: np.ndarray) -> int:
    parity = 0
    for k in range(12):
        if (data >> (11 - k)) & 1:
            parity ^= generator[k]
    return parity


@nb.njit(cache=True)
def _golay_fix(word: int, generator: np.ndarray, matrix: np.ndarray) -> int:
    """The Golay(23,12) codeword the hard decoder finds for `word` (bit j = codeword bit j; data in
    bits 11..22), parity recomputed."""
    data = word >> 11
    syndrome = _golay_parity(data, generator) ^ (word & 0x7FF)
    data ^= int(matrix[syndrome])
    return int((data << 11) | _golay_parity(data, generator))


@nb.njit(cache=True)
def _hamming_fix(word: int, generator: np.ndarray, matrix: np.ndarray) -> int:
    """The Hamming(15,11) codeword the hard decoder finds for `word` (bit j = codeword bit j)."""
    syndrome = 0
    for i in range(4):
        x = word & generator[i]
        parity = 0
        while x:
            parity ^= x & 1
            x >>= 1
        syndrome = (syndrome << 1) | parity
    return int(word ^ matrix[syndrome])


@nb.njit(cache=True)
def _chase(
    word: int,
    rel: np.ndarray,
    width: int,
    p: int,
    golay: bool,
    generator: np.ndarray,
    matrix: np.ndarray,
) -> tuple[int, float]:
    order = np.argsort(rel[:width])
    best = word
    best_d = 1e30
    for pattern in range(1 << p):
        trial = word
        for b in range(p):
            if (pattern >> b) & 1:
                trial ^= 1 << order[b]
        if golay:
            c = _golay_fix(trial, generator, matrix)
        else:
            c = _hamming_fix(trial, generator, matrix)
        diff = c ^ word
        d = 0.0
        for j in range(width):
            if (diff >> j) & 1:
                d += rel[j]
        if d < best_d:
            best_d = d
            best = c
    return best, best_d


@nb.njit(cache=True)
def soft_correct(
    fr: np.ndarray,
    rel: np.ndarray,
    golay_generator: np.ndarray,
    golay_matrix: np.ndarray,
    hamming_generator: np.ndarray,
    hamming_matrix: np.ndarray,
    p_golay: int,
    p_hamming: int,
) -> tuple[np.ndarray, int, float, float]:
    """The (8, 23) received codewords `fr` (bits) with their reliabilities `rel` into the 88-bit
    parameter vector. Returns (vector, errors as mbelib counts them: data bits changed in c0..c3 and
    Hamming words corrected, c0's soft distance, the frame's soft distance)."""
    out = np.empty(88, dtype=np.int8)
    word = 0
    for j in range(23):
        word |= int(fr[0, j]) << j
    c0, d0 = _chase(word, rel[0], 23, p_golay, True, golay_generator, golay_matrix)
    errors = 0
    changed = (c0 ^ word) >> 11
    while changed:
        errors += changed & 1
        changed >>= 1
    dt = d0
    seed = c0 >> 11
    for j in range(12):
        out[j] = (seed >> (11 - j)) & 1
    pr = 16 * seed
    pos = 12
    for i in range(1, 7):
        width = 23 if i < 4 else 15
        word = 0
        for j in range(width - 1, -1, -1):
            pr = (173 * pr + 13849) & 0xFFFF
            word |= (int(fr[i, j]) ^ (pr >> 15)) << j
        if i < 4:
            c, d = _chase(word, rel[i], 23, p_golay, True, golay_generator, golay_matrix)
            changed = (c ^ word) >> 11
            while changed:
                errors += changed & 1
                changed >>= 1
            data = c >> 11
            for j in range(12):
                out[pos + j] = (data >> (11 - j)) & 1
            pos += 12
        else:
            c, d = _chase(word, rel[i], 15, p_hamming, False, hamming_generator, hamming_matrix)
            if c != word:
                errors += 1
            for j in range(11):
                out[pos + j] = (c >> (14 - j)) & 1
            pos += 11
        dt += d
    for j in range(7):
        out[81 + j] = fr[7, 6 - j]
    return out, errors, d0, dt
