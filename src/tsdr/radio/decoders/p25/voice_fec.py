"""P25 Phase 1 voice-frame error control: Hamming(10,6,3), Golay(24,12,8) and its shortened (18,6,8),
and the Reed-Solomon codes over GF(64) that protect link control (LDU1, TDULC), the encryption sync
(LDU2) and the header (HDU).

Reference: TIA-102.BAAA. Words are integers in the order sent: the first bit sent is the most
significant, data bits before parity bits.

- Hamming(10,6,3): 6 data bits, then 4 parity bits; corrects one error, detects some of two
  (syndromes no single error gives).
- Golay(24,12,8): 12 data bits, then 12 parity bits (the cyclic Golay(23,12) code with generator
  x^11 + x^9 + x^7 + x^6 + x^5 + x + 1, the data's first bit its constant term, plus an overall
  parity bit); corrects three errors, detects four. Golay(18,6,8) is the same with the first six
  data bits zero (not sent).
- Reed-Solomon (24,12,13), (24,16,9) and (36,20,17): shortened from (63, 63 - r) over GF(64)
  (primitive x^6 + x + 1, the generator's roots alpha^1..alpha^r), systematic; the 6-bit words are
  sent from the highest degree down, data words first. Decoding: syndromes, Berlekamp-Massey, a
  root search over the positions sent, Forney.
"""

from collections.abc import Sequence
from itertools import combinations

from tsdr.radio.decoders.p25.fec import _GF_EXP, _GF_LOG

# --- Hamming(10,6,3) -----------------------------------------------------------------------------

# The parity bits each data bit sets, first data bit first (the first parity bit most significant).
_HAMMING_PARITY = (0b1110, 0b1101, 0b1011, 0b0111, 0b0011, 0b1100)
_HAMMING_FIX = {p: 1 << (9 - i) for i, p in enumerate(_HAMMING_PARITY)} | {
    1 << j: 1 << j for j in range(4)
}


def _hamming_parity(data: int) -> int:
    parity = 0
    for i, p in enumerate(_HAMMING_PARITY):
        if data >> (5 - i) & 1:
            parity ^= p
    return parity


def hamming_10_6_encode(data: int) -> int:
    return data << 4 | _hamming_parity(data)


def hamming_10_6_decode(word: int) -> tuple[int, int]:
    """(data, errors): errors 0 or 1 corrected, 2 when the syndrome shows more (the data as
    received)."""
    syndrome = (word & 0xF) ^ _hamming_parity(word >> 4)
    if syndrome == 0:
        return word >> 4, 0
    fix = _HAMMING_FIX.get(syndrome)
    if fix is None:
        return word >> 4, 2
    return (word ^ fix) >> 4, 1


# --- Golay(24,12,8) and (18,6,8) -----------------------------------------------------------------

_GOLAY_POLY = 0xAE3  # x^11 + x^9 + x^7 + x^6 + x^5 + x + 1, bit k = x^k


def _golay_parity_of(data: int) -> int:
    """The 12 parity bits (first sent most significant) of 12 data bits (first sent most
    significant)."""
    # As a polynomial the first data bit is the constant term: reverse into that order, take the
    # remainder of data(x) x^11 by the generator, and append the overall parity.
    poly = int(f"{data:012b}"[::-1], 2)
    reg = poly
    for _ in range(12):
        if reg & 1:
            reg ^= _GOLAY_POLY
        reg >>= 1
    check = reg & 0x7FF
    overall = (bin(poly).count("1") + bin(check).count("1")) & 1
    # check bit j (x^j of the remainder) is the j-th parity bit sent; the overall parity is last.
    return int(f"{check:011b}"[::-1], 2) << 1 | overall


_GOLAY_PARITY = tuple(_golay_parity_of(1 << (11 - i)) for i in range(12))


def _golay_parity(data: int) -> int:
    parity = 0
    for i, p in enumerate(_GOLAY_PARITY):
        if data >> (11 - i) & 1:
            parity ^= p
    return parity


def _golay_syndrome(word: int) -> int:
    return (word & 0xFFF) ^ _golay_parity(word >> 12)


# Every pattern of up to three errors has a syndrome of its own (minimum distance 8).
_GOLAY_FIX: dict[int, int] = {}
for _n in range(1, 4):
    for _bits in combinations(range(24), _n):
        _e = sum(1 << b for b in _bits)
        _GOLAY_FIX[_golay_syndrome(_e)] = _e
assert len(_GOLAY_FIX) == 24 + 276 + 2024


def golay_24_12_encode(data: int) -> int:
    return data << 12 | _golay_parity(data)


def golay_24_12_decode(word: int) -> tuple[int, int]:
    """(data, errors): up to 3 errors corrected; 4 when more are seen (the data as received)."""
    syndrome = _golay_syndrome(word)
    if syndrome == 0:
        return word >> 12, 0
    fix = _GOLAY_FIX.get(syndrome)
    if fix is None:
        return word >> 12, 4
    return (word ^ fix) >> 12, bin(fix).count("1")


def golay_18_6_encode(data: int) -> int:
    return golay_24_12_encode(data) & 0x3FFFF


def golay_18_6_decode(word: int) -> tuple[int, int]:
    """(data, errors) of the shortened code: the six missing data bits are zero, so a correction
    that lands on them means more than three errors (4)."""
    syndrome = _golay_syndrome(word)
    if syndrome == 0:
        return word >> 12, 0
    fix = _GOLAY_FIX.get(syndrome)
    if fix is None or fix >> 18:
        return word >> 12 & 0x3F, 4
    return (word ^ fix) >> 12 & 0x3F, bin(fix).count("1")


# --- Reed-Solomon over GF(64) --------------------------------------------------------------------


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_GF_EXP[_GF_LOG[a] + _GF_LOG[b]])


def _div(a: int, b: int) -> int:
    if a == 0:
        return 0
    return int(_GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 63])


def _pow_alpha(k: int) -> int:
    return int(_GF_EXP[k % 63])


def _rs_generator(r: int) -> list[int]:
    """g(x) = (x - alpha)(x - alpha^2)...(x - alpha^r), lowest degree first."""
    g = [1]
    for i in range(1, r + 1):
        root = _pow_alpha(i)
        nxt = [0] * (len(g) + 1)
        for k, c in enumerate(g):
            nxt[k + 1] ^= c
            nxt[k] ^= _mul(c, root)
        g = nxt
    return g


_GENERATORS = {r: _rs_generator(r) for r in (8, 12, 16)}


def rs_encode(data: Sequence[int], parity_words: int) -> list[int]:
    """The parity words (highest degree first) for the data words (highest degree first)."""
    g = _GENERATORS[parity_words]
    reg = [0] * parity_words  # the remainder, lowest degree first
    for d in data:
        feedback = d ^ reg[-1]
        reg = [0] + reg[:-1]
        if feedback:
            for k in range(parity_words):
                reg[k] ^= _mul(g[k], feedback)
    return reg[::-1]


def rs_decode(words: Sequence[int], data_words: int) -> tuple[list[int], int] | None:
    """The data words of a received codeword (words highest degree first) and the words corrected,
    or None when there are more errors than the code can correct (as far as it can tell)."""
    n = len(words)
    r = n - data_words
    coef = list(words[::-1])  # coefficient of x^i
    syndromes = []
    for j in range(1, r + 1):
        s = 0
        for i, c in enumerate(coef):
            if c:
                s ^= _mul(c, _pow_alpha(i * j))
        syndromes.append(s)
    if not any(syndromes):
        return list(words[:data_words]), 0
    # Berlekamp-Massey: the error locator lam(x) = prod (1 - X_k x), lowest degree first.
    lam, prev = [1], [1]
    length, shift, prev_d = 0, 1, 1
    for k in range(r):
        d = syndromes[k]
        for i in range(1, min(length, len(lam) - 1) + 1):
            d ^= _mul(lam[i], syndromes[k - i])
        if d == 0:
            shift += 1
            continue
        scale = _div(d, prev_d)
        new = lam + [0] * max(0, shift + len(prev) - len(lam))
        for i, c in enumerate(prev):
            new[shift + i] ^= _mul(scale, c)
        if 2 * length <= k:
            prev, length, prev_d, shift = lam, k + 1 - length, d, 1
        else:
            shift += 1
        lam = new
    while len(lam) > 1 and lam[-1] == 0:
        lam.pop()
    if len(lam) - 1 != length or length > r // 2:
        return None
    # The roots among the positions sent: lam(alpha^-i) = 0 means an error at degree i.
    positions = []
    for i in range(n):
        x = _pow_alpha(-i)
        v, p = 0, 1
        for c in lam:
            v ^= _mul(c, p)
            p = _mul(p, x)
        if v == 0:
            positions.append(i)
    if len(positions) != length:
        return None
    # Forney (first root alpha^1): e = omega(X^-1) / lam'(X^-1), omega = S(x) lam(x) mod x^r.
    omega = [0] * r
    for a, s in enumerate(syndromes):
        for b, c in enumerate(lam):
            if a + b < r:
                omega[a + b] ^= _mul(s, c)
    for i in positions:
        x = _pow_alpha(-i)
        num, p = 0, 1
        for c in omega:
            num ^= _mul(c, p)
            p = _mul(p, x)
        den, p = 0, 1  # lam'(x): the odd terms, in characteristic 2
        for k in range(1, len(lam), 2):
            den ^= _mul(lam[k], _pow_alpha(-i * (k - 1)))
        if den == 0:
            return None
        coef[i] ^= _div(num, den)
    fixed = coef[::-1]
    return fixed[:data_words], len(positions)
