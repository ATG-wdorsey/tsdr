"""P25 Phase 1 error control: the NID's BCH code, the data blocks' rate-1/2 trellis and interleave,
and their CRC.

Reference: TIA-102.BAAA (FDMA common air interface).

- NID: 64 bits sent MSB first: NAC (12) and DUID (4), then 47 parity bits of a BCH(63,16,23) code
  over them, then a parity bit (1 for LDU1 and LDU2, else 0). The generator is built here from
  GF(64) (primitive x^6 + x + 1, roots alpha^1..alpha^22). Decoding is maximum likelihood: the
  received word against all 65,536 codewords.
- Data blocks (TSBK, PDU): 12 bytes = 48 dibits and a flush dibit, coded by a 4-state rate-1/2
  trellis into 49 four-bit constellation words (196 bits), sent interleaved. The decoder is a
  Viterbi decoder on bit distances.
- CRC: CRC-16 with polynomial 0x1021, initial value 0, final XOR 0xFFFF, over the block's first 10
  bytes (found to validate every TSBK OP25 logged from the local ICORRS control channel, 23,961 of
  them). A PDU's data blocks end with a CRC-32 over the packet (`crc32`).
"""

import numpy as np

# --- GF(64) and the BCH(63,16,23) code ------------------------------------------------------------

_GF_POLY = 0b1000011  # x^6 + x + 1
_GF_EXP = np.zeros(126, dtype=np.int64)
_GF_LOG = np.zeros(64, dtype=np.int64)
_x = 1
for _i in range(63):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x40:
        _x ^= _GF_POLY
_GF_EXP[63:] = _GF_EXP[:63]


def _poly_mul_gf2(a: int, b: int) -> int:
    """Product of two GF(2) polynomials held as integers (bit i = coefficient of x^i)."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _minimal_poly(power: int) -> int:
    """Minimal polynomial over GF(2) of alpha^power: the product of (x - alpha^j) over its
    cyclotomic coset."""
    coset = []
    j = power % 63
    while j not in coset:
        coset.append(j)
        j = (2 * j) % 63
    # Multiply out in GF(64): coefficients as field elements, lowest degree first.
    poly = [1]
    for root in coset:
        r = int(_GF_EXP[root])
        nxt = [0] * (len(poly) + 1)
        for k, c in enumerate(poly):
            nxt[k + 1] ^= c
            if c:
                nxt[k] ^= int(_GF_EXP[(_GF_LOG[c] + _GF_LOG[r]) % 63])
        poly = nxt
    assert all(c in (0, 1) for c in poly)
    return sum(c << k for k, c in enumerate(poly))


def _bch_generator() -> int:
    gen, seen = 1, set()
    for power in range(1, 23):
        m = _minimal_poly(power)
        if m not in seen:
            seen.add(m)
            gen = _poly_mul_gf2(gen, m)
    return gen


BCH_GENERATOR = _bch_generator()
BCH_N, BCH_K = 63, 16
assert BCH_GENERATOR.bit_length() - 1 == BCH_N - BCH_K


def bch_encode(info: int) -> int:
    """The 63-bit codeword for 16 information bits (NAC << 4 | DUID): information first, then 47
    parity bits."""
    reg = info << (BCH_N - BCH_K)
    for bit in range(BCH_N - 1, BCH_N - BCH_K - 1, -1):
        if reg >> bit & 1:
            reg ^= BCH_GENERATOR << (bit - (BCH_N - BCH_K))
    return (info << (BCH_N - BCH_K)) | reg


def _codebook() -> np.ndarray:
    basis = np.array([bch_encode(1 << i) for i in range(BCH_K)], dtype=np.uint64)
    book = np.zeros(1 << BCH_K, dtype=np.uint64)
    info = np.arange(1 << BCH_K, dtype=np.uint64)
    for i in range(BCH_K):
        book ^= np.where((info >> np.uint64(i)) & np.uint64(1), basis[i], np.uint64(0))
    return book


_CODEBOOK = _codebook()

# DUIDs (TIA-102.BAAA) and the parity bit each carries.
DUID_HDU, DUID_TDU, DUID_LDU1, DUID_TSDU, DUID_LDU2, DUID_PDU, DUID_TDULC = 0, 3, 5, 7, 10, 12, 15
_DUID_PARITY = {
    DUID_HDU: 0,
    DUID_TDU: 0,
    DUID_TSDU: 0,
    DUID_PDU: 0,
    DUID_TDULC: 0,
    DUID_LDU1: 1,
    DUID_LDU2: 1,
}
# The BCH code corrects 11 errors for certain; maximum likelihood finds the nearest codeword beyond
# that too, but a word that far from every codeword is more likely noise than a NID.
NID_MAX_ERRORS = 11


def nid_encode(nac: int, duid: int) -> int:
    """The 64-bit NID for a NAC and DUID."""
    return bch_encode((nac << 4) | duid) << 1 | _DUID_PARITY.get(duid, 0)


def nid_decode(
    word: int, nac: int | None = None, max_errors: int = NID_MAX_ERRORS
) -> tuple[int, int, int] | None:
    """(NAC, DUID, bit errors) of a received 64-bit NID, or None.

    With `nac`, only that NAC's codewords are candidates (a locked channel). A word more than
    `max_errors` from the nearest candidate (a parity bit that doesn't match the DUID counts as one
    more error) or a DUID that isn't defined rejects the word.
    """
    received = np.uint64((word >> 1) & ((1 << BCH_N) - 1))
    if nac is None:
        dist = np.bitwise_count(_CODEBOOK ^ received)
        best = int(np.argmin(dist))
    else:
        cands = _CODEBOOK[(nac << 4) : (nac << 4) + 16]
        dist = np.bitwise_count(cands ^ received)
        best = (nac << 4) + int(np.argmin(dist))
        dist = dist[best - (nac << 4) : best - (nac << 4) + 1]
    duid = best & 0xF
    parity = _DUID_PARITY.get(duid)
    if parity is None:
        return None
    errors = int(dist.min()) + (parity != (word & 1))
    if errors > max_errors:
        return None
    return best >> 4, duid, errors


def nid_words() -> np.ndarray:
    """Every NID with a defined DUID, as 64-bit words (each NAC's, DUIDs in increasing order)."""
    duids = np.arange(1 << BCH_K) & 0xF
    keep = np.isin(duids, list(_DUID_PARITY))
    parity = np.array([_DUID_PARITY.get(d, 0) for d in range(16)], dtype=np.uint64)
    words: np.ndarray = (_CODEBOOK[keep] << np.uint64(1)) | parity[duids[keep]]
    return words


# --- Data blocks: interleave, rate-1/2 trellis, CRC -----------------------------------------------

BLOCK_BITS = 196
BLOCK_DIBITS = 98
BLOCK_BYTES = 12

# The 196 coded bits are sent in four columns of 4-bit groups: group g of the coded stream
# (row g // 4, column g % 4) is sent at bit 4 * row + {0, 52, 100, 148}[column] (column 0 holds
# 13 groups, the others 12).
_COLUMN_START = (0, 52, 100, 148)
DEINTERLEAVE = np.array(
    [_COLUMN_START[(b // 4) % 4] + 4 * ((b // 4) // 4) + b % 4 for b in range(BLOCK_BITS)],
    dtype=np.int64,
)
assert sorted(DEINTERLEAVE.tolist()) == list(range(BLOCK_BITS))
INTERLEAVE = np.argsort(DEINTERLEAVE)

# The rate-1/2 trellis: the constellation word sent for each (state, input dibit); the state is the
# previous input dibit, and a flush dibit 0 ends the block in state 0.
TRELLIS_WORDS = np.array(
    [[0x2, 0xC, 0x1, 0xF], [0xE, 0x0, 0xD, 0x3], [0x9, 0x7, 0xA, 0x4], [0x5, 0xB, 0x6, 0x8]],
    dtype=np.int64,
)
_POPCOUNT4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.int64)


def crc16(data: bytes) -> int:
    reg = 0
    for byte in data:
        reg ^= byte << 8
        for _ in range(8):
            reg = ((reg << 1) ^ 0x1021) & 0xFFFF if reg & 0x8000 else (reg << 1) & 0xFFFF
    return reg ^ 0xFFFF


def block_crc_ok(block: bytes) -> bool:
    return crc16(block[:10]) == int.from_bytes(block[10:12], "big")


def crc32(data: bytes) -> int:
    """The packet CRC of a PDU's data blocks: poly 0x04C11DB7, init 0, not reflected, xorout
    0xFFFFFFFF."""
    reg = 0
    for byte in data:
        reg ^= byte << 24
        for _ in range(8):
            reg = (
                ((reg << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                if reg & 0x80000000
                else (reg << 1) & 0xFFFFFFFF
            )
    return reg ^ 0xFFFFFFFF


def _bytes_to_dibits(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return (bits[0::2] << 1 | bits[1::2]).astype(np.int64)


def trellis_encode(block: bytes) -> np.ndarray:
    """196 coded bits (before interleaving) for a 12-byte block."""
    dibits = np.concatenate([_bytes_to_dibits(block), [0]])
    words = np.empty(len(dibits), dtype=np.int64)
    state = 0
    for i, d in enumerate(dibits):
        words[i] = TRELLIS_WORDS[state, d]
        state = int(d)
    bits: np.ndarray = (words[:, None] >> np.array([3, 2, 1, 0])) & 1
    return bits.reshape(-1).astype(np.uint8)


def block_encode(block: bytes) -> np.ndarray:
    """The 98 dibits sent for a 12-byte block (coded and interleaved)."""
    coded = trellis_encode(block)
    sent = coded[INTERLEAVE]
    dibits: np.ndarray = sent[0::2] << 1 | sent[1::2]
    return dibits.astype(np.uint8)


def trellis_decode(coded: np.ndarray) -> tuple[bytes, int]:
    """Viterbi decoding of 196 coded bits: the 12 bytes and the number of coded bits that disagree
    with them."""
    words = (coded.reshape(-1, 4).astype(np.int64) * np.array([8, 4, 2, 1])).sum(axis=1)
    n = len(words)
    inf = 1 << 30
    metric = np.array([0, inf, inf, inf], dtype=np.int64)
    back = np.zeros((n, 4), dtype=np.int64)
    # cost[s, d]: distance of word i from the word sent leaving state s with input d (which leads to
    # state d).
    for i in range(n):
        cost = metric[:, None] + _POPCOUNT4[TRELLIS_WORDS ^ words[i]]
        back[i] = np.argmin(cost, axis=0)
        metric = cost[back[i], np.arange(4)]
    state = 0  # the flush dibit
    dibits = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        dibits[i] = state
        state = int(back[i, state])
    data = dibits[:-1]
    bits = np.stack([data >> 1, data & 1], axis=1).reshape(-1).astype(np.uint8)
    return np.packbits(bits).tobytes(), int(metric[0])


def block_decode(dibits: np.ndarray) -> tuple[bytes, int]:
    """A data block from its 98 received dibits (status symbols removed): its 12 bytes and the bits
    corrected."""
    bits = np.stack([dibits >> 1 & 1, dibits & 1], axis=1).reshape(-1)
    return trellis_decode(bits[DEINTERLEAVE])


_WORD_BITS = ((np.arange(16)[:, None] >> np.array([3, 2, 1, 0])) & 1).astype(np.float64)
# A bit's confidence is capped at this, so one symbol far out (a noise spike) can't outweigh its
# neighbors.
_SOFT_CAP = 2.0


def block_decode_soft(symbols: np.ndarray) -> tuple[bytes, int]:
    """A data block from its 98 received symbols in units of 45 degrees (ideally +3, +1, -1, -3),
    status symbols removed: soft-decision Viterbi decoding. Returns the 12 bytes and the coded bits
    that disagree with them.

    A dibit's first bit is its sign (1 for -1 and -3), its second whether it is an outer symbol (1
    for +3 and -3), so each bit's confidence that it is 0 is, for the second, 2 - |s| and, for the
    first, s's distance to the nearer sign boundary: 0 and also +-4 (+-180 degrees, where +3 and -3
    meet), with s's sign.
    """
    s = np.asarray(symbols, dtype=np.float64)
    a = np.abs(s)
    sign_conf = np.sign(s) * np.minimum(a, 4.0 - a)
    conf = np.clip(np.stack([sign_conf, 2.0 - a], axis=1).reshape(-1), -_SOFT_CAP, _SOFT_CAP)[
        DEINTERLEAVE
    ]
    words = conf.reshape(-1, 4)
    # cost of each of the 16 constellation words at each step: sum of the confidences against its
    # bits
    word_cost = (words[:, None, :] * (2.0 * _WORD_BITS[None, :, :] - 1.0)).sum(axis=2)
    n = len(words)
    metric = np.array([0.0, np.inf, np.inf, np.inf])
    back = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        cost = metric[:, None] + word_cost[i][TRELLIS_WORDS]
        back[i] = np.argmin(cost, axis=0)
        metric = cost[back[i], np.arange(4)]
    state = 0
    dibits = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        dibits[i] = state
        state = int(back[i, state])
    data = dibits[:-1]
    bits = np.stack([data >> 1, data & 1], axis=1).reshape(-1).astype(np.uint8)
    block = np.packbits(bits).tobytes()
    hard = (conf < 0).astype(np.uint8)
    errors = int(np.count_nonzero(trellis_encode(block) != hard))
    return block, errors
