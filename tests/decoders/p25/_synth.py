"""Synthesized P25 Phase 1 channels for the tests: TSDU and MBT frames of a made-up system, voice
data units, as C4FM or CQPSK IQ, and receiver noise."""

from __future__ import annotations

import numpy as np

from tsdr.radio.decoders.p25 import fec, voice
from tsdr.radio.decoders.p25 import voice_fec as vf
from tsdr.radio.decoders.p25.phase1 import _FS_WORD, FS_DIBITS, STATUS_PERIOD
from tsdr.radio.dsp import rrc_taps

SYMBOL_RATE = 4800
RATE = 240_000.0
SPS = 50  # samples per symbol at RATE
_LEVEL = {
    1: 3.0,
    0: 1.0,
    2: -1.0,
    3: -3.0,
}  # dibit -> symbol, in units of 45 degrees (C4FM: 600 Hz)
_STATUS = 1


def pack(*fields: tuple[int, int]) -> bytes:
    """Fields (value, width in bits) packed most significant first into whole bytes."""
    value = width = 0
    for v, w in fields:
        assert 0 <= v < 1 << w
        value, width = value << w | v, width + w
    assert width % 8 == 0
    return value.to_bytes(width // 8, "big")


def tsbk(opcode: int, args: bytes, *, last: bool, mfid: int = 0x00) -> bytes:
    """A 12-byte TSBK with its CRC."""
    assert len(args) == 8
    body = bytes([(0x80 if last else 0) | opcode, mfid]) + args
    return body + fec.crc16(body).to_bytes(2, "big")


def mbt(opcode: int, address: bytes, octets: bytes, payload: bytes) -> list[bytes]:
    """An alternate-format MBT: its header block and data blocks (the packet CRC in the last 4
    bytes)."""
    blocks = (len(payload) + 4 + 11) // 12
    data = payload.ljust(blocks * 12 - 4, b"\0")
    data += fec.crc32(data).to_bytes(4, "big")
    header = bytes([0x37, 0xFD, 0x00]) + address + bytes([0x80 | blocks, opcode]) + octets
    header += fec.crc16(header).to_bytes(2, "big")
    return [header] + [data[i : i + 12] for i in range(0, len(data), 12)]


def frame(nac: int, duid: int, blocks: list[bytes], status: int = _STATUS) -> list[int]:
    """A frame's dibits: frame sync, NID, the blocks coded and interleaved, status symbols every
    36th dibit."""
    coded: list[int] = []
    for block in blocks:
        coded += [int(d) for d in fec.block_encode(block)]
    return raw_frame(nac, duid, coded, status)


def raw_frame(nac: int, duid: int, coded: list[int], status: int = _STATUS) -> list[int]:
    """A frame of these data dibits after the NID."""
    data = [(_FS_WORD >> (46 - 2 * i)) & 3 for i in range(FS_DIBITS)]
    nid = fec.nid_encode(nac, duid)
    data += [(nid >> (62 - 2 * i)) & 3 for i in range(32)] + coded
    out: list[int] = []
    for d in data:
        if (len(out) + 1) % STATUS_PERIOD == 0:
            out.append(status)
        out.append(d)
    while len(out) % STATUS_PERIOD:  # null dibits to the next status symbol, then the status symbol
        out.append(status if (len(out) + 1) % STATUS_PERIOD == 0 else 0)
    return out


def _rc_pulse(alpha: float, span: int) -> np.ndarray:
    """Raised-cosine pulse at SPS samples per symbol: 1 at its center, 0 at the other symbol
    instants."""
    t = np.arange(-span * SPS // 2, span * SPS // 2 + 1) / SPS
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t) ** 2)
    edge = np.isclose(np.abs(t), 1 / (2 * alpha))
    g[edge] = np.pi / 4 * np.sinc(1 / (2 * alpha))
    return g


def _c4fm_pulse() -> np.ndarray:
    """The frequency pulse (phase increment per sample) whose sum over any symbol period is the
    raised-cosine (0.2) pulse: P25's Nyquist filter followed by its inverse-sinc shaping filter.
    Centered in its array."""
    g = _rc_pulse(0.2, 16)
    n = 1 << int(np.ceil(np.log2(4 * len(g))))
    circular = np.roll(np.pad(g, (0, n - len(g))), -(len(g) // 2))
    w = 2 * np.pi * np.fft.rfftfreq(n)
    moving_sum = np.full(len(w), float(SPS), dtype=np.complex128)
    moving_sum[1:] = (1 - np.exp(-1j * w[1:] * SPS)) / (1 - np.exp(-1j * w[1:]))
    band = (
        w * SPS / (2 * np.pi) < 0.8
    )  # the pulse occupies 0.6 of the symbol rate; the moving sum's zeros lie at 1, 2, ...
    spectrum = np.where(band, np.fft.rfft(circular) / np.where(band, moving_sum, 1), 0)
    return np.roll(np.fft.irfft(spectrum, n), n // 2)


def c4fm(dibits: list[int]) -> np.ndarray:
    """C4FM: FM whose phase advance over each symbol period follows the raised-cosine pulse train of
    the levels."""
    impulses = np.zeros(len(dibits) * SPS)
    impulses[::SPS] = [_LEVEL[d] * np.pi / 4 for d in dibits]
    phase = np.cumsum(np.convolve(impulses, _c4fm_pulse(), mode="same"))
    return np.exp(1j * phase).astype(np.complex64)


def cqpsk(dibits: list[int]) -> np.ndarray:
    """CQPSK (LSM): each symbol turns the carrier phase by its level times 45 degrees;
    root-raised-cosine (0.2) pulses."""
    steps = np.array([_LEVEL[d] * np.pi / 4 for d in dibits])
    symbols = np.exp(1j * np.cumsum(steps))
    impulses = np.zeros(len(dibits) * SPS, dtype=np.complex128)
    impulses[::SPS] = symbols
    x = np.convolve(
        impulses, rrc_taps(SPS, 0.2, 8) * np.sqrt(SPS), mode="same"
    )  # unit-energy taps: power about 1
    return x.astype(np.complex64)


def receive(
    iq: np.ndarray, rng: np.random.Generator, *, snr_db: float, offset_hz: float
) -> np.ndarray:
    """A frequency offset and white noise: `snr_db` is the signal over the noise in 12.5 kHz."""
    n = np.arange(len(iq))
    shifted = iq * np.exp(2j * np.pi * offset_hz / RATE * n)
    power = float(np.mean(np.abs(iq) ** 2))
    sigma = np.sqrt(power / 10 ** (snr_db / 10) * RATE / 12_500 / 2)
    noise = rng.normal(0, sigma, len(iq)) + 1j * rng.normal(0, sigma, len(iq))
    return (shifted + noise).astype(np.complex64)


# --- Voice data units ----------------------------------------------------------------------------


def _dibits(value: int, count: int) -> list[int]:
    return [(value >> (2 * (count - 1 - i))) & 3 for i in range(count)]


def _hex_words(data: bytes) -> list[int]:
    value = int.from_bytes(data, "big")
    n = len(data) * 8 // 6
    return [(value >> (6 * (n - 1 - i))) & 0x3F for i in range(n)]


def imbe_dibits(codewords: np.ndarray) -> list[int]:
    """One IMBE frame's (8, 23) codewords interleaved into its 72 dibits."""
    flat = np.asarray(codewords).reshape(-1)
    return [
        int(flat[a]) << 1 | int(flat[b]) for a, b in zip(voice._FIRST, voice._SECOND, strict=True)
    ]


def ldu(imbe: list[np.ndarray], info: bytes, second: bool, lsd: bytes = b"\0\0") -> list[int]:
    """An LDU's 784 data dibits: nine IMBE frames, LDU1's link control (9 octets) or LDU2's
    encryption sync (12) as Reed-Solomon coded Hamming words, the low-speed data."""
    data = _hex_words(info)
    words = data + vf.rs_encode(data, 8 if second else 12)
    coded = [_dibits(vf.hamming_10_6_encode(w), 5) for w in words]
    out: list[int] = []
    for k in range(9):
        out += imbe_dibits(imbe[k])
        if 1 <= k <= 6:
            for w in coded[4 * (k - 1) : 4 * k]:
                out += w
        if k == 7:
            for octet in lsd:
                out += _dibits(octet, 4) + [0, 0, 0, 0]  # the cyclic code's parity: not checked
    return out


def hdu(mi: bytes, mfid: int, algid: int, kid: int, talkgroup: int) -> list[int]:
    raw = mi + bytes([mfid, algid]) + kid.to_bytes(2, "big") + talkgroup.to_bytes(2, "big")
    data = _hex_words(raw)
    out: list[int] = []
    for w in data + vf.rs_encode(data, 16):
        out += _dibits(vf.golay_18_6_encode(w), 9)
    return out


def tdulc(lc: bytes) -> list[int]:
    data = _hex_words(lc)
    words = data + vf.rs_encode(data, 12)
    out: list[int] = []
    for k in range(12):
        out += _dibits(vf.golay_24_12_encode(words[2 * k] << 6 | words[2 * k + 1]), 12)
    return out
