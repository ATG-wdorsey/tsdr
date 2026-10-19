# Derived from DSD (https://github.com/szechyjs/dsd), whose license follows:
#
# Copyright (C) 2010 DSD Author
# GPG Key ID: 0x3F1D7FD0 (74EF 430D F7F2 0A48 FCE6  F630 FAA2 635D 3F1D 7FD0)
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

"""P25 Phase 1 voice data units: the header (HDU), the logical link data units (LDU1, LDU2) and the
terminator with link control (TDULC), from their data dibits after the NID (status symbols removed).

- LDU1 and LDU2 (784 dibits each, 180 ms): nine IMBE frames of 72 dibits, with 24 Hamming(10,6,3)
  words between frames 2-8 and the low-speed data before frame 9:

      IMBE1 IMBE2 w w w w IMBE3 w w w w IMBE4 w w w w IMBE5 w w w w IMBE6 w w w w IMBE7 w w w w
      IMBE8 LSD IMBE9

  The 24 words are a Reed-Solomon codeword: LDU1's link control (RS(24,12,13): 72 bits), LDU2's
  encryption sync (RS(24,16,9): message indicator 72 bits, algorithm 8, key 16).
- HDU (324 dibits): 36 Golay(18,6,8) words, an RS(36,20,17) codeword: message indicator 72 bits,
  manufacturer 8, algorithm 8, key 16, talkgroup 16.
- TDULC (144 dibits): 12 Golay(24,12,8) words, each two 6-bit words of an RS(24,12,13) codeword
  holding link control.

An IMBE frame's 144 bits are sent interleaved; the deinterleave tables below come from DSD
(p25p1_const.h; its license above).

Link control fields: TIA-102.AABF; the octets after the LCF, the MFID first.

Reference: TIA-102.BAAA.
"""

from dataclasses import dataclass

import numpy as np

from tsdr.radio.decoders.p25 import voice_fec as vf

HDU_DIBITS = 324
LDU_DIBITS = 784
TDULC_DIBITS = 144
IMBE_DIBITS = 72
_IMBE_STARTS = (0, 72, 164, 256, 348, 440, 532, 624, 712)
_WORD_STARTS = (144, 236, 328, 420, 512, 604)  # four Hamming words each
_LSD_START = 696

# Where each dibit's two bits go in mbelib's (8, 23) codeword layout: (row, bit) for the first bit,
# then for the second (DSD's iW, iX, iY, iZ).
# fmt: off
_IW = (0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4,
       1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 4, 7,
       0, 3, 5, 2, 4, 7)
_IX = (22, 20, 10, 20, 18, 0, 20, 18, 8, 18, 16, 13, 18, 16, 6, 16, 14, 11, 16, 14, 4, 14, 12, 9, 14,
       12, 2, 12, 10, 7, 12, 10, 0, 10, 8, 5, 10, 8, 13, 8, 6, 3, 8, 6, 11, 6, 4, 1, 6, 4, 9, 4, 2, 6, 4,
       2, 7, 2, 0, 4, 2, 0, 5, 0, 13, 2, 0, 21, 3, 21, 11, 0)
_IY = (1, 3, 5, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6, 0, 2, 4, 1, 3, 6,
       0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 6, 0, 2, 5, 1, 3, 7, 0, 2, 5, 1, 4, 7, 0, 3, 5,
       2, 4, 7, 1, 3, 5)
_IZ = (21, 19, 1, 21, 19, 9, 19, 17, 14, 19, 17, 7, 17, 15, 12, 17, 15, 5, 15, 13, 10, 15, 13, 3, 13,
       11, 8, 13, 11, 1, 11, 9, 6, 11, 9, 14, 9, 7, 4, 9, 7, 12, 7, 5, 2, 7, 5, 10, 5, 3, 0, 5, 3, 8, 3,
       1, 5, 3, 1, 6, 1, 14, 3, 1, 22, 4, 22, 12, 1, 22, 20, 2)
# fmt: on
_FIRST = np.array(_IW) * 23 + np.array(_IX)
_SECOND = np.array(_IY) * 23 + np.array(_IZ)

# Algorithm ID of a clear call (TIA-102.AAAD).
ALGID_CLEAR = 0x80
# LCOs (TIA-102.AABF)
LCO_GROUP_VOICE = 0x00
LCO_GROUP_UPDATE = 0x02
LCO_UNIT_TO_UNIT = 0x03
LCO_GROUP_UPDATE_EXPLICIT = 0x04
LCO_CALL_TERMINATION = 0x0F
# Service options: emergency and encrypted (as in grants).
SVC_EMERGENCY = 0x80
SVC_ENCRYPTED = 0x40


def imbe_codewords(dibits: np.ndarray) -> np.ndarray:
    """One IMBE frame's 72 dibits as mbelib's (8, 23) codewords (int8)."""
    d = np.asarray(dibits, dtype=np.int8)
    flat = np.zeros(8 * 23, dtype=np.int8)
    flat[_FIRST] = d >> 1
    flat[_SECOND] = d & 1
    return flat.reshape(8, 23)


def _bits(dibits: np.ndarray, start: int, count: int) -> int:
    """`count` dibits from `start` as one integer, the first bit most significant."""
    value = 0
    for d in dibits[start : start + count]:
        value = value << 2 | int(d)
    return value


def _words_to_bytes(words: list[int]) -> bytes:
    value = 0
    for w in words:
        value = value << 6 | w
    return value.to_bytes(len(words) * 6 // 8, "big")


@dataclass(frozen=True)
class LinkControl:
    """A link control word: 9 octets, the LCF (protected bit, standard-format bit, LCO) first."""

    data: bytes

    @property
    def protected(self) -> bool:
        """The rest is encrypted (not decoded)."""
        return bool(self.data[0] & 0x80)

    @property
    def lco(self) -> int:
        return self.data[0] & 0x3F

    @property
    def mfid(self) -> int:
        """The manufacturer, when the standard-format bit leaves it explicit (else 0, standard)."""
        return 0 if self.data[0] & 0x40 else self.data[1]

    @property
    def service_options(self) -> int | None:
        if self.protected or self.mfid or self.lco not in (LCO_GROUP_VOICE, LCO_UNIT_TO_UNIT):
            return None
        return self.data[2]

    @property
    def talkgroup(self) -> int | None:
        if self.protected or self.mfid or self.lco != LCO_GROUP_VOICE:
            return None
        return int.from_bytes(self.data[4:6], "big")

    @property
    def target(self) -> int | None:
        """The called unit of a unit-to-unit call."""
        if self.protected or self.mfid or self.lco != LCO_UNIT_TO_UNIT:
            return None
        return int.from_bytes(self.data[3:6], "big")

    @property
    def source(self) -> int | None:
        """The talking unit (group, unit-to-unit) or the unit released (call termination)."""
        if self.protected or self.mfid:
            return None
        if self.lco in (LCO_GROUP_VOICE, LCO_UNIT_TO_UNIT, LCO_CALL_TERMINATION):
            return int.from_bytes(self.data[6:9], "big")
        return None


@dataclass(frozen=True)
class EncryptionSync:
    """The encryption sync (LDU2 or HDU): message indicator, algorithm and key."""

    mi: bytes  # 9 octets
    algid: int
    kid: int

    @property
    def clear(self) -> bool:
        return self.algid == ALGID_CLEAR


@dataclass(frozen=True)
class Header:
    """A call's header data unit."""

    sync: EncryptionSync
    mfid: int
    talkgroup: int


@dataclass(frozen=True)
class Ldu:
    """A logical link data unit: its nine IMBE frames (and their bits' reliabilities, given the
    symbols), and LDU1's link control or LDU2's encryption sync when the Reed-Solomon code could
    correct them."""

    second: bool  # LDU2
    imbe: tuple[np.ndarray, ...]  # nine (8, 23) codeword arrays
    reliability: tuple[np.ndarray, ...] | None  # nine (8, 23) arrays, or None
    link_control: LinkControl | None
    sync: EncryptionSync | None
    lsd: bytes  # the two low-speed data octets, uncorrected
    word_errors: int  # Hamming words that needed correction or failed


def bit_reliability(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each dibit's two bits' confidence from its symbol (units of 45 degrees): the first bit
    (the sign) by the distance to the nearer sign boundary, 0 or +-4 (where +3 and -3 meet); the
    second (outer or inner) by the distance of |s| from 2."""
    a = np.abs(np.asarray(symbols, dtype=np.float64))
    return np.clip(np.minimum(a, 4.0 - a), 0.0, None), np.abs(a - 2.0)


def imbe_reliability(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """One IMBE frame's 72 dibits' bit reliabilities in mbelib's (8, 23) layout."""
    flat = np.zeros(8 * 23)
    flat[_FIRST] = first
    flat[_SECOND] = second
    return flat.reshape(8, 23)


def parse_ldu(dibits: np.ndarray, second: bool, symbols: np.ndarray | None = None) -> Ldu:
    imbe = tuple(imbe_codewords(dibits[s : s + IMBE_DIBITS]) for s in _IMBE_STARTS)
    reliability = None
    if symbols is not None and len(symbols) >= LDU_DIBITS:
        first, second_bits = bit_reliability(symbols)
        reliability = tuple(
            imbe_reliability(first[s : s + IMBE_DIBITS], second_bits[s : s + IMBE_DIBITS])
            for s in _IMBE_STARTS
        )
    words, flagged = [], 0
    for start in _WORD_STARTS:
        for k in range(4):
            data, errors = vf.hamming_10_6_decode(_bits(dibits, start + 5 * k, 5))
            words.append(data)
            flagged += errors > 0
    lsd = bytes([_bits(dibits, _LSD_START, 4), _bits(dibits, _LSD_START + 8, 4)])
    link_control, sync = None, None
    if second:
        fixed = vf.rs_decode(words, 16)
        if fixed is not None:
            raw = _words_to_bytes(fixed[0])
            sync = EncryptionSync(raw[:9], raw[9], int.from_bytes(raw[10:12], "big"))
    else:
        fixed = vf.rs_decode(words, 12)
        if fixed is not None:
            link_control = LinkControl(_words_to_bytes(fixed[0]))
    return Ldu(second, imbe, reliability, link_control, sync, lsd, flagged)


def parse_hdu(dibits: np.ndarray) -> Header | None:
    words = [vf.golay_18_6_decode(_bits(dibits, 9 * k, 9))[0] for k in range(36)]
    fixed = vf.rs_decode(words, 20)
    if fixed is None:
        return None
    raw = _words_to_bytes(fixed[0])
    sync = EncryptionSync(raw[:9], raw[10], int.from_bytes(raw[11:13], "big"))
    return Header(sync, raw[9], int.from_bytes(raw[13:15], "big"))


def parse_tdulc(dibits: np.ndarray) -> LinkControl | None:
    words = []
    for k in range(12):
        data, _ = vf.golay_24_12_decode(_bits(dibits, 12 * k, 12))
        words += [data >> 6, data & 0x3F]
    fixed = vf.rs_decode(words, 12)
    return None if fixed is None else LinkControl(_words_to_bytes(fixed[0]))
