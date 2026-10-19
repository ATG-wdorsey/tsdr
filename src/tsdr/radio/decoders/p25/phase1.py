"""P25 Phase 1 framing: frame sync, NID, status symbols, trunking signaling blocks (TSDU) and
multi-block trunking messages (PDU).

A frame starts with 24 dibits of frame sync and 32 of NID; from the frame's start, every 36th dibit
is a status symbol, not data. A TSDU carries one to three 98-dibit trunking signaling blocks
(TSBKs), the last with its last-block flag set. A PDU carries a header block, which gives the number
of data blocks after it, and those blocks; on a control channel, multi-block trunking (MBT) PDUs
carry the messages too long for one TSBK, such as those naming the uplink and downlink channels
separately (VHF and UHF systems). Voice data units (HDU, LDU1, LDU2, TDULC, TDU) have fixed lengths:
their data dibits are collected and handed on whole (`VoiceFrame`; `voice.py` parses them). Other
data units are recognized by their NID and passed over until the next frame sync.

Reference: TIA-102.BAAA.
"""

from dataclasses import dataclass

import numpy as np

from tsdr.radio.decoders.p25 import fec

FS_DIBITS = 24
NID_DIBITS = 32
STATUS_PERIOD = 36
_FS_WORD = 0x5575F5FF77FF
_FS_INVERTED = _FS_WORD ^ 0xAAAAAAAAAAAA  # every symbol's sign flipped (a spectrum inverted)
_MASK48 = (1 << 48) - 1
# Bit errors allowed in the 48-bit frame sync. Random data matches within 6, in either polarity,
# about once in 1e7 dibits (35 minutes at 4800 baud).
FS_MAX_ERRORS = 6
MAX_TSBKS = 3
# MBT PDUs: the service access point and the formats (unconfirmed, alternate) sent as rate-1/2
# blocks; data blocks after the header (one or two seen; three is the largest message defined).
MBT_SAP = 61
MBT_FORMATS = (0x15, 0x17)
MAX_MBT_BLOCKS = 3
# A NID beyond the code's sure correction (fec.NID_MAX_ERRORS) is tried again against the NAC
# already heard on the channel: among its 16 codewords, which lie 23 or more apart, a word this
# close is still a NID.
NID_KNOWN_NAC_MAX_ERRORS = 16
# NIDs in a row with one NAC before it counts as the channel's.
_NAC_CONFIRM = 2
# Data units carrying rate-1/2 blocks with a CRC (TSBKs, a PDU's header).
_BLOCK_DUIDS = (fec.DUID_TSDU, fec.DUID_PDU)
# Voice data units and their data dibits after the NID (the null dibits at their ends left out).
VOICE_DIBITS = {
    fec.DUID_HDU: 324,
    fec.DUID_LDU1: 784,
    fec.DUID_LDU2: 784,
    fec.DUID_TDULC: 144,
    fec.DUID_TDU: 0,
}
# Inside a voice data unit, whose length is known, a frame sync counts only this close: IMBE bits
# would match within FS_MAX_ERRORS about once in 40 minutes of voice, within 2 never in practice,
# while a frame cut short and followed by the next still shows its sync.
VOICE_FS_MAX_ERRORS = 2


def _data_count(raw: int) -> int:
    """Data dibits among a frame's first `raw` dibits after its sync (status symbols left out)."""
    return raw - (FS_DIBITS + raw) // STATUS_PERIOD


def _frame_data(inverted: bool, recorded: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    """A candidate frame's data dibits and symbols from those recorded after its sync."""
    dibits: list[int] = []
    soft: list[float] = []
    for i, (d, s) in enumerate(recorded):
        if (FS_DIBITS + i + 1) % STATUS_PERIOD:
            dibits.append(d ^ 2 if inverted else d)
            soft.append(-s if inverted else s)
    return dibits, soft


def _first_block_ok(soft: list[float]) -> bool:
    block = soft[NID_DIBITS : NID_DIBITS + fec.BLOCK_DIBITS]
    return fec.block_crc_ok(fec.block_decode_soft(np.array(block))[0])


def slice_symbols(symbols: np.ndarray) -> np.ndarray:
    """Symbols in units of 45 degrees (+3, +1, -1, -3) to P25 dibits (1, 0, 2, 3)."""
    dibits = np.full(len(symbols), 3, dtype=np.uint8)
    dibits[symbols > -2.0] = 2
    dibits[symbols > 0.0] = 0
    dibits[symbols > 2.0] = 1
    return dibits


@dataclass(frozen=True)
class Nid:
    nac: int
    duid: int
    errors: int


@dataclass(frozen=True)
class Mbt:
    """One multi-block trunking PDU whose header and packet CRCs passed."""

    # 12 bytes: format, SAP, MFID, address, blocks to follow, opcode, 2 argument bytes, CRC
    header: bytes
    data: bytes  # the data blocks without the 4-byte packet CRC
    nac: int

    @property
    def format(self) -> int:
        return self.header[0] & 0x1F

    @property
    def mfid(self) -> int:
        return self.header[2]

    @property
    def opcode(self) -> int:
        return (self.header[7] if self.format == 0x17 else self.data[0]) & 0x3F


@dataclass(frozen=True)
class Tsbk:
    """One trunking signaling block that passed its CRC."""

    data: bytes  # 12 bytes: LB/P/opcode, MFID, 8 argument bytes, CRC
    nac: int
    corrected: int  # coded bits that disagree with the decoded block

    @property
    def last(self) -> bool:
        return bool(self.data[0] & 0x80)

    @property
    def opcode(self) -> int:
        return self.data[0] & 0x3F

    @property
    def mfid(self) -> int:
        return self.data[1]


@dataclass(frozen=True)
class VoiceFrame:
    """A voice data unit: its data dibits after the NID, status symbols removed (VOICE_DIBITS), and
    their symbols (for soft-decision decoding)."""

    duid: int
    nac: int
    dibits: bytes  # one dibit per byte
    symbols: bytes = b""  # float32 per dibit, in units of 45 degrees, the polarity corrected


Event = Nid | Tsbk | Mbt | VoiceFrame


class Phase1Framer:
    """Streaming: symbols in units of 45 degrees in (P25SymbolDemod's output); NIDs, TSBKs and MBTs
    out, in the order they were sent."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._window = 0
        self._seen = 0
        self._in_frame = False
        self._inverted = False
        self._frame_pos = 0  # dibits of the frame so far, sync and status symbols included
        self._data: list[int] = []  # the frame's data dibits after the sync (sign corrected)
        self._soft: list[float] = []  # and their symbols
        self._nid: Nid | None = None
        self._blocks = 0
        self._pdu_header: bytes | None = None
        self._pdu_blocks = 0  # data blocks the PDU header announced
        self._pdu_data: list[bytes] = []
        # A frame sync seen while the NID was being collected: (inverted, dibits and symbols since,
        # the sync window then).
        self._candidate: tuple[bool, list[tuple[int, float]], int] | None = None
        # This frame's NID (and whether it decoded without the NAC-restricted retry), held back
        # while the candidate is judged (_settle); and the candidate's, once decoded.
        self._held: tuple[Nid, bool] | None = None
        self._rival: tuple[Nid, bool] | None = None
        self._nac: int | None = None
        self._nac_run = 0
        # The channel's NAC as the caller has settled it (by majority); it takes precedence.
        self.expected_nac: int | None = None
        # Statistics
        self.syncs = 0
        self.nids_ok = 0
        self.nids_bad = 0
        self.blocks_ok = 0
        self.blocks_bad = 0
        self.mbts_ok = 0
        self.pdus_bad = 0  # header or packet CRC failed
        self.voice_frames = 0

    @property
    def nac(self) -> int | None:
        """The NAC of this channel, if known: the caller's, else two NIDs in a row."""
        if self.expected_nac is not None:
            return self.expected_nac
        return self._nac if self._nac_run >= _NAC_CONFIRM else None

    def process(self, symbols: np.ndarray) -> list[Event]:
        events: list[Event] = []
        dibits = slice_symbols(symbols)
        for d, s in zip(dibits.tolist(), symbols.tolist(), strict=True):
            self._seen += 1
            self._step(d, s, events)
        return events

    def _step(self, d: int, s: float, events: list[Event]) -> None:
        self._window = ((self._window << 2) | d) & _MASK48
        inverted = None
        if self._seen >= FS_DIBITS:
            nid = self._nid if self._in_frame and self._held is None else None
            voice = nid is not None and nid.duid in VOICE_DIBITS
            inverted = self._match_sync(VOICE_FS_MAX_ERRORS if voice else FS_MAX_ERRORS)
        if inverted is not None and not (self._in_frame and self._nid is None):
            # A frame sync starts a new frame even inside one: after a block that failed its CRC
            # the frame's length is unknown, and its next frame may already have begun.
            if self._held is not None:
                self._settle(events, interrupted=True)
            self._start_frame(inverted)
            return
        if inverted is not None and self._candidate is None:
            # Inside the NID being collected: only a candidate (_end_nid, _settle). For some NACs a
            # TSDU's own NID, with its status symbol, matches the inverted sync; after a false sync
            # just before a frame, the real one shows here. (The dibit belongs to both.) A later
            # sync in this NID is among the candidate's dibits: replayed, its frame meets it.
            self._candidate = (inverted, [], self._window)
        elif self._candidate is not None:
            self._candidate[1].append((d, s))
            if self._held is not None and self._settle(events):
                return  # the candidate's frame took over, this dibit included
        if self._in_frame:
            self._frame_dibit(d, s, events)

    def _match_sync(self, max_errors: int) -> bool | None:
        """Whether the latest 24 dibits are a frame sync: None, or True when inverted."""
        if bin(self._window ^ _FS_WORD).count("1") <= max_errors:
            return False
        if bin(self._window ^ _FS_INVERTED).count("1") <= max_errors:
            return True
        return None

    def _start_frame(self, inverted: bool) -> None:
        self.syncs += 1
        self._inverted = inverted
        self._candidate = None
        self._held = None
        self._rival = None
        self._in_frame = True
        self._frame_pos = FS_DIBITS
        self._data = []
        self._soft = []
        self._nid = None
        self._blocks = 0
        self._pdu_header = None
        self._pdu_data = []

    def _frame_dibit(self, d: int, s: float, events: list[Event]) -> None:
        self._frame_pos += 1
        if self._frame_pos % STATUS_PERIOD == 0:
            return
        if self._inverted:
            d, s = d ^ 2, -s
        self._data.append(d)
        self._soft.append(s)
        if self._nid is None:
            if len(self._data) == NID_DIBITS:
                self._end_nid(events)
            return
        if self._held is not None:
            return  # the blocks wait until the frame is settled (_settle)
        if self._nid.duid in VOICE_DIBITS:
            if len(self._data) == NID_DIBITS + VOICE_DIBITS[self._nid.duid]:
                self._end_voice(events)
        elif len(self._data) == NID_DIBITS + fec.BLOCK_DIBITS * (self._blocks + 1):
            self._end_frame_block(events)

    def _end_frame_block(self, events: list[Event]) -> None:
        assert self._nid is not None
        if self._nid.duid == fec.DUID_TSDU:
            self._end_block(events)
        else:
            self._end_pdu_block(events)

    def _end_nid(self, events: list[Event]) -> None:
        decoded = self._decode_nid(self._data)
        if decoded is None:
            self.nids_bad += 1
            self._in_frame = False
            candidate, self._candidate = self._candidate, None
            if candidate is not None:  # the sync seen during this NID starts the frame after all
                self._replay(*candidate, events)
            return
        self._nid = decoded[0]
        if self._nid.duid not in _BLOCK_DUIDS and self._nid.duid not in VOICE_DIBITS:
            # Other data units (packet data): not decoded; look for the next frame sync.
            self._in_frame = False
        if self._candidate is None:
            self._take(self._nid, events)
            if self._in_frame and VOICE_DIBITS.get(self._nid.duid) == 0:
                self._end_voice(events)  # a TDU: nothing after its NID
        else:
            self._held = decoded  # either may be the real frame (_settle)

    def _settle(self, events: list[Event], interrupted: bool = False) -> bool:
        """Judge the held frame against the candidate's, once the candidate's NID is in: a failed
        NID loses at once. Two NIDs: the frame whose first block passes its CRC wins, which waits
        for the first block of each that carries blocks. Without that evidence, the better NID: one
        decoded without the retry, then fewer errors, then the channel's NAC (the held one on a
        tie). Interrupted by a new frame sync, it decides on what is in: the held frame stands if
        the candidate's NID isn't. Whether the candidate's frame took over."""
        assert self._candidate is not None and self._held is not None
        inverted, recorded, window = self._candidate
        have = _data_count(len(recorded))
        if have < NID_DIBITS:
            if interrupted:
                self._keep_held(events)
            return False
        if self._rival is None:
            self._rival = self._decode_nid(_frame_data(inverted, recorded)[0][:NID_DIBITS])
            if self._rival is None:
                self._keep_held(events)
                return False
        (held, held_sure), (rival, rival_sure) = self._held, self._rival
        held_blocks, rival_blocks = held.duid in _BLOCK_DUIDS, rival.duid in _BLOCK_DUIDS
        block_end = NID_DIBITS + fec.BLOCK_DIBITS
        held_ready, rival_ready = len(self._data) >= block_end, have >= block_end
        if not interrupted and (
            (held_blocks and not held_ready) or (rival_blocks and not rival_ready)
        ):
            return False
        # A block not complete when a new sync interrupts gives no evidence.
        held_ok = held_blocks and held_ready and _first_block_ok(self._soft)
        rival_ok = (
            rival_blocks and rival_ready and _first_block_ok(_frame_data(inverted, recorded)[1])
        )
        if held_ok != rival_ok:
            candidate_wins = rival_ok
        else:
            candidate_wins = self._rank(rival, rival_sure) > self._rank(held, held_sure)
        if not candidate_wins:
            self._keep_held(events)
            return False
        self.nids_bad += 1
        self._replay(inverted, recorded, window, events)
        return True

    def _keep_held(self, events: list[Event]) -> None:
        assert self._held is not None
        held = self._held[0]
        self._held = self._rival = self._candidate = None
        self._take(held, events)
        if held.duid in VOICE_DIBITS:
            if self._in_frame and len(self._data) >= NID_DIBITS + VOICE_DIBITS[held.duid]:
                self._end_voice(events)
            return
        while self._in_frame and len(self._data) >= NID_DIBITS + fec.BLOCK_DIBITS * (
            self._blocks + 1
        ):
            self._end_frame_block(events)

    def _rank(self, nid: Nid, sure: bool) -> tuple[bool, int, bool]:
        # Errors before the NAC: a frame that repeats exactly (a TDU) makes the same garbage
        # candidate every time, which must not outrank a clean NID because its NAC is the old one.
        return sure, -nid.errors, nid.nac == self.nac

    def _decode_nid(self, dibits: list[int]) -> tuple[Nid, bool] | None:
        """The NID and whether it decoded without the NAC-restricted retry."""
        word = 0
        for v in dibits:
            word = (word << 2) | v
        result = fec.nid_decode(word)
        if result is not None:
            return Nid(*result), True
        if self.nac is not None:
            result = fec.nid_decode(word, nac=self.nac, max_errors=NID_KNOWN_NAC_MAX_ERRORS)
            if result is not None:
                return Nid(*result), False
        return None

    def _replay(
        self, inverted: bool, recorded: list[tuple[int, float]], window: int, events: list[Event]
    ) -> None:
        """Start the candidate's frame and take its dibits through again as they came, from its
        sync on (frame syncs among them included)."""
        live = self._window
        self._start_frame(inverted)
        self._window = window
        for d, s in recorded:
            self._step(d, s, events)
        self._window = live

    def _take(self, nid: Nid, events: list[Event]) -> None:
        self.nids_ok += 1
        if nid.nac == self._nac:
            self._nac_run += 1
        else:
            self._nac, self._nac_run = nid.nac, 1
        events.append(nid)

    def _end_voice(self, events: list[Event]) -> None:
        assert self._nid is not None
        self._in_frame = False
        self.voice_frames += 1
        length = VOICE_DIBITS[self._nid.duid]
        data = bytes(self._data[NID_DIBITS : NID_DIBITS + length])
        symbols = np.asarray(self._soft[NID_DIBITS : NID_DIBITS + length], dtype=np.float32)
        events.append(VoiceFrame(self._nid.duid, self._nid.nac, data, symbols.tobytes()))

    def _decode_next_block(self) -> tuple[bytes, int]:
        start = NID_DIBITS + fec.BLOCK_DIBITS * self._blocks
        self._blocks += 1
        return fec.block_decode_soft(np.array(self._soft[start : start + fec.BLOCK_DIBITS]))

    def _end_block(self, events: list[Event]) -> None:
        assert self._nid is not None
        data, corrected = self._decode_next_block()
        if fec.block_crc_ok(data):
            self.blocks_ok += 1
            tsbk = Tsbk(data, self._nid.nac, corrected)
            events.append(tsbk)
            if tsbk.last:
                self._in_frame = False
                return
        else:
            # Not knowing whether it was the last, go on to the next block; a frame sync ends the
            # frame anyway.
            self.blocks_bad += 1
        if self._blocks >= MAX_TSBKS:
            self._in_frame = False

    def _end_pdu_block(self, events: list[Event]) -> None:
        assert self._nid is not None
        block, _ = self._decode_next_block()
        if self._pdu_header is None:
            if not fec.block_crc_ok(block):
                # The header gives the PDU's length; without it, wait for the next frame sync.
                self.pdus_bad += 1
                self._in_frame = False
                return
            blocks = block[6] & 0x7F
            if (
                (block[1] & 0x3F) != MBT_SAP
                or (block[0] & 0x1F) not in MBT_FORMATS
                or not 1 <= blocks <= MAX_MBT_BLOCKS
            ):
                self._in_frame = False  # packet data, or another PDU not decoded here
                return
            self._pdu_header, self._pdu_blocks = block, blocks
            return
        self._pdu_data.append(block)
        if len(self._pdu_data) < self._pdu_blocks:
            return
        self._in_frame = False
        data = b"".join(self._pdu_data)
        if fec.crc32(data[:-4]) != int.from_bytes(data[-4:], "big"):
            self.pdus_bad += 1
            return
        self.mbts_ok += 1
        events.append(Mbt(self._pdu_header, data[:-4], self._nid.nac))
