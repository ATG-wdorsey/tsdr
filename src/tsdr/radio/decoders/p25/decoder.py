"""P25 decoder: a Phase 1 control channel at the dial (M2 of plan_p25.md; voice and following come
later).

IQ -> P25SymbolDemod (C4FM or CQPSK) -> dibits -> Phase1Framer (sync, NID, TSBKs, MBTs)
   -> messages -> system state
"""

import collections
import logging
from dataclasses import dataclass

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.radio.decoders.p25 import tsbk
from tsdr.radio.decoders.p25.demod import OFFSET_AMBIGUITY_HZ, P25SymbolDemod
from tsdr.radio.decoders.p25.phase1 import Nid, Phase1Framer, Tsbk
from tsdr.radio.demodulators import Demodulator

logger = logging.getLogger(__name__)

# A grant for a talkgroup not heard for this long is reported as a new call.
_NEW_CALL_S = 5.0
# A snapshot lists up to this many calls, those heard within the last minute.
_RECENT_CALLS = 8
_RECENT_S = 60.0
_MAX_CALLS = 64
# An adjacent site no longer announced for this long is dropped.
_ADJACENT_EXPIRY_S = 120.0
# Without a block passing its CRC for this long (a control channel sends dozens a second), the next
# coarse frequency shift is tried: the offset tracking measures only modulo OFFSET_AMBIGUITY_HZ, so
# these multiples of it cover a receiver up to +-3 kHz off. A shift that has decoded is held longer
# (a fade) and tried again between the others. NIDs don't count: with up to 16 bit errors allowed
# they decode from a signal half outside the channel filter. Nor does a block completed within
# _STEP_GUARD_S of a step: its symbols came before it. Times are in samples processed, so a pause
# or a clock step changes nothing.
_SEARCH_STEP_S = 0.5
_SEARCH_HOLD_S = 3.0
_STEP_GUARD_S = 0.08
_COARSE_STEPS = tuple(k * OFFSET_AMBIGUITY_HZ for k in (0, 1, -1, 2, -2))
# The channel's NAC: the majority of the latest _NAC_WINDOW NIDs, at least _NAC_LOCK of them. It
# gives way only when absent from a whole window (another system holds the channel now), and the
# system state goes with it. Messages count only with the locked NAC.
_NAC_WINDOW = 16
_NAC_LOCK = 3
# The block pass rate is counted over the latest this many blocks.
_QUALITY_WINDOW = 200


@dataclass(frozen=True)
class P25Call:
    group: int
    channel: int  # 16-bit channel number
    source: int | None
    frequency_hz: int | None  # None while the channel's identifier is unknown
    slot: int | None  # TDMA slot; None on an FDMA channel or while the identifier is unknown
    svcopts: int | None  # None until a grant gives them (updates don't): flags unknown, not clear
    last_seen: float

    @property
    def encrypted(self) -> bool:
        return bool(self.svcopts and self.svcopts & tsbk.SVC_ENCRYPTED)

    @property
    def emergency(self) -> bool:
        return bool(self.svcopts and self.svcopts & tsbk.SVC_EMERGENCY)


@dataclass(frozen=True)
class P25Site:
    """An adjacent site as the control channel announces it."""

    sysid: int
    rfss: int
    site: int
    frequency_hz: int | None  # the site's control channel; None while its identifier is unknown
    flags: int  # conventional 8, failure 4, valid 2, active 1


@dataclass(frozen=True)
class P25Snapshot:
    """Immutable decoder state for the P25 panel."""

    locked: bool
    nac: int | None
    wacn: int | None
    sysid: int | None
    rfss: int | None
    site: int | None
    tsbk_pct: float
    tsbks: int
    mbts: int
    frequency_offset_hz: float
    calls: tuple[P25Call, ...]
    adjacent: tuple[P25Site, ...]
    time: float  # capture time of the snapshot (the calls' last_seen are on the same clock)


class P25Decoder(Demodulator):
    LABEL = "P25"
    MODULATION = "C4FM/CQPSK"
    MESSAGE_TYPE = "p25"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500.0
    _SNAPSHOT_INTERVAL = 0.5

    def __init__(self, sample_rate: float = 250_000):
        super().__init__()
        self.sample_rate = sample_rate
        self._demod = P25SymbolDemod(sample_rate)
        self._framer = Phase1Framer()
        # The last coarse shift that decoded: a property of the receiver, kept across resets.
        self._good_coarse = 0.0
        self._reset_state()

    def _reset_system(self) -> None:
        """What belongs to the system on the channel."""
        self._plan = tsbk.ChannelPlan()
        self._wacn: int | None = None
        self._sysid: int | None = None
        self._rfss: int | None = None
        self._site: int | None = None
        self._calls: dict[int, P25Call] = {}
        # (sysid, rfss, site) -> (channel, flags, last heard)
        self._adjacent: dict[tuple[int, int, int], tuple[int, int, float]] = {}
        self._dirty = True

    def _reset_state(self) -> None:
        self._reset_system()
        self._nac: int | None = None
        self._nacs: collections.deque[int] = collections.deque(maxlen=_NAC_WINDOW)
        self._block_results: list[bool] = []
        self._tsbks = 0
        self._mbts = 0
        self._last_message_time = 0.0
        self._last_symbol = 0.0
        self._constellation: np.ndarray | None = None
        self._pending: list[DecodedMessage] = []
        self._last_snapshot = 0.0
        self._dirty = False
        # What status() returns: rebuilt by the pipeline thread, one immutable object.
        self._status = DemodStatus()
        self._search_samples = 0  # samples since the last decoded block or step
        self._since_step = 1 << 62  # samples since the last step
        self._search_wait = _SEARCH_STEP_S
        self._search_pos = 0

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        symbols = self._demod.process(iq_samples)
        if len(symbols) == 0:
            return
        pairs = np.concatenate([[self._last_symbol], symbols]) / 3.0
        self._constellation = (pairs[1:] + 1j * pairs[:-1]).astype(np.complex64)
        self._last_symbol = float(symbols[-1])
        ok_before, bad_before = self._framer.blocks_ok, self._framer.blocks_bad
        events = self._framer.process(symbols)
        self._count_blocks(self._framer.blocks_ok - ok_before, self._framer.blocks_bad - bad_before)
        self._search(any(not isinstance(e, Nid) for e in events), len(iq_samples))
        # In the order sent: a later message overrides an earlier one (a grant, then its update).
        for event in events:
            if isinstance(event, Nid):
                self._track_nac(event.nac)
            elif event.nac == self._nac:
                self._last_message_time = capture_utc_s
                if isinstance(event, Tsbk):
                    self._tsbks += 1
                    self._handle(tsbk.parse(event.data), capture_utc_s)
                else:
                    self._mbts += 1
                    self._handle(tsbk.parse_mbt(event.header, event.data), capture_utc_s)
        self._maybe_snapshot(capture_utc_s)
        self._status = self._build_status()

    def _track_nac(self, nac: int) -> None:
        self._nacs.append(nac)
        if self._nac is not None and len(self._nacs) == _NAC_WINDOW and self._nac not in self._nacs:
            self._nac = None
            self._reset_system()
        if self._nac is None:
            top, n = collections.Counter(self._nacs).most_common(1)[0]
            if n >= _NAC_LOCK and 2 * n > len(self._nacs):
                self._nac = top
                self._dirty = True
        self._framer.expected_nac = self._nac

    def _search(self, decoded: bool, samples: int) -> None:
        self._search_samples += samples
        self._since_step += samples
        if decoded and self._since_step > _STEP_GUARD_S * self.sample_rate:
            self._search_samples = 0
            self._search_wait = _SEARCH_HOLD_S
            self._search_pos = 0
            self._good_coarse = self._demod.coarse_hz
            return
        if self._search_samples > self._search_wait * self.sample_rate:
            # The others in turn, the last good one again between them.
            others = [c for c in _COARSE_STEPS if c != self._good_coarse]
            order = [c for other in others for c in (other, self._good_coarse)]
            self._demod.set_coarse(order[self._search_pos % len(order)])
            self._search_pos += 1
            self._search_samples = 0
            self._since_step = 0
            self._search_wait = _SEARCH_STEP_S

    def _count_blocks(self, ok: int, bad: int) -> None:
        if ok == 0 and bad == 0:
            return
        self._block_results.extend([True] * ok + [False] * bad)
        del self._block_results[:-_QUALITY_WINDOW]

    def _handle(self, msg: tsbk.Message | tsbk.Iden | None, now: float) -> None:
        if msg is None:
            return
        if isinstance(msg, tsbk.Iden):
            if self._plan.idens.get(msg.iden) != msg:
                self._plan.update(msg)
                self._dirty = True
            return
        f = msg.fields
        if msg.opcode == tsbk.NET_STS_BCST:
            self._set_system(wacn=f["wacn"], sysid=f["sysid"])
        elif msg.opcode == tsbk.RFSS_STS_BCST:
            self._set_system(sysid=f["sysid"], rfss=f["rfss"], site=f["site"])
        elif msg.opcode == tsbk.ADJ_STS_BCST:
            key = (f["sysid"], f["rfss"], f["site"])
            old = self._adjacent.get(key)
            if old is None or old[:2] != (f["channel"], f["flags"]):
                self._dirty = True
            self._adjacent[key] = (f["channel"], f["flags"], now)
        elif msg.opcode in (tsbk.GRP_V_CH_GRANT, tsbk.GRP_V_CH_GRANT_UPDT_EXP):
            self._grant(f["group"], f["channel"], f["svcopts"], f.get("source"), now)
        elif msg.opcode == tsbk.GRP_V_CH_GRANT_UPDT:
            self._grant(f["group1"], f["channel1"], None, None, now)
            if f["group2"] != f["group1"]:
                self._grant(f["group2"], f["channel2"], None, None, now)

    def _set_system(self, **values: int) -> None:
        for name, value in values.items():
            if getattr(self, f"_{name}") != value:
                setattr(self, f"_{name}", value)
                self._dirty = True

    def _grant(
        self, group: int, channel: int, svcopts: int | None, source: int | None, now: float
    ) -> None:
        """A grant (svcopts given: 0x00, MBT 0x00, 0x03) or an update (svcopts None: 0x02)."""
        resolved = self._plan.resolve(channel)
        freq, slot = resolved if resolved is not None else (None, None)
        old = self._calls.get(group)
        # Capture time can step back (a clock correction): no reason for a new call.
        known = old if old is not None and now - old.last_seen <= _NEW_CALL_S else None
        call = P25Call(
            group=group,
            channel=channel,
            source=source if source is not None else (known.source if known else None),
            frequency_hz=freq,
            slot=slot,
            svcopts=svcopts if svcopts is not None else (known.svcopts if known else None),
            last_seen=now,
        )
        self._calls[group] = call
        if len(self._calls) > _MAX_CALLS:
            others = sorted(
                (g for g in self._calls if g != group), key=lambda g: self._calls[g].last_seen
            )
            for g in others[: len(self._calls) - _MAX_CALLS]:
                del self._calls[g]
        if known is None:
            self._report(call, "new")
        elif known.frequency_hz is None and freq is not None:
            self._report(call, "channel_known")  # the identifier came after the grant
        elif (call.svcopts is None, call.encrypted, call.emergency) != (
            known.svcopts is None,
            known.encrypted,
            known.emergency,
        ):
            self._report(call, "flags")  # a grant shows flags an update didn't carry
        elif channel != known.channel:
            self._report(call, "moved")  # updates repeat the current channel
        elif source is not None and known.source is not None and source != known.source:
            self._report(call, "talker")

    def _report(self, call: P25Call, reason: str) -> None:
        self._dirty = True
        if call.frequency_hz is not None:
            where = f"{call.frequency_hz / 1e6:.4f} MHz"
            if call.slot is not None:
                where += f" slot {call.slot}"
        else:
            where = f"channel 0x{call.channel:04x}"
        if call.svcopts is None:
            flags = " ENC?"  # no grant seen yet: unknown, not clear
        else:
            flags = (" ENC" if call.encrypted else "") + (" EMERG" if call.emergency else "")
        src = f" from {call.source}" if call.source is not None else ""
        self._pending.append(
            DecodedMessage(text=f"TG {call.group}{src} on {where}{flags}", timestamp=call.last_seen)
        )
        # Unit IDs (the source) stay out of the log file.
        logger.info(
            "p25_call reason=%s group=%d frequency_hz=%s slot=%s svcopts=%s",
            reason,
            call.group,
            call.frequency_hz,
            call.slot,
            "unknown" if call.svcopts is None else f"0x{call.svcopts:02x}",
        )

    def _maybe_snapshot(self, now: float) -> None:
        if now < self._last_snapshot:  # capture time stepped back
            self._last_snapshot = now
        for key, (_, _, heard) in list(self._adjacent.items()):
            if abs(now - heard) > _ADJACENT_EXPIRY_S:
                del self._adjacent[key]
                self._dirty = True
        if not self._dirty and now - self._last_snapshot < self._SNAPSHOT_INTERVAL:
            return
        self._dirty = False
        self._last_snapshot = now
        self._pending.append(DecodedMessage(text="", timestamp=now, data=self._to_snapshot(now)))

    def _tsbk_rate(self) -> float:
        results = self._block_results
        return sum(results) / len(results) * 100 if results else 0.0

    def _to_snapshot(self, now: float) -> P25Snapshot:
        calls = sorted(
            (c for c in self._calls.values() if now - c.last_seen <= _RECENT_S),
            key=lambda c: c.last_seen,
            reverse=True,
        )[:_RECENT_CALLS]
        adjacent = tuple(
            P25Site(sysid, rfss, site, r[0] if (r := self._plan.resolve(ch)) else None, flags)
            for (sysid, rfss, site), (ch, flags, _) in sorted(self._adjacent.items())
        )
        return P25Snapshot(
            locked=0.0 <= now - self._last_message_time < 2.0,
            nac=self._nac,
            wacn=self._wacn,
            sysid=self._sysid,
            rfss=self._rfss,
            site=self._site,
            tsbk_pct=self._tsbk_rate(),
            tsbks=self._tsbks,
            mbts=self._mbts,
            frequency_offset_hz=self._demod.frequency_offset_hz,
            calls=tuple(calls),
            adjacent=adjacent,
            time=now,
        )

    def get_messages(self) -> list[DecodedMessage]:
        messages, self._pending = self._pending, []
        return messages

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        points, self._constellation = self._constellation, None
        return None if points is None else (points, "4FSK")

    def _build_status(self) -> DemodStatus:
        """Pipeline thread only."""
        results = self._block_results
        quality = sum(results) / len(results) if results else None
        parts = []
        if self._nac is not None:
            parts.append(f"NAC {self._nac:03X}")
        if self._sysid is not None:
            parts.append(f"SYS {self._sysid:03X}")
        return DemodStatus(
            quality_label=f"TSBK {quality * 100:.0f}%" if quality is not None else None,
            quality=quality,
            description=" ".join(parts) or None,
        )

    def status(self) -> DemodStatus:
        """Thread-safe: one immutable object, replaced whole by the pipeline thread."""
        return self._status

    def set_sample_rate(self, rate: float) -> None:
        if rate != self.sample_rate:
            self.sample_rate = rate
            coarse = self._demod.coarse_hz
            self._demod = P25SymbolDemod(rate)
            self._demod.set_coarse(coarse)

    def reset(self) -> None:
        super().reset()
        self._demod.reset()
        self._demod.set_coarse(self._good_coarse)
        self._framer.reset()
        self._reset_state()
