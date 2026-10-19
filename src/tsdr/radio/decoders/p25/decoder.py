"""P25 decoder: the Phase 1 channel at the dial, a control channel (M2 of plan_p25.md) or a voice
channel, trunked or conventional (M3), and the calls a control channel grants, followed on a second
channel inside the capture (M4); Phase 2 comes later.

IQ -> P25SymbolDemod (C4FM or CQPSK) -> dibits -> Phase1Framer (sync, NID, TSBKs, MBTs, voice
   data units) -> messages -> system state; LDUs' IMBE frames -> ImbeDecoder -> 8 kHz audio
IQ shifted to a granted channel -> a second P25Decoder (its voice at its dial) -> 8 kHz audio

A call heard at the dial starts with its first voice data unit and ends with a terminator or
_VOICE_TIMEOUT_S without one. It plays unless it shows encryption: link control's encrypted flag,
or an algorithm other than clear in the header or an LDU2 (the plan's rule G5); a call's first
LDU1 may play before its first LDU2 shows the algorithm.

Following (the control channel at the dial): the decoder gets the whole capture, so a granted voice
channel inside it is decoded beside the control channel, which keeps decoding. The channel is
shifted to 0 Hz with the receiver's error the control channel measures (scaled to its frequency),
so it starts centered; the second decoder's coarse search stays as a fallback. One call at a time,
of a talkgroup the system's table follows (`talkgroups.py`). A call holds the channel from its grant
until its voice ends; in the hang time after a terminator the channel goes to another call (the same
talkgroup first within _HANG_S of its last transmission). An
encrypted call is followed, muted, only until its algorithm and key ID are known (the header or an
LDU2), which the table keeps for its talkgroup; afterwards that talkgroup's flagged grants are
skipped, unless what it learned was clear (the flag is wrong there). Every call, followed or not,
goes in the call log with what happened to it.
"""

import collections
import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, DemodStatus
from tsdr.radio.decoders.p25 import fec, talkgroups, tsbk, voice
from tsdr.radio.decoders.p25.demod import OFFSET_AMBIGUITY_HZ, P25SymbolDemod
from tsdr.radio.decoders.p25.phase1 import Nid, Phase1Framer, Tsbk, VoiceFrame
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp._kernels import apply_freq_shift_c64
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.imbe import ImbeDecoder

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
# these multiples of it and the half steps between cover a receiver up to about +-3.3 kHz off. A
# shift that has decoded is held longer (a fade) and tried again between the others. A block of any
# NAC counts (the CRC proves the shift), NIDs don't: with up to 16 bit errors allowed they decode
# from a signal half outside the channel filter (review 25: a signal 3 kHz off gave NIDs of NAC
# 0xFFF that locked the channel's NAC). Nor does a block completed within _STEP_GUARD_S of a step:
# its symbols came before it, nor one while the signal's mean frequency is _CENTER_MAX_HZ or more
# from the shift: a shift 4800 Hz from the signal decodes a block now and then (the differential
# detector can't see a turn per symbol), and must not hold the search (reviews 28, 29). An offset on
# the edge between two shifts is the demod's (`_EDGE_ENTER`). Without a carrier in the channel (the
# demod's two filters) the shift that decoded is held: _IDLE_RETRY_S once it has decoded on this
# frequency (a voice channel that drops its carrier between calls keeps the shift for the next
# call), else _SEARCH_STEP_S through the first round of the others and _SEARCH_HOLD_S after it. A
# trial shift is left after _SEARCH_STEP_S either way: its carrier is measured through the shift,
# which can put the signal outside the narrow filter. Times are in samples processed, so a pause or
# a clock step changes nothing.
_SEARCH_STEP_S = 0.5
_SEARCH_HOLD_S = 3.0
_IDLE_RETRY_S = 30.0
_CENTER_MAX_HZ = 900.0
# Decodes read farther than this from the shift move the shift a whole step toward them.
_STEP_HZ = 900.0
# Nothing decoded for _LOCATE_WAIT_S while the demod locates a transmission (`located_hz`) _STEP_HZ or more from the
# shift, or half a step or more from where the reading puts it (the frame syncs haven't found it on its branch:
# review 35, a reply at -1000 Hz read -750 beside a neighbor at 10 dB): the shift moves to the nearest whole step there,
# where the frame syncs can find it (review 34: a reply 2 kHz from the shift the last transmitter left, beyond what the
# timing loop follows, was lost while the shift was held). Once located for _LOCATE_FAR_S when _LOCATE_FAR_HZ or more
# away (it can't decode where it is), else for _LOCATE_NEAR_S: nearer, it decodes where it is at a good C/N (a fresh
# call at +900 Hz moved at once left the next reply, at -900 Hz, 2.1 kHz from the shift). Located nearer the same step
# since (its reading wanders: a reply at -1200 Hz read -760 to -1210 at 10 dB), it still counts, and so it does across
# chunks not located for up to _LOCATE_GAP_S (review 35: beside a 50 dB neighbor a reply was located in one chunk of two
# or three, and the count, restarting at each, moved at 10 ms chunks never). The shift is the whole step nearest the
# location (from a half-step trial too). Never more than half a step beyond the search's own range (_COARSE_STEPS;
# +-3600 Hz centers a signal 3.3 kHz off): a neighbor 6.25 kHz away, its bump within reach of the locator (review 34,
# when the locator looked around the shift), was decoded as this channel once the shift moved onto it (+-4800, +-6000).
_LOCATE_WAIT_S = 0.25
_LOCATE_FAR_HZ = 1500.0
_LOCATE_FAR_S = 0.1
_LOCATE_NEAR_S = 0.25
_LOCATE_GAP_S = 0.1
_STEP_GUARD_S = 0.08
# The whole steps first, then the half steps between them (an offset on the edge between two whole
# steps decodes at neither: CQPSK 1800 Hz off).
_COARSE_STEPS = tuple(k * OFFSET_AMBIGUITY_HZ / 2 for k in (0, 2, -2, 4, -4, 1, -1, 3, -3, 5, -5))
# The channel's NAC: the majority of the latest _NAC_WINDOW NIDs, at least _NAC_LOCK of them. It
# gives way only when absent from a whole window (another system holds the channel now), and the
# system state goes with it. Messages count only with the locked NAC.
_NAC_WINDOW = 16
_NAC_LOCK = 3
# The block pass rate is counted over the latest this many blocks.
_QUALITY_WINDOW = 200
# A call at the dial ends this long after its last voice data unit without a terminator.
_VOICE_TIMEOUT_S = 1.0
_PCM_RATE = 8000
# The voice level control, as DMR's (the vocoder's output on mbelib's scale / 32768).
_AGC_MAX_GAIN = 50.0
_AGC_START_GAIN = 25.0
_AGC_PEAK = 30000 / 32768
# A call is active while its grants or updates keep coming: sites repeat them for every call up
# (the reference capture: every 0.4 s, 99 % within 1.1 s, at most 2.3 s with 14 % of blocks lost).
# Its grants after a longer silence start a new call; so does a grant more than _REGRANT_S after the
# last (a site repeats a grant within half a second, and the next transmission's comes 1.5 s or more
# later: the reference capture).
_ACTIVE_S = 2.5
_REGRANT_S = 1.0
# A call is taken only while fresh, its latest grant or update within this long (a call whose updates
# have stopped has likely ended: the grid's "three", calls left while another held the channel).
_FRESH_S = 1.2
# A followed call whose voice hasn't come this long after it was taken is left.
_VOICE_WAIT_S = 1.0
# After a transmission's terminator the channel stays on the call (its talkgroup's next transmission
# may come on it without a new grant) unless another call wants it, and that talkgroup is preferred
# for this long.
_HANG_S = 2.0
# The usable share of the capture (the decimation's and the analog filter's edges stay out), and a
# channel's half width.
_CAPTURE_USABLE = 0.45
_CHANNEL_HALF_HZ = 6_250.0
_LOG_CALLS = 64


@dataclass(frozen=True)
class P25Call:
    group: int
    channel: int  # 16-bit channel number
    source: int | None
    frequency_hz: int | None  # None while the channel's identifier is unknown
    slot: int | None  # TDMA slot; None on an FDMA channel or while the identifier is unknown
    svcopts: int | None  # None until a grant gives them (updates don't): flags unknown, not clear
    last_seen: float
    first_seen: float = 0.0  # since its grants began on this channel
    granted: float | None = None  # its latest grant (not an update): a transmission starting

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
class P25Voice:
    """The call heard at the dial."""

    nac: int
    group: int | None
    source: int | None  # the talking unit (shown, never logged)
    target: int | None  # the called unit of a unit-to-unit call
    algid: int | None  # None until a header or LDU2 gives it
    flagged: bool  # link control's encrypted flag
    emergency: bool
    started: float
    last_heard: float
    kid: int | None = None  # the key ID, with the algorithm
    # Followed from a grant flagged encrypted: muted until its algorithm shows (link control can be
    # lost with the header).
    granted_encrypted: bool = False

    @property
    def encrypted(self) -> bool:
        if self.algid is None:
            return self.flagged or self.granted_encrypted
        return self.flagged or self.algid != voice.ALGID_CLEAR


@dataclass(frozen=True)
class P25Channel:
    """An active call as the control channel shows it, for the active channels table."""

    call: P25Call
    followed: bool  # the call on the second channel now
    inside: bool | None  # its frequency inside the capture; None while unknown
    algid: int | None = None  # what following this call learned of its encryption, if anything
    kid: int | None = None


@dataclass(frozen=True)
class P25LogEntry:
    """A call that has ended, followed or not. Outcomes: played (its voice heard), learned (encrypted,
    followed until its algorithm and key were known), encrypted (skipped: its talkgroup's encryption is
    known), not_followed (its talkgroup isn't), busy (another call held the channel), outside (the
    capture), no_voice (followed, no voice came), muted (followed, its voice encrypted but left before
    its algorithm showed), unknown (its channel's frequency never known), tdma (a Phase 2 channel:
    M5)."""

    start: float
    end: float
    group: int
    frequency_hz: int | None
    svcopts: int | None
    outcome: str
    ldus: int = 0
    algid: int | None = None
    kid: int | None = None
    # The system it was heard on (WACN, system ID), None while unknown: the log outlives retunes, and
    # a talkgroup number means something else on another system (review 41, R41-2).
    system: tuple[int, int] | None = None


@dataclass(frozen=True)
class P25Following:
    group: int
    frequency_hz: int
    since: float
    voice: P25Voice | None


@dataclass
class _Followed:
    """The call on the second channel (pipeline thread only)."""

    call: P25Call  # as when taken; (group, first_seen) names it
    since: float
    shift_hz: float  # where its channel lies in the input
    phase: float = 0.0
    ldus: int = 0
    last_voice: float = 0.0
    holding: bool = False  # a transmission under way: from a voice frame to its terminator
    learned: tuple[int, int] | None = None
    audible: bool = False  # some of its voice played (not all muted)
    before: tuple[int, bool] = (
        0,
        False,
    )  # LDUs and audible from before its talkgroup was turned off

    @property
    def key(self) -> tuple[int, float]:
        return (self.call.group, self.call.first_seen)


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
    voice: P25Voice | None = None  # the call at the dial (or followed), while it lasts
    voice_frames: int = 0  # LDUs decoded
    channels: tuple[P25Channel, ...] = ()  # the active calls
    talkgroups: tuple[talkgroups.P25Talkgroup, ...] = ()
    follow_new: bool = True
    following: P25Following | None = None
    log: tuple[P25LogEntry, ...] = ()  # the latest calls ended, oldest first
    capture: tuple[float, float] | None = None  # the usable band, Hz
    # The site's channels heard (its control channel and every FDMA channel granted), lowest and
    # highest, Hz: against `capture`, the center and rate that would hold them all.
    site_span: tuple[int, int] | None = None


class P25Decoder(Demodulator):
    HAS_AUDIO = True
    CODEC_AUDIO = True
    LABEL = "P25"
    MODULATION = "C4FM/CQPSK"
    MESSAGE_TYPE = "p25"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500.0
    _SNAPSHOT_INTERVAL = 0.5

    @property
    def audio_prebuffer_seconds(self) -> float:
        return 0.3  # an LDU brings 180 ms at once

    def __init__(self, sample_rate: float = 250_000, *, follow: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self._demod = P25SymbolDemod(sample_rate)
        self._framer = Phase1Framer()
        # The last coarse shift that decoded: a property of the receiver, kept across resets.
        self._good_coarse = 0.0
        self._dial_hz: int | None = None
        self._center_hz: int | None = None
        # The voice channel calls are followed on (a decoder of its own, never following).
        self._follower = P25Decoder(sample_rate, follow=False) if follow else None
        # The call log outlives resets (a retune, a rate change): it is history.
        self._log: collections.deque[P25LogEntry] = collections.deque(maxlen=_LOG_CALLS)
        self._dirty = False
        self._reset_state()

    def _reset_system(self) -> None:
        """What belongs to the system on the channel."""
        if hasattr(self, "_wacn"):
            self._close_calls("system")  # into the log, while the system they belong to is known
        self._plan = tsbk.ChannelPlan()
        self._wacn: int | None = None
        self._sysid: int | None = None
        self._rfss: int | None = None
        self._site: int | None = None
        self._calls: dict[int, P25Call] = {}
        # (sysid, rfss, site) -> (channel, flags, last heard)
        self._adjacent: dict[tuple[int, int, int], tuple[int, int, float]] = {}
        self._cc_channel: int | None = None  # the control channel's own, from its broadcasts
        self._span: tuple[int, int] | None = (
            None  # the site's channels granted: lowest, highest (Hz)
        )
        self._dirty = True

    def _reset_state(self) -> None:
        if hasattr(self, "_open"):
            self._close_calls("reset")
        self._following: _Followed | None = None
        self._last_played: tuple[int, float] | None = (
            None  # (group, time) of the latest transmission followed
        )
        # Active calls not yet in the log, by (group, first seen): the latest of each, and what happened to it.
        self._open: dict[tuple[int, float], P25Call] = {}
        self._results: dict[tuple[int, float], tuple[str, int, int | None, int | None]] = {}
        self._results_open: dict[tuple[int, float], str] = {}  # why a call isn't followed (latest)
        # Calls stopped because their talkgroup was turned off: (LDUs, audible) so far. They may be
        # taken again if it is turned back on.
        self._partial: dict[tuple[int, float], tuple[int, bool]] = {}
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
        self._idle_samples = 0  # of those, without a carrier at the shift that decoded
        self._coarse_known = False  # a shift has decoded on this frequency
        self._far = 0  # decodes in a row read more than _STEP_HZ from the shift
        self._since_decode = 1 << 62  # samples since the last decode that held the search
        # The shift to a transmission the demod locates, the samples it has shown there, and since it last did.
        self._located = (0.0, 0, 0)
        self._since_step = 1 << 62  # samples since the last step
        self._search_wait = _SEARCH_STEP_S
        self._search_pos = 0
        self._voice: P25Voice | None = None
        self._voice_frames = 0
        self._ended: tuple[str, P25Voice] | None = None  # the latest call ended at the dial and why
        self._expect_encrypted = (
            False  # the call followed was granted flagged (P25Voice.granted_encrypted)
        )
        self._imbe = ImbeDecoder()
        self._agc = VoiceAGC(max_gain=_AGC_MAX_GAIN, start_gain=_AGC_START_GAIN, peak=_AGC_PEAK)

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        symbols = self._demod.process(iq_samples)
        if len(symbols) == 0:  # a chunk shorter than a symbol: the voice channel still gets it
            if self._follower is not None:
                self._follow(iq_samples, capture_utc_s)
            return
        pairs = np.concatenate([[self._last_symbol], symbols]) / 3.0
        self._constellation = (pairs[1:] + 1j * pairs[:-1]).astype(np.complex64)
        self._last_symbol = float(symbols[-1])
        ok_before, bad_before = self._framer.blocks_ok, self._framer.blocks_bad
        events = self._framer.process(symbols)
        self._count_blocks(self._framer.blocks_ok - ok_before, self._framer.blocks_bad - bad_before)
        decoded = False
        pcm: list[np.ndarray] = []
        # In the order sent: a later message overrides an earlier one (a grant, then its update).
        for event in events:
            if isinstance(event, Nid):
                self._track_nac(event.nac)
            elif isinstance(event, VoiceFrame):
                # A call follows its own NAC: any NAC starts one (before the channel's locks too, and
                # a radio on another NAC is heard), and while it lasts other NACs' frames wait.
                if self._voice is None or event.nac == self._voice.nac:
                    checked = self._handle_voice(event, capture_utc_s, pcm)
                    decoded = decoded or checked
                    if checked:
                        self._last_message_time = capture_utc_s
            else:
                decoded = True
                if event.nac != self._nac:
                    continue
                self._last_message_time = capture_utc_s
                if isinstance(event, Tsbk):
                    self._tsbks += 1
                    self._handle(tsbk.parse(event.data), capture_utc_s)
                else:
                    self._mbts += 1
                    self._handle(tsbk.parse_mbt(event.header, event.data), capture_utc_s)
        self._search(decoded, len(iq_samples))
        if self._voice is not None and not (
            0.0 <= capture_utc_s - self._voice.last_heard <= _VOICE_TIMEOUT_S
        ):
            self._end_voice("timeout")
        if self._follower is not None:
            self._follow(iq_samples, capture_utc_s)
        if pcm:
            self._audio_batches.append(
                AudioBatch(
                    samples=np.concatenate(pcm),
                    sample_rate=_PCM_RATE,
                    stereo=False,
                    prebuffer_seconds=self.audio_prebuffer_seconds,
                    codec=True,
                )
            )
        self._maybe_snapshot(capture_utc_s)
        self._status = self._build_status()

    def _handle_voice(self, event: VoiceFrame, now: float, pcm: list[np.ndarray]) -> bool:
        """A voice data unit: the call's state, its audio into `pcm`. Whether a code checked it
        (Reed-Solomon of the header, link control or encryption sync), which holds the coarse
        frequency search as a control channel's CRCs do."""
        dibits = np.frombuffer(event.dibits, dtype=np.uint8)
        if event.duid == fec.DUID_TDU:
            self._end_voice("terminator")
            return False
        if event.duid == fec.DUID_TDULC:
            lc = voice.parse_tdulc(dibits)
            self._end_voice("terminator")
            return lc is not None
        if event.duid == fec.DUID_HDU:
            header = voice.parse_hdu(dibits)
            if header is None:
                return False
            # A header starts a transmission: one heard before has ended (its terminator lost).
            self._end_voice("new_header")
            self._voice_update(
                now,
                event.nac,
                {"group": header.talkgroup, "algid": header.sync.algid, "kid": header.sync.kid},
            )
            return True
        symbols = np.frombuffer(event.symbols, dtype=np.float32) if event.symbols else None
        ldu = voice.parse_ldu(dibits, second=event.duid == fec.DUID_LDU2, symbols=symbols)
        lc, sync = ldu.link_control, ldu.sync
        changes: dict[str, Any] = {}
        if (
            lc is not None
            and lc.lco in (voice.LCO_GROUP_VOICE, voice.LCO_UNIT_TO_UNIT)
            and not lc.protected
            and lc.mfid == 0
        ):
            old = self._voice
            known = None if old is None else (old.group, old.target)
            if known not in (None, (None, None), (lc.talkgroup, lc.target)):
                # Another call's link control without its header: the last one's terminator was
                # lost.
                self._end_voice("new_call")
            changes = {"group": lc.talkgroup, "target": lc.target, "source": lc.source}
            options = lc.service_options
            if options is not None:  # flags not given leave what is known as it is
                changes["flagged"] = bool(options & voice.SVC_ENCRYPTED)
                changes["emergency"] = bool(options & voice.SVC_EMERGENCY)
        elif (
            lc is not None
            and lc.protected
            and lc.lco
            in (
                voice.LCO_GROUP_VOICE,
                voice.LCO_UNIT_TO_UNIT,
            )
        ):
            changes = {"flagged": True}  # link control encrypted: an encrypted call
        elif sync is not None:
            changes = {"algid": sync.algid, "kid": sync.kid}
        self._voice_update(now, event.nac, changes)
        self._voice_frames += 1
        assert self._voice is not None
        if not self._voice.encrypted:
            for k, codewords in enumerate(ldu.imbe):
                reliability = None if ldu.reliability is None else ldu.reliability[k]
                speech = self._imbe.decode_frame_float(codewords, reliability)
                pcm.append(self._agc.process(speech / 32768.0))
        return lc is not None or sync is not None

    def _voice_update(self, now: float, nac: int, changes: dict[str, Any]) -> None:
        """The call at the dial with `changes` applied: a new one if none is heard, or the last
        voice data unit came more than _VOICE_TIMEOUT_S ago (a gap between batches, a clock step)."""
        old = self._voice
        if old is not None and not 0.0 <= now - old.last_heard <= _VOICE_TIMEOUT_S:
            self._end_voice("timeout")
            old = None
        if old is None:
            self._imbe.reset()
            self._agc.reset()
            base = P25Voice(
                nac=nac,
                group=None,
                source=None,
                target=None,
                algid=None,
                flagged=False,
                emergency=False,
                started=now,
                last_heard=now,
                granted_encrypted=self._expect_encrypted,
            )
        else:
            base = old
        new = replace(base, last_heard=now, **changes)
        self._voice = new
        if old is None or (old.group, old.source, old.target, old.encrypted, old.emergency) != (
            new.group,
            new.source,
            new.target,
            new.encrypted,
            new.emergency,
        ):
            self._report_voice(new, "start" if old is None else "update")

    def _end_voice(self, reason: str) -> None:
        call, self._voice = self._voice, None
        if call is None:
            return
        self._ended = (reason, call)
        self._dirty = True
        logger.info(
            "p25_voice reason=%s group=%s seconds=%.1f",
            reason,
            call.group,
            call.last_heard - call.started,
        )

    def _report_voice(self, call: P25Voice, reason: str) -> None:
        self._dirty = True
        who = f"TG {call.group}" if call.group is not None else "voice"
        if call.target is not None:
            who = f"unit {call.target}"
        src = f" from {call.source}" if call.source is not None else ""
        if call.encrypted:
            state = " ENC" + (f" (ALGID 0x{call.algid:02x})" if call.algid is not None else "")
        else:
            state = ""
        self._pending.append(
            DecodedMessage(
                text=f"Voice {who}{src}{state}{' EMERG' if call.emergency else ''}",
                timestamp=call.last_heard,
            )
        )
        # Unit IDs (source, target) stay out of the log file.
        logger.info(
            "p25_voice reason=%s group=%s unit_to_unit=%s algid=%s encrypted=%s emergency=%s",
            reason,
            call.group,
            call.target is not None,
            "unknown" if call.algid is None else f"0x{call.algid:02x}",
            call.encrypted,
            call.emergency,
        )

    def _track_nac(self, nac: int) -> None:
        self._nacs.append(nac)
        if self._nac is not None and len(self._nacs) == _NAC_WINDOW and self._nac not in self._nacs:
            if self._voice is not None and self._voice.nac == self._nac:
                self._end_voice("nac")  # another system holds the channel now
            self._nac = None
            self._reset_system()
        if self._nac is None:
            top, n = collections.Counter(self._nacs).most_common(1)[0]
            if n >= _NAC_LOCK and 2 * n > len(self._nacs):
                self._nac = top
                self._dirty = True
        self._framer.expected_nac = self._nac

    def _search(self, decoded: bool, samples: int) -> None:
        self._since_step += samples
        if (
            decoded
            and self._since_step > _STEP_GUARD_S * self.sample_rate
            and abs(self._demod.center_hz) < _CENTER_MAX_HZ
        ):
            # Decodes read a branch away (two in a row: not a reading still settling): the shift
            # follows, keeping the filters centered.
            rest = self._demod.frequency_offset_hz - self._demod.coarse_hz
            self._far = self._far + 1 if abs(rest) > _STEP_HZ else 0
            if self._far >= 2:
                self._demod.step_coarse(math.copysign(OFFSET_AMBIGUITY_HZ, rest))
                self._far = 0
            self._since_decode = 0
            self._located = (0.0, 0, 0)
            self._search_samples = 0
            self._idle_samples = 0
            self._search_wait = _SEARCH_HOLD_S
            self._search_pos = 0
            self._good_coarse = self._demod.coarse_hz
            self._coarse_known = True
            return
        self._since_decode += samples
        located = self._demod.located_hz
        coarse = self._demod.coarse_hz
        target = coarse
        if located is not None:
            target = round((coarse + located) / OFFSET_AMBIGUITY_HZ) * OFFSET_AMBIGUITY_HZ
            limit = max(abs(c) for c in _COARSE_STEPS) + OFFSET_AMBIGUITY_HZ / 2
            if abs(target) > limit:
                target = coarse
        target_, count, gap = self._located
        rest = self._demod.frequency_offset_hz - coarse
        if target != coarse and count and target == target_:
            self._located = (target, count + samples, 0)
        elif (
            located is not None
            and target != coarse
            and (abs(located) >= _STEP_HZ or abs(located - rest) >= OFFSET_AMBIGUITY_HZ / 2)
        ):
            self._located = (target, samples, 0)
        elif located is None and count and gap + samples <= _LOCATE_GAP_S * self.sample_rate:
            self._located = (target_, count, gap + samples)
        else:
            self._located = (0.0, 0, 0)
        hold = (
            _LOCATE_FAR_S
            if located is not None and abs(located) >= _LOCATE_FAR_HZ
            else _LOCATE_NEAR_S
        )
        if (
            target != coarse
            and self._located[1] >= hold * self.sample_rate
            and self._since_decode > _LOCATE_WAIT_S * self.sample_rate
        ):
            self._demod.set_coarse(target)
            self._located = (0.0, 0, 0)
            self._search_samples = 0
            self._idle_samples = 0
            self._since_step = 0
            self._search_wait = _SEARCH_STEP_S
            return
        # The others in turn, the last good one again between them.
        others = [c for c in _COARSE_STEPS if c != self._good_coarse]
        order = [c for other in others for c in (other, self._good_coarse)]
        if self._demod.coarse_hz == self._good_coarse and not self._demod.carrier:
            # Nothing to find (an idle channel), but the shift may be stale: after a retune, or for
            # another transmitter.
            self._idle_samples += samples
            if self._coarse_known:
                retry = _IDLE_RETRY_S
            elif self._search_pos < len(order):
                retry = _SEARCH_STEP_S
            else:
                retry = _SEARCH_HOLD_S
            if self._idle_samples <= retry * self.sample_rate:
                return
        else:
            self._search_samples += samples
            if self._search_samples <= self._search_wait * self.sample_rate:
                return
        self._demod.set_coarse(order[self._search_pos % len(order)])
        self._search_pos += 1
        self._located = (0.0, 0, 0)
        self._search_samples = 0
        self._idle_samples = 0
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
            self._set_system(wacn=f["wacn"], sysid=f["sysid"], cc_channel=f["channel"])
        elif msg.opcode == tsbk.RFSS_STS_BCST:
            self._set_system(
                sysid=f["sysid"], rfss=f["rfss"], site=f["site"], cc_channel=f["channel"]
            )
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
        # Capture time can step back (a clock correction): no reason for a new call report.
        known = old if old is not None and now - old.last_seen <= _NEW_CALL_S else None
        # A call keeps what was heard of it (its flags, its talker) only while it is the same call: a
        # continuation of its grants and updates on one channel. Anything else is a new transmission,
        # whose flags are unknown until a grant gives them; seen only through updates (its grant
        # missed), it is followed and its own voice decides (G5 rule 2). An encrypted call's 0x44 taken
        # by the clear call after it, on the same channel or another, skipped it (reviews 41, 42).
        same = (
            known is not None
            and known.channel == channel
            and 0.0 <= now - known.last_seen <= _ACTIVE_S
            and not (svcopts is not None and now - (known.granted or known.first_seen) > _REGRANT_S)
        )
        prev = known if same else None
        call = P25Call(
            group=group,
            channel=channel,
            source=source if source is not None else (prev.source if prev else None),
            frequency_hz=freq,
            slot=slot,
            svcopts=svcopts if svcopts is not None else (prev.svcopts if prev else None),
            last_seen=now,
            first_seen=prev.first_seen if prev is not None else now,
            granted=now if svcopts is not None else (prev.granted if prev else None),
        )
        self._calls[group] = call
        if freq is not None and slot is None:
            low, high = self._span or (freq, freq)
            self._span = (min(low, freq), max(high, freq))
        table = self._table()
        if table is not None:
            table.heard(group, now, call.first_seen == now)
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

    # --- Following ---------------------------------------------------------------------------------

    def set_tuned_frequency(self, frequency_hz: int) -> None:
        if (
            self._dial_hz is not None
            and frequency_hz != self._dial_hz
            and self._following is not None
        ):
            self._stop_following("retuned")  # its shift was from the old dial
        self._dial_hz = int(frequency_hz)

    def set_capture_center(self, frequency_hz: int) -> None:
        self._center_hz = int(frequency_hz)

    @property
    def talkgroup_table(self) -> talkgroups.TalkgroupTable | None:
        """The table of the system heard (shared: thread-safe), or None until its broadcasts name it."""
        return self._table()

    def _table(self) -> talkgroups.TalkgroupTable | None:
        system = self._system()
        return None if system is None else talkgroups.table_for(*system)

    def _cc_hz(self) -> int | None:
        resolved = self._plan.resolve(self._cc_channel) if self._cc_channel is not None else None
        return resolved[0] if resolved is not None else None

    def _dial(self) -> int | None:
        """The dial's frequency: as tuned, else the control channel's own (from its broadcasts)."""
        return self._dial_hz if self._dial_hz is not None else self._cc_hz()

    def _capture(self) -> tuple[float, float] | None:
        dial = self._dial()
        if dial is None:
            return None
        center = self._center_hz if self._center_hz is not None else dial
        half = _CAPTURE_USABLE * self.sample_rate
        return (center - half, center + half)

    def _site_span(self) -> tuple[int, int] | None:
        cc = self._cc_hz()
        if self._span is None:
            return None if cc is None else (cc, cc)
        low, high = self._span
        return (low, high) if cc is None else (min(low, cc), max(high, cc))

    def _inside(self, frequency_hz: int | None) -> bool | None:
        capture = self._capture()
        if capture is None or frequency_hz is None:
            return None
        return capture[0] + _CHANNEL_HALF_HZ <= frequency_hz <= capture[1] - _CHANNEL_HALF_HZ

    def _position(self, frequency_hz: int) -> float:
        """Where a channel at `frequency_hz` lies in the input: its distance from the dial plus the
        receiver's error there, the control channel's (its measured offset less its own distance
        from the dial) scaled by frequency."""
        dial = self._dial()
        assert dial is not None
        cc = self._cc_hz() or dial
        error = self._demod.frequency_offset_hz - (cc - dial)
        return (frequency_hz - dial) + error * frequency_hz / cc

    def _verdict(self, call: P25Call, table: talkgroups.TalkgroupTable) -> str | None:
        """Why `call` can't be followed (a P25LogEntry outcome), or None when it can."""
        if call.frequency_hz is None:
            return "unknown"
        if call.slot is not None:
            return "tdma"
        inside = self._inside(call.frequency_hz)
        if inside is None:
            return "unknown"  # the dial unknown: nothing to shift from
        if not inside:
            return "outside"
        if not table.follows(call.group):
            return "not_followed"
        tg = table.get(call.group)
        if call.encrypted and tg is not None and tg.algid not in (None, voice.ALGID_CLEAR):
            return "encrypted"
        return None

    def _follow(self, iq: np.ndarray, now: float) -> None:
        """The second channel: feed the call followed and see whether it has ended; take the best
        call wanting the channel; log the calls ended."""
        table = self._table()
        if self._following is not None:
            self._feed(iq, now, table)
        active = {
            (c.group, c.first_seen): c
            for c in self._calls.values()
            if 0.0 <= now - c.last_seen <= _ACTIVE_S
        }
        self._open.update(active)
        f = self._following
        best: P25Call | None = None
        if table is not None:
            for key, call in active.items():
                if f is not None and key == f.key or key in self._results:
                    continue  # followed now, or before (its grant repeating)
                verdict = self._verdict(call, table)
                if verdict is None and f is not None and not self._yields(f, call, now):
                    verdict = "busy"
                if verdict is not None:
                    self._results_open[key] = verdict
                elif now - call.last_seen <= _FRESH_S and (
                    best is None or self._better(call, best, now)
                ):
                    best = call
        if best is not None and (f is None or self._yields(f, best, now)):
            if f is not None:
                self._stop_following("yielded")
            self._start_following(best, now)
            self._feed(iq, now, table)
        for key in [k for k, c in self._open.items() if k not in active]:
            if self._following is None or key != self._following.key:
                self._log_call(key, self._open.pop(key))

    def _better(self, a: P25Call, b: P25Call, now: float) -> bool:
        """Whether `a` should have the channel before `b`: the talkgroup just heard within _HANG_S,
        then the newer call."""
        last = self._last_played
        if (
            last is not None
            and 0.0 <= now - last[1] <= _HANG_S
            and (a.group == last[0]) != (b.group == last[0])
        ):
            return a.group == last[0]
        return a.first_seen > b.first_seen

    def _yields(self, f: _Followed, call: P25Call, now: float) -> bool:
        """Whether the call followed gives the channel to `call`: in the hang time after a
        transmission's terminator, yes; not while a transmission is under way, nor while waiting for
        its first voice (it comes a few tenths of a second after the grant; up to _VOICE_WAIT_S)."""
        return not f.holding and f.ldus > 0

    def _start_following(self, call: P25Call, now: float) -> None:
        assert self._follower is not None and call.frequency_hz is not None
        # The shift centers the channel: the second decoder's search starts from 0, whatever an
        # earlier call's transmitter needed.
        self._follower._good_coarse = 0.0
        self._follower.reset()
        table = self._table()
        tg = table.get(call.group) if table is not None else None
        # A flagged grant mutes its call until its algorithm shows, unless the flag was learned wrong.
        self._follower._expect_encrypted = call.encrypted and not (
            tg is not None and tg.algid == voice.ALGID_CLEAR
        )
        key = (call.group, call.first_seen)
        self._following = _Followed(
            call, now, self._position(call.frequency_hz), before=self._partial.pop(key, (0, False))
        )
        self._dirty = True
        logger.info(
            "p25_follow reason=start group=%d frequency_hz=%d shift_hz=%.0f",
            call.group,
            call.frequency_hz,
            self._following.shift_hz,
        )

    def _feed(self, iq: np.ndarray, now: float, table: talkgroups.TalkgroupTable | None) -> None:
        """The capture shifted to the call followed, through the second decoder; its audio and texts
        join this decoder's. Ends the call when it is over (see _yields for giving it up early)."""
        f, inner = self._following, self._follower
        assert f is not None and inner is not None
        shifted, f.phase = apply_freq_shift_c64(iq, f.shift_hz, self.sample_rate, f.phase)
        inner.demodulate(shifted, now)
        audio, texts = inner.get_audio(), [m for m in inner.get_messages() if m.text]
        if table is not None and not table.follows(f.call.group):
            self._stop_following("unfollowed")  # turned off while followed: its audio stops now
            return
        on_air = inner._voice
        if on_air is not None and (
            on_air.target is not None or on_air.group not in (None, f.call.group)
        ):
            # Another call on the channel (the site gave it to another talkgroup in the hang time):
            # not this one's to play.
            self._stop_following("other_call")
            return
        self._audio_batches.extend(audio)
        self._pending.extend(texts)
        f.audible = f.audible or any(len(b.samples) for b in audio)
        if inner._voice_frames > f.ldus:
            f.ldus = inner._voice_frames
            f.last_voice = now
        heard = inner._voice
        if heard is None and inner._ended is not None:
            heard = inner._ended[1]
        if f.holding and inner._voice is None:
            self._last_played = (f.call.group, now)  # a transmission ended
        f.holding = inner._voice is not None
        if (
            heard is not None
            and heard.algid is not None
            and heard.kid is not None
            and f.learned is None
            and (heard.algid != voice.ALGID_CLEAR or f.call.encrypted)
        ):
            f.learned = (heard.algid, heard.kid)
            if table is not None:
                table.learn(f.call.group, heard.algid, heard.kid)
            if heard.algid == voice.ALGID_CLEAR:
                inner._expect_encrypted = False  # the flag is wrong: its next transmission plays
            if heard.algid != voice.ALGID_CLEAR:
                self._stop_following("learned")
                return
        latest = self._calls.get(f.call.group)
        moved = latest is None or latest.first_seen != f.call.first_seen
        updated = latest is not None and 0.0 <= now - latest.last_seen <= _ACTIVE_S
        if f.holding:
            return
        # Elapsed times out of range (capture time stepped back) count as over.
        waited, quiet = now - f.since, now - f.last_voice
        if f.ldus == 0:
            if moved or not 0.0 <= waited <= _VOICE_WAIT_S:
                self._stop_following("no_voice")
        elif moved or not 0.0 <= quiet <= _HANG_S or (not updated and quiet > _VOICE_TIMEOUT_S):
            self._stop_following("ended")

    def _stop_following(self, reason: str) -> None:
        f, self._following = self._following, None
        if f is None:
            return
        self._dirty = True
        ldus, audible = f.ldus + f.before[0], f.audible or f.before[1]
        if reason == "unfollowed":
            # Its talkgroup turned off: not a final result, it may be turned back on during the call.
            self._partial[f.key] = (ldus, audible)
            self._results_open[f.key] = "not_followed"
            outcome = "not_followed"
        else:
            outcome = self._outcome(f, ldus, audible)
            algid, kid = f.learned if f.learned is not None else (None, None)
            self._results[f.key] = (outcome, ldus, algid, kid)
        logger.info(
            "p25_follow reason=%s group=%d outcome=%s ldus=%d",
            reason,
            f.call.group,
            outcome,
            f.ldus,
        )

    def _outcome(self, f: _Followed, ldus: int, audible: bool) -> str:
        if f.learned is not None and f.learned[0] != voice.ALGID_CLEAR:
            return "learned"
        if audible:
            return "played"
        if ldus:
            return "muted"  # its voice encrypted, left before its algorithm showed
        return "busy" if self._results_open.get(f.key) == "busy" else "no_voice"

    def _close_calls(self, reason: str) -> None:
        """A reset: the call followed ends and every open call goes in the log."""
        if self._following is not None:
            self._stop_following(reason)
        for key, call in list(self._open.items()):
            self._log_call(key, call)
        self._open.clear()
        self._results.clear()
        self._results_open.clear()
        self._partial.clear()

    def _log_call(self, key: tuple[int, float], call: P25Call) -> None:
        result = self._results.pop(key, None)
        partial = self._partial.pop(key, None)
        if result is None and partial is not None:  # followed until its talkgroup was turned off
            result = ("played" if partial[1] else "not_followed", partial[0], None, None)
        if result is None:  # never followed: why not
            result = (self._results_open.get(key, "unknown"), 0, None, None)
        outcome, ldus, algid, kid = result
        self._results_open.pop(key, None)
        self._log.append(
            P25LogEntry(
                start=call.first_seen,
                end=call.last_seen,
                group=call.group,
                frequency_hz=call.frequency_hz,
                svcopts=call.svcopts,
                outcome=outcome,
                ldus=ldus,
                algid=algid,
                kid=kid,
                system=self._system(),
            )
        )
        self._dirty = True

    def _system(self) -> tuple[int, int] | None:
        return None if self._wacn is None or self._sysid is None else (self._wacn, self._sysid)

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

    def _channel(self, call: P25Call, f: _Followed | None) -> P25Channel:
        key = (call.group, call.first_seen)
        followed = f is not None and key == f.key
        algid: int | None = None
        kid: int | None = None
        if followed and f is not None and f.learned is not None:
            algid, kid = f.learned
        elif key in self._results:
            _, _, algid, kid = self._results[key]
        return P25Channel(call, followed, self._inside(call.frequency_hz), algid, kid)

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
        f, inner = self._following, self._follower
        active = sorted(
            (c for c in self._calls.values() if 0.0 <= now - c.last_seen <= _ACTIVE_S),
            key=lambda c: c.first_seen,
            reverse=True,
        )
        channels = tuple(self._channel(c, f) for c in active)
        followed_voice = inner._voice if f is not None and inner is not None else None
        following = (
            P25Following(f.call.group, f.call.frequency_hz or 0, f.since, followed_voice)
            if f is not None
            else None
        )
        table = self._table()
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
            voice=self._voice or followed_voice,
            voice_frames=self._voice_frames
            + (inner._voice_frames if f is not None and inner else 0),
            channels=channels,
            talkgroups=table.snapshot() if table is not None else (),
            follow_new=table.follow_new if table is not None else True,
            following=following,
            log=tuple(self._log),
            capture=self._capture(),
            site_span=self._site_span(),
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
            if self._follower is not None:
                self._stop_following("sample_rate")
                self._follower.set_sample_rate(rate)

    def reset(self) -> None:
        super().reset()
        self._demod.reset()
        self._demod.set_coarse(self._good_coarse)
        self._framer.reset()
        self._reset_state()
