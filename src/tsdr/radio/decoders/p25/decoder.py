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
channel inside it is decoded beside the control channel, which keeps decoding (`voice_channel.py`).
The channel is shifted to 0 Hz with the receiver's error the control channel measures (scaled to its
frequency), so it starts centered; the second decoder's coarse search stays as a fallback. The calls
are records (`calls.py`: one per call, by rule C1), the follower decides which one has the channel and
when it ends (`follower.py`: FREE, WAIT, HOLD, HANG), and the system's talkgroups hold the choices and
what was learned (`systems.py`). One call at a time; in the hang time after a transmission the channel
goes to another call (the same talkgroup first within its hang time). An encrypted call is followed,
muted, only until its algorithm and key ID are known (the header or an LDU2), which its talkgroup keeps;
afterwards that talkgroup's flagged grants are skipped, unless a flagged call showed clear (the flag is
wrong there) or a rescan asks for another look. Every call, followed or not, goes in the call log with
what happened to it (the one outcome rule, `CallRecord.outcome`).
"""

import collections
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, DemodStatus
from tsdr.radio.decoders.p25 import fec, systems, tsbk, voice
from tsdr.radio.decoders.p25.calls import CallRecord, CallTracker, P25Call, ReportWindow
from tsdr.radio.decoders.p25.demod import OFFSET_AMBIGUITY_HZ, P25SymbolDemod
from tsdr.radio.decoders.p25.follower import Follower, VoiceDecision
from tsdr.radio.decoders.p25.phase1 import Nid, Phase1Framer, Tsbk, VoiceFrame
from tsdr.radio.decoders.p25.voice_channel import VoiceChannel
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.imbe import ImbeDecoder

logger = logging.getLogger(__name__)

# A snapshot lists up to this many calls, those heard within the last minute.
_RECENT_CALLS = 8
_RECENT_S = 60.0
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
# The channel at the dial is a control, voice or data channel by the data units of its NIDs in the
# last _KIND_S (a NID decodes even when its block doesn't); none for that long: no P25 signal.
_KIND_S = 3.0
_KIND_OF_DUID = {
    fec.DUID_TSDU: "control",
    fec.DUID_HDU: "voice",
    fec.DUID_LDU1: "voice",
    fec.DUID_LDU2: "voice",
    fec.DUID_TDU: "voice",
    fec.DUID_TDULC: "voice",
}
# The block pass rate is counted over the latest this many blocks.
_QUALITY_WINDOW = 200
# A call at the dial ends this long after its last voice data unit without a terminator.
_VOICE_TIMEOUT_S = 1.0
_PCM_RATE = 8000
# The voice level control, as DMR's (the vocoder's output on mbelib's scale / 32768).
_AGC_MAX_GAIN = 50.0
_AGC_START_GAIN = 25.0
_AGC_PEAK = 30000 / 32768
# A call taken and stopped at once (learned, another call's voice) leaves the channel to the next candidate in the same
# chunk, at most this many times.
_TAKES_PER_CHUNK = 3
# The usable share of the capture (the decimation's and the analog filter's edges stay out), and a
# channel's half width.
_CAPTURE_USABLE = 0.45
_CHANNEL_HALF_HZ = 6_250.0
# The capture center a fit chooses keeps the LO (the hardware's DC, its leakage and near-DC products) this far from every
# channel of the site heard: 50 kHz from a channel's edge (the owner, 2026-10-20); it moves in raster half steps.
_LO_CLEAR_HZ = 50_000.0 + _CHANNEL_HALF_HZ
_LO_SEARCH_STEPS = 400  # up to 5 MHz either way
# When the rate can't hold the site, the dial's channel is kept this much (of its frequency) further inside the usable
# band than a channel's half width: the envelope's receiver error (review 56, L2).
_DIAL_MARGIN_PPM = 2.5
_LOG_CALLS = 64


def rate_text(rate: float) -> str:
    """A sample rate a fit needs, in MS/s, rounded up to the tenth, never below the need (review 56, R56-2)."""
    return f"{math.ceil(round(rate / 1e5, 6)) / 10:.1f}"


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
        """Muted: its algorithm decides once shown, clear playing whatever the flags say (D8); before, a flag
        mutes (link control's, or its grant's)."""
        if self.algid is None:
            return self.flagged or self.granted_encrypted
        return self.algid != voice.ALGID_CLEAR


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
    cid: int = 0  # the call's serial


@dataclass(frozen=True)
class P25Following:
    group: int
    frequency_hz: int
    since: float
    voice: P25Voice | None


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
    # (the talkgroups are read from `systems.system_for(wacn, sysid)` directly, at the panel's own pace: 10.2)
    follow_new: bool = True
    following: P25Following | None = None
    log: tuple[P25LogEntry, ...] = ()  # the latest calls ended, oldest first
    capture: tuple[float, float] | None = None  # the usable band, Hz
    # The site's channels heard (its control channel and every FDMA channel granted), lowest and
    # highest, Hz: against `capture`, the center and rate that would hold them all.
    site_span: tuple[int, int] | None = None
    # `fit()` while the site's channels heard don't all fit the capture: the panel's advice, as `p25 fit` and `c` place it
    fit: tuple[int, int, float, float] | None = None
    sample_rate: float = 0.0  # the capture's
    channel_kind: str | None = None  # the dial's: control, voice, data; None without a NID for 3 s
    cc_hz: int | None = None  # the site's control channel, as its broadcasts name it
    dial_hz: int | None = None


class P25Decoder(Demodulator):
    HAS_AUDIO = True
    CODEC_AUDIO = True
    LABEL = "P25"
    MODULATION = "C4FM/CQPSK"
    MESSAGE_TYPE = "p25"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500.0
    FREE_TUNING = True  # D7: the capture stays on the site, the dial on its control channel
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
        self._channel = VoiceChannel(P25Decoder(sample_rate, follow=False)) if follow else None
        # The calls and the call log outlive resets (a retune, a rate change): a call's serial is never reused, and the
        # log is history.
        self._tracker = CallTracker()
        self._reports = ReportWindow()
        self._follower = Follower()
        self._log: collections.deque[P25LogEntry] = collections.deque(maxlen=_LOG_CALLS)
        self._dirty = False
        self._reset_state()

    def _reset_system(self) -> None:
        """What belongs to the system on the channel."""
        if hasattr(self, "_wacn"):
            self._close_calls("system")  # into the log, each with its own system
        self._follower.hang = None  # R52-2: a system inherits nothing, its hang talkgroup included
        self._tracker.recent.clear()
        self._reports.clear()
        self._plan = tsbk.ChannelPlan()
        self._wacn: int | None = None
        self._sysid: int | None = None
        self._rfss: int | None = None
        self._site: int | None = None
        # (sysid, rfss, site) -> (channel, flags, last heard)
        self._adjacent: dict[tuple[int, int, int], tuple[int, int, float]] = {}
        self._cc_channel: int | None = None  # the control channel's own, from its broadcasts
        self._span: tuple[int, int] | None = (
            None  # the site's channels granted: lowest, highest (Hz)
        )
        # every channel granted (Hz), for the LO's place in a fit
        self._heard_hz: frozenset[int] = frozenset()
        self._dirty = True

    def _reset_state(self) -> None:
        if hasattr(self, "_wacn"):
            self._close_calls("reset")
        self._follower.hang = None
        self._reset_system()
        self._nac: int | None = None
        self._nacs: collections.deque[int] = collections.deque(maxlen=_NAC_WINDOW)
        # What the channel at the dial carries, from its latest data units (the panel's first line)
        self._kinds: collections.deque[tuple[float, str]] = collections.deque(maxlen=64)
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
        self._chunk_ended: list[
            tuple[str, P25Voice]
        ] = []  # those ended in this chunk (the second channel's report)
        self._expect_encrypted = (
            False  # the call followed was granted flagged (P25Voice.granted_encrypted)
        )
        self._imbe = ImbeDecoder()
        self._agc = VoiceAGC(max_gain=_AGC_MAX_GAIN, start_gain=_AGC_START_GAIN, peak=_AGC_PEAK)

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        self._chunk_ended = []  # R52-4: this chunk's ended transmissions only
        symbols = self._demod.process(iq_samples)
        if len(symbols) == 0:  # a chunk shorter than a symbol: the voice channel still gets it
            if self._channel is not None:
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
                self._kinds.append((capture_utc_s, _KIND_OF_DUID.get(event.duid, "data")))
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
                    self._kinds.append(
                        (capture_utc_s, "control")
                    )  # a trunking MBT (its PDU's NID: data)
                    self._handle(tsbk.parse_mbt(event.header, event.data), capture_utc_s)
        self._search(decoded, len(iq_samples))
        if self._voice is not None and not (
            0.0 <= capture_utc_s - self._voice.last_heard <= _VOICE_TIMEOUT_S
        ):
            self._end_voice("timeout")
        if self._channel is not None:
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
            old = self._voice
            if (
                old is not None
                and old.source
                and lc.source
                and lc.source != old.source
                and old.algid not in (None, voice.ALGID_CLEAR)
            ):
                # Another talker, its header and the last one's terminator lost: an encrypted transmission's algorithm
                # isn't this one's (D10; never a clear one's, which would only mute: the owner's "favor playing").
                changes.update(algid=None, kid=None)
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
        self._chunk_ended.append((reason, call))
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
        system = self._system()
        if system is not None:
            self._tracker.name_system(system)
            if self._nac is not None:
                systems.system_for(*system).heard_nac(self._nac, self._last_message_time)

    def _grant(
        self, group: int, channel: int, svcopts: int | None, source: int | None, now: float
    ) -> None:
        """A grant (svcopts given: 0x00, MBT 0x00, 0x03) or an update (svcopts None: 0x02): the call it belongs to
        (`calls.py`, rule C1), the talkgroup heard, the report."""
        resolved = self._plan.resolve(channel)
        call, new = self._tracker.message(
            self._system(), group, channel, svcopts, source, now, resolved
        )
        freq, slot = call.frequency_hz, call.slot
        if freq is not None and slot is None:
            low, high = self._span or (freq, freq)
            self._span = (min(low, freq), max(high, freq))
        if freq is not None and freq not in self._heard_hz:
            # replaced whole: fits read it on the main thread
            self._heard_hz = self._heard_hz | {freq}
        state = self._state()
        if state is not None:
            flagged = None if svcopts is None else bool(svcopts & tsbk.SVC_ENCRYPTED)
            state.heard(group, now, new, flagged, self._nac)
        reason = self._reports.reason(call, source, now)
        if reason is not None:
            self._report(call.view(), reason)

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
            and self._follower.call is not None
        ):
            self._stop("retuned")  # its shift was from the old dial
        self._dial_hz = int(frequency_hz)

    def set_capture_center(self, frequency_hz: int) -> None:
        self._center_hz = int(frequency_hz)

    @property
    def system_state(self) -> systems.SystemState | None:
        """The system heard (shared: thread-safe), or None until its broadcasts name it."""
        return self._state()

    def _state(self) -> systems.SystemState | None:
        system = self._system()
        return None if system is None else systems.system_for(*system)

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

    def fit(self) -> tuple[int, int, float, float] | None:
        """What `p25 fit` needs (any thread): the site's channels heard, lowest and highest, and the capture center and
        least sample rate whose usable band holds them; None before the site is heard. The center is the raster half
        step nearest the site's middle that keeps the LO _LO_CLEAR_HZ from every channel heard."""
        span = self._site_span()
        if span is None:
            return None
        low, high = span
        dial = self._dial()
        if dial is not None:  # the dial's channel held too (it is the control channel's, normally)
            low, high = min(low, dial), max(high, dial)
        center = self._clear_lo((low + high) / 2, lambda c: True)
        need = 2 * (max(center - low, high - center) + _CHANNEL_HALF_HZ) / (2 * _CAPTURE_USABLE)
        return low, high, center, need

    def center_for(self, rate: float) -> tuple[int, int, float, float] | None:
        """`c` while P25 runs (any thread): `fit()` with the center for the capture's rate as it is. When that rate can't
        hold the site, the center comes as near the site's middle as keeps the dial's channel inside the usable band,
        the LO clear of the channels heard as `fit()` keeps it; None before the site is heard or while the dial is
        unknown."""
        fit = self.fit()
        dial = self._dial()
        if fit is None or dial is None:
            return None
        low, high, _, need = fit
        if rate >= need:
            return fit
        # the farthest the dial's channel may be from the capture's center, the receiver's error kept as a margin; the
        # center as near the site's middle as that lets it
        reach = _CAPTURE_USABLE * rate - _CHANNEL_HALF_HZ - _DIAL_MARGIN_PPM * 1e-6 * dial
        target = min(max((low + high) / 2, dial - reach), dial + reach)
        return low, high, self._clear_lo(target, lambda c: abs(c - dial) <= reach), need

    @staticmethod
    def holds(low: float, high: float, center: float, rate: float) -> bool:
        """Whether a capture at `center` and `rate` holds the channels from `low` to `high` (as `_inside` judges them)."""
        half = _CAPTURE_USABLE * rate - _CHANNEL_HALF_HZ
        return center - half <= low and high <= center + half

    def _clear_lo(self, target: float, allowed: Callable[[float], bool]) -> float:
        """The capture's center for a fit: a raster half step (between two channels: the hardware's DC on none of
        them), the one nearest `target` (ties: the lower) that `allowed` takes and that is _LO_CLEAR_HZ from every
        channel of the site heard; when none within _LO_SEARCH_STEPS is that clear, the nearest one `allowed` takes
        (review 56, R56-1: by the distance from the target itself, not from a half step rounded near it)."""
        channels = set(self._heard_hz)
        cc = self._cc_hz()
        if cc is not None:
            channels.add(cc)
        if self._dial_hz is not None:
            channels.add(self._dial_hz)
        below = (
            math.floor((target - 6_250) / 12_500) * 12_500 + 6_250
        )  # the half step at or below the target
        steps = range(-_LO_SEARCH_STEPS, _LO_SEARCH_STEPS + 2)
        nearest = None
        for c in sorted((below + 12_500 * k for k in steps), key=lambda c: (abs(c - target), c)):
            if not allowed(c):
                continue
            if all(abs(c - f) >= _LO_CLEAR_HZ for f in channels):
                return float(c)
            if nearest is None:
                nearest = c
        return float(nearest if nearest is not None else below)

    def _site_span(self) -> tuple[int, int] | None:
        cc = self._cc_hz()
        span = self._span  # once: the pipeline thread may reset it meanwhile (review 56, L4)
        if span is None:
            return None if cc is None else (cc, cc)
        low, high = span
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

    def _verdict(self, call: CallRecord, state: systems.SystemState) -> str | None:
        """Why `call` can't be followed (DESIGN-M4b 4.3; a P25LogEntry outcome), or None when it can."""
        if call.frequency_hz is None:
            return "unknown"
        if call.slot is not None:
            return "tdma"
        inside = self._inside(call.frequency_hz)
        if inside is None:
            return "unknown"  # the dial unknown: nothing to shift from
        if not inside:
            return "outside"
        if not state.follows(call.group):
            return "not_followed"
        if call.flagged and state.skips_flagged(call.group):
            return "encrypted"
        return None

    def _follow(self, iq: np.ndarray, now: float) -> None:
        """The second channel (DESIGN-M4b 4.3): feed the call followed and see whether it goes on; take the best call
        wanting the channel (fed this same chunk); close the calls ended into the log."""
        state = self._state()
        f = self._follower
        if f.call is not None:
            self._feed(iq, now, state)
        if state is not None:  # no system named yet: nothing taken (nor in M4a)
            for _ in range(_TAKES_PER_CHUNK):
                best = f.choose(
                    now,
                    self._tracker.active(now),
                    lambda c: self._verdict(c, state),
                    self._tracker.moved,
                )
                if best is None:
                    break
                if f.call is not None:
                    self._stop("yielded")  # in its hang time
                self._take(best, now, state)
                self._feed(iq, now, state)
                if f.call is not None:
                    break
            if f.call is not None:
                # The calls that wanted the channel and didn't get it: busy, as the next chunk would find them
                f.choose(
                    now,
                    self._tracker.active(now),
                    lambda c: self._verdict(c, state),
                    self._tracker.moved,
                )
        for call in self._tracker.close_inactive(now, f.call):
            self._log_call(call)

    def _take(self, call: CallRecord, now: float, state: systems.SystemState) -> None:
        assert self._channel is not None and call.frequency_hz is not None
        call.look = state.look(call.group)
        self._channel.start(
            call.frequency_hz, self._position(call.frequency_hz), self._mute_flag(call, state)
        )
        self._follower.take(call, now)
        self._dirty = True
        logger.info(
            "p25_follow reason=start group=%d frequency_hz=%d shift_hz=%.0f",
            call.group,
            call.frequency_hz,
            self._channel.shift_hz,
        )

    @staticmethod
    def _mute_flag(call: CallRecord, state: systems.SystemState | None) -> bool:
        """The call's grant flags encryption, and its talkgroup's flag isn't known wrong (R47-4)."""
        row = state.get(call.group) if state is not None else None
        return call.flagged and not (row is not None and row.flag_wrong)

    def _feed(self, iq: np.ndarray, now: float, state: systems.SystemState | None) -> None:
        """The capture shifted to the call followed, through the second channel; its audio and texts join this
        decoder's unless the follower drops them (4.3 step 2)."""
        f, channel = self._follower, self._channel
        call = f.call
        assert call is not None and channel is not None
        mute = self._mute_flag(call, state)
        if mute != channel.muted_by_flag:
            channel.set_mute_flag(mute)  # D5: a grant's flag arriving after the take
        report = channel.feed(iq, now)
        chosen = state.follows(call.group) if state is not None else True
        decision: VoiceDecision = f.voice(
            report, now, chosen, self._tracker.moved(call), call.active(now)
        )
        for v in report.voices:
            if v.group == call.group and v.source:
                call.source = v.source  # C2: its own voice names its talker
                call.talkers.add(v.source)
        if decision.learn is not None:
            call.learned = decision.learn
            if state is not None:
                state.learn(call.group, *decision.learn, now, call.flagged, call.look)
        elif decision.clear_voice and state is not None:
            state.clear_voice(call.group, now)
        if decision.keep_audio:
            self._audio_batches.extend(report.audio)
            self._pending.extend(report.texts)
        if decision.stop is not None:
            self._stop(decision.stop)

    def _stop(self, reason: str) -> None:
        f = self._follower
        if f.call is None:
            return
        call = f.stop(reason)
        if self._channel is not None:
            self._channel.stop()
        self._dirty = True
        logger.info(
            "p25_follow reason=%s group=%d outcome=%s ldus=%d",
            reason,
            call.group,
            call.outcome,
            call.ldus,
        )

    def _close_calls(self, reason: str) -> None:
        """A reset, a retune, a system change: the call followed ends and every call goes in the log."""
        self._stop(reason)
        for call in self._tracker.close_all():
            self._log_call(call)

    def _log_call(self, call: CallRecord) -> None:
        algid, kid = call.learned if call.learned is not None else (None, None)
        self._log.append(
            P25LogEntry(
                start=call.first_seen,
                end=call.last_seen,
                group=call.group,
                frequency_hz=call.frequency_hz,
                svcopts=call.svcopts,
                outcome=call.outcome,
                ldus=call.ldus,
                algid=algid,
                kid=kid,
                system=call.system,
                cid=call.cid,
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

    def _channel_row(self, call: CallRecord) -> P25Channel:
        algid, kid = call.learned if call.learned is not None else (None, None)
        return P25Channel(
            call.view(), call is self._follower.call, self._inside(call.frequency_hz), algid, kid
        )

    def channel_positions(self) -> tuple[tuple[int, int, float], ...]:
        if self._dial() is None:
            return ()
        return tuple(
            (c.cid, c.frequency_hz, self._position(c.frequency_hz))
            for c in self._tracker.open.values()
            if c.frequency_hz is not None and c.slot is None and self._inside(c.frequency_hz)
        )

    def _channel_kind(self, now: float) -> str | None:
        """control, voice or data: the most of the latest data units within _KIND_S; None without any."""
        counts = collections.Counter(k for t, k in self._kinds if 0.0 <= now - t <= _KIND_S)
        if not counts:
            return None
        if counts["control"]:
            counts["data"] -= counts["control"]  # each trunking MBT came with a PDU's NID
        return max(("control", "voice", "data"), key=lambda k: counts[k])

    def _tsbk_rate(self) -> float:
        results = self._block_results
        return sum(results) / len(results) * 100 if results else 0.0

    def _to_snapshot(self, now: float) -> P25Snapshot:
        records = list(self._tracker.open.values())
        calls = sorted(
            (c.view() for c in self._tracker.recent.values() if now - c.last_seen <= _RECENT_S),
            key=lambda c: c.last_seen,
            reverse=True,
        )[:_RECENT_CALLS]
        adjacent = tuple(
            P25Site(sysid, rfss, site, r[0] if (r := self._plan.resolve(ch)) else None, flags)
            for (sysid, rfss, site), (ch, flags, _) in sorted(self._adjacent.items())
        )
        f, channel = self._follower, self._channel
        active = sorted(
            (c for c in records if c.active(now)), key=lambda c: (c.first_seen, c.cid), reverse=True
        )
        channels = tuple(self._channel_row(c) for c in active)
        inner = channel.decoder if channel is not None else None
        followed_voice = inner._voice if f.call is not None and inner is not None else None
        following = (
            P25Following(
                f.call.group, f.call.frequency_hz or 0, f.call.taken or now, followed_voice
            )
            if f.call is not None
            else None
        )
        state = self._state()
        capture, span = self._capture(), self._site_span()
        fit = None
        if (
            capture is not None
            and span is not None
            and (span[0] - _CHANNEL_HALF_HZ < capture[0] or span[1] + _CHANNEL_HALF_HZ > capture[1])
        ):
            fit = self.fit()
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
            + (inner._voice_frames if f.call is not None and inner else 0),
            channels=channels,
            follow_new=state.follow_new if state is not None else True,
            following=following,
            log=tuple(self._log),
            capture=capture,
            site_span=span,
            fit=fit,
            sample_rate=self.sample_rate,
            channel_kind=self._channel_kind(now),
            cc_hz=self._cc_hz(),
            dial_hz=self._dial_hz,
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
            if self._channel is not None:
                self._stop("sample_rate")
                self._channel.set_sample_rate(rate)

    def reset(self) -> None:
        super().reset()
        self._demod.reset()
        self._demod.set_coarse(self._good_coarse)
        self._framer.reset()
        self._reset_state()
