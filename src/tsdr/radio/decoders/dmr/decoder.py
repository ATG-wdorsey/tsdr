"""DMR protocol decoder.

IQ -> decimate -> FM discriminator -> M&M timing -> 4-level slicer -> dibit stream -> sync search -> burst parse

Reference: ETSI TS 102 361-1
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch, DemodStatus
from tsdr.radio.decoders.dmr.constants import (
    BURST_DIBITS,
    CACH_START,
    DEVIATION,
    FIRST_HALF_DIBITS,
    SECOND_HALF_DIBITS,
    SLOT_TYPE_DIBITS,
    ST1_START,
    ST2_START,
    SUPERFRAME_LEN,
    SYMBOL_RATE,
    SYNC_DIBITS,
    SYNC_MAX_ERRORS,
    SYNC_PATTERNS,
    SYNC_START,
    VOICE_SYNC_TYPES,
    DataType,
    DecoderState,
    SyncType,
)
from tsdr.radio.decoders.dmr.fec import (
    decode_cach,
    decode_emb,
    decode_slot_type,
    extract_voice_frames,
)
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.dsp import DCBlocker, FMChannelizer, MuellerMuller
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.ambe import AmbePlus2Decoder

logger = logging.getLogger(__name__)

_TARGET_RATE = SYMBOL_RATE * 10
_DMR_PCM_RATE = 8000
# A signal off frequency adds its offset to every symbol, and the timing loop and the
# sync search decide by sign: 500 Hz off, 14 % of the outer symbols came out with the
# wrong sign and no sync was found. The 4FSK data averages to zero over a few hundred
# symbols, so a slow high-pass (80 ms) takes the offset off first.
_DC_CUTOFF_HZ = 2.0
_VOICE_TIMEOUT_SEC = 0.5
# Require this many consecutive voice bursts before decoding audio.
# Filters out false VOICE sync matches on noise.
_VOICE_CONFIDENCE_BURSTS = 3
_SLOT_ACTIVE_WINDOW_SEC = 2.0
_SLOT_SWITCH_HOLD_SEC = 0.5
# What is heard (DemodSpec.dmr_slot): None = auto, "1" / "2" one timeslot (TS1 is CACH
# TC 0), "both" TS1 left and TS2 right.
AUDIO_SLOT_CHOICES: tuple[str | None, ...] = (None, "1", "2", "both")
# Both slots: a slot that has carried no voice for this long is not waited for (its
# channel plays silence); one that is talking is waited for up to _BOTH_MAX_LAG_SEC.
_BOTH_TALKING_SEC = 0.25
_BOTH_MAX_LAG_SEC = 0.25
# A timeslot's voice superframe is 6 bursts: A carries the voice sync, B-F
# embedded signalling instead. In lock, a slot in a voice call is decoded by
# timing for up to 11 bursts since its own last sync: B-F, one missed A, B-F.
_MAX_CONTINUATION_BURSTS = 11
# Bursts in a row without a sync, both slots, before the lock is dropped. With
# voice on both slots up to 10 carry none (11 with one missed A). Bounds what is
# decoded after the signal is lost mid-call.
_MAX_SYNCLESS_RUN = 12
# A burst taken by timing must hold what its sync/EMB field should: at B-F a valid EMB
# (within 2 bit errors of a QR(16,7,6) word, with the repeater's color code: noise
# passes about 1 time in 60), at A (its sync missed) the voice sync within 6 of 24
# symbols (noise: about 1 in 90). One that fails is not decoded; this many in a row
# drop the lock (the signal is gone).
_MAX_BAD_TIMED = 2
_MISSED_SYNC_MAX_ERRORS = 6
# The reference harness's speech gain (DSD, mbelib's decode_ambe): starts at x25, at most x50.
_AGC_MAX_GAIN = 50.0
_AGC_START_GAIN = 25.0
# DSD's target: the 8 -> 48 kHz resampler overshoots full-scale 8 kHz peaks (clipped
# 100 of 342,720 samples on the 443.7 MHz recording, by up to 0.45 dB).
_AGC_PEAK = 30000 / 32768


@dataclass
class TimeslotState:
    """Per-timeslot voice call tracking."""

    in_voice_call: bool = False
    confirmed: bool = False  # True after VOICE_LC_HEADER or enough consecutive bursts
    voice_burst_count: int = 0
    last_voice_ts: float = 0.0
    voice_sync_type: SyncType = field(default=SyncType.BS_VOICE)


@dataclass(frozen=True)
class SlotSnapshot:
    """Immutable per-timeslot state for the DMR widget."""

    timeslot: int
    in_voice_call: bool
    voice_burst_count: int


@dataclass(frozen=True)
class QualitySnapshot:
    """Immutable quality metrics for the DMR widget."""

    locked: bool
    lock_pct: float
    cach_pct: float
    slot_type_pct: float
    burst_count: int


@dataclass(frozen=True)
class DMRSnapshot:
    """Immutable decoder state snapshot for the DMR widget."""

    color_code: int | None
    slots: tuple[SlotSnapshot, SlotSnapshot]
    quality: QualitySnapshot
    audio: str = "auto"  # what is heard: "auto", "auto: TS1", "TS2", "both"


def _slice_dibit(symbol: float) -> int:
    """Map a normalized FM symbol to a DMR dibit."""
    return 1 if symbol > 0 else 3


def _slice_4level(symbols: np.ndarray) -> np.ndarray:
    """Map normalized FM symbols to 4-level dibits for data extraction.

    4FSK levels (ETSI TS 102 361-1 S4.2.2), thresholds at the midpoints (0, +-2/3):
        +3 (+1944 Hz, +1)   -> dibit 1
        +1  (+648 Hz, +1/3) -> dibit 0
        -1  (-648 Hz, -1/3) -> dibit 2
        -3 (-1944 Hz, -1)   -> dibit 3
    """
    dibits = np.full(len(symbols), 3, dtype=np.uint8)
    dibits[symbols > -2 / 3] = 2
    dibits[symbols > 0] = 0
    dibits[symbols > 2 / 3] = 1
    return dibits


# The +3 symbols of each sync pattern (the rest are -3).
_SYNC_HIGH = {t: np.frombuffer(p, dtype=np.uint8) == 1 for t, p in SYNC_PATTERNS.items()}


def _match_sync(dibits: np.ndarray) -> tuple[SyncType, int] | None:
    """Match 24 dibits against all DMR sync patterns."""
    best_type: SyncType | None = None
    best_errors = SYNC_DIBITS

    for sync_type, pattern in SYNC_PATTERNS.items():
        errors = 0
        for i in range(SYNC_DIBITS):
            if dibits[i] != pattern[i]:
                errors += 1
                if errors > SYNC_MAX_ERRORS:
                    break
        if errors <= SYNC_MAX_ERRORS and errors < best_errors:
            best_errors = errors
            best_type = sync_type

    if best_type is not None:
        return best_type, best_errors
    return None


class DMRDecoder(Demodulator):
    """DMR protocol decoder with voice output.

    Processes raw IQ samples through:
    decimate -> FM discriminator -> M&M timing -> 4-level slicer -> sync detection -> burst parse

    Voice bursts are decoded through AMBE+2 vocoder to 8 kHz PCM audio.
    """

    HAS_AUDIO = True
    LABEL = "DMR"
    MODULATION = "4FSK"
    MESSAGE_TYPE = "dmr"
    HAS_TEXT = True
    FIXED_CHANNEL_BANDWIDTH = 12_500.0

    @property
    def audio_prebuffer_seconds(self) -> float:
        return 0.25

    def __init__(self, sample_rate: float = 250_000, audio_slot: str | None = None):
        super().__init__()
        self.sample_rate = sample_rate
        self._audio_slot: str | None = None
        self.set_audio_slot(audio_slot)

        # Decimate to ~48 kHz (10 samples/symbol) and FM-demodulate; output
        # normalized: +/-1.0 = +/-DEVIATION Hz.
        self._channelizer = FMChannelizer(
            sample_rate, DEVIATION, target_rate=_TARGET_RATE, channel_cutoff=DEVIATION * 3
        )
        self._decimated_rate = self._channelizer.audio_rate
        self._sps = self._decimated_rate / SYMBOL_RATE
        self._dc = DCBlocker(self._decimated_rate, cutoff_hz=_DC_CUTOFF_HZ)

        # Mueller-Muller symbol timing recovery
        self._mm = MuellerMuller(self._sps, gain=0.01)

        # Constellation delay line: one symbol period of FM samples
        self._constellation_delay = round(self._sps)
        self._fm_tail = np.array([], dtype=np.float64)

        # Ring buffers: sign dibits for the sync search, and the symbols themselves (sliced
        # when a burst is complete: the ring then holds exactly that burst).
        self._sync_buf = np.zeros(BURST_DIBITS, dtype=np.uint8)
        self._sym_buf = np.zeros(BURST_DIBITS, dtype=np.float32)
        self._sync_pos = 0
        self._dibit_count = 0

        # Burst collection state machine
        self._state = DecoderState.SEARCHING
        self._burst_buf = np.zeros(BURST_DIBITS, dtype=np.uint8)
        self._burst_collect_pos = 0
        self._burst_sync_type = SyncType.BS_DATA
        self._lock_remaining = 0
        # Continuation bursts per timeslot since that slot's last real sync match.
        self._continuation = {0: 0, 1: 0}
        self._syncless_run = 0
        self._bad_timed_run = 0
        # The burst being collected: came with a sync; was framed by the lock.
        self._burst_synced = True
        self._burst_in_lock = False

        # Voice call state per timeslot (DMR has 2: TS0, TS1)
        self._ts_state: dict[int, TimeslotState] = {0: TimeslotState(), 1: TimeslotState()}
        self._last_slot = -1

        # AMBE+2 vocoders -- one per timeslot, state must not be shared
        self._vocoders: dict[int, AmbePlus2Decoder] = {
            0: AmbePlus2Decoder(),
            1: AmbePlus2Decoder(),
        }
        self._pcm_per_ts: dict[int, list[np.ndarray]] = {0: [], 1: []}
        # Speech level per timeslot (each carries its own talkers), as mbelib's reference
        # harness sets it: at most x50 on the synthesis scale (the old fixed gain was x56).
        self._agc: dict[int, VoiceAGC] = {
            0: VoiceAGC(max_gain=_AGC_MAX_GAIN, start_gain=_AGC_START_GAIN, peak=_AGC_PEAK),
            1: VoiceAGC(max_gain=_AGC_MAX_GAIN, start_gain=_AGC_START_GAIN, peak=_AGC_PEAK),
        }
        self._selected_ts: int | None = None
        self._candidate_ts: int | None = None
        self._candidate_since: float = 0.0
        self._audio_batches: list[AudioBatch] = []

        # Statistics
        self._syncs_found = 0
        self._bursts_decoded = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        self._lock_hits = 0
        self._lock_misses = 0
        self._color_code_counts: dict[int, int] = {}
        self._data_type_counts: dict[int, int] = {}
        self._sync_type_counts: dict[SyncType, int] = {}
        self._dominant_cc: int | None = None
        self._dominant_cc_count: int = 0

        # Constellation
        self._constellation_points: np.ndarray | None = None

        # Output
        self._pending_messages: list[DecodedMessage] = []
        self._dirty = False
        self._last_snapshot_ts: float = 0.0

    def demodulate(self, iq_samples: np.ndarray, capture_utc_s: float) -> None:
        fm = self._dc.process(self._channelizer.process(iq_samples))

        # Transition constellation: plot each FM sample
        # against the sample one symbol period earlier. Shows a 4x4 grid
        # of clusters for the 16 possible dibit transitions.
        delay = self._constellation_delay
        if len(self._fm_tail) > 0:
            fm_full = np.concatenate([self._fm_tail, fm])
        else:
            fm_full = fm
        if len(fm_full) > delay:
            current = fm_full[delay:]
            delayed = fm_full[: len(current)]
            self._constellation_points = (current + 1j * delayed).astype(np.complex64)
        self._fm_tail = fm[-delay:].copy() if len(fm) >= delay else fm.copy()

        symbols = self._mm.process(fm)
        if len(symbols) < 2:
            return

        for sym in symbols:
            sym_f = float(sym)
            pos = self._sync_pos % BURST_DIBITS
            self._sync_buf[pos] = _slice_dibit(sym_f)
            self._sym_buf[pos] = sym_f
            self._sync_pos = (self._sync_pos + 1) % BURST_DIBITS
            self._dibit_count += 1

            match self._state:
                case DecoderState.SEARCHING:
                    self._do_sync_search(capture_utc_s)
                case DecoderState.COLLECTING:
                    self._do_collect(capture_utc_s)
                case DecoderState.LOCKED:
                    self._do_locked(capture_utc_s)

        self._flush_selected_audio(capture_utc_s)
        self._maybe_emit_snapshot(capture_utc_s)

    # Burst framing state machine

    def _do_sync_search(self, timestamp: float) -> None:
        if self._dibit_count < SYNC_DIBITS:
            return

        sync_dibits = self._get_ring_history(self._sync_buf, SYNC_DIBITS)
        result = _match_sync(sync_dibits)
        if result is None:
            return

        sync_type, _errors = result
        self._syncs_found += 1
        self._burst_sync_type = sync_type

        if self._dibit_count < FIRST_HALF_DIBITS + SYNC_DIBITS:
            return

        self._begin_collecting(synced=True, in_lock=False)

    def _do_collect(self, timestamp: float) -> None:
        self._burst_collect_pos += 1

        if self._burst_collect_pos >= SECOND_HALF_DIBITS:
            # The ring now holds exactly this burst, CACH first.
            self._burst_buf[:] = _slice_4level(self._get_ring_history(self._sym_buf, BURST_DIBITS))

            valid = self._process_burst(self._burst_buf, self._burst_sync_type, timestamp)
            # A sync where the lock expected one holds the framing, whatever the
            # burst's own content (e.g. an idle burst's slot type lost to noise).
            if valid or (self._burst_synced and self._burst_in_lock):
                self._state = DecoderState.LOCKED
                self._lock_remaining = FIRST_HALF_DIBITS + SYNC_DIBITS
            else:
                self._drop_lock()

    def _do_locked(self, timestamp: float) -> None:
        """Frame-locked tracking with voice continuation support.

        Voice continuation bursts (B-F in a superframe) have EMB instead of a
        sync: a slot in a voice call is decoded by frame timing, up to
        _MAX_CONTINUATION_BURSTS since its own last sync. The other slot's
        sync-less bursts (its call not yet found, after a late entry or a lost
        lock) are passed by on the same timing, so its next A sync is met in
        lock. A run of more than _MAX_SYNCLESS_RUN bursts without a sync, or one
        with no call to follow, drops the lock.
        """
        self._lock_remaining -= 1
        if self._lock_remaining > 0:
            return

        sync_dibits = self._get_ring_history(self._sync_buf, SYNC_DIBITS)
        result = _match_sync(sync_dibits)

        if result is not None:
            sync_type, _errors = result
            self._lock_hits += 1
            self._syncs_found += 1
            self._syncless_run = 0
            self._burst_sync_type = sync_type
            self._begin_collecting(synced=True, in_lock=True)
            return

        self._syncless_run += 1
        slot = 1 - self._last_slot
        if self._syncless_run > _MAX_SYNCLESS_RUN or not self._any_voice_call(timestamp):
            self._lock_misses += 1
            self._drop_lock()
        elif self._voice_continuation_expected(slot, timestamp):
            self._lock_hits += 1
            self._continuation[slot] += 1
            self._burst_sync_type = self._ts_state[slot].voice_sync_type
            self._begin_collecting(synced=False, in_lock=True)
        else:
            self._last_slot = slot
            self._lock_remaining = BURST_DIBITS

    def _drop_lock(self) -> None:
        """Return to SEARCHING and reset all transient state so the UI
        doesn't show stale info (quality percentages, voice call status)."""
        self._state = DecoderState.SEARCHING
        self._continuation = {0: 0, 1: 0}
        self._syncless_run = 0
        self._bad_timed_run = 0
        self._lock_hits = 0
        self._lock_misses = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        for ts in self._ts_state.values():
            ts.in_voice_call = False
            ts.confirmed = False
            ts.voice_burst_count = 0
        self._dirty = True

    def _in_voice_call(self, slot: int, timestamp: float) -> bool:
        ts = self._ts_state[slot]
        return ts.in_voice_call and (timestamp - ts.last_voice_ts) < _VOICE_TIMEOUT_SEC

    def _any_voice_call(self, timestamp: float) -> bool:
        return self._in_voice_call(0, timestamp) or self._in_voice_call(1, timestamp)

    def _voice_continuation_expected(self, slot: int, timestamp: float) -> bool:
        if self._continuation[slot] >= _MAX_CONTINUATION_BURSTS:
            return False
        return self._in_voice_call(slot, timestamp)

    def _begin_collecting(self, *, synced: bool, in_lock: bool) -> None:
        self._state = DecoderState.COLLECTING
        self._burst_collect_pos = 0
        self._burst_synced = synced
        self._burst_in_lock = in_lock

    def _get_ring_history(self, buf: np.ndarray, n: int) -> np.ndarray:
        start = (self._sync_pos - n) % BURST_DIBITS
        if start < self._sync_pos:
            return buf[start : self._sync_pos].copy()
        return np.concatenate([buf[start:], buf[: self._sync_pos]])

    # Burst processing

    def _process_burst(self, burst: np.ndarray, sync_type: SyncType, timestamp: float) -> bool:
        """Parse a complete 144-dibit burst. Returns True to stay locked."""
        self._bursts_decoded += 1
        self._sync_type_counts[sync_type] = self._sync_type_counts.get(sync_type, 0) + 1

        cach = decode_cach(burst[CACH_START : CACH_START + 12])
        if cach.clean:
            self._cach_ok += 1
        else:
            self._cach_fail += 1

        # The slot: in lock, the TDMA alternation, with a clean TACT as a check (a
        # synced burst's corrects it; a sync-less burst's contradicting it means the
        # framing is lost). Acquiring, a clean TACT: two bit errors "correct" to the
        # wrong slot, and noise passes a clean TACT only 1 time in 8.
        if self._burst_in_lock:
            timeslot = 1 - self._last_slot
            if cach.clean and cach.timeslot != timeslot:
                if not self._burst_synced:
                    return False
                timeslot = cach.timeslot
        elif cach.clean:
            timeslot = cach.timeslot
        else:
            return False
        self._last_slot = timeslot
        if self._burst_synced:
            self._continuation[timeslot] = 0
            self._bad_timed_run = 0
        elif self._timed_burst_ok(burst, timeslot):
            self._bad_timed_run = 0
        else:
            self._bad_timed_run += 1
            return self._bad_timed_run < _MAX_BAD_TIMED

        if sync_type in VOICE_SYNC_TYPES:
            return self._process_voice_burst(burst, sync_type, timeslot, timestamp)
        return self._process_data_burst(burst, sync_type, timeslot, cach.clean, timestamp)

    def _timed_burst_ok(self, burst: np.ndarray, slot: int) -> bool:
        """A burst taken by timing holds what its sync/EMB field should: the voice sync
        (loosely) at the superframe's A, else an EMB with the repeater's color code."""
        field = burst[SYNC_START : SYNC_START + SYNC_DIBITS]
        if self._continuation[slot] % SUPERFRAME_LEN == 0:
            high = _SYNC_HIGH[self._ts_state[slot].voice_sync_type]
            errors = int(np.count_nonzero((field <= 1) != high))  # dibits 0, 1: positive
            return errors <= _MISSED_SYNC_MAX_ERRORS
        emb = decode_emb(np.concatenate([field[:4], field[-4:]]))
        if emb is None:
            return False
        # Counted before the check: a noise EMB that decoded first must not lock out the
        # repeater's own color code.
        self._count_color_code(emb.color_code)
        return emb.color_code == self._dominant_cc

    def _count_color_code(self, cc: int) -> None:
        count = self._color_code_counts.get(cc, 0) + 1
        self._color_code_counts[cc] = count
        if count > self._dominant_cc_count:
            self._dominant_cc = cc
            self._dominant_cc_count = count

    def _process_voice_burst(
        self, burst: np.ndarray, sync_type: SyncType, timeslot: int, timestamp: float
    ) -> bool:
        """Handle a voice burst -- extract AMBE frames, decode to PCM.

        Only decodes audio after confidence is established (VOICE_LC_HEADER
        seen, or enough consecutive voice bursts at locked timing). This
        prevents false VOICE sync matches on noise from producing audio.
        """
        ts = self._ts_state[timeslot]
        if not ts.in_voice_call:
            ts.in_voice_call = True
            ts.voice_burst_count = 0
            ts.voice_sync_type = sync_type
            self._agc[timeslot].reset()  # a new talker: not the last call's level
            self._dirty = True
            logger.debug("dmr_voice_call_started slot=%d sync_type=%s", timeslot, sync_type)

        ts.voice_burst_count += 1
        ts.last_voice_ts = timestamp

        if not ts.confirmed and ts.voice_burst_count >= _VOICE_CONFIDENCE_BURSTS:
            ts.confirmed = True
            logger.debug(
                "dmr_voice_call_confirmed slot=%d bursts=%d", timeslot, ts.voice_burst_count
            )

        if ts.confirmed:
            for frame_bytes in extract_voice_frames(burst):
                speech = self._vocoders[timeslot].decode_frame_float(frame_bytes)
                self._pcm_per_ts[timeslot].append(self._agc[timeslot].process(speech / 32768.0))

        if ts.voice_burst_count <= 3 or ts.voice_burst_count % 100 == 0:
            self._pending_messages.append(
                DecodedMessage(
                    text=f"{sync_type} slot={timeslot} voice burst#{ts.voice_burst_count}",
                    timestamp=timestamp,
                )
            )

        return True

    def _process_data_burst(
        self,
        burst: np.ndarray,
        sync_type: SyncType,
        timeslot: int,
        cach_ok: bool,
        timestamp: float,
    ) -> bool:
        """Handle a data burst -- decode Slot Type, track voice call lifecycle."""
        st1 = burst[ST1_START : ST1_START + SLOT_TYPE_DIBITS // 2]
        st2 = burst[ST2_START : ST2_START + SLOT_TYPE_DIBITS // 2]
        slot_type = decode_slot_type(st1, st2)

        if slot_type is None:
            self._slot_type_fail += 1
            text = f"{sync_type} slot={timeslot} [SlotType err]"
            if not cach_ok:
                text += " [CACH err]"
            self._pending_messages.append(DecodedMessage(text=text, timestamp=timestamp))
            return False

        self._slot_type_ok += 1
        self._count_color_code(slot_type.color_code)
        self._data_type_counts[slot_type.data_type] = (
            self._data_type_counts.get(slot_type.data_type, 0) + 1
        )

        # Voice call lifecycle
        ts = self._ts_state[timeslot]
        if slot_type.data_type == DataType.VOICE_LC_HEADER:
            ts.in_voice_call = True
            ts.confirmed = True  # header from Slot Type = high confidence
            self._agc[timeslot].reset()
            ts.voice_burst_count = 0
            ts.last_voice_ts = timestamp
            self._dirty = True
        elif slot_type.data_type == DataType.TERMINATOR_WITH_LC and ts.in_voice_call:
            logger.debug("dmr_voice_call_ended slot=%d bursts=%d", timeslot, ts.voice_burst_count)
            ts.in_voice_call = False
            ts.confirmed = False
            self._dirty = True
            self._vocoders[timeslot] = AmbePlus2Decoder()
            self._agc[timeslot].reset()

        try:
            dt_name = DataType(slot_type.data_type).name
        except ValueError:
            dt_name = f"UNKNOWN({slot_type.data_type})"

        if self._bursts_decoded <= 5 or self._bursts_decoded % 100 == 0:
            logger.debug(
                "dmr_burst index=%d sync_type=%s slot=%d cc=%d type=%s",
                self._bursts_decoded,
                sync_type,
                timeslot,
                slot_type.color_code,
                dt_name,
            )

        if slot_type.data_type != DataType.IDLE:
            text = f"{sync_type} slot={timeslot} cc={slot_type.color_code} type={dt_name}"
            if not cach_ok:
                text += " [CACH err]"
            self._pending_messages.append(DecodedMessage(text=text, timestamp=timestamp))

        return True

    # Snapshot emission

    _SNAPSHOT_INTERVAL = 0.25  # ~4 Hz

    def _maybe_emit_snapshot(self, timestamp: float) -> None:
        due = (timestamp - self._last_snapshot_ts) >= self._SNAPSHOT_INTERVAL
        if not (self._dirty or due):
            return
        self._dirty = False
        self._last_snapshot_ts = timestamp
        snap = self._to_snapshot()
        self._pending_messages.append(DecodedMessage(text="", timestamp=timestamp, data=snap))

    def _to_snapshot(self) -> DMRSnapshot:
        cc = self._dominant_cc

        slots = tuple(
            SlotSnapshot(
                timeslot=ts,
                in_voice_call=state.in_voice_call,
                voice_burst_count=state.voice_burst_count,
            )
            for ts, state in sorted(self._ts_state.items())
        )

        total_lock = self._lock_hits + self._lock_misses
        total_cach = self._cach_ok + self._cach_fail
        total_st = self._slot_type_ok + self._slot_type_fail

        quality = QualitySnapshot(
            locked=self._state != DecoderState.SEARCHING,
            lock_pct=self._lock_hits / total_lock * 100 if total_lock > 0 else 0.0,
            cach_pct=self._cach_ok / total_cach * 100 if total_cach > 0 else 0.0,
            slot_type_pct=self._slot_type_ok / total_st * 100 if total_st > 0 else 0.0,
            burst_count=self._bursts_decoded,
        )

        return DMRSnapshot(
            color_code=cc,
            slots=slots,  # type: ignore[arg-type]
            quality=quality,
            audio=self._audio_label(),
        )

    def _audio_label(self) -> str:
        if self._audio_slot == "both":
            return "both"
        if self._audio_slot is not None:
            return f"TS{self._audio_slot}"
        return f"auto: TS{self._selected_ts + 1}" if self._selected_ts is not None else "auto"

    def set_audio_slot(self, choice: str | None) -> None:
        if choice not in AUDIO_SLOT_CHOICES:
            raise ValueError(f"unknown DMR timeslot choice: {choice!r}")
        self._audio_slot = choice
        self._dirty = True

    # Audio output

    def _select_active_ts(self, now: float) -> int | None:
        """Pick timeslot with freshest speech, with hysteresis to prevent flapping."""
        fresh = [
            (slot, state.last_voice_ts)
            for slot, state in self._ts_state.items()
            if state.last_voice_ts > 0.0 and now - state.last_voice_ts <= _SLOT_ACTIVE_WINDOW_SEC
        ]
        if not fresh:
            self._selected_ts = None
            self._candidate_ts = None
            return None

        newest_ts = max(fresh, key=lambda x: x[1])[0]
        fresh_set = {ts for ts, _ in fresh}

        if self._selected_ts is None or self._selected_ts not in fresh_set:
            self._selected_ts = newest_ts
            self._candidate_ts = None
            return newest_ts

        if newest_ts == self._selected_ts:
            self._candidate_ts = None
            return self._selected_ts

        if self._candidate_ts != newest_ts:
            self._candidate_ts = newest_ts
            self._candidate_since = now
            return self._selected_ts
        if now - self._candidate_since >= _SLOT_SWITCH_HOLD_SEC:
            self._selected_ts = newest_ts
            self._candidate_ts = None
        return self._selected_ts

    def _flush_selected_audio(self, timestamp: float) -> None:
        """Package the selected slot's buffered PCM into AudioBatch."""
        if self._audio_slot == "both":
            self._flush_both_slots(timestamp)
            return
        selected = self._select_active_ts(timestamp)
        if self._audio_slot is not None:
            selected = int(self._audio_slot) - 1

        # Discard non-selected slot's PCM immediately to prevent backlog
        # buildup when both slots have active voice calls
        for slot in self._pcm_per_ts:
            if slot != selected:
                self._pcm_per_ts[slot].clear()

        if selected is None:
            return
        chunks = self._pcm_per_ts[selected]
        if not chunks:
            return
        samples = np.concatenate(chunks)
        self._pcm_per_ts[selected] = []

        self._audio_batches.append(
            AudioBatch(
                samples=samples,
                sample_rate=_DMR_PCM_RATE,
                stereo=False,
                prebuffer_seconds=self.audio_prebuffer_seconds,
            )
        )

    def _flush_both_slots(self, timestamp: float) -> None:
        """TS1 left, TS2 right, in step: each slot yields 60 ms of speech per 60 ms, a
        burst apart. A talking slot is waited for (up to _BOTH_MAX_LAG_SEC), a silent
        one is not: its channel plays silence."""
        pending = [
            np.concatenate(self._pcm_per_ts[ts]) if self._pcm_per_ts[ts] else np.zeros(0)
            for ts in (0, 1)
        ]
        lens = [len(p) for p in pending]
        if max(lens) == 0:
            return
        talking = [
            timestamp - self._ts_state[ts].last_voice_ts < _BOTH_TALKING_SEC for ts in (0, 1)
        ]
        n = min(lens)
        if not all(talking) or max(lens) - n > _BOTH_MAX_LAG_SEC * _DMR_PCM_RATE:
            n = max(lens)
        if n == 0:
            return
        out = np.zeros((n, 2), dtype=np.float32)
        for ts in (0, 1):
            k = min(n, lens[ts])
            out[:k, ts] = pending[ts][:k]
            self._pcm_per_ts[ts] = [pending[ts][k:]] if lens[ts] > k else []
        self._audio_batches.append(
            AudioBatch(
                samples=out,
                sample_rate=_DMR_PCM_RATE,
                stereo=True,
                prebuffer_seconds=self.audio_prebuffer_seconds,
            )
        )

    # Demodulator interface

    def get_audio(self) -> list[AudioBatch]:
        batches = self._audio_batches
        self._audio_batches = []
        return batches

    def get_messages(self) -> list[DecodedMessage]:
        messages = self._pending_messages
        self._pending_messages = []
        return messages

    def get_constellation(self) -> tuple[np.ndarray, str] | None:
        points = self._constellation_points
        self._constellation_points = None
        if points is None:
            return None
        return points, "4FSK"

    def status(self) -> DemodStatus:
        """Thread-safe: callable from any thread. Reads scalar counters only."""
        quality = None
        quality_label = None
        total_lock = self._lock_hits + self._lock_misses
        if total_lock > 0:
            quality = self._lock_hits / total_lock
            quality_label = f"Lock {quality * 100:.0f}%"
        description = None
        cc = self._dominant_cc
        if cc is not None:
            description = f"CC{cc}"

        return DemodStatus(
            quality_label=quality_label,
            quality=quality,
            description=description,
        )

    def reset(self) -> None:
        self._channelizer.reset()
        self._dc.reset()
        self._mm.reset()
        self._fm_tail = np.array([], dtype=np.float64)
        self._sync_buf[:] = 0
        self._sym_buf[:] = 0
        self._sync_pos = 0
        self._dibit_count = 0
        self._state = DecoderState.SEARCHING
        self._burst_buf[:] = 0
        self._burst_collect_pos = 0
        self._lock_remaining = 0
        self._continuation = {0: 0, 1: 0}
        self._syncless_run = 0
        self._bad_timed_run = 0
        self._last_slot = -1
        for ts in self._ts_state.values():
            ts.in_voice_call = False
            ts.confirmed = False
            ts.voice_burst_count = 0
            ts.last_voice_ts = 0.0
        self._vocoders = {ts: AmbePlus2Decoder() for ts in self._vocoders}
        for agc in self._agc.values():
            agc.reset()
        for buf in self._pcm_per_ts.values():
            buf.clear()
        self._selected_ts = None
        self._candidate_ts = None
        self._audio_batches.clear()
        self._syncs_found = 0
        self._bursts_decoded = 0
        self._cach_ok = 0
        self._cach_fail = 0
        self._slot_type_ok = 0
        self._slot_type_fail = 0
        self._lock_hits = 0
        self._lock_misses = 0
        self._color_code_counts.clear()
        self._data_type_counts.clear()
        self._sync_type_counts.clear()
        self._dominant_cc = None
        self._dominant_cc_count = 0
        self._pending_messages.clear()
