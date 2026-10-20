"""Following trunked calls on the second channel (M4 of plan_p25.md; DESIGN-M4b section 4): which call has the channel,
its phase, and when it ends. Pure logic: its inputs are the call records, the choices and the second channel's
`VoiceReport`; its outputs are takes, stops and what to do with a chunk's audio.

- **FREE:** no call followed.
- **WAIT:** a call taken, no voice yet (up to VOICE_WAIT_S).
- **HOLD:** a transmission under way: from a voice frame to its terminator or the inner decoder's voice timeout.
- **HANG:** after a transmission, up to HANG_S, the call still followed; another call wanting the channel takes it.

The phase is explicit state, and the timers count from an anchor: the take, or the call's last voice frame since the
take, whichever is later, so no timer ends a call with time that passed while it wasn't on the channel. A call taken
again after a pause that had voice before it waits out a hang time from the take rather than ending for want of voice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tsdr.radio.decoders.p25 import voice
from tsdr.radio.decoders.p25.calls import DONE, FOLLOWED, NEVER, PAUSED, CallRecord
from tsdr.radio.decoders.p25.voice_channel import VoiceReport

# A followed call whose voice hasn't come this long after it was taken is left.
VOICE_WAIT_S = 1.0
# A transmission without a voice frame for this long has ended (the inner decoder's timeout).
VOICE_TIMEOUT_S = 1.0
# After a transmission the channel stays on the call (its talkgroup's next transmission may come on it without a new
# grant) unless another call wants it, and that talkgroup is preferred for this long.
HANG_S = 2.0

FREE, WAIT, HOLD, HANG = "FREE", "WAIT", "HOLD", "HANG"
# Stops that only pause a call: it may be taken again while active.
PAUSES = ("unfollowed", "preempted")


@dataclass(frozen=True)
class VoiceDecision:
    """The follower's reading of one chunk of the followed call's voice (DESIGN-M4b 4.3 step 2)."""

    keep_audio: bool
    stop: str | None = None
    learn: tuple[int, int] | None = (
        None  # an algorithm and key of its own transmission (clear only on a flagged call)
    )
    clear_voice: bool = False  # its own algorithm showed clear on an unflagged call


class Follower:
    def __init__(self) -> None:
        self.call: CallRecord | None = None
        self.phase = FREE
        self._take = 0.0
        self._anchor = 0.0  # the take, or its last voice since
        self._resumed_with_voice = False
        # (transmission, algorithm, key) judged since the take
        self._judged: set[tuple[float, int, int]] = set()
        # the system and talkgroup of the latest transmission followed, and when it ended (cleared with the system)
        self.hang: tuple[tuple[int, int] | None, int, float] | None = None

    def take(self, call: CallRecord, now: float) -> None:
        assert self.call is None
        call.verdict_at_take = call.verdict
        call.state, call.reason, call.taken = FOLLOWED, None, now
        self.call, self.phase = call, WAIT
        self._take = self._anchor = now
        self._resumed_with_voice = call.had_voice
        self._judged = set()

    def stop(self, reason: str) -> CallRecord:
        call = self.call
        assert call is not None
        call.state = PAUSED if reason in PAUSES else DONE
        call.reason = reason
        self.call, self.phase = None, FREE
        return call

    def voice(
        self, report: VoiceReport, now: float, chosen: bool, moved: bool, updated: bool
    ) -> VoiceDecision:
        """One chunk of the followed call's voice: Table B rows 4-10 and 16, checked in 4.3's order."""
        call = self.call
        assert call is not None
        if not chosen:
            return VoiceDecision(False, "unfollowed")
        if any(v.target is not None or v.group not in (None, call.group) for v in report.voices):
            return VoiceDecision(False, "other_call")
        learn, clear_voice, learned = None, False, False
        for v in report.voices:
            # Its own transmission once named (an HDU, or link control): one joined mid-way may be another talkgroup's
            # until its link control says so. Each algorithm a transmission shows is judged once, as it shows (its
            # grant's flag then).
            if v.group != call.group or v.algid is None or v.kid is None:
                continue
            if (v.started, v.algid, v.kid) in self._judged:
                continue
            self._judged.add((v.started, v.algid, v.kid))
            if v.algid != voice.ALGID_CLEAR:
                learn, learned = (v.algid, v.kid), True
                break
            if call.flagged:
                learn = (v.algid, v.kid)
            else:
                clear_voice = True
        if report.frames:
            call.ldus += report.frames
            call.had_voice = True
            self._anchor = now
        if any(len(b.samples) for b in report.audio):
            call.audible = True
        if learned:
            return VoiceDecision(True, "learned", learn)
        if report.on_air is not None:
            self.phase = HOLD
        elif self.phase == HOLD or (self.phase == WAIT and report.frames):
            self.phase = HANG  # a terminator, or the voice's own timeout
            self.hang = (call.system, call.group, now)
        stop = None
        if self.phase == WAIT and (moved or not 0.0 <= now - self._anchor <= VOICE_WAIT_S):
            if self._resumed_with_voice and now >= self._take:
                self.phase = HANG  # D2: its hang time counted from the take
                self.hang = (call.system, call.group, self._take)
            else:
                stop = "no_voice"
        if self.phase == HANG:
            quiet = now - self._anchor
            if moved or not 0.0 <= quiet <= HANG_S or (not updated and quiet > VOICE_TIMEOUT_S):
                stop = "ended"
        return VoiceDecision(True, stop, learn, clear_voice)

    def choose(
        self,
        now: float,
        calls: Iterable[CallRecord],
        verdict: Callable[[CallRecord], str | None],
        moved: Callable[[CallRecord], bool],
    ) -> CallRecord | None:
        """Verdicts for every active call not followed (4.3 step 3), and the best fresh eligible one if the channel
        would go to it (step 4): free, or in its hang time. The hang talkgroup first within HANG_S of its last
        transmission, then the newest."""
        busy = self.phase in (WAIT, HOLD)
        best: CallRecord | None = None
        for c in calls:
            if c is self.call or c.state not in (NEVER, PAUSED):
                continue
            v = verdict(c)
            if v is None and busy:
                v = "busy"
            if v is not None:
                c.verdict = v
                continue
            if not c.fresh(now) or moved(c):
                continue
            if best is None or self._better(c, best, now):
                best = c
        return None if busy else best

    def _better(self, a: CallRecord, b: CallRecord, now: float) -> bool:
        hang = self.hang
        if hang is not None and 0.0 <= now - hang[2] <= HANG_S:
            # the hang talkgroup: the same talkgroup of the same system (R52-2: never another system's)
            a_hang, b_hang = (a.system, a.group) == hang[:2], (b.system, b.group) == hang[:2]
            if a_hang != b_hang:
                return a_hang
        # the newest; between calls first heard in one batch, the first heard (as M4a)
        return (a.first_seen, -a.cid) > (b.first_seen, -b.cid)
