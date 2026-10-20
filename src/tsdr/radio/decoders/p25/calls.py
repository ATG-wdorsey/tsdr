"""The calls a P25 control channel grants (M4 of plan_p25.md): one record per call, created and continued by grants and
updates, closed into the call log.

A call's identity is a serial (`cid`), never reused while the decoder lives. A grant or update continues its talkgroup's
latest call only while that call is open, on the same channel, active, and (for a grant) within _REGRANT_S of its last
grant; anything else is a new call, which inherits nothing: its service options and talker stay unknown until a grant
gives them (rule C1, C2). A call is active while its grants or updates keep coming (_ACTIVE_S); a negative elapsed time
(capture time stepped back) counts as expired. An inactive call not being followed closes, and a closed call is never
continued (C3, C4).

M2's message texts ("TG 1001 on 851.0125 MHz ENC?") keep their own 5 s window by talkgroup (`ReportWindow`), apart from
the calls' identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tsdr.radio.decoders.p25 import tsbk, voice

# A call is active while its grants or updates keep coming: sites repeat them for every call up (the reference capture:
# every 0.4 s, 99 % within 1.1 s, at most 2.3 s with 14 % of blocks lost). Its grants after a longer silence start a
# new call; so does a grant more than _REGRANT_S after the last (a site repeats a grant within half a second, and the
# next transmission's comes 1.5 s or more later: the reference capture).
ACTIVE_S = 2.5
REGRANT_S = 1.0
# A call is taken only while fresh, its latest grant or update within this long (a call whose updates have stopped has
# likely ended).
FRESH_S = 1.2
# A grant for a talkgroup not reported for this long is reported as a new call.
REPORT_S = 5.0
# The latest calls kept for the panel's list, one per talkgroup.
RECENT_GROUPS = 64

NEVER, FOLLOWED, PAUSED, DONE = "never", "followed", "paused", "done"


@dataclass(frozen=True)
class P25Call:
    """A call as the control channel shows it (an immutable view of a record)."""

    group: int
    channel: int  # 16-bit channel number
    source: int | None
    frequency_hz: int | None  # None while the channel's identifier is unknown
    slot: int | None  # TDMA slot; None on an FDMA channel or while the identifier is unknown
    svcopts: int | None  # None until a grant gives them (updates don't): flags unknown, not clear
    last_seen: float
    first_seen: float = 0.0
    granted: float | None = None  # its latest grant (not an update): a transmission starting
    cid: int = 0
    system: tuple[int, int] | None = None

    @property
    def encrypted(self) -> bool:
        return bool(self.svcopts and self.svcopts & tsbk.SVC_ENCRYPTED)

    @property
    def emergency(self) -> bool:
        return bool(self.svcopts and self.svcopts & tsbk.SVC_EMERGENCY)


@dataclass
class CallRecord:
    """One call (pipeline thread only): what the control channel said of it, its follow state, and the facts its
    outcome is decided from (DESIGN-M4b 4.2, 4.4)."""

    cid: int
    system: tuple[int, int] | None
    group: int
    channel: int
    first_seen: float
    last_seen: float
    frequency_hz: int | None = None
    slot: int | None = None
    svcopts: int | None = None
    source: int | None = None
    granted: float | None = None
    talkers: set[int] = field(default_factory=set)
    state: str = NEVER
    reason: str | None = None  # PAUSED, DONE: why
    verdict: str | None = (
        None  # the latest reason it wasn't taken (busy, not_followed, encrypted, outside, ...)
    )
    verdict_at_take: str | None = None
    taken: float | None = None  # its latest take
    look: int | None = None  # the rescan pending for its talkgroup when it was taken
    ldus: int = 0  # voice frames of its own heard while followed
    audible: bool = False  # some of its audio played
    learned: tuple[int, int] | None = (
        None  # an algorithm and key learned from it (clear: on a flagged call)
    )
    had_voice: bool = False
    closed: bool = False  # in the call log: never continued (C3)

    @property
    def outcome(self) -> str:
        """The one outcome rule (DESIGN-M4b 4.4): learned, played, muted; then why it stopped or wasn't taken."""
        if self.learned is not None and self.learned[0] != voice.ALGID_CLEAR:
            return "learned"
        if self.audible:
            return "played"
        if self.ldus:
            return "muted"
        if self.state == PAUSED:
            return "not_followed" if self.reason == "unfollowed" else "busy"
        if self.taken is not None:
            return "busy" if self.verdict_at_take == "busy" else "no_voice"
        return self.verdict or "unknown"

    @property
    def flagged(self) -> bool:
        return bool(self.svcopts is not None and self.svcopts & tsbk.SVC_ENCRYPTED)

    def active(self, now: float) -> bool:
        return 0.0 <= now - self.last_seen <= ACTIVE_S

    def fresh(self, now: float) -> bool:
        return 0.0 <= now - self.last_seen <= FRESH_S

    def view(self) -> P25Call:
        return P25Call(
            group=self.group,
            channel=self.channel,
            source=self.source,
            frequency_hz=self.frequency_hz,
            slot=self.slot,
            svcopts=self.svcopts,
            last_seen=self.last_seen,
            first_seen=self.first_seen,
            granted=self.granted,
            cid=self.cid,
            system=self.system,
        )


class CallTracker:
    """The call records of the system on the channel (pipeline thread only)."""

    def __init__(self) -> None:
        self._serial = 0
        self.open: dict[int, CallRecord] = {}  # by cid
        self._latest: dict[int, CallRecord] = {}  # each talkgroup's latest call, open
        # each talkgroup's latest call, closed or not (M2's latest calls)
        self.recent: dict[int, CallRecord] = {}

    def message(
        self,
        system: tuple[int, int] | None,
        group: int,
        channel: int,
        svcopts: int | None,
        source: int | None,
        now: float,
        resolved: tuple[int, int | None] | None,
    ) -> tuple[CallRecord, bool]:
        """A grant (`svcopts` given) or an update (None) for `group` on `channel`; the record it belongs to (rule C1,
        Table A), and whether it is new."""
        last = self._latest.get(group)
        grant = svcopts is not None
        same = (
            last is not None
            and not last.closed
            and last.channel == channel
            and 0.0 <= now - last.last_seen <= ACTIVE_S
            and not (
                grant
                and now - (last.granted if last.granted is not None else last.first_seen)
                > REGRANT_S
            )
        )
        if same:
            assert last is not None
            call, new = last, False
            call.last_seen = now
            if call.system is None:
                call.system = system
        else:
            self._serial += 1
            call, new = CallRecord(self._serial, system, group, channel, now, now), True
            self.open[call.cid] = call
            self._latest[group] = call
        if grant:
            call.svcopts, call.granted = svcopts, now
            if source is not None:
                call.source = source
                call.talkers.add(source)
        call.frequency_hz, call.slot = resolved if resolved is not None else (None, None)
        self.recent[group] = call
        if len(self.recent) > RECENT_GROUPS:
            oldest = min(self.recent.values(), key=lambda c: c.last_seen)
            del self.recent[oldest.group]
        return call, new

    def name_system(self, system: tuple[int, int]) -> None:
        """The system's broadcasts named it: calls heard before were its calls."""
        for call in self.open.values():
            if call.system is None:
                call.system = system

    def latest(self, group: int) -> CallRecord | None:
        return self._latest.get(group)

    def moved(self, call: CallRecord) -> bool:
        """Its talkgroup has a newer call (another channel, a new transmission)."""
        return self._latest.get(call.group) is not call

    def active(self, now: float) -> list[CallRecord]:
        return [c for c in self.open.values() if c.active(now)]

    def close_inactive(self, now: float, followed: CallRecord | None) -> list[CallRecord]:
        """C3: every call inactive and not followed closes (the call log's)."""
        out = [c for c in self.open.values() if c is not followed and not c.active(now)]
        for c in out:
            self._close(c)
        return out

    def close_all(self) -> list[CallRecord]:
        """A reset, a retune, a system change: every call closes."""
        out = list(self.open.values())
        for c in out:
            self._close(c)
        return out

    def _close(self, call: CallRecord) -> None:
        call.closed = True
        del self.open[call.cid]
        if self._latest.get(call.group) is call:
            del self._latest[call.group]


@dataclass(frozen=True)
class _Reported:
    channel: int
    frequency_hz: int | None
    svcopts: int | None
    source: int | None
    last_seen: float


class ReportWindow:
    """M2's message texts: a talkgroup's grants and updates reported when new (none within REPORT_S), when its channel's
    frequency becomes known, when it moves to another channel, when its flags change, or its talker. Never read for a
    call's identity or for following."""

    def __init__(self) -> None:
        self._seen: dict[int, _Reported] = {}

    def reason(self, call: CallRecord, source: int | None, now: float) -> str | None:
        old = self._seen.get(call.group)
        self._seen[call.group] = _Reported(
            call.channel, call.frequency_hz, call.svcopts, call.source, now
        )
        if old is None or not 0.0 <= now - old.last_seen <= REPORT_S:
            return "new"
        if old.frequency_hz is None and call.frequency_hz is not None:
            return "channel_known"  # the identifier came after the grant
        if call.channel != old.channel:
            return "moved"  # updates repeat the current channel (Q147: before a flags change)
        flags = (
            call.svcopts is None,
            bool(call.svcopts and call.svcopts & tsbk.SVC_ENCRYPTED),
            bool(call.svcopts and call.svcopts & tsbk.SVC_EMERGENCY),
        )
        old_flags = (
            old.svcopts is None,
            bool(old.svcopts and old.svcopts & tsbk.SVC_ENCRYPTED),
            bool(old.svcopts and old.svcopts & tsbk.SVC_EMERGENCY),
        )
        if flags != old_flags:
            return "flags"  # a grant shows flags an update didn't carry
        if source is not None and old.source is not None and source != old.source:
            return "talker"
        return None

    def clear(self) -> None:
        self._seen.clear()
