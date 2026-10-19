"""The talkgroups of a P25 system: which ones are followed, and what has been learned about each (M4 of plan_p25.md).

One table per system (WACN and system ID), shared by the decoder (the pipeline thread), which reads the follow choices and
records what it hears, and the P25 panel and console commands (the main thread), which change them; a lock guards it, and
what it hands out is immutable. A talkgroup's encryption is learned from its calls, not its grants: a grant says only
whether a call is encrypted, while the algorithm and key ID travel in the call (the header and LDU2). It is taken as fixed
per talkgroup, so it is learned once, from a call of a followed talkgroup, until a rescan forgets it.
"""

import threading
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class P25Talkgroup:
    group: int
    follow: bool
    algid: int | None = None  # learned from a call; None until then (or after a rescan)
    kid: int | None = None
    first_heard: float = 0.0
    last_heard: float = 0.0
    calls: int = 0  # grants heard for it after a pause (a new call), followed or not

    @property
    def learned(self) -> bool:
        return self.algid is not None


class TalkgroupTable:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: dict[int, P25Talkgroup] = {}
        self._follow_new = True

    @property
    def follow_new(self) -> bool:
        return self._follow_new

    def set_follow_new(self, on: bool) -> None:
        with self._lock:
            self._follow_new = on

    def get(self, group: int) -> P25Talkgroup | None:
        return self._groups.get(group)

    def follows(self, group: int) -> bool:
        tg = self._groups.get(group)
        return self._follow_new if tg is None else tg.follow

    def heard(self, group: int, now: float, new_call: bool) -> None:
        """A grant or update for `group` at capture time `now`; `new_call` when it starts a call."""
        with self._lock:
            tg = self._groups.get(group)
            if tg is None:
                tg = P25Talkgroup(group, self._follow_new, first_heard=now)
            self._groups[group] = replace(tg, last_heard=now, calls=tg.calls + new_call)

    def learn(self, group: int, algid: int, kid: int) -> None:
        with self._lock:
            tg = self._groups.get(group) or P25Talkgroup(group, self._follow_new)
            self._groups[group] = replace(tg, algid=algid, kid=kid)

    def set_follow(self, group: int, on: bool) -> None:
        with self._lock:
            tg = self._groups.get(group) or P25Talkgroup(group, on)
            self._groups[group] = replace(tg, follow=on)

    def set_all(self, on: bool) -> None:
        """Follow all talkgroups heard, or none; newly heard ones the same way."""
        with self._lock:
            self._groups = {g: replace(tg, follow=on) for g, tg in self._groups.items()}
            self._follow_new = on

    def rescan(self, group: int) -> None:
        """Forget what was learned of its encryption: its next encrypted call is looked at again."""
        with self._lock:
            tg = self._groups.get(group)
            if tg is not None:
                self._groups[group] = replace(tg, algid=None, kid=None)

    def snapshot(self) -> tuple[P25Talkgroup, ...]:
        with self._lock:
            return tuple(self._groups[g] for g in sorted(self._groups))


_tables: dict[tuple[int, int], TalkgroupTable] = {}
_tables_lock = threading.Lock()


def table_for(wacn: int, sysid: int) -> TalkgroupTable:
    with _tables_lock:
        table = _tables.get((wacn, sysid))
        if table is None:
            table = _tables[(wacn, sysid)] = TalkgroupTable()
        return table


def clear() -> None:
    """Forget every system's table (tests)."""
    with _tables_lock:
        _tables.clear()
