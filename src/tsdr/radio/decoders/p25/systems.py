"""The P25 systems heard and their talkgroups: which ones are followed, and what has been learned of each (M4 of
plan_p25.md; DESIGN-M4b section 3).

One `SystemState` per system (WACN and system ID), shared by the decoder (the pipeline thread), which reads the follow
choices and records what it hears and learns, and the P25 panel and console commands (the main thread), which change the
choices. A lock guards it, and its rows are immutable.

- **Follow:** a talkgroup with a row follows its override if it has one, else the system's `follow_default`; one without
  a row (never heard, or deleted) follows `follow_new`. A row created when heard takes `follow_new` (an override only
  when that differs from the default). `all`/`none` set both defaults and clear every override.
- **Encryption** is learned from the calls, not the grants: a grant says only whether a call is encrypted, while the
  algorithm and key ID travel in the call itself. Every key learned is kept (with when it was last seen); `flag_wrong`
  marks a talkgroup whose flagged call showed clear (its flagged calls are then always looked at); a rescan (`e`) asks
  for one more look at the next flagged call, until a call taken after it adds or refreshes a key.
- **The NACs** each talkgroup was heard on are kept, so the panel can show the current NAC's talkgroups.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

from tsdr.radio.decoders.p25 import voice


class Talkgroup(NamedTuple):
    group: int
    follow: bool | None = None  # the user's override; None: the system's default
    keys: tuple[tuple[int, int, float], ...] = ()  # (algorithm, key ID, last seen), oldest first
    flag_wrong: bool = False  # a flagged call's own algorithm showed clear
    seen_flagged: bool = False  # a grant flagged encrypted
    seen_clear_flag: bool = False  # a grant not flagged
    seen_clear_voice: bool = False  # a followed call's own algorithm showed clear
    first_heard: float = 0.0
    last_heard: float = 0.0
    calls: int = 0  # new calls heard
    nacs: tuple[tuple[int, float], ...] = ()  # (NAC, last heard)

    @property
    def encrypted_key(self) -> bool:
        """A key other than clear learned."""
        return any(alg != voice.ALGID_CLEAR for alg, _, _ in self.keys)

    @property
    def latest_key(self) -> tuple[int, int] | None:
        """The latest non-clear key."""
        keys = [k for k in self.keys if k[0] != voice.ALGID_CLEAR]
        if not keys:
            return None
        alg, kid, _ = max(keys, key=lambda k: k[2])
        return alg, kid


class SystemState:
    def __init__(self, wacn: int, sysid: int) -> None:
        self.key = (wacn, sysid)
        self._lock = threading.Lock()
        self._rows: dict[int, Talkgroup] = {}
        self.follow_default = True
        self.follow_new = True
        # rescans waiting for a look: group -> the rescan's serial (a call taken under it clears that one only)
        self._pending: dict[int, int] = {}
        self._rescans = 0
        self.nacs: dict[int, float] = {}  # the NACs its control channels used: NAC -> last heard

    # --- reads (any thread: one dict lookup of an immutable row) ---

    def get(self, group: int) -> Talkgroup | None:
        return self._rows.get(group)

    def follows(self, group: int) -> bool:
        row = self._rows.get(group)
        if row is None:
            return self.follow_new
        return self.follow_default if row.follow is None else row.follow

    def pending(self, group: int) -> bool:
        return group in self._pending

    def look(self, group: int) -> int | None:
        """The rescan a call taken now looks for, if one is pending."""
        return self._pending.get(group)

    def skips_flagged(self, group: int) -> bool:
        """A flagged grant of this talkgroup isn't taken: a key other than clear known, the flag not known wrong, and no
        look pending (DESIGN-M4b 3.3)."""
        row = self._rows.get(group)
        return (
            row is not None
            and row.encrypted_key
            and not row.flag_wrong
            and group not in self._pending
        )

    def rows(self) -> list[Talkgroup]:
        with self._lock:
            return [self._rows[g] for g in sorted(self._rows)]

    # --- the decoder's (pipeline thread) ---

    def heard(
        self, group: int, now: float, new_call: bool, flagged: bool | None, nac: int | None
    ) -> None:
        """A grant (`flagged` its encrypted flag) or update (None) for `group`; `new_call` when it starts a call."""
        with self._lock:
            row = self._row(group, now)
            nacs = row.nacs
            if nac is not None:
                nacs = tuple((n, t) for n, t in nacs if n != nac) + ((nac, now),)
            self._rows[group] = row._replace(
                last_heard=max(row.last_heard, now),
                calls=row.calls + new_call,
                seen_flagged=row.seen_flagged or flagged is True,
                seen_clear_flag=row.seen_clear_flag or flagged is False,
                nacs=nacs,
            )

    def learn(
        self, group: int, algid: int, kid: int, now: float, call_flagged: bool, look: int | None
    ) -> None:
        """A followed call's own algorithm and key: a non-clear one, or clear on a flagged call (which marks the flag
        wrong). `look`: the rescan pending when the call was taken, which this clears (not a later one)."""
        with self._lock:
            row = self._row(group, now)
            keys = tuple(k for k in row.keys if k[:2] != (algid, kid)) + ((algid, kid, now),)
            clear = algid == voice.ALGID_CLEAR
            self._rows[group] = row._replace(
                keys=keys,
                flag_wrong=row.flag_wrong or (clear and call_flagged),
                # R52-5: a followed call's own clear algorithm, flagged or not (3.3)
                seen_clear_voice=row.seen_clear_voice or clear,
            )
            if look is not None and self._pending.get(group) == look:
                del self._pending[group]

    def clear_voice(self, group: int, now: float) -> None:
        """A followed, unflagged call's own algorithm showed clear."""
        with self._lock:
            row = self._row(group, now)
            if not row.seen_clear_voice:
                self._rows[group] = row._replace(seen_clear_voice=True)

    def heard_nac(self, nac: int, now: float) -> None:
        with self._lock:
            self.nacs[nac] = now

    def _row(self, group: int, now: float) -> Talkgroup:
        """The row, created as Table C row 1 says (`follow_new`, stored as an override only when it differs from the
        default)."""
        row = self._rows.get(group)
        if row is None:
            follow = None if self.follow_new == self.follow_default else self.follow_new
            row = Talkgroup(group, follow, first_heard=now)
        return row

    # --- the user's (main thread) ---

    def set_follow(self, group: int, on: bool) -> None:
        with self._lock:
            row = self._rows.get(group) or Talkgroup(group)
            self._rows[group] = row._replace(follow=on)

    def set_all(self, on: bool) -> None:
        """Follow every talkgroup, or none, newly heard ones the same way: one swap of the defaults and the rows."""
        with self._lock:
            self.follow_default = self.follow_new = on
            self._rows = {g: r._replace(follow=None) for g, r in self._rows.items()}

    def set_follow_new(self, on: bool) -> None:
        with self._lock:
            self.follow_new = on

    def rescan(self, group: int, forget: bool = False) -> None:
        """Look again at the talkgroup's next flagged call; `forget` instead resets what was learned (its keys and
        `flag_wrong`)."""
        with self._lock:
            if forget:
                row = self._rows.get(group)
                if row is not None:
                    self._rows[group] = row._replace(keys=(), flag_wrong=False)
                return
            self._rescans += 1
            self._pending[group] = self._rescans


_systems: dict[tuple[int, int], SystemState] = {}
_systems_lock = threading.Lock()


def system_for(wacn: int, sysid: int) -> SystemState:
    with _systems_lock:
        state = _systems.get((wacn, sysid))
        if state is None:
            state = _systems[(wacn, sysid)] = SystemState(wacn, sysid)
        return state


def known() -> list[SystemState]:
    with _systems_lock:
        return list(_systems.values())


def clear() -> None:
    """Forget every system (tests, harnesses)."""
    with _systems_lock:
        _systems.clear()
