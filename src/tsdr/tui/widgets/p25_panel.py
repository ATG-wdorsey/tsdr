"""The P25 panel's logic, without Textual (DESIGN-M4b 3.7, Table E, 10.1-10.3): the rows of each view, the cursor's key,
what a key acts on, the confirmation pending. The widget (`p25_widget.py`) renders it; the keyboard calls it.

- **The cursor holds a row's key, never an index:** rows come and go and reorder under it. When its row has gone, it
  shows the row now at its place, and the next action key does nothing but take that row (Table E).
- **Every key carries its system,** so an equal number on another system never takes the cursor, and a key press acts
  on the target taken when it was pressed: `a`/`n` capture the system for their `y` (D3).
- **Tab re-anchors** the cursor on the row shown (D4).
- **Rows fit the view's width:** columns by priority, the lowest dropped first; the cursor row in full on the detail line.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from rich.cells import cell_len
from rich.markup import escape

from tsdr.radio.decoders.p25 import systems, voice
from tsdr.radio.decoders.p25.decoder import P25Channel, P25LogEntry, P25Snapshot, rate_text

VIEWS = ("channels", "talkgroups", "log")
TITLES = {"channels": "CHANNELS", "talkgroups": "TALKGROUPS", "log": "CALL LOG"}
MIN_ROWS = 12
CONFIRM_S = 5.0
FOLLOWED = "cyan"
# The call log's outcomes as short tags (the legend is in the key list, `?`).
OUTCOME_TAGS = {
    "played": "heard",
    "busy": "busy",
    "not_followed": "off",
    "encrypted": "enc",
    "learned": "learn",
    "muted": "muted",
    "no_voice": "quiet",
    "outside": "out",
    "unknown": "?ch",
    "tdma": "ph2",
}
OUTCOME_LEGEND = (
    ("heard", "followed; its audio played"),
    ("busy", "missed: another call had the one voice channel"),
    ("off", "its talkgroup isn't followed"),
    ("enc", "skipped: its talkgroup's encryption is known"),
    ("learn", "followed muted until its algorithm and key showed"),
    ("muted", "encrypted voice, left before its algorithm showed"),
    ("quiet", "followed; no voice came"),
    ("out", "outside the capture"),
    ("?ch", "its channel's frequency unknown"),
    ("ph2", "a Phase 2 channel"),
)
_SHORT = {
    0x80: "clr",
    0x81: "DES",
    0x83: "DES",
    0x84: "AES",
    0x85: "AES",
    0x88: "AES",
    0x89: "AES",
    0xAA: "ADP",
}


def alg_short(algid: int) -> str:
    return _SHORT.get(algid, f"ALG {algid:02X}")


def alg_name(algid: int) -> str:
    return voice.ALGID_NAMES.get(algid, f"ALG 0x{algid:02X}")


def tg_encryption(row: systems.Talkgroup | None, pending: bool, wide: bool) -> str:
    """A talkgroup's encryption as the talkgroups view shows it (10.1 item 2), first match wins."""
    if pending:
        return "e… look" if wide else "e…"
    if row is None:
        return "?"
    nonclear = [k for k in row.keys if k[0] != voice.ALGID_CLEAR]
    if nonclear:
        latest = max(nonclear, key=lambda k: k[2])[0]  # the latest key's algorithm (R49-7)
        text = alg_name(latest) if wide else alg_short(latest)
        if len(nonclear) > 1:
            text = f"{text}, {len(nonclear)} keys" if wide else f"{len(nonclear)} keys"
        if row.seen_clear_voice:
            text += "+clear" if wide else "+clr"
        if row.flag_wrong:
            text += "!"
        return text
    if row.seen_flagged and row.seen_clear_flag:
        return "mixed ?" if wide else "mix?"
    if row.seen_flagged:
        return "ENC ?" if wide else "ENC?"
    if row.seen_clear_flag or row.keys:
        return "clear" if wide else "clr"
    return "?"


def ago(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.0f}h"


def system_name(system: tuple[int, int], shared_sysid: bool) -> str:
    """SYS 2A1, with the WACN when another retained system has the same system ID (I24)."""
    if shared_sysid:
        return f"WACN {system[0]:05X} SYS {system[1]:03X}"
    return f"SYS {system[1]:03X}"


@dataclass(frozen=True)
class Cell:
    text: str  # plain text (escaped when rendered)
    priority: int  # higher stays longer
    align: str = "<"  # "<" or ">"
    width: int = 0  # padded to this many cells (0: its own)
    style: str = ""


def fit(cells: list[Cell], width: int) -> str:
    """The cells that fit `width` (one space between), the lowest priority dropped first, in their order."""
    kept = list(range(len(cells)))

    def size(idx: list[int]) -> int:
        return sum(max(int(cell_len(cells[i].text)), cells[i].width) for i in idx) + max(
            0, len(idx) - 1
        )

    for i in sorted(kept, key=lambda i: cells[i].priority):
        if size(kept) <= width:
            break
        kept.remove(i)
    parts = []
    for i in kept:
        c = cells[i]
        pad = max(0, c.width - cell_len(c.text))
        text = escape(c.text)
        text = (" " * pad + text) if c.align == ">" else (text + " " * pad)
        parts.append(f"[{c.style}]{text}[/{c.style}]" if c.style else text)
    line = " ".join(parts)
    return line


@dataclass(frozen=True)
class Target:
    """What a row's key acts on: a talkgroup of a system (units in M4b3)."""

    system: tuple[int, int] | None
    group: int


@dataclass(frozen=True)
class Row:
    key: object
    target: Target
    followed: bool  # its talkgroup chosen (the row's color)
    build: Callable[[int], str]  # its line at a width
    detail: Callable[[], str]  # the detail line, for the cursor row


class P25PanelState:
    def __init__(self) -> None:
        self.snapshot: P25Snapshot | None = None
        self.view = 0
        self.focused = False
        self._key: object | None = None  # the cursor's row key
        self._index = 0  # where that key was last found
        self.sort_heard = False
        self.whole_system = False  # `t`: the talkgroups of the whole system, not this NAC's
        self.pending: tuple[bool, tuple[int, int], float] | None = (
            None  # (all?, system captured, when)
        )
        self.width = 42
        self.shown = MIN_ROWS
        self._alerted: set[int] = set()  # calls already announced outside the capture
        self.channel_snr: dict[
            int, float
        ] = {}  # the active channels' C/N by granted frequency (the stats stage)

    # --- the decoder's state ---

    def update(self, snapshot: P25Snapshot) -> None:
        self.snapshot = snapshot

    @property
    def system(self) -> tuple[int, int] | None:
        snap = self.snapshot
        if snap is None or snap.wacn is None or snap.sysid is None:
            return None
        return (snap.wacn, snap.sysid)

    def shown_system(self) -> tuple[int, int] | None:
        """The system the view shows, which `a`, `n` and `N` act on: the talkgroups view's is its NAC's when no
        broadcast has named one (3.1b)."""
        if VIEWS[self.view] == "talkgroups":
            return self.nac_system()[0]
        return self.system

    def follow_new(self) -> bool | None:
        """`new:` in the header: the shown system's setting for newly heard talkgroups, the one `N` toggles (review 53,
        R53-5); None when no system is shown."""
        system = self.shown_system()
        return None if system is None else systems.system_for(*system).follow_new

    def nac_system(self) -> tuple[tuple[int, int] | None, str | None]:
        """The system the talkgroups view shows (3.1b): the current one, else the one known system whose control
        channels used the channel's NAC; else None and why."""
        system = self.system
        if system is not None:
            return system, None
        snap = self.snapshot
        if snap is None or snap.nac is None:
            return None, "No P25 signal"
        users = [s.key for s in systems.known() if snap.nac in s.nacs]
        if len(users) == 1:
            return users[0], None
        if not users:
            return None, f"No system known for NAC {snap.nac:03X}"
        shared = len({k[1] for k in users}) < len(users)
        names = " and ".join(system_name(k, shared).removeprefix("SYS ") for k in users)
        return None, f"NAC {snap.nac:03X} is used by systems {names}"

    def _shared_sysid(self, system: tuple[int, int]) -> bool:
        return sum(1 for s in systems.known() if s.key[1] == system[1]) > 1

    # --- rows ---

    def rows(self) -> list[Row]:
        snap = self.snapshot
        if snap is None:
            return []
        view = VIEWS[self.view]
        if view == "channels":
            return self._channel_rows(snap)
        if view == "talkgroups":
            return self._talkgroup_rows(snap)
        return self._log_rows(snap)

    def _state(self, system: tuple[int, int] | None) -> systems.SystemState | None:
        return systems.system_for(*system) if system is not None else None

    def _channel_rows(self, snap: P25Snapshot) -> list[Row]:
        system = self.system
        state = self._state(system)
        out, seen, freqs = [], set(), set()
        for c in snap.channels:  # newest first: one row per talkgroup, its latest call (39-F1)
            key = (system, c.call.group)
            if key in seen:
                continue
            seen.add(key)
            # a frequency's C/N on its newest row only (a call ended within _ACTIVE_S, and the new one)
            freq = c.call.frequency_hz
            snr = self.channel_snr.get(freq) if freq is not None and freq not in freqs else None
            freqs.add(freq)
            followed = state.follows(c.call.group) if state is not None else snap.follow_new
            out.append(
                Row(
                    key,
                    Target(system, c.call.group),
                    followed,
                    partial(self._channel_line, c, snap, snr=snr),
                    partial(self._channel_detail, c, snap),
                )
            )
        return out

    def _call_encryption(self, c: P25Channel, snap: P25Snapshot, wide: bool = False) -> str:
        """The type only (10.1 item 8): clr, ENC?, AES, DES, ADP, ALG xx; `?` before a grant gave flags; in full
        words when wide."""
        name = alg_name if wide else alg_short
        heard = snap.following.voice if c.followed and snap.following is not None else None
        if heard is not None and heard.algid is not None:
            return name(heard.algid)
        if c.algid is not None:
            return name(c.algid)
        state = self._state(self.system)
        row = state.get(c.call.group) if state is not None else None
        key = row.latest_key if row is not None else None
        if c.call.encrypted or (heard is not None and heard.encrypted):
            return name(key[0]) if key is not None else ("ENC ?" if wide else "ENC?")
        if c.call.svcopts is None:
            return "?"
        return name(key[0]) + "?" if key is not None else ("clear" if wide else "clr")

    def _channel_line(self, c: P25Channel, snap: P25Snapshot, width: int, snr: float | None) -> str:
        call = c.call
        wide = width >= 60
        freq = f"{call.frequency_hz / 1e6:.4f}" if call.frequency_hz is not None else "unknown"
        cells = [
            Cell("▶" if c.followed else " ", 9),
            Cell(freq, 8, width=8),
            Cell(f"TG {call.group}" if wide else str(call.group), 9, width=8 if wide else 5),
            Cell("EMERG" if call.emergency else "", 7, style="bold red"),
            Cell(self._call_encryption(c, snap, wide), 6, width=7 if wide else 4),
            Cell(
                ("SNR " if wide else "") + (f"{snr:.0f}dB" if snr is not None else "-"),
                5,
                ">",
                width=4,
            ),
            Cell(f"{max(0.0, snap.time - call.first_seen):.0f}s", 4, ">", width=3),
            Cell("outside" if c.inside is False else "", 3, style="dim"),
        ]
        return fit([x for x in cells if x.text or x.width], width)

    def _channel_detail(self, c: P25Channel, snap: P25Snapshot) -> str:
        call = c.call
        parts = [f"TG {call.group}"]
        if call.frequency_hz is not None:
            parts.append(
                f"{call.frequency_hz / 1e6:.4f} MHz"
                + (f" slot {call.slot}" if call.slot is not None else "")
            )
        parts.append(self._call_encryption(c, snap))
        if call.emergency:
            parts.append("[bold red]EMERG[/bold red]")
        if c.inside is False and snap.capture is not None:
            parts.append(
                f"outside the capture ({snap.capture[0] / 1e6:.1f}-{snap.capture[1] / 1e6:.1f})"
            )
        return "  ".join(parts)

    def _talkgroup_rows(self, snap: P25Snapshot) -> list[Row]:
        system, _ = self.nac_system()
        state = self._state(system)
        if state is None:
            return []
        rows = state.rows()
        if not self.whole_system and snap.nac is not None:
            rows = [r for r in rows if any(n == snap.nac for n, _ in r.nacs)]
        if self.sort_heard:
            rows.sort(key=lambda r: -r.last_heard)
        return [
            Row(
                (system, r.group),
                Target(system, r.group),
                state.follows(r.group),
                partial(self._talkgroup_line, r, state, snap),
                partial(self._talkgroup_detail, r, state, snap),
            )
            for r in rows
        ]

    def _talkgroup_line(
        self, r: systems.Talkgroup, state: systems.SystemState, snap: P25Snapshot, width: int
    ) -> str:
        wide = width >= 60
        heard = ago(max(0.0, snap.time - r.last_heard)) if r.calls else "-"
        cells = [
            Cell(f"TG {r.group}" if wide else str(r.group), 9, width=8 if wide else 5),
            Cell(tg_encryption(r, state.pending(r.group), wide), 8, width=13 if wide else 8),
            Cell(heard + (" ago" if wide and r.calls else ""), 6, ">", width=4),
            Cell(f"{r.calls}" + (" calls" if wide else "x"), 5, ">", width=4),
        ]
        return fit(cells, width)

    def _talkgroup_detail(
        self, r: systems.Talkgroup, state: systems.SystemState, snap: P25Snapshot
    ) -> str:
        parts = [f"TG {r.group}", "followed" if state.follows(r.group) else "not followed"]
        keys = [
            f"{alg_name(a)} key {k:04X}" + f" {ago(max(0.0, snap.time - t))} ago"
            for a, k, t in r.keys
            if a != voice.ALGID_CLEAR
        ]
        if keys:
            parts.append(", ".join(keys))
        elif r.keys:
            parts.append("clear")
        if r.flag_wrong:
            parts.append("flag wrong: its flagged calls looked at")
        if state.pending(r.group):
            parts.append("rescan pending")
        if r.nacs:
            parts.append("NAC " + ",".join(f"{n:03X}" for n, _ in r.nacs))
        return "  ".join(parts)

    def _log_rows(self, snap: P25Snapshot) -> list[Row]:
        current = self.system
        out = []
        for e in reversed(snap.log):
            state = self._state(e.system)
            followed = state.follows(e.group) if state is not None else snap.follow_new
            out.append(
                Row(
                    ("log", e.cid, e.start),
                    Target(e.system, e.group),
                    followed,
                    partial(self._log_line, e, current),
                    partial(self._log_detail, e),
                )
            )
        return out

    def _log_line(self, e: P25LogEntry, current: tuple[int, int] | None, width: int) -> str:
        wide = width >= 60
        other = e.system is not None and e.system != current
        cells = [
            Cell(time.strftime("%H:%M:%S", time.localtime(e.start)), 9, width=8),
            Cell(f"TG {e.group}" if wide else str(e.group), 9, width=8 if wide else 5),
            Cell(OUTCOME_TAGS.get(e.outcome, e.outcome), 8, width=5),
            Cell(f"{max(0.0, e.end - e.start):.0f}s", 6, ">", width=3),
            Cell("°" if other else "", 8),  # reserved before the optional columns
            Cell(f"{e.frequency_hz / 1e6:.4f}" if e.frequency_hz is not None else "", 3),
            Cell(
                system_name(e.system, self._shared_sysid(e.system)) if other and e.system else "",
                2,
                style="italic",
            ),
        ]
        return fit([x for x in cells if x.text or x.width], width)

    def _log_detail(self, e: P25LogEntry) -> str:
        parts = [
            f"TG {e.group}",
            time.strftime("%H:%M:%S", time.localtime(e.start)),
            e.outcome.replace("_", " "),
        ]
        if e.frequency_hz is not None:
            parts.append(f"{e.frequency_hz / 1e6:.4f} MHz")
        if e.algid is not None:
            parts.append(
                alg_name(e.algid)
                + (f" key {e.kid:04X}" if e.kid is not None and e.algid != 0x80 else "")
            )
        if e.ldus:
            parts.append(f"{e.ldus} LDUs")
        if e.system is not None:
            parts.append(system_name(e.system, self._shared_sysid(e.system)))
        return "  ".join(parts)

    # --- the cursor ---

    def cursor(self, rows: list[Row]) -> int | None:
        """The cursor's row: where its key is now; when that row has gone, the row at its last place (the key kept, so
        an action key won't act on another row unasked)."""
        if not rows:
            return None
        keys = [r.key for r in rows]
        if self._key in keys:
            self._index = keys.index(self._key)
        else:
            self._index = min(self._index, len(rows) - 1)
            if self._key is None:
                self._key = keys[self._index]
        return self._index

    def _move(self, rows: list[Row], to: int) -> None:
        if rows:
            self._index = max(0, min(to, len(rows) - 1))
            self._key = rows[self._index].key

    def anchor(self) -> None:
        """D4: the cursor's key becomes the row shown at its place."""
        rows = self.rows()
        i = self.cursor(rows)
        if i is not None:
            self._key = rows[i].key

    def click(self, key: object | None) -> None:
        """A click in the panel: on a row (the key of the row drawn there), the cursor goes to it; elsewhere, or on a
        row since gone, it's re-anchored, as Tab does."""
        rows = self.rows()
        keys = [r.key for r in rows]
        if key is not None and key in keys:
            self._move(rows, keys.index(key))
        else:
            self.anchor()

    def _target(self) -> tuple[Target | None, str | None]:
        """The target of an action key: the cursor row's, or None and why (the row has gone: the cursor takes the row
        at its place, for the next key)."""
        rows = self.rows()
        i = self.cursor(rows)
        if i is None:
            return None, "P25: no row here"
        row = rows[i]
        if row.key != self._key:
            self._key = row.key
            return None, f"P25: that row has gone; the cursor is on TG {row.target.group} now"
        if row.target.system is None:
            return None, "P25: that call's system wasn't known"
        return row.target, None

    # --- keys ---

    def key(self, key: str, now: float | None = None) -> str | None:
        """One key while the panel has the keys (Table E); a status line, or None."""
        now = time.monotonic() if now is None else now
        if self.pending is not None:
            on, captured, since = self.pending
            self.pending = None
            if now - since < CONFIRM_S:
                if key != "y":
                    return "P25: cancelled"  # the key does nothing else (R44-8)
                systems.system_for(*captured).set_all(on)
                where = "" if captured == self.system else f" ({self.system_label(captured)})"
                if on:
                    return f"P25: following every talkgroup heard, and newly heard ones{where}"
                return f"P25: following none, nor newly heard ones{where}"
        rows = self.rows()
        match key:
            case "up" | "k":
                self._move(rows, (self.cursor(rows) or 0) - 1)
            case "down" | "j":
                self._move(rows, (self.cursor(rows) or 0) + 1)
            case "pageup":
                self._move(rows, (self.cursor(rows) or 0) - (self.shown - 1))
            case "pagedown":
                self._move(rows, (self.cursor(rows) or 0) + (self.shown - 1))
            case "home":
                self._move(rows, 0)
            case "end":
                self._move(rows, len(rows) - 1)
            case "left" | "right":
                self.view = (self.view + (1 if key == "right" else -1)) % len(VIEWS)
                self._key, self._index = None, 0
                self.cursor(self.rows())
                return f"P25: {TITLES[VIEWS[self.view]].lower()}"
            case "space" | "enter":
                return self._toggle()
            case "e":
                return self._rescan()
            case "a" | "n":
                system = self.shown_system()
                if system is None:
                    return "P25: no system yet"
                self.pending = (key == "a", system, now)
                which = "every talkgroup heard" if key == "a" else "none"
                return f"P25: follow {which}? Press y to confirm"
            case "N":
                system = self.shown_system()
                if system is None:
                    return "P25: no system yet"
                state = systems.system_for(*system)
                state.set_follow_new(not state.follow_new)
                return f"P25: newly heard talkgroups {'followed' if state.follow_new else 'not followed'}"
            case "s":
                self.sort_heard = not self.sort_heard
                return (
                    "P25 talkgroups: by last heard"
                    if self.sort_heard
                    else "P25 talkgroups: by number"
                )
            case "t":
                self.whole_system = not self.whole_system
                if self.whole_system:
                    return "P25 talkgroups: the whole system's"
                nac = self.snapshot.nac if self.snapshot is not None else None
                return (
                    f"P25 talkgroups: NAC {nac:03X}'s"
                    if nac is not None
                    else "P25 talkgroups: this NAC's"
                )
            case _:
                return None
        return None

    def cancel_pending(self, now: float | None = None) -> bool:
        """The confirmation's time ran out (the keyboard's timer): whether one was pending."""
        if self.pending is None:
            return False
        self.pending = None
        return True

    def system_label(self, system: tuple[int, int]) -> str:
        return system_name(system, self._shared_sysid(system))

    def _toggle(self) -> str:
        target, why = self._target()
        if target is None:
            return why or "P25: no talkgroup here"
        assert target.system is not None
        state = systems.system_for(*target.system)
        on = not state.follows(target.group)
        state.set_follow(target.group, on)
        where = "" if target.system == self.system else f" ({self.system_label(target.system)})"
        return f"P25: TG {target.group}{where} {'followed' if on else 'not followed'}"

    def _rescan(self) -> str:
        target, why = self._target()
        if target is None:
            return why or "P25: no talkgroup here"
        assert target.system is not None
        systems.system_for(*target.system).rescan(target.group)
        where = "" if target.system == self.system else f" ({self.system_label(target.system)})"
        return f"P25: TG {target.group}{where}: its next encrypted call is looked at"

    # --- the alert ---

    def outside_alert(self, wide: bool = True) -> tuple[str, str] | None:
        """A chosen call granted outside the capture (10.1 item 10): the view's top line and its remedy, each one line
        of at most 42 cells when not `wide`."""
        snap = self.snapshot
        if snap is None or snap.capture is None:
            return None
        state = self._state(self.system)
        for c in snap.channels:
            if (
                c.inside is False
                and c.call.slot is None
                and c.call.frequency_hz is not None
                and (state is None or state.follows(c.call.group))
            ):
                low, high = snap.capture
                freq = c.call.frequency_hz / 1e6
                if wide:
                    line = (
                        f"TG {c.call.group} on {freq:.4f} is outside the capture "
                        f"({low / 1e6:.1f}-{high / 1e6:.1f})"
                    )
                else:
                    line = f"TG {c.call.group} {freq:.4f} outside {low / 1e6:.1f}-{high / 1e6:.1f}"
                remedy = ""
                if snap.fit is not None and snap.fit[3] <= snap.sample_rate:
                    # the capture's rate holds the site: `c` centers it there (review 56, R56-5)
                    remedy = (
                        "press c to center the capture on the site"
                        if wide
                        else "c: center on the site"
                    )
                elif snap.fit is not None:  # what `p25 fit` would set
                    _, _, center, rate = snap.fit
                    remedy = (
                        f"center {center / 1e6:.1f}, {rate_text(rate)} MS/s or more (p25 fit)"
                        if wide
                        else f"p25 fit: {center / 1e6:.1f}, {rate_text(rate)} MS/s"
                    )
                return line, remedy
        return None

    def new_alerts(self) -> list[str]:
        """Each chosen call outside the capture once, for the status line."""
        snap = self.snapshot
        if snap is None:
            return []
        out = []
        state = self._state(self.system)
        for c in snap.channels:
            call = c.call
            if c.inside is not False or call.frequency_hz is None or call.cid in self._alerted:
                continue
            if state is not None and not state.follows(call.group):
                continue
            self._alerted.add(call.cid)
            fits = snap.fit is not None and snap.fit[3] <= snap.sample_rate  # `c` is enough (R56-5)
            out.append(
                f"P25: TG {call.group} on {call.frequency_hz / 1e6:.4f} is outside the capture "
                + ("(c: center on the site)" if fits else "(p25 fit)")
            )
        return out
