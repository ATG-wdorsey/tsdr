"""P25 domain-specific display widget.

Reads a P25Snapshot payload off DecoderOutputEvent and renders three columns: the system and site,
a view of the trunked system (the active channels, the talkgroups, or the call log), and signal
quality. Tab gives the panel the keyboard (`keybindings.Context.P25_PANEL`): a cursor moves over the
view's rows and the keys follow talkgroups on and off, all or none; a followed talkgroup's row is
colored, the call followed marked with ▶.
"""

from __future__ import annotations

from rich.markup import escape
from textual.reactive import reactive

from tsdr.core.events.events import DecoderOutputEvent, StatsUpdateEvent
from tsdr.radio.decoders.p25 import talkgroups, voice
from tsdr.radio.decoders.p25.decoder import P25Channel, P25LogEntry, P25Snapshot, P25Voice
from tsdr.tui.widgets.section_panel import Section, SectionPanel

VIEWS = ("channels", "talkgroups", "log")
_VIEW_TITLES = {"channels": "CHANNELS", "talkgroups": "TALKGROUPS", "log": "CALL LOG"}
_ROWS = 10  # rows of a view shown at once, around the cursor
_PAGE = _ROWS - 1
_FOLLOWED = "cyan"
# The decoder's usable share of a capture, on either side of its center, and a channel's half width.
_USABLE = 0.45
_HALF_CHANNEL_HZ = 6_250.0
_OUTCOMES = {
    "played": "played",
    "learned": "enc learned",
    "encrypted": "enc skipped",
    "not_followed": "not followed",
    "busy": "busy",
    "outside": "outside",
    "no_voice": "no voice",
    "muted": "enc muted",
    "unknown": "unknown ch",
    "tdma": "Phase 2",
}


def encryption(algid: int | None, kid: int | None) -> str:
    """An algorithm and key as the panel shows them (ALGID names as TIA's assignments and OP25 list
    them)."""
    if algid is None:
        return "ENC ?"
    name = voice.ALGID_NAMES.get(algid, f"ALG 0x{algid:02X}")
    if algid == voice.ALGID_CLEAR or kid is None:
        return name
    return f"{name} key {kid:04X}"


def _ago(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.0f}h"


class P25Widget(SectionPanel):
    """Three-column P25 display.

    Col 1: System (NAC, system ID, WACN, RFSS and site, adjacent sites)
    Col 2: The view: active channels, talkgroups or call log (Tab: the keys)
    Col 3: Signal quality (lock, TSBK pass rate, messages, frequency offset, SNR)
    """

    panel_focused: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: P25Snapshot | None = None
        self._snr_db: float | None = None
        self._view = 0
        # The cursor is on a row's key (a call, a talkgroup, a log entry), not its index: rows come
        # and go and reorder under it (review 38, F2). `_cursor` is where that key was last found.
        self._selected: object | None = None
        self._cursor = 0
        self._sort_heard = False

    def watch_panel_focused(self, _focused: bool) -> None:
        self.refresh()

    # event handlers

    def update_messages(self, event: DecoderOutputEvent) -> None:
        latest: P25Snapshot | None = None
        for msg in event.messages:
            if isinstance(msg.data, P25Snapshot):
                latest = msg.data
        if latest is None:
            return
        self._snapshot = latest
        self.refresh()

    def update_stats(self, event: StatsUpdateEvent) -> None:
        if not self.display:
            return
        self._snr_db = event.channel_snr
        self.refresh()

    # keys (the keyboard mixin calls these while the panel has the keys); each returns a status line

    def move(self, rows: int) -> None:
        keys = [key for key, _, _ in self._rows()]
        if keys:
            self._cursor = max(0, min(self._find(keys) + rows, len(keys) - 1))
            self._selected = keys[self._cursor]
        self.refresh()

    def page(self, pages: int) -> None:
        self.move(pages * _PAGE)

    def switch_view(self, step: int) -> str:
        self._view = (self._view + step) % len(VIEWS)
        self._selected, self._cursor = None, 0
        self.refresh()
        return f"P25: {_VIEW_TITLES[VIEWS[self._view]].lower()}"

    def toggle_sort(self) -> str:
        self._sort_heard = not self._sort_heard  # the cursor stays on its talkgroup
        self.refresh()
        return "P25 talkgroups: by last heard" if self._sort_heard else "P25 talkgroups: by number"

    def toggle_follow(self) -> str:
        group, system, why = self._target()
        if group is None or system is None:
            return why or "P25: no talkgroup here"
        table = talkgroups.table_for(*system)
        on = not table.follows(group)
        table.set_follow(group, on)
        self.refresh()
        return f"P25: TG {group}{self._label(system)} {'followed' if on else 'not followed'}"

    def follow_all(self, on: bool) -> str:
        table = self._table()
        if table is None:
            return "P25: no system yet"
        table.set_all(on)
        self.refresh()
        if on:
            return "P25: following every talkgroup heard, and newly heard ones"
        return "P25: following none, nor newly heard ones"

    def toggle_follow_new(self) -> str:
        table = self._table()
        if table is None:
            return "P25: no system yet"
        table.set_follow_new(not table.follow_new)
        self.refresh()
        return f"P25: newly heard talkgroups {'followed' if table.follow_new else 'not followed'}"

    def rescan(self) -> str:
        group, system, why = self._target()
        if group is None or system is None:
            return why or "P25: no talkgroup here"
        talkgroups.table_for(*system).rescan(group)
        self.refresh()
        return (
            f"P25: TG {group}{self._label(system)}'s encryption is learned again at its next "
            "encrypted call"
        )

    # rows

    def _table(self) -> talkgroups.TalkgroupTable | None:
        """The current system's table."""
        snap = self._snapshot
        if snap is None or snap.wacn is None or snap.sysid is None:
            return None
        return talkgroups.table_for(snap.wacn, snap.sysid)

    def _current(self) -> tuple[int, int] | None:
        snap = self._snapshot
        if snap is None or snap.wacn is None or snap.sysid is None:
            return None
        return (snap.wacn, snap.sysid)

    def _label(self, system: tuple[int, int]) -> str:
        """ " (SYS xxx)" for a system other than the current one (a call log row from before a
        retune)."""
        return "" if system == self._current() else f" ({self._system_name(system)})"

    def _system_name(self, system: tuple[int, int]) -> str:
        """ "SYS xxx", with the WACN when the current system has the same system ID (review 42)."""
        current = self._current()
        if current is not None and current[1] == system[1]:
            return f"WACN {system[0]:05X} SYS {system[1]:03X}"
        return f"SYS {system[1]:03X}"

    def _rows(self) -> list[tuple[object, int, str]]:
        """The view's rows: (key, talkgroup, markup). Log rows carry their system in the key."""
        snap = self._snapshot
        if snap is None:
            return []
        table = self._table()
        view = VIEWS[self._view]
        if view == "channels":
            # One row per talkgroup (its latest call): keyed by the talkgroup, so its next
            # transmission, a new call on top, keeps the cursor on it (review 39, F1).
            return [
                (c.call.group, c.call.group, self._channel_row(c, snap, table))
                for c in snap.channels
            ]
        if view == "talkgroups":
            groups = sorted(
                snap.talkgroups,
                key=(lambda t: -t.last_heard) if self._sort_heard else (lambda t: t.group),
            )
            return [(t.group, t.group, self._talkgroup_row(t, snap)) for t in groups]
        return [
            ((e.system, e.group, e.start), e.group, self._log_row(e, snap))
            for e in reversed(snap.log)
        ]

    def _find(self, keys: list[object]) -> int:
        """The cursor's row: where its key is now; when that row has gone, the row at its last place
        (the key kept: `_target` won't act on another row unasked)."""
        if self._selected in keys:
            self._cursor = keys.index(self._selected)
        elif keys:
            self._cursor = min(self._cursor, len(keys) - 1)
        return self._cursor

    def _target(self) -> tuple[int | None, tuple[int, int] | None, str | None]:
        """The talkgroup the keys act on and its system (a call log row's own: it may have been heard
        before a retune, and a talkgroup number means something else on another system), or None
        and why not. When the row the cursor was on has gone, nothing is done: the cursor takes the
        row now at its place, for the next press."""
        rows = self._rows()
        if not rows:
            return None, None, None
        keys = [key for key, _, _ in rows]
        i = self._find(keys)
        if self._selected is not None and self._selected not in keys:
            self._selected = keys[i]
            self.refresh()
            return None, None, f"P25: that row has gone; the cursor is on TG {rows[i][1]} now"
        self._selected = key = keys[i]
        if VIEWS[self._view] == "log" and isinstance(key, tuple):
            if key[0] is None:
                return None, None, "P25: that call's system wasn't known"
            return rows[i][1], key[0], None
        current = self._current()
        return (
            (rows[i][1], current, None)
            if current is not None
            else (None, None, "P25: no system yet")
        )

    @staticmethod
    def _learned(snap: P25Snapshot, group: int) -> talkgroups.P25Talkgroup | None:
        return next((t for t in snap.talkgroups if t.group == group), None)

    def _channel_row(
        self, c: P25Channel, snap: P25Snapshot, table: talkgroups.TalkgroupTable | None
    ) -> str:
        call = c.call
        where = f"{call.frequency_hz / 1e6:9.4f}" if call.frequency_hz is not None else "  unknown"
        tg = self._learned(snap, call.group)
        heard = snap.following.voice if c.followed and snap.following is not None else None
        if heard is not None and heard.algid is not None:
            enc = encryption(heard.algid, heard.kid)  # the call's own algorithm, as it plays
        elif c.algid is not None:
            enc = encryption(c.algid, c.kid)  # learned from this call
        elif heard is not None and heard.encrypted:
            # Muted (its link control's flag), its algorithm not yet heard: its talkgroup's, if learned.
            known = tg is not None and tg.algid not in (None, voice.ALGID_CLEAR)
            enc = encryption(tg.algid, tg.kid) if known and tg is not None else "ENC ?"
        elif call.encrypted:
            enc = encryption(tg.algid, tg.kid) if tg is not None and tg.learned else "ENC ?"
        elif tg is not None and tg.algid not in (None, voice.ALGID_CLEAR):
            # Not flagged, but its talkgroup encrypts (a system that leaves the flag off): unsure.
            enc = encryption(tg.algid, tg.kid) + "?"
        else:
            enc = "?" if call.svcopts is None else "clear"
        flags = " [bold red]EMERG[/bold red]" if call.emergency else ""
        if c.inside is False:
            flags += " [dim]outside[/dim]"
        mark = "▶" if c.followed else " "
        line = f"{mark} {where} TG {call.group:<5} {escape(enc):<14} {snap.time - call.first_seen:3.0f}s{flags}"
        followed = table.follows(call.group) if table is not None else True
        return f"[{_FOLLOWED}]{line}[/{_FOLLOWED}]" if followed else f"[dim]{line}[/dim]"

    def _talkgroup_row(self, t: talkgroups.P25Talkgroup, snap: P25Snapshot) -> str:
        enc = encryption(t.algid, t.kid) if t.learned else "-"
        # "-": set before ever heard
        heard = _ago(max(0.0, snap.time - t.last_heard)) if t.calls else "-"
        line = f"TG {t.group:<6} {escape(enc):<16} {heard:>4} {t.calls:>3}x"
        return f"[{_FOLLOWED}]{line}[/{_FOLLOWED}]" if t.follow else f"[dim]{line}[/dim]"

    def _log_row(self, e: P25LogEntry, snap: P25Snapshot) -> str:
        where = f"{e.frequency_hz / 1e6:9.4f}" if e.frequency_hz is not None else "  unknown"
        outcome = _OUTCOMES.get(e.outcome, e.outcome)
        if e.algid is not None:
            outcome += f" {encryption(e.algid, e.kid)}"
        line = (
            f"{_ago(max(0.0, snap.time - e.end)):>4} TG {e.group:<5} {where} "
            f"{e.end - e.start:3.0f}s {escape(outcome)}"
        )
        current = (snap.wacn, snap.sysid)
        if e.system is not None and e.system != current:
            # Heard on another system (before a retune): its own table, and said so.
            line += f" [italic]{self._system_name(e.system)}[/italic]"
            followed = talkgroups.table_for(*e.system).follows(e.group)
        else:
            tg = self._learned(snap, e.group)
            followed = tg.follow if tg is not None else snap.follow_new
        return f"[{_FOLLOWED}]{line}[/{_FOLLOWED}]" if followed else f"[dim]{line}[/dim]"

    # render

    def build_sections(self) -> list[Section]:
        snap = self._snapshot
        if snap is None:
            return [Section("Waiting...")]
        return [
            Section(self._render_system(snap), width=20, min_width=16),
            Section(self._render_view(snap), width=48, min_width=36),
            Section(self._render_quality(snap), width=18, min_width=14),
        ]

    @staticmethod
    def _render_system(snap: P25Snapshot) -> str:
        lines: list[str] = []
        lines.append(
            f"NAC [bold]{snap.nac:03X}[/bold]" if snap.nac is not None else "[dim]no NAC[/dim]"
        )
        if snap.sysid is not None:
            wacn = f"  WACN {snap.wacn:05X}" if snap.wacn is not None else ""
            lines.append(f"SYS {snap.sysid:03X}{wacn}")
        if snap.rfss is not None and snap.site is not None:
            lines.append(f"RFSS {snap.rfss}  SITE {snap.site}")
        if snap.adjacent:
            lines.append(f"[dim]{len(snap.adjacent)} adjacent sites[/dim]")
        outside = sum(c.inside is False for c in snap.channels)
        if outside:
            lines.append(f"[yellow]{outside} call(s) outside the capture[/yellow]")
        if snap.capture is not None:
            low, high = snap.capture
            lines.append(f"[dim]capture {low / 1e6:.3f}-{high / 1e6:.3f}[/dim]")
            span = snap.site_span
            if span is not None and (
                span[0] - _HALF_CHANNEL_HZ < low or span[1] + _HALF_CHANNEL_HZ > high
            ):
                # The center and rate whose usable band holds every channel heard.
                center = (span[0] + span[1]) / 2
                rate = (span[1] - span[0] + 2 * _HALF_CHANNEL_HZ) / (2 * _USABLE)
                lines.append(
                    f"[yellow]site {span[0] / 1e6:.4f}-{span[1] / 1e6:.4f}: center "
                    f"{center / 1e6:.4f}, {rate / 1e6:.1f} MS/s or more[/yellow]"
                )
        return "\n".join(lines)

    def _render_view(self, snap: P25Snapshot) -> str:
        view = VIEWS[self._view]
        tabs = "  ".join(
            f"[bold]{_VIEW_TITLES[v]}[/bold]"
            if v == view
            else f"[dim]{_VIEW_TITLES[v].lower()}[/dim]"
            for v in VIEWS
        )
        hint = "[reverse] Tab: back [/reverse]" if self.panel_focused else "[dim]Tab: keys[/dim]"
        new = "new: follow" if snap.follow_new else "new: skip"
        lines = [f"{tabs}  [dim]{new}[/dim]  {hint}"]
        if snap.voice is not None:
            lines.append(_voice_line(snap.voice, snap.nac))
        rows = self._rows()
        if not rows:
            lines.append("[dim]none yet[/dim]")
            return "\n".join(lines)
        keys = [key for key, _, _ in rows]
        cursor = self._find(keys)
        if self._selected is None:
            self._selected = keys[cursor]  # what is shown under the cursor is what the keys act on
        first = max(0, min(cursor - _ROWS // 2, len(rows) - _ROWS))
        for i in range(first, min(first + _ROWS, len(rows))):
            text = rows[i][2]
            if self.panel_focused and i == cursor:
                text = f"[reverse]{text}[/reverse]"
            lines.append(text)
        if len(rows) > _ROWS:
            lines.append(f"[dim]{cursor + 1}/{len(rows)}[/dim]")
        return "\n".join(lines)

    def _render_quality(self, snap: P25Snapshot) -> str:
        lines = ["[green]●[/green] LOCKED" if snap.locked else "[red]○[/red] SEARCHING"]
        if snap.tsbks + snap.mbts == 0 and snap.voice_frames:
            lines.append(f"LDU {snap.voice_frames}")  # a voice channel: no blocks to count
        else:
            color = "green" if snap.tsbk_pct >= 80 else "yellow" if snap.tsbk_pct >= 50 else "red"
            lines.append(f"TSBK [{color}]{snap.tsbk_pct:.0f}%[/{color}]")
        if self._snr_db is not None:
            snr = self._snr_db
            snr_color = "green" if snr > 20 else "yellow" if snr > 10 else "red"
            lines.append(f"SNR [{snr_color}]{snr:.1f}dB[/{snr_color}]")
        if snap.locked:  # without frames the tracked offset means nothing
            lines.append(f"[dim]offset {snap.frequency_offset_hz:+.0f} Hz[/dim]")
        lines.append(f"[dim]{snap.tsbks + snap.mbts} messages[/dim]")
        return "\n".join(lines)


# render helpers


def _voice_line(call: P25Voice, channel_nac: int | None) -> str:
    """The call heard (at the dial, or followed): muted if encrypted; its NAC when not the
    channel's."""
    who = f"TG {call.group}" if call.group is not None else "voice"
    if call.target is not None:
        who = f"unit {call.target}"
    if channel_nac is not None and call.nac != channel_nac:
        who += f" NAC {call.nac:03X}"
    flags = (" [red]ENC muted[/red]" if call.encrypted else "") + (
        " [bold red]EMERG[/bold red]" if call.emergency else ""
    )
    return f"[green bold]▶ {who}[/green bold]{flags}"
