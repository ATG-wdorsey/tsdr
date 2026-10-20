"""P25 domain-specific display widget: rendering only (the logic is `p25_panel.P25PanelState`).

Reads a P25Snapshot payload off DecoderOutputEvent and renders three columns: the system and site, a view of the
trunked system (the active channels, the talkgroups, or the call log), and signal quality. Tab, or a click in the
panel, gives it the keyboard (`keybindings.Context.P25_PANEL`): its title, its top border and the status line then
say so. In a side column the view's rows fill the height (at least 12); on the bottom bar it shows 12 and its rows fit
the width left beside the other columns.
"""

from __future__ import annotations

from typing import NamedTuple

from rich.console import Group
from rich.markup import escape
from rich.text import Text
from textual import events
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive

from tsdr.core.events.events import DecoderOutputEvent, StatsUpdateEvent
from tsdr.radio.decoders.p25.decoder import P25Snapshot, P25Voice
from tsdr.tui.widgets.p25_panel import FOLLOWED, MIN_ROWS, TITLES, VIEWS, P25PanelState
from tsdr.tui.widgets.section_panel import Section, SectionPanel, _SectionSurface

# The decoder's usable share of a capture, on either side of its center, and a channel's half width.
_USABLE = 0.45
_HALF_CHANNEL_HZ = 6_250.0
_SYSTEM_WIDTH, _QUALITY_WIDTH = 22, 16
_SIDE_WIDTH = 42  # the side column's usable width
FOCUSED_TITLE = "[reverse] P25 · KEYS · Tab/Esc to leave [/reverse]"
# the bottom bar's columns: Rich's grid puts one column of padding between them
_GAP = 1


class _Hits(NamedTuple):
    """Where the last render drew the view's rows, in the render's own coordinates (its surface's): a click selects the
    row drawn under it, by its key, or none (review 53, R53-2 and R53-3)."""

    top: int  # the line of the first row drawn
    x: tuple[int, int] | None  # the view column's x range (None: the whole width)
    keys: tuple[object, ...]  # the rows drawn, top down (none when their place isn't known exactly)


class P25Widget(SectionPanel):
    """Three-column P25 display.

    Col 1: System (what the channel at the dial is, its control channel, NAC, system, site)
    Col 2: The view: active channels, talkgroups or call log (Tab: the keys)
    Col 3: Signal quality (lock, TSBK pass rate, messages, frequency offset, SNR)
    """

    panel_focused: reactive[bool] = reactive(False)

    # the handler's name stated (`on_p25_widget_clicked`): Textual would derive `on_p25widget_clicked` (review 53, R53-1);
    # the lint's mypy has no Textual, so it checks the keyword against `object`
    class Clicked(Message, namespace="p25_widget"):  # type: ignore[call-arg]
        """A click in the panel: it takes the keys, and the cursor goes to the row clicked, by its key (None: not on a
        row)."""

        def __init__(self, key: object | None) -> None:
            self.key = key
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.state = P25PanelState()
        self._snr_db: float | None = None
        self._hits: _Hits | None = None
        # the view's lines above its rows, and the rows drawn
        self._view_rows: tuple[int, tuple[object, ...]] = (0, ())

    def watch_panel_focused(self, focused: bool) -> None:
        self.state.focused = focused
        self.set_panel_title(FOCUSED_TITLE if focused else None)
        self.refresh()

    def on_mount(self) -> None:
        if self.panel_focused:
            self.set_panel_title(FOCUSED_TITLE)

    # event handlers

    def update_messages(self, event: DecoderOutputEvent) -> None:
        latest: P25Snapshot | None = None
        for msg in event.messages:
            if isinstance(msg.data, P25Snapshot):
                latest = msg.data
        if latest is None:
            return
        self.state.update(latest)
        for alert in self.state.new_alerts():
            self.app.show_status(f"[yellow]{escape(alert)}[/yellow]")  # type: ignore[attr-defined]
        self.refresh()

    def update_stats(self, event: StatsUpdateEvent) -> None:
        if not self.display:
            return
        self._snr_db = event.channel_snr
        self.state.channel_snr = dict(event.channel_snrs)
        self.refresh()

    def on_click(self, event: events.Click) -> None:
        """The row drawn under the click, by its key (None: not on a row). The click's place is taken in the render's
        coordinates: the screen's less the surface's region, which moves with the scroll."""
        key: object | None = None
        try:
            surface = self.query_one(_SectionSurface).region
        except NoMatches:
            surface = None
        hits = self._hits
        if surface is not None and hits is not None:
            x, y = event.screen_x - surface.x, event.screen_y - surface.y
            i = y - hits.top
            low, high = hits.x if hits.x is not None else (0, surface.width)
            if 0 <= i < len(hits.keys) and low <= x < high:
                key = hits.keys[i]
        self.post_message(self.Clicked(key))

    # render

    def build_sections(self) -> list[Section]:
        snap = self.state.snapshot
        if snap is None:
            return [Section("Waiting...")]
        side = self.dock_edge != "bottom"
        system = self._render_system(snap, side)
        quality = self._render_quality(snap)
        self._system_lines = len(system)
        if not side:
            # the three columns fit the surface (a scrollbar excluded), so Rich keeps their widths and the view's x
            # range is known; on a bar too narrow for them it isn't, and a click there selects no row
            room = self._surface_width() - _SYSTEM_WIDTH - _QUALITY_WIDTH - 2 * _GAP
            width, exact = max(30, room), room >= 30
            rows = MIN_ROWS
        else:
            width = max(30, min(self.size.width, 200) - 2 if self.size.width else _SIDE_WIDTH)
            # the other columns stacked above and below, the gaps, the header, the detail and position lines
            used = self._system_lines + quality.count("\n") + 1 + 2 + 5
            rows = max(MIN_ROWS, self.size.height - used)
        view = self._render_view(snap, width, rows)
        above, keys = self._view_rows
        if side:  # the system lines and a blank line above the view (`_render_vertical`)
            self._hits = _Hits(len(system) + 1 + above, None, keys)
        else:
            start = _SYSTEM_WIDTH + _GAP
            self._hits = _Hits(above, (start, start + width), keys if exact else ())
        # R52-1: in a side column the system lines stand above the view's rows, so none may wrap (the rows' place is
        # counted in lines); on the bottom bar it is a column of its own beside them and may wrap
        system_body = _lines(system) if side else "\n".join(system)
        return [
            Section(system_body, width=_SYSTEM_WIDTH, min_width=_SYSTEM_WIDTH),
            Section(view, width=width, min_width=30),
            Section(quality, width=_QUALITY_WIDTH, min_width=14),
        ]

    def _render_system(self, snap: P25Snapshot, side: bool) -> list[str]:
        lines: list[str] = []
        kind = snap.channel_kind
        if kind == "control":
            dial, cc = snap.dial_hz, snap.cc_hz
            if cc is not None and dial is not None and abs(dial - cc) > 1_000:
                lines.append(f"[bold]Control channel[/bold] (the site's: {cc / 1e6:.4f})")
            elif cc is not None or dial is not None:
                lines.append(f"[bold]Control channel[/bold]  CC {(cc or dial or 0) / 1e6:.4f}")
            else:
                lines.append("[bold]Control channel[/bold]")
        elif kind == "voice":
            lines.append("[bold]Voice channel[/bold]")
        elif kind == "data":
            lines.append("[bold]Data channel[/bold]")
        else:
            lines.append("[dim]No P25 signal[/dim]")
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
                center = (span[0] + span[1]) / 2
                rate = (span[1] - span[0] + 2 * _HALF_CHANNEL_HZ) / (2 * _USABLE)
                lines.append(f"[yellow]site {span[0] / 1e6:.4f}-{span[1] / 1e6:.4f}[/yellow]")
                lines.append(
                    f"[yellow]p25 fit: {center / 1e6:.4f}, {rate / 1e6:.1f} MS/s[/yellow]"
                    if side
                    else f"[yellow]center {center / 1e6:.4f}, {rate / 1e6:.1f} MS/s or more "
                    "(p25 fit)[/yellow]"
                )
        return lines

    def _render_view(self, snap: P25Snapshot, width: int, shown: int) -> Group:
        """The view: every line down to the last row one line on screen (R52-1: a click's row is counted in lines),
        then the cursor row's detail, which may wrap."""
        state = self.state
        state.width, state.shown = width, shown
        view = VIEWS[state.view]
        wide = width >= 60
        follow_new = state.follow_new()
        extra = [] if follow_new is None else [f"new: {'follow' if follow_new else 'skip'}"]
        if view == "talkgroups":
            extra.append("all" if state.whole_system or snap.nac is None else f"NAC {snap.nac:03X}")
            known, _ = state.nac_system()
            if known is not None and state.system is None:
                # no broadcasts here: the rows are the system its NAC names (3.1b), said so
                extra.append(state.system_label(known))
        hint = "" if self.panel_focused else "  [dim]Tab: keys[/dim]"
        if wide:
            tabs = " ".join(
                f"[bold]{TITLES[v]}[/bold]" if v == view else f"[dim]{TITLES[v].lower()}[/dim]"
                for v in VIEWS
            )
            lines = [f"{tabs}  [dim]{' · '.join(extra)}[/dim]{hint}"]
        else:  # a side column: the view's title alone (←→ for the others), its settings below
            lines = [
                f"[dim]◂[/dim] [bold]{TITLES[view]}[/bold] [dim]▸[/dim]{hint}",
                f"[dim]{' · '.join(extra)}[/dim]",
            ]
        alert = state.outside_alert(wide)
        if alert is not None:
            lines.append(f"[yellow]{escape(alert[0])}[/yellow]")
            if alert[1]:
                lines.append(f"[yellow]{escape(alert[1])}[/yellow]")
        if snap.voice is not None:
            lines.append(_voice_line(snap.voice, snap.nac))
        if state.pending is not None:
            lines.append("[reverse] y to confirm [/reverse]")
        if snap.channel_kind in ("voice", "data"):
            lines.append("[dim]Following needs the control channel[/dim]")
        rows = state.rows()
        tail: list[str] = []
        if not rows:
            _, why = state.nac_system() if view == "talkgroups" else (None, None)
            self._view_rows = (len(lines), ())
            # the view's last line: it may wrap (R53-6)
            return _lines(lines, [f"[dim]{escape(why) if why else 'none yet'}[/dim]"])
        cursor = state.cursor(rows)
        assert cursor is not None
        first = max(0, min(cursor - shown // 2, len(rows) - shown))
        drawn = rows[first : first + shown]
        self._view_rows = (len(lines), tuple(r.key for r in drawn))
        for i in range(first, first + len(drawn)):
            row = rows[i]
            text = row.build(width)
            text = f"[{FOLLOWED}]{text}[/{FOLLOWED}]" if row.followed else f"[dim]{text}[/dim]"
            if self.panel_focused and i == cursor:
                text = f"[reverse]{text}[/reverse]"
            lines.append(text)
        if self.panel_focused:
            tail.append(f"[italic]{rows[cursor].detail()}[/italic]")
        if len(rows) > shown:
            tail.append(f"[dim]{cursor + 1}/{len(rows)}[/dim]")
        return _lines(lines, tail)

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

    def _surface_width(self) -> int:
        """The width the render is drawn in: the inner surface's (the panel's less a scrollbar)."""
        try:
            width = self.query_one(_SectionSurface).size.width
        except NoMatches:
            width = 0
        return int(width or self.size.width)

    _system_lines = 0


# render helpers


def _lines(lines: list[str], wrapping: list[str] = ()) -> Group:  # type: ignore[assignment]
    """`lines` one screen line each (cropped with an ellipsis, never wrapped), then `wrapping` as they come."""
    texts = []
    for line in lines:
        text = Text.from_markup(line, overflow="ellipsis")
        text.no_wrap = True
        texts.append(text)
    return Group(*texts, *(Text.from_markup(line) for line in wrapping))


def _voice_line(call: P25Voice, channel_nac: int | None) -> str:
    """The call heard (at the dial, or followed): muted if encrypted; its NAC when not the channel's."""
    who = f"TG {call.group}" if call.group is not None else "voice"
    if call.target is not None:
        who = f"unit {call.target}"
    if channel_nac is not None and call.nac != channel_nac:
        who += f" NAC {call.nac:03X}"
    flags = (" [red]ENC muted[/red]" if call.encrypted else "") + (
        " [bold red]EMERG[/bold red]" if call.emergency else ""
    )
    return f"[green bold]▶ {who}[/green bold]{flags}"
