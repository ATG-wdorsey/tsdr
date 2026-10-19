"""P25 domain-specific display widget.

Reads a P25Snapshot payload off DecoderOutputEvent and renders three columns: the system and
site, the call heard at the dial and the latest calls granted, and signal quality.
"""

from __future__ import annotations

from tsdr.core.events.events import DecoderOutputEvent, StatsUpdateEvent
from tsdr.radio.decoders.p25.decoder import P25Call, P25Snapshot, P25Voice
from tsdr.tui.widgets.section_panel import Section, SectionPanel

_CALL_ROWS = 5
# A call whose grants came within this long of the snapshot is shown as active.
_ACTIVE_S = 3.0


class P25Widget(SectionPanel):
    """Three-column P25 control channel display.

    Col 1: System (NAC, system ID, WACN, RFSS and site, adjacent sites)
    Col 2: Latest calls (talkgroup, frequency and slot, ENC/EMERG)
    Col 3: Signal quality (lock, TSBK pass rate, messages, frequency offset, SNR)
    """

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: P25Snapshot | None = None
        self._snr_db: float | None = None

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

    # render

    def build_sections(self) -> list[Section]:
        snap = self._snapshot
        if snap is None:
            return [Section("Waiting...")]
        return [
            Section(self._render_system(snap), width=20, min_width=16),
            Section(self._render_calls(snap), width=34, min_width=26),
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
        return "\n".join(lines)

    @staticmethod
    def _render_calls(snap: P25Snapshot) -> str:
        lines = ["[bold]CALLS[/bold]"]
        if snap.voice is not None:
            lines.append(_voice_line(snap.voice, snap.nac))
        if not snap.calls and snap.voice is None:
            lines.append("[dim]none yet[/dim]")
        for call in snap.calls[:_CALL_ROWS]:
            lines.append(_call_line(call, snap.time))
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
    """The call at the dial: heard now, muted if encrypted; its NAC when not the channel's."""
    who = f"TG {call.group}" if call.group is not None else "voice"
    if call.target is not None:
        who = f"unit {call.target}"
    if call.nac != channel_nac:
        who += f" NAC {call.nac:03X}"
    flags = (" [red]ENC muted[/red]" if call.encrypted else "") + (
        " [bold red]EMERG[/bold red]" if call.emergency else ""
    )
    return f"[green bold]▶ {who}[/green bold]{flags}"


def _call_line(call: P25Call, now: float) -> str:
    if call.frequency_hz is not None:
        where = f"{call.frequency_hz / 1e6:.4f}"
        if call.slot is not None:
            where += f"/{call.slot}"
    else:
        where = "[dim]unknown ch[/dim]"
    if call.svcopts is None:
        flags = " [dim]?[/dim]"  # no grant seen yet: encryption unknown, not clear
    else:
        flags = ("[red] ENC[/red]" if call.encrypted else "") + (
            " [bold red]EMERG[/bold red]" if call.emergency else ""
        )
    group = f"TG {call.group}"
    if 0 <= now - call.last_seen <= _ACTIVE_S:
        group = f"[green bold]{group}[/green bold]"
    return f"{group} {where}{flags}"
