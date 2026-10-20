"""The key list, as a panel (`?`): the same table the `keys` command prints."""

from __future__ import annotations

from itertools import groupby

from rich.table import Table
from rich.text import Text

from tsdr.tui.keybindings import BINDINGS, Context, format_keys
from tsdr.tui.widgets.p25_panel import OUTCOME_LEGEND
from tsdr.tui.widgets.section_panel import Section, SectionPanel


def key_rows() -> list[tuple[str, list[tuple[str, str]]]]:
    """(category, [(keys, description)]) for the keys that work outside the console: the global
    ones, and the P25 panel's while it has the keys."""
    rows = [b for b in BINDINGS if b.context in (Context.GLOBAL, Context.P25_PANEL)]
    return [
        (category, [(format_keys(b), b.description) for b in group])
        for category, group in groupby(rows, key=lambda b: b.category)
    ]


class KeysWidget(SectionPanel):
    def on_mount(self) -> None:
        self.refresh()

    def build_sections(self) -> list[Section]:
        sections: list[Section] = []
        for category, rows in key_rows():
            # Descriptions wrap within their own column, beside the keys.
            table = Table.grid(padding=(0, 2))
            table.add_column(no_wrap=True, style="cyan")
            table.add_column()
            for keys, description in rows:
                table.add_row(Text(keys), Text(description))
            sections.append(Section(table, title=f"[bold]{category}:[/bold]", min_width=40))
        # The P25 call log's outcome tags (DESIGN-M4b 10.1 item 4)
        legend = Table.grid(padding=(0, 2))
        legend.add_column(no_wrap=True, style="cyan")
        legend.add_column()
        for tag, meaning in OUTCOME_LEGEND:
            legend.add_row(Text(tag), Text(meaning))
        sections.append(Section(legend, title="[bold]P25 call log:[/bold]", min_width=40))
        return sections
