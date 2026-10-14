"""A one-line prompt on the status line, vi style (`f` types a frequency there).

It edits like the console (the same readline keys) with a history of its own, and
no command completion or highlighting. It is always mounted, over the status bar;
`derive_tree` shows it (class `active`) while `UIModel.status_prompt` names a prompt.
The app owns the keyboard: `KeyboardMixin` hands it every key while it is shown.
"""

from __future__ import annotations

from rich.style import Style
from textual import events
from textual.reactive import reactive

from tsdr.tui.console.history import CommandHistory
from tsdr.tui.console.terminal_input import TerminalInput

_LABEL_STYLE = Style(color="white", dim=True)


class StatusPrompt(TerminalInput):
    can_focus = False

    label: reactive[str] = reactive("")

    def __init__(self) -> None:
        super().__init__()
        self._histories: dict[str, CommandHistory] = {}

    def watch_label(self, label: str) -> None:
        self.prompt_segments = [(label, _LABEL_STYLE)]

    @property
    def _history(self) -> CommandHistory:
        # One history per prompt, kept in the config dir beside the console's.
        name = self.label.strip().rstrip(":").lower() or "prompt"
        history = self._histories.get(name)
        if history is None:
            history = self._histories[name] = CommandHistory(f"{name}_history")
        return history

    def open(self) -> None:
        self._history_nav = None
        self.value = ""
        self.active = True

    def close(self) -> str:
        """Leave the prompt; return what was typed."""
        value = self.value
        self.active = False
        self._history_nav = None
        self.value = ""
        return value

    def remember(self, line: str) -> None:
        self._history.add(line)

    def edit_key(self, event: events.Key) -> None:
        """Edit the line with one key (Enter and Escape are the app's to handle)."""
        if event.key in ("up", "ctrl+p"):
            self.history_up()
        elif event.key in ("down", "ctrl+n"):
            self.history_down()
        else:
            self._on_key(event)

    def _update_suggestion(self) -> None:
        self._suggestion = ""
        self._suggestion_source = ""

    def _spans_for(self, text: str) -> list[tuple[str, Style]]:
        return [(text, Style())]
