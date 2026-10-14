#!/usr/bin/env python3
"""Check that every logger.<level>(...) call in src/tsdr/ uses a stable
`snake_case_event_name` as the first token of its message string (a pre-commit hook).

See `## Logging` in CLAUDE.md for the convention.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "tsdr"

EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

LOG_LEVELS = {"debug", "info", "warning", "error", "critical", "exception", "log"}


def _iter_log_calls(tree: ast.AST):
    """Yield (level, msg_node) for every logger.<level>(msg, ...) call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in LOG_LEVELS:
            continue
        # Match logger.* and self._<...>logger.* — accept any attribute access
        # whose final name matches a log level. False positives (e.g.
        # spec.debug(...)) are unlikely in this codebase.
        value = func.value
        if isinstance(value, ast.Name) and not value.id.endswith("logger"):
            # Skip non-logger calls (helpers like _log_once)
            continue
        # First arg is the message; for logger.log it's the second arg.
        args = node.args
        if func.attr == "log":
            if len(args) < 2:
                continue
            msg_node = args[1]
        else:
            if not args:
                continue
            msg_node = args[0]
        yield func.attr, msg_node


def _extract_first_token(msg_node: ast.AST) -> str | None:
    """Return the first whitespace-delimited token of the message string, or None."""
    if isinstance(msg_node, ast.Constant) and isinstance(msg_node.value, str):
        text = msg_node.value
    elif isinstance(msg_node, ast.JoinedStr):
        # f-string: take the literal head if any.
        if not msg_node.values:
            return None
        head = msg_node.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            text = head.value
        else:
            return None
    else:
        return None
    parts = text.split(None, 1)
    return parts[0] if parts else None


def _collect_names() -> tuple[set[str], list[tuple[Path, int, str]]]:
    """Walk src/tsdr/, collect event names, return (names, violations)."""
    names: set[str] = set()
    violations: list[tuple[Path, int, str]] = []

    for path in sorted(SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for _level, msg_node in _iter_log_calls(tree):
            token = _extract_first_token(msg_node)
            if token is None:
                # Dynamic message (e.g. logger.log(level, msg, ...) where msg is a
                # variable). Skip.
                continue
            if not EVENT_NAME_RE.match(token):
                violations.append((path, msg_node.lineno, token))
                continue
            names.add(token)
    return names, violations


def main() -> int:
    _, violations = _collect_names()
    for path, lineno, token in violations:
        print(
            f"{path.relative_to(SRC_ROOT.parent.parent)}:{lineno}: log event name not snake_case: {token!r}"
        )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
