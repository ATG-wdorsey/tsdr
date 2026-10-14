"""CLI dispatch: `doctor` subcommand must not break the bare/legacy invocation."""

from __future__ import annotations

from tsdr.__main__ import _build_parser


def test_bare_invocation_has_no_command() -> None:
    args = _build_parser().parse_args([])
    assert args.command is None


def test_legacy_exec_flag_still_works() -> None:
    args = _build_parser().parse_args(["-e", "start rtl0"])
    assert args.command is None
    assert args.startup_commands == ["start rtl0"]
