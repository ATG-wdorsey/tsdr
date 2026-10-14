"""Saved text files the app reads back at startup or on a key (histories)."""

from __future__ import annotations

from pathlib import Path

from tsdr.core import storage


def test_a_history_that_is_not_utf8_still_reads(tmp_path: Path) -> None:
    # A Latin-1 edit, or a damaged file: `f` + Up ended the app with UnicodeDecodeError.
    (tmp_path / "frequency_history").write_bytes(b"146.52\ncaf\xe9\n7074k\n")
    lines = storage.read_text("frequency_history").splitlines()
    assert lines[0] == "146.52" and lines[2] == "7074k"
