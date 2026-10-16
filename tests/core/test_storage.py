"""Saved text files the app reads back at startup or on a key (histories)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsdr.core import storage


def test_a_history_that_is_not_utf8_still_reads(tmp_path: Path) -> None:
    # A Latin-1 edit, or a damaged file: `f` + Up ended the app with UnicodeDecodeError.
    (tmp_path / "frequency_history").write_bytes(b"146.52\ncaf\xe9\n7074k\n")
    lines = storage.read_text("frequency_history").splitlines()
    assert lines[0] == "146.52" and lines[2] == "7074k"


def test_a_store_that_is_not_utf8_keeps_its_entries(tmp_path: Path) -> None:
    # A memory named in Latin-1 stopped tsdr from starting (UnicodeDecodeError); skipping
    # the file would lose every other memory, and the next save would write it empty.
    (tmp_path / "memories.toml").write_bytes(
        b'[[memory]]\nname = "Caf\xe9"\nfrequency = 146520000\n'
        b'[[memory]]\nname = "Repeater"\nfrequency = 147360000\n'
    )
    memories = storage.load_toml("memories.toml")["memory"]
    assert [m["frequency"] for m in memories] == [146520000, 147360000]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32", "latin-1"])
def test_a_bandplan_in_another_encoding_loads(tmp_path: Path, encoding: str) -> None:
    # JSON's own encodings (a BOM, UTF-16, UTF-32) as json.load read them; Latin-1 as damaged.
    (tmp_path / "plan.json").write_bytes('{"name": "Café", "bands": []}'.encode(encoding))
    assert storage.load_json("plan.json")["bands"] == []
