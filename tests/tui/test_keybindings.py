"""Binding-table integrity and table-driven dispatch."""

from __future__ import annotations

from tsdr.tui.keybindings import BINDINGS, build_lookup, format_keys


def test_lookup_builds_without_duplicates_or_gaps():
    assert build_lookup()


def test_every_binding_has_display_label():
    for binding in BINDINGS:
        assert format_keys(binding), binding.description
