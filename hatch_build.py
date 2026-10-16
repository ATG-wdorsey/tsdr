"""Wheel build hook: record which commit a wheel was built from, as `tsdr/_build.py`.

A wheel built at a release tag (`vX.Y.Z`, nothing uncommitted) shows its version alone; one built
anywhere else shows how far past the last release it is, as a PEP 440 local label:
`0.2.0+106.ga767567` (106 commits past v0.2.0, at a767567), `.dirty` with uncommitted changes
(or new files under `src` not yet added, which the wheel would carry). Without git or a release
tag it records nothing. Editable installs get nothing: they run the source, whose label changes.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _git(root: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()


def local_label(root: str) -> str:
    try:
        described = _git(root, "describe", "--tags", "--long", "--dirty", "--match", "v[0-9]*")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", "src")
        dirty = described.endswith("-dirty") or bool(untracked)
        _tag, ahead, commit = described.removesuffix("-dirty").rsplit("-", 2)
    except OSError, subprocess.SubprocessError, ValueError:
        return ""
    if ahead == "0" and not dirty:
        return ""
    return f"+{ahead}.{commit}" + (".dirty" if dirty else "")


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        self._dir = tempfile.TemporaryDirectory()
        path = Path(self._dir.name) / "_build.py"
        path.write_text(f'LOCAL = "{local_label(self.root)}"\n')
        build_data["force_include"][str(path)] = "tsdr/_build.py"

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if hasattr(self, "_dir"):
            self._dir.cleanup()
