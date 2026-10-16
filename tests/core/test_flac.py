from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr import flac


def test_a_reader_stopped_by_an_error_raises_instead_of_hanging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After libFLAC aborts (a failure in the frame callback, or a read error), its
    process_single returns true and does nothing: the next read() looped forever, an
    iq-file device spun a core and tsdr couldn't exit. Every read raises now."""
    path = tmp_path / "x.flac"
    writer = flac.FlacWriter(path, sample_rate=250e3)
    writer.write(np.arange(2 * 50_000, dtype=np.int32) % 1000)
    writer.finish()

    def no_memory(self: flac.FlacReader, n: int) -> None:
        raise MemoryError("transient")

    reader = flac.FlacReader(path)
    monkeypatch.setattr(flac.FlacReader, "_reserve", no_memory)
    outcomes: list[str] = []

    def read_twice() -> None:
        for _ in range(2):
            try:
                reader.read(10_000)
                outcomes.append("read")
            except MemoryError:
                outcomes.append("raised")

    thread = threading.Thread(target=read_twice, daemon=True)
    thread.start()
    thread.join(10.0)
    assert not thread.is_alive(), "read() hung after the abort"
    assert outcomes == ["raised", "raised"]
    reader.close()
