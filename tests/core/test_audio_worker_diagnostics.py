"""Diagnostic preservation tests for the SoundCard-based AudioOutputWorker.

These exercise the worker's underflow detection, max-depth cap, default-follow,
and stall heuristics without touching real audio hardware. We drive a fake
``_Player._queue`` directly to simulate the audio thread's drain behavior.
"""

from __future__ import annotations

import logging
import queue
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tsdr.core.sdr.workers import audio_worker as audio_worker_module
from tsdr.core.sdr.workers.audio_worker import AudioOutputWorker


@pytest.fixture(autouse=True)
def _client_queue_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests model CoreAudio's client-side deque, whatever the host OS."""
    monkeypatch.setattr(audio_worker_module, "_CLIENT_QUEUE_BACKEND", True)


def _make_worker(output_device: str | None = None) -> AudioOutputWorker:
    w = AudioOutputWorker(source_id="rtl0", audio_queue=queue.Queue(), output_device=output_device)
    # Stand in a fake player with a real deque; no real audio thread.
    fake_queue: deque = deque()
    fake_player = SimpleNamespace(
        _queue=fake_queue, play=lambda data, wait=True: fake_queue.append(None)
    )
    w.player = fake_player  # type: ignore[assignment]
    return w


def test_no_underflow_during_prebuffer_phase(caplog: pytest.LogCaptureFixture) -> None:
    """While _prebuffered is False, depth==0 doesn't count as underflow — we're
    accumulating in _pending_blocks; the audio thread plays silence."""
    w = _make_worker()
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underflow_count == 0
    assert not w._prebuffered
    assert len(w._pending_blocks) == 1


def test_prebuffer_flushes_all_at_once_when_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When _pending_blocks reaches _pre_buffer_blocks, all are pushed in one
    burst and the worker transitions to steady-state."""
    w = _make_worker()
    w._pre_buffer_blocks = 4  # small for the test
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            w._push_block(block, max_depth=16)
    assert w._prebuffered
    assert w._pending_blocks == []
    assert len(w.player._queue) == 4  # type: ignore[union-attr]
    assert any("audio_prebuffered" in r.getMessage() for r in caplog.records)


def test_no_underflow_when_queue_non_empty(caplog: pytest.LogCaptureFixture) -> None:
    w = _make_worker()
    w._prebuffered = True
    w._push_count = w._pre_buffer_blocks + 5
    # Queue has stuff → no underflow.
    w.player._queue.extend([None] * 3)  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        w._push_block(block, max_depth=w._pre_buffer_blocks * 4)
    assert w._underflow_count == 0


def test_max_depth_cap_drops_oldest() -> None:
    w = _make_worker()
    w._prebuffered = True
    max_depth = 4
    # Pre-fill queue past the cap.
    w.player._queue.extend([f"old-{i}" for i in range(10)])  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    w._push_block(block, max_depth=max_depth)
    # After push: trim down to max_depth-1 oldest, then append this new push.
    assert len(w.player._queue) == max_depth  # type: ignore[union-attr]
    assert w._drop_count == 10 - (max_depth - 1)


def test_max_depth_no_drop_when_under_cap() -> None:
    w = _make_worker()
    w._prebuffered = True
    w.player._queue.extend([None] * 2)  # type: ignore[union-attr]
    block = np.zeros((w.BLOCK_SIZE, 2), dtype=np.float32)
    w._push_block(block, max_depth=10)
    assert w._drop_count == 0
    # Queue grew by 1 from the push, no drops.
    assert len(w.player._queue) == 3  # type: ignore[union-attr]


def test_maybe_follow_default_reopens_on_change(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = _make_worker(output_device=None)
    w._speaker_id = "old"
    fake_sc = SimpleNamespace(
        default_speaker=lambda: SimpleNamespace(id="new", name="new"),
    )
    monkeypatch.setattr(audio_worker_module, "soundcard", fake_sc)
    open_spy = MagicMock()
    close_spy = MagicMock()
    w._open_player = open_spy  # type: ignore[method-assign]
    w._close_player = close_spy  # type: ignore[method-assign]
    w._last_default_check = 0.0
    with caplog.at_level(logging.INFO):
        w._maybe_follow_default()
    close_spy.assert_called_once()
    open_spy.assert_called_once()
    assert any("audio_default_changed" in r.getMessage() for r in caplog.records)


def test_fractional_source_rate_bounds_resampler() -> None:
    """A GPS-corrected fractional rate (KiwiSDR: 12001.116 Hz) must not build a
    coprime up/down like 48000/12001 with a ~million-tap prototype filter."""
    w = _make_worker()
    audio = (np.random.randn(600, 2) * 0.1).astype(np.float32)
    out = w._resample_streaming(audio, 12001.116)
    assert w._resampler is not None
    assert w._resampler.up == 4
    assert w._resampler.down == 1
    # ~4x upsample toward 48 kHz, not the degenerate 48000/12001 ratio.
    assert out.shape[1] == 2
    assert 2200 <= out.shape[0] <= 2500


def test_standard_source_rate_ratio_preserved() -> None:
    """44.1 kHz must still resolve to the exact 160/147 ratio, unchanged."""
    w = _make_worker()
    audio = (np.random.randn(441, 2) * 0.1).astype(np.float32)
    w._resample_streaming(audio, 44100.0)
    assert w._resampler is not None
    assert (w._resampler.up, w._resampler.down) == (160, 147)
