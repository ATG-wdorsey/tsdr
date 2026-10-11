"""AudioOutputWorker on backends that buffer in the device (PulseAudio, WASAPI).

SoundCard sizes their device buffer from the player's blocksize and blocks in
play() until there is room. A 512-frame block left ~10 ms of headroom, so any
longer stall of the worker thread (a GC pause, the UI holding the GIL) was an
audible gap: the TUI at 1 MS/s dropped audio several times per second.
"""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core.sdr.workers import audio_worker as audio_worker_module
from tsdr.core.sdr.workers.audio_worker import AudioOutputWorker

RATE = AudioOutputWorker.TARGET_RATE


@pytest.fixture(autouse=True)
def _device_buffer_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_worker_module, "_CLIENT_QUEUE_BACKEND", False)


class _DevicePlayer:
    """A device buffer of `capacity` frames that starts playing in real time
    once it first fills (PulseAudio's prebuffer) and frees room a `quantum`
    at a time; play() blocks until there is room. `stalls` delays the given
    calls (1-based) although there may be room: a GC pause, the GIL. Waits
    busy-wait, since tests patch time.sleep to observe the worker's pacing."""

    def __init__(
        self, capacity_frames: int, *, quantum: int = 512, stalls: dict[int, float] | None = None
    ) -> None:
        self.capacity = capacity_frames
        self.quantum = quantum
        self.stalls = stalls or {}
        self.queued = 0
        self.playing_since: float | None = None
        self.calls: list[int] = []

    def _played(self) -> int:
        if self.playing_since is None:
            return 0
        frames = int((time.perf_counter() - self.playing_since) * RATE)
        return min(self.queued, frames // self.quantum * self.quantum)

    def play(self, data) -> None:
        self.calls.append(len(data))
        until = time.perf_counter() + self.stalls.get(len(self.calls), 0.0)
        while time.perf_counter() < until:
            pass
        while self.queued - self._played() + len(data) > self.capacity:
            pass
        self.queued += len(data)
        if self.playing_since is None and self.queued + 512 > self.capacity:
            self.playing_since = time.perf_counter()


class _Speaker:
    def __init__(self, player: _DevicePlayer) -> None:
        self.id = "fake"
        self.name = "fake"
        self._player = player
        self.requested_blocksize: int | None = None

    def player(self, *, samplerate, channels, blocksize):
        self.requested_blocksize = blocksize
        player = self._player
        return SimpleNamespace(__enter__=lambda: player, __exit__=lambda *a: None)


def _open(
    monkeypatch: pytest.MonkeyPatch,
    capacity_s: float,
    *,
    quantum: int = 512,
    stalls: dict[int, float] | None = None,
) -> tuple[AudioOutputWorker, _Speaker]:
    player = _DevicePlayer(int(capacity_s * RATE), quantum=quantum, stalls=stalls)
    speaker = _Speaker(player)
    monkeypatch.setattr(
        audio_worker_module,
        "soundcard",
        SimpleNamespace(default_speaker=lambda: speaker, get_speaker=lambda spec: speaker),
    )
    w = AudioOutputWorker(source_id="d", audio_queue=queue.Queue())
    w._open_player()
    return w, speaker


def test_open_requests_a_deep_device_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    _, speaker = _open(monkeypatch, capacity_s=0.25)
    assert speaker.requested_blocksize is not None
    assert speaker.requested_blocksize / RATE >= audio_worker_module._MIN_DEVICE_BUFFER_SECONDS


def test_open_measures_granted_capacity_and_starts_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """PipeWire grants ~60% of the request; pacing must use what it really holds."""
    w, _ = _open(monkeypatch, capacity_s=0.24)
    assert w._device_buffer_seconds == pytest.approx(0.24, abs=3 * w.BLOCK_SIZE / RATE)
    assert w._prebuffered
    assert w._estimated_buffered_seconds() > 0.2


@pytest.mark.parametrize(
    "stalls",
    [
        pytest.param({2: 0.004}, id="one-slow-write"),
        pytest.param({1: 0.004, 2: 0.004}, id="two-slow-writes"),
        pytest.param({5: 0.12}, id="gc-pause"),
    ],
)
def test_slow_writes_while_priming_do_not_shrink_the_buffer(
    monkeypatch: pytest.MonkeyPatch, stalls: dict[int, float]
) -> None:
    """A stall on a write with room once ended priming early: 21 ms of
    cushion for the whole session (a real session measured 139 ms of 245)."""
    w, _ = _open(monkeypatch, capacity_s=0.24, stalls=stalls)
    assert w._device_buffer_seconds == pytest.approx(0.24, abs=3 * w.BLOCK_SIZE / RATE)


def test_a_server_that_frees_a_quantum_at_a_time_is_measured_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once full, PipeWire makes every 4th write wait, never several in a row."""
    w, _ = _open(monkeypatch, capacity_s=0.24, quantum=2048)
    assert w._device_buffer_seconds == pytest.approx(0.24, abs=(2048 + 512) / RATE)


def test_write_into_full_buffer_sleeps_instead_of_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.24)
    sleeps: list[float] = []
    monkeypatch.setattr(audio_worker_module.time, "sleep", sleeps.append)
    chunk = np.zeros((2400, 2), dtype=np.float32)  # 50 ms

    w._raw_push(chunk, max_depth=0)

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.05 - audio_worker_module._PACE_SLACK_SECONDS, abs=0.01)


def test_write_with_room_does_not_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.24)
    w._buffered_seconds = 0.05
    w._buffered_at = time.perf_counter()
    sleeps: list[float] = []
    monkeypatch.setattr(audio_worker_module.time, "sleep", sleeps.append)

    w._raw_push(np.zeros((2400, 2), dtype=np.float32), max_depth=0)

    assert sleeps == []


def test_underflow_counted_when_device_buffer_ran_dry(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.24)
    block = np.zeros((2400, 2), dtype=np.float32)
    w._push_block(block, max_depth=0)
    assert w._underflow_count == 0

    w._buffered_at -= 1.0  # the worker stalled for a second
    w._push_block(block, max_depth=0)
    assert w._underflow_count == 1


def _batch_queue(n: int) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    for _ in range(n):
        q.put(object())
    return q


def test_backlog_within_two_buffers_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.25)
    w.audio_queue = _batch_queue(9)  # 0.45 s of 50 ms batches
    w._trim_backlog(0.05)
    assert w.audio_queue.qsize() == 9
    assert w._drop_count == 0


def test_excess_backlog_is_trimmed_to_half_a_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    w, _ = _open(monkeypatch, capacity_s=0.25)
    w.audio_queue = _batch_queue(20)  # 1 s backlog after a long upstream stall
    w._trim_backlog(0.05)
    assert w.audio_queue.qsize() * 0.05 <= w._device_buffer_seconds / 2
    assert w._drop_count > 0
