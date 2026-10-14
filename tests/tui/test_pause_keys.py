"""Space pauses and resumes a running radio (and starts a stopped one). While paused,
keys and clicks that operate the radio are ignored; the display's own keys, the
panels and the console still work."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
from textual.pilot import Pilot

from tsdr.core.sdr.device_context import DeviceState
from tsdr.core.sdr.engine import get_engine
from tsdr.tui.app import TSDRApp
from tsdr.tui.model.store import get_ui_store
from tsdr.tui.pause import PAUSED_HINT
from tsdr.tui.widgets.keys_widget import KeysWidget
from tsdr.tui.widgets.spectrum_widget import SpectrumWidget


def _noise_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 1_000_000
    iq = ((rng.normal(size=n) + 1j * rng.normal(size=n)) * 1e-3).astype(np.complex64)
    path = tmp_path / "noise.cf32"
    iq.tofile(path)
    return path


def _run(tmp_path: Path, start: bool, body: Callable[[TSDRApp, Pilot], Awaitable[None]]) -> None:
    commands = [f"add f --type iq-file --path {_noise_file(tmp_path)} --frequency 146M"]
    if start:
        commands.append("start f")

    async def go() -> None:
        app = TSDRApp(startup_commands=commands)
        async with app.run_test(size=(160, 40)) as pilot:
            await _until(pilot, lambda: "f" in get_engine().devices)
            if start:
                await _until(pilot, lambda: _device().state == DeviceState.RUNNING)
            await body(app, pilot)
            get_engine().stop_device("f")

    asyncio.run(go())


async def _until(pilot: Pilot, predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert predicate()


def _device():
    return get_engine().devices["f"]


def _statuses(app: TSDRApp) -> list[str]:
    seen: list[str] = []
    show = app.show_status

    def record(message: str) -> None:
        seen.append(message)
        show(message)

    app.show_status = record  # type: ignore[method-assign]
    return seen


def test_space_pauses_and_operating_keys_and_clicks_are_ignored(tmp_path: Path) -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        seen = _statuses(app)
        # Running, a click on the spectrum tunes there.
        before = _device().config.tuned_frequency
        await pilot.click(SpectrumWidget, offset=(40, 5))
        await pilot.pause()
        assert _device().config.tuned_frequency != before

        await pilot.press("space")
        await pilot.pause()
        assert _device().paused and _device().state == DeviceState.RUNNING
        tuner_state = str(app.query_one("#tuner-state").render())
        assert "PAUSED" in tuner_state

        dial = _device().config.tuned_frequency
        await pilot.press("right", "f")
        await pilot.click(SpectrumWidget, offset=(100, 5))
        await pilot.pause()
        assert _device().config.tuned_frequency == dial
        assert get_ui_store().model.status_prompt is None  # `f` did not open
        assert seen[-1] == PAUSED_HINT

        # The display, the panels and the console still answer.
        span = _device().config.spectrum_span
        await pilot.press("k", "question_mark")
        await pilot.pause()
        assert _device().config.spectrum_span != span
        assert app.query(KeysWidget)

        await pilot.press("space")
        await pilot.pause()
        assert not _device().paused
        await pilot.press("right")
        await pilot.pause()
        assert _device().config.tuned_frequency != dial

    _run(tmp_path, True, body)


def test_space_starts_a_stopped_radio(tmp_path: Path) -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        assert _device().state == DeviceState.STOPPED
        await pilot.press("space")
        await _until(pilot, lambda: _device().state == DeviceState.RUNNING)
        assert not _device().paused

    _run(tmp_path, False, body)
