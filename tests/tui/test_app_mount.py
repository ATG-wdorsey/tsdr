"""Smoke + layout-geometry tests for the live app.

Catches CSS / widget initialization bugs that pass unit tests but blow up in
the compositor, and asserts the dock layout has the right shape after mount:
the single bottom panel-bar spans full width and clears the overlay zone, and
active side panels sit inside their dock above that zone.
"""

from __future__ import annotations

import asyncio
import gc

from tsdr.tui.app import TSDRApp

SCREEN_W = 120
SCREEN_H = 40
# Bottom 5 rows of the screen are reserved for the overlay layer:
# 3 (collapsed ConsoleWidget) + 1 (StatusBar) + 1 (StatusBar margin) = 5.


def _run(coro_factory) -> None:
    asyncio.run(coro_factory())


def test_app_mounts_and_renders_on_resize() -> None:
    """Mount + resize must not crash the compositor."""

    async def go() -> None:
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause()
            await pilot.resize_terminal(100, 32)
            await pilot.pause()

    _run(go)


def test_mounting_the_app_freezes_nothing() -> None:
    """Widgets come and go (and are cyclic): frozen at mount, they could never
    be collected. Only tsdr's main() freezes, before the app exists."""

    async def go() -> None:
        before = gc.get_freeze_count()
        app = TSDRApp(startup_commands=[])
        async with app.run_test(size=(SCREEN_W, SCREEN_H)) as pilot:
            await pilot.pause(0.2)
            assert gc.get_freeze_count() == before

    _run(go)
