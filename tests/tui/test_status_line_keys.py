"""Operating keys that use the status line: `f` types a frequency there, `D` picks a decoder,
`?` shows the key list."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from textual.pilot import Pilot

from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import parse_dial
from tsdr.tui.app import TSDRApp
from tsdr.tui.widgets.status_bar import StatusBar
from tsdr.tui.widgets.status_prompt import StatusPrompt


@pytest.mark.parametrize(
    ("typed", "hz"),
    [
        ("146.52", 146_520_000),
        ("146.52M", 146_520_000),
        ("146.52m", 146_520_000),
        (" 146.52 mhz ", 146_520_000),
        ("7074k", 7_074_000),
        ("7074K", 7_074_000),
        ("1.09G", 1_090_000_000),
        ("1.09g", 1_090_000_000),
        ("162550000hz", 162_550_000),
        ("929.6125", 929_612_500),
        ("144.", 144_000_000),
        ("2.01M", 2_010_000),  # 2.01e6 is 2009999.9999999998: rounded, not truncated
    ],
)
def test_a_bare_number_is_mhz_and_suffixes_any_case(typed: str, hz: int) -> None:
    assert parse_dial(typed) == hz


@pytest.mark.parametrize(
    "typed", ["", "abc", "146.52x", "nan", "inf", "k", "1e303", "9" * 303, "1e302k", "2e6G"]
)
def test_what_is_not_a_frequency_is_refused(typed: str) -> None:
    with pytest.raises(ValueError):
        parse_dial(typed)


def _run(body: Callable[[TSDRApp, Pilot], Awaitable[None]], size: tuple[int, int] = (160, 40)):
    async def go() -> None:
        app = TSDRApp(startup_commands=["add m --type mock"])
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(go())


def _dial() -> float:
    return get_engine().devices["m"].config.tuned_frequency


def _on_status_line(app: TSDRApp) -> object:
    """The widget the user sees on the status line (the topmost one there)."""
    status = app.query_one(StatusBar)
    widget, _ = app.screen.get_widget_at(status.region.x + 2, status.region.y)
    return widget


def test_f_types_a_frequency_on_the_status_line_and_enter_tunes() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        span = get_engine().devices["m"].config.spectrum_span
        await pilot.press("f")
        await pilot.pause()
        prompt = app.query_one(StatusPrompt)
        assert _on_status_line(app) is prompt
        # Typed keys edit the line; none reaches a shortcut (`k` narrows the span).
        await pilot.press(*"146.53", "backspace", "2", "k")
        await pilot.pause()
        assert prompt.value == "146.52k"
        await pilot.press("backspace", "enter")
        await pilot.pause()
        assert _dial() == 146_520_000
        assert get_engine().devices["m"].config.spectrum_span == span
        assert _on_status_line(app) is app.query_one(StatusBar)

    _run(body)


def _mode() -> str | None:
    audio = get_engine().devices["m"].config.pipelines.get("audio")
    if audio is None or audio.audio_spec is None:
        return None
    return audio.audio_spec.mode


@pytest.mark.parametrize(("second", "mode"), [("d", "DMR"), ("D", "DMR"), ("p", "APRS")])
def test_shift_d_then_a_key_picks_a_decoder(second: str, mode: str) -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        await pilot.press("D", second)
        await pilot.pause()
        assert _mode() == mode

    _run(body)
