"""Operating keys that use the status line: `f` types a frequency there, `D` picks a decoder,
`?` shows the key list."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from textual.content import Content
from textual.pilot import Pilot

from tsdr.core.sdr.engine import get_engine
from tsdr.core.units import parse_dial
from tsdr.tui.app import TSDRApp
from tsdr.tui.keybindings import DECODER_CHORD_MODES, DEMOD_CHORD_MODES, chord_choice
from tsdr.tui.widgets.keys_widget import KeysWidget, key_rows
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
    ],
)
def test_a_bare_number_is_mhz_and_suffixes_any_case(typed: str, hz: int) -> None:
    assert parse_dial(typed) == hz


@pytest.mark.parametrize("typed", ["", "abc", "146.52x", "nan", "inf", "k"])
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


def test_escape_leaves_the_dial_alone_and_up_recalls_an_entry() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        await pilot.press("f", *"162.55", "enter")
        await pilot.pause()
        assert _dial() == 162_550_000
        await pilot.press("f", *"118.6", "escape")
        await pilot.pause()
        assert _dial() == 162_550_000
        await pilot.press("f", "up")
        await pilot.pause()
        assert app.query_one(StatusPrompt).value == "162.55"
        await pilot.press("ctrl+u", *"121.5M", "enter")
        await pilot.pause()
        assert _dial() == 121_500_000

    _run(body)


def test_a_bad_entry_is_reported_and_the_dial_stays() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        errors: list[str] = []
        app._show_error = errors.append  # type: ignore[method-assign]
        before = _dial()
        await pilot.press("f", *"14x", "enter")
        await pilot.pause()
        assert app.is_running
        assert _dial() == before
        assert errors and "Not a frequency: 14x" in errors[-1]

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


def test_another_key_cancels_the_decoder_chord() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        await pilot.press("D", "x")
        await pilot.pause()
        assert _mode() is None
        # The next key is a shortcut again.
        await pilot.press("D", "o", "d", "n")
        await pilot.pause()
        assert _mode() == "NFM"

    _run(body)


@pytest.mark.parametrize("modes", [DECODER_CHORD_MODES, DEMOD_CHORD_MODES])
def test_each_choice_shows_its_key_in_place_emphasized(modes: dict[str, str]) -> None:
    for key, mode in modes.items():
        content = Content.from_markup(chord_choice(key, mode))
        emphasized = [
            content.plain[span.start : span.end]
            for span in content.spans
            if "bold" in str(span.style)
        ]
        assert emphasized == [key.upper()], (key, mode)


def test_the_choices_take_a_second_line_when_the_screen_is_narrow() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        await pilot.press("D")
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.region.height >= 2
        lines = ["".join(seg.text for seg in app.screen.get_widget_at(0, 0)[0].render_line(0))]
        assert lines  # rendered
        text = str(status.render())
        for name in ("ads-b", "tetra", "ft8", "off"):
            assert name in text.lower()
        await pilot.press("escape")
        await pilot.pause()
        assert status.region.height == 1

    _run(body, size=(70, 40))


def test_the_choices_fit_on_one_line_when_wide() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        await pilot.press("D")
        await pilot.pause()
        assert app.query_one(StatusBar).region.height == 1

    _run(body, size=(160, 40))


def test_question_mark_shows_and_hides_the_key_list() -> None:
    async def body(app: TSDRApp, pilot: Pilot) -> None:
        assert not app.query(KeysWidget)
        await pilot.press("question_mark")
        await pilot.pause()
        assert app.query_one(KeysWidget).build_sections()
        listed = {description for _, rows in key_rows() for _, description in rows}
        assert any(d.startswith("Type a frequency") for d in listed)
        assert any(d.startswith("Switch decoder") for d in listed)
        assert any(d.startswith("Show / hide this list") for d in listed)
        await pilot.press("question_mark")
        await pilot.pause()
        assert not app.query(KeysWidget)

    _run(body)
