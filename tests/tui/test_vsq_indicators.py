"""VSQ's indicators: the meter's squelch label, the SEARCH badge, the denoiser
tag and the Stats row. Text and linkage only."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from rich.style import Style

from tsdr.core.sdr.search import SearchConfig, SearchStatus
from tsdr.core.sdr.voice_squelch import VsqStatus
from tsdr.radio.voice.detector import Classification
from tsdr.tui.widgets.snr_widget import SNRWidget
from tsdr.tui.widgets.stats_widget import StatsWidget
from tsdr.tui.widgets.tuner_widget import TunerWidget, _search_lines

DIAL = 146.52e6
SEARCH = SearchConfig(low_hz=146e6, high_hz=147e6, step_hz=12_500.0)


def test_the_meter_reads_vsq_while_vsq_runs() -> None:
    meter = SNRWidget()
    meter.update_squelch(-82.0, True)

    def text() -> str:
        return "".join(s.text for s in meter._squelch_segments(Style())).strip()

    assert text() == "SQ -82 ●"
    meter.update_vsq(True, False)
    assert text() == "VSQ -82 ●"
    meter.update_vsq(True, True)  # not voice: muted though the carrier holds the squelch open
    assert text() == "VSQ -82 ○"


def test_a_hold_vsq_cut_short_shows_vsq_with_its_countdown() -> None:
    assert "HOLD" in _search_lines(SearchStatus("holding", DIAL, -62.0, 2.0, 1, "listening"))[0]
    badge, _ = _search_lines(SearchStatus("holding", DIAL, -62.0, 1.5, 1, "not_voice"))
    assert "VSQ" in badge and "1.5s" in badge and "HOLD" not in badge


@pytest.mark.parametrize(
    ("search", "voice", "expected"),
    [
        (replace(SEARCH, voice_squelch=True), "not_voice", (True, True)),
        (replace(SEARCH, voice_squelch=True), "listening", (True, False)),
        (SEARCH, None, (False, False)),
        (None, None, (False, False)),
    ],
)
def test_the_tuner_tells_the_meter_when_vsq_runs_and_mutes(
    monkeypatch: pytest.MonkeyPatch, search, voice, expected
) -> None:
    calls: list[tuple[bool, bool]] = []
    tuner = TunerWidget()
    meter = SimpleNamespace(update_vsq=lambda active, muted: calls.append((active, muted)))
    monkeypatch.setattr(tuner, "query_one", lambda *_args, **_kw: meter)
    tuner._search_status = SearchStatus("holding", DIAL, -62.0, 1.0, 1, voice)
    tuner._update_vsq(search)
    assert calls == [expected]


def test_the_volume_line_tags_the_denoiser() -> None:
    tuner = TunerWidget()
    tuner._volume, tuner._denoise = 0.5, True
    assert tuner._vol_line().startswith("VOL 50%") and "DN" in tuner._vol_line()
    tuner._denoise = False
    assert "DN" not in tuner._vol_line()


def test_stats_show_vsq_and_the_detectors_last_label() -> None:
    stats = StatsWidget()
    assert "off" in stats._vsq_line()
    stats._vsq = VsqStatus("listening", 2.5, Classification(0.19))
    assert all(part in stats._vsq_line() for part in ("listening", "2.5s", "not speech", "0.19"))
    stats._vsq = VsqStatus("not_voice", 6.0, None)
    assert "not voice" in stats._vsq_line()
