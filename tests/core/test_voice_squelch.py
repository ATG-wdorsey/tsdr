"""VSQ, the voice squelch for SEARCH: verdicts per transmission, the mute, and
the DenoiserStage that feeds it."""

from __future__ import annotations

import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core.sdr import voice_squelch
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.datatypes import AudioBatch, DemodStatus
from tsdr.core.sdr.io import load_iq
from tsdr.core.sdr.pipeline.stages.denoiser_stage import DenoiserStage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import SearchConfig
from tsdr.core.sdr.voice_squelch import DECIDE_S, LATE_S, VoiceSquelch
from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.voice.detector import Classification
from tsdr.radio.voice.frontend import FS

DIAL = 146.52e6
RATE = 50_000.0  # NFM's audio rate from a 1 MS/s capture
# Audio the fake stream tells apart by its mean; QUIET is a dead carrier (about -60 dBFS).
SPEECH, DATA, QUIET = 0.5, -0.5, 0.001
SAMPLES = Path(__file__).parents[1] / "samples"


class _FakeStream:
    """VoiceStream's schedule (every 0.5 s from 1 s) with a verdict from the mean of
    the last half second: speech above 0.25."""

    def __init__(self, net: object = None) -> None:
        self.times: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._audio = np.zeros(0, np.float32)
        self._next = 1.0

    def feed(self, a8: np.ndarray) -> list[tuple[float, Classification]]:
        self._audio = np.concatenate([self._audio, a8])
        out = []
        while len(self._audio) >= self._next * FS:
            recent = self._audio[int((self._next - 0.5) * FS) : int(self._next * FS)]
            out.append((self._next, Classification(0.9 if recent.mean() > 0.25 else 0.1)))
            self.times.append(self._next)
            self._next += 0.5
        return out


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> list[_FakeStream]:
    made: list[_FakeStream] = []

    def make(net: object = None) -> _FakeStream:
        made.append(_FakeStream())
        return made[-1]

    monkeypatch.setattr(voice_squelch, "VoiceStream", make)
    monkeypatch.setattr(voice_squelch, "detector", lambda: None)
    return made


def _feed(vsq: VoiceSquelch, value: float, seconds: float) -> None:
    for _ in range(round(seconds / 0.05)):
        vsq.feed(np.full(int(0.05 * RATE), value, dtype=np.float32), RATE, True)


def _vsq() -> VoiceSquelch:
    vsq = VoiceSquelch("dev")
    vsq.set_target((1, DIAL))
    return vsq


@pytest.mark.usefixtures("fake")
def test_speech_makes_a_transmission_voice() -> None:
    vsq = _vsq()
    _feed(vsq, SPEECH, 1.5)
    assert vsq.verdict_for(1, DIAL) == "voice" and not vsq.muted


@pytest.mark.usefixtures("fake")
def test_no_speech_by_the_decision_time_mutes_until_speech() -> None:
    vsq = _vsq()
    _feed(vsq, DATA, DECIDE_S - 0.4)
    assert vsq.verdict_for(1, DIAL) == "listening" and not vsq.muted
    _feed(vsq, DATA, 0.6)
    assert vsq.verdict_for(1, DIAL) == "not_voice" and vsq.muted
    _feed(vsq, SPEECH, 1.0)
    assert vsq.verdict_for(1, DIAL) == "voice" and not vsq.muted


@pytest.mark.usefixtures("fake")
def test_a_quiet_carrier_gets_until_the_late_time() -> None:
    # Someone keyed up and has not spoken yet: a late talker is kept.
    vsq = _vsq()
    _feed(vsq, QUIET, DECIDE_S + 0.6)
    assert vsq.verdict_for(1, DIAL) == "listening" and not vsq.muted
    _feed(vsq, SPEECH, 1.0)
    assert vsq.verdict_for(1, DIAL) == "voice"


@pytest.mark.usefixtures("fake")
def test_a_quiet_carrier_without_speech_is_let_go_at_the_late_time() -> None:
    vsq = _vsq()
    _feed(vsq, QUIET, LATE_S - 0.4)
    assert vsq.verdict_for(1, DIAL) == "listening"
    _feed(vsq, QUIET, 0.6)
    assert vsq.verdict_for(1, DIAL) == "not_voice" and vsq.muted


@pytest.mark.usefixtures("fake")
def test_quietness_is_judged_once_at_the_decision_time() -> None:
    # Loud data that starts after a quiet decision time still waits for LATE_S.
    vsq = _vsq()
    _feed(vsq, QUIET, DECIDE_S + 0.2)
    _feed(vsq, DATA, 1.0)
    assert vsq.verdict_for(1, DIAL) == "listening"
    _feed(vsq, DATA, 1.0)
    assert vsq.verdict_for(1, DIAL) == "not_voice"


@pytest.mark.usefixtures("fake")
def test_a_closed_squelch_ends_the_transmission() -> None:
    vsq = _vsq()
    _feed(vsq, DATA, DECIDE_S + 1.0)
    vsq.feed(np.zeros(2500, dtype=np.float32), RATE, False)
    assert vsq.verdict_for(1, DIAL) == "listening" and not vsq.muted


@pytest.mark.usefixtures("fake")
def test_a_verdict_belongs_to_one_run_and_channel() -> None:
    vsq = _vsq()
    _feed(vsq, DATA, DECIDE_S + 1.0)
    assert vsq.verdict_for(2, DIAL) is None and vsq.verdict_for(1, DIAL + 12_500) is None
    vsq.set_target((1, DIAL + 12_500))  # SEARCH moved on
    assert vsq.verdict_for(1, DIAL + 12_500) == "listening" and not vsq.muted
    vsq.set_target(None)
    assert vsq.status is None and not vsq.muted


def test_windows_are_scored_every_half_second_from_one_second(fake: list[_FakeStream]) -> None:
    vsq = _vsq()
    _feed(vsq, DATA, 3.05)  # the resampler's delay aside
    assert fake[-1].times == [1.0, 1.5, 2.0, 2.5, 3.0]


def _demodulated_noise(demod_class: type, seconds: float) -> np.ndarray:
    """What a demodulator plays with its squelch open on noise alone (no carrier)."""
    rng = np.random.default_rng(4)
    n = int(seconds * RATE)
    iq = (0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)
    demod = demod_class(sample_rate=RATE)
    out = []
    for i in range(0, n, 3072):
        demod.demodulate(iq[i : i + 3072], 0.0)
        out += [b.samples for b in demod.get_audio()]
    return np.concatenate(out).astype(np.float32)


@pytest.mark.parametrize(
    "audio",
    [
        pytest.param(
            lambda: _demodulated_noise(NarrowbandFMDemodulator, LATE_S + 1), id="nfm-noise"
        ),
        pytest.param(lambda: _demodulated_noise(AMDemodulator, LATE_S + 1), id="am-noise"),
        pytest.param(lambda: np.zeros(int((LATE_S + 1) * RATE), np.float32), id="dead-carrier"),
    ],
)
def test_noise_and_a_dead_carrier_are_not_voice(audio) -> None:
    vsq = _vsq()
    a = audio()
    for i in range(0, len(a), int(0.05 * RATE)):
        vsq.feed(a[i : i + int(0.05 * RATE)], RATE, True)
    assert vsq.verdict_for(1, DIAL) == "not_voice"


def test_weather_radio_is_voice() -> None:
    iq = load_iq(SAMPLES / "freq=162.4M_sr=25k_dur=3s_gain=40_noaa.cu8.zst").astype(np.complex64)
    demod = NarrowbandFMDemodulator(sample_rate=25_000.0)
    vsq = _vsq()
    for i in range(0, len(iq), 2048):
        demod.demodulate(iq[i : i + 2048], 0.0)
        for b in demod.get_audio():
            vsq.feed(b.samples, b.sample_rate, True)
    assert vsq.verdict_for(1, DIAL) == "voice"


def _process(
    *, denoise: bool, vsq_on: bool, seconds: float
) -> tuple[list[AudioBatch], VoiceSquelch]:
    vsq = VoiceSquelch("dev")
    stage = DenoiserStage(enabled=False)
    stage.on_config_change(SDRConfig(denoise=denoise))
    stage._set_denoise(False)  # the denoiser's flag set, RNNoise out: it would blank the test audio
    search = SearchConfig(146e6, 147e6, 12_500, voice_squelch=vsq_on, run=1)
    stage.on_config_change(DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, search=search))
    context = SimpleNamespace(
        audio_queue=queue.Queue(), device_context=SimpleNamespace(voice_squelch=vsq)
    )
    heard = []
    for _ in range(round(seconds / 0.05)):
        batch = AudioBatch(
            samples=np.full(int(0.05 * RATE), DATA, dtype=np.float32), sample_rate=RATE
        )
        stage.process(
            SamplesBatch(audio_batches=(batch,), demod_status=DemodStatus(squelch_open=True)),
            context,
        )
        while not context.audio_queue.empty():
            heard.append(context.audio_queue.get_nowait())
    return heard, vsq


@pytest.mark.usefixtures("fake")
@pytest.mark.parametrize("denoise", [False, True])
def test_vsq_mutes_a_transmission_without_speech_whatever_the_denoiser(denoise: bool) -> None:
    heard, vsq = _process(denoise=denoise, vsq_on=True, seconds=DECIDE_S + 1.0)
    assert np.all(heard[0].samples == DATA) and np.all(heard[-1].samples == 0.0)
    assert vsq.verdict_for(1, DIAL) == "not_voice"


def test_wfm_audio_is_not_classified(fake: list[_FakeStream]) -> None:
    # WFM sends two channels, flagged stereo only while its pilot is locked
    vsq = VoiceSquelch("dev")
    stage = DenoiserStage(enabled=False)
    search = SearchConfig(88e6, 108e6, 200_000, voice_squelch=True, run=1)
    stage.on_config_change(DeviceConfig(tuned_frequency=DIAL, center_frequency=DIAL, search=search))
    context = SimpleNamespace(audio_queue=None, device_context=SimpleNamespace(voice_squelch=vsq))
    two_channels = np.full((int(0.05 * RATE), 2), DATA, dtype=np.float32)
    for _ in range(round((DECIDE_S + 1.0) / 0.05)):
        batch = AudioBatch(samples=two_channels, sample_rate=RATE, stereo=False)
        stage.process(
            SamplesBatch(audio_batches=(batch,), demod_status=DemodStatus(squelch_open=True)),
            context,
        )
    assert fake == [] and vsq.verdict_for(1, DIAL) == "listening"


@pytest.mark.usefixtures("fake")
def test_without_vsq_nothing_is_muted() -> None:
    heard, vsq = _process(denoise=True, vsq_on=False, seconds=DECIDE_S + 1.0)
    assert np.all(heard[-1].samples == DATA) and vsq.status is None
