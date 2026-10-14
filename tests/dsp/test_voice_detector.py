"""The voice detector (radio/voice): a small CNN on log-mel windows of demodulated audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.dsp._kernels import make_rational_resampler
from tsdr.radio.voice.detector import (
    FIRST_S,
    HOP_S,
    MODEL,
    Classification,
    VoiceStream,
    detector,
    load_model,
)
from tsdr.radio.voice.frontend import FS, HOP, N_MELS, WIN_FRAMES, logmel, normalize

SAMPLES = Path(__file__).parents[1] / "samples"


def _probabilities(sample: str, rate: float) -> list[float]:
    """tsdr's NFM demod of a sample, resampled to 8 kHz as VSQ does, scored as a
    transmission from its first sample."""
    iq = load_iq(SAMPLES / sample).astype(np.complex64)
    demod = NarrowbandFMDemodulator(sample_rate=rate, channel_bandwidth=12_500.0)
    out = []
    for i in range(0, len(iq), 2048):
        demod.demodulate(iq[i : i + 2048], 0.0)
        out.extend(b.samples for b in demod.get_audio())
    audio = np.concatenate(out).astype(np.float32).reshape(-1, 1)
    a8 = make_rational_resampler(FS, demod.decimated_rate).process(audio)[:, 0]
    return [c.probability for _, c in VoiceStream().feed(a8)]


def test_weather_radio_speech_is_speech() -> None:
    p = _probabilities("freq=162.4M_sr=25k_dur=3s_gain=40_noaa.cu8.zst", 25_000.0)
    assert len(p) >= 4 and all(x >= 0.5 for x in p)


def test_aprs_packets_are_rarely_speech() -> None:
    # Unsquelched and scored as one 22 s transmission: the windows mix packets with
    # the hiss between them (two of 42 read 0.50-0.58). Under SEARCH each packet is
    # its own transmission, over before the first window at 1 s.
    p = _probabilities("freq=144.8M_sr=30k_dur=22s_gain=0_aprs.cf32.zst", 30_000.0)
    assert len(p) > 30 and sum(x >= 0.5 for x in p) <= len(p) // 10
    assert np.median(p) < 0.05


def test_hiss_a_tone_and_silence_are_not_speech() -> None:
    t = np.arange(3 * FS) / FS
    for audio in (
        np.random.default_rng(1).normal(scale=0.1, size=3 * FS),
        0.3 * np.sin(2 * np.pi * 1000 * t),
        np.zeros(3 * FS),
    ):
        assert not any(c.is_speech for _, c in VoiceStream().feed(audio))


def test_numpy_network_matches_the_trained_one() -> None:
    # voice.npz carries reference windows and the training framework's logits for them.
    with np.load(MODEL) as m:
        x, expected = m["check_x"], m["check_logit"]
    got = np.array([detector().logit(w) for w in x])
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_a_model_for_another_front_end_is_refused(tmp_path: Path) -> None:
    with np.load(MODEL) as m:
        arrays = dict(m)
    arrays["n_mels"] = np.array(N_MELS + 8)
    path = tmp_path / "other.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="n_mels"):
        load_model(path)


def test_stream_scores_the_same_windows_as_framing_it_all_at_once() -> None:
    rng = np.random.default_rng(3)
    audio = rng.normal(scale=0.1, size=int(4.3 * FS)).astype(np.float32)
    stream = VoiceStream()
    got: list[tuple[float, Classification]] = []
    for i in range(0, len(audio), 333):  # chunks unrelated to the frame hop
        got += stream.feed(audio[i : i + 333])
    frames = logmel(audio)
    times = [t for t, _ in got]
    assert times == [FIRST_S + HOP_S * k for k in range(len(times))] and times[-1] == 4.0
    for t, c in got:
        end = int(round(t * FS / HOP))
        win = frames[max(0, end - WIN_FRAMES) : end]
        if len(win) < WIN_FRAMES:
            win = np.resize(win, (WIN_FRAMES, N_MELS))
        assert c.probability == pytest.approx(detector().prob(normalize(win).T), abs=1e-6)


def test_reset_starts_a_new_transmission() -> None:
    audio = np.random.default_rng(4).normal(scale=0.1, size=2 * FS).astype(np.float32)
    stream = VoiceStream()
    first = stream.feed(audio)
    stream.reset()
    assert [c.probability for _, c in stream.feed(audio)] == [c.probability for _, c in first]
