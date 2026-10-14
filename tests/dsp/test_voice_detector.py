"""The voice detector (radio/voice): a small CNN on log-mel windows of demodulated audio."""

from __future__ import annotations

import os
import subprocess
import sys
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
from tsdr.radio.voice.frontend import (
    FS,
    HOP,
    N_MELS,
    NORM_FLOOR,
    WIN_FRAMES,
    logmel,
    normalize,
)

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


def test_a_model_without_the_normalization_floor_is_refused(tmp_path: Path) -> None:
    with np.load(MODEL) as m:
        arrays = {k: m[k] for k in m.files if k != "norm_floor"}
    path = tmp_path / "old.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="norm_floor"):
        load_model(path)


@pytest.mark.parametrize("damage", ["truncated", "empty", "flipped_byte", "missing_array"])
def test_a_damaged_model_is_refused_not_raised_raw(tmp_path: Path, damage: str) -> None:
    # `v` turns VSQ on through load_model; anything but OSError/ValueError ended the app.
    data = MODEL.read_bytes()
    path = tmp_path / "voice.npz"
    if damage == "truncated":
        path.write_bytes(data[: len(data) // 2])
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "flipped_byte":
        i = len(data) // 2
        path.write_bytes(data[:i] + bytes([data[i] ^ 0xFF]) + data[i + 1 :])
    else:
        with np.load(MODEL) as m:
            np.savez(path, **{k: m[k] for k in m.files if k != "w2"})
    with pytest.raises(ValueError):
        load_model(path)


def test_a_quiet_window_keeps_its_level() -> None:
    rng = np.random.default_rng(5)
    loud = logmel(0.03 * rng.normal(size=2 * FS))[:WIN_FRAMES]  # -30 dBFS
    quiet = logmel(0.0003 * rng.normal(size=2 * FS))[:WIN_FRAMES]  # -70 dBFS
    assert abs(normalize(loud).mean()) < 1e-6
    assert normalize(quiet).mean() == pytest.approx(quiet.mean() - NORM_FLOOR, abs=1e-5)
    assert normalize(quiet).mean() < -1.5


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


# In a fresh interpreter, as the app starts: import tsdr (before numpy), score a window, then ask every OpenBLAS the
# process loaded (numpy's, scipy's) how many threads it runs.
_BLAS_THREADS = """
import ctypes
import tsdr
import numpy as np
from tsdr.radio.voice.detector import detector
detector().prob(np.zeros((32, 124), np.float32))
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "openblas" in l.lower() and ".so" in l})
for path in libs:
    lib = ctypes.CDLL(path)
    for name in ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                 "openblas_get_num_threads64_", "openblas_get_num_threads"):
        if hasattr(lib, name):
            print(getattr(lib, name)())
            break
"""


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="reads /proc/self/maps")
def test_blas_runs_on_one_thread() -> None:
    # OpenBLAS's pool spins after each product: with it, the detector cost ~0.5 s
    # of CPU per second of audio, nearly all on other cores (vs 9 ms on one thread).
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    out = subprocess.run(
        [sys.executable, "-c", _BLAS_THREADS, "test"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout.split()
    if not out:
        pytest.skip("no OpenBLAS loaded")
    assert out == ["1"] * len(out)
