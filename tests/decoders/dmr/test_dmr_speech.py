"""DMR speech from a live repeater (443.700 MHz, colour code 8, 2026-10-14) comes out as
speech, unclipped. The decoder's fixed +18 dB clipped 7 % of this talker's samples;
the speech gain now follows the reference decoders (DSD, mbelib's decode_ambe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.dmr.decoder import DMRDecoder
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.voice.detector import VoiceStream

SAMPLE = (
    Path(__file__).resolve().parents[2] / "samples" / "freq=443.7M_sr=62k5_dur=8s_gain=35_dmr.cu8.zst"
)
RATE = 62_500.0
FULL_SCALE = 32767 / 32768


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    if not SAMPLE.exists():
        pytest.skip(f"Sample file not found: {SAMPLE}")
    iq = load_iq(SAMPLE)
    decoder = DMRDecoder(sample_rate=RATE)
    chunk = int(RATE * 0.05)
    audio = []
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)
        audio += decoder.get_audio()
        decoder.get_messages()
    return np.concatenate([b.samples for b in audio])


def test_a_live_repeater_decodes_to_speech(speech: np.ndarray) -> None:
    assert len(speech) / 8000 > 7.0  # 8 s of a call in progress
    p = np.array([c.probability for _, c in VoiceStream().feed(speech.astype(np.float32))])
    assert np.mean(p >= 0.5) > 0.8


def test_loud_speech_is_not_clipped(speech: np.ndarray) -> None:
    assert np.max(np.abs(speech)) <= FULL_SCALE + 1e-6
    assert np.mean(np.abs(speech) >= 0.999) < 0.005  # was 7 % with the fixed +18 dB
    assert 0.05 < np.sqrt(np.mean(speech**2)) < 0.4  # and still a good level


def test_the_gain_falls_at_once_and_rises_slowly() -> None:
    agc = VoiceAGC(max_gain=50.0, start_gain=25.0)
    loud = np.full(160, 0.5, dtype=np.float32)
    assert np.max(agc.process(loud)) == pytest.approx(FULL_SCALE)  # 25x would be 12.5
    quiet = np.full(160, 0.001, dtype=np.float32)
    gains = []
    for _ in range(120):
        agc.process(quiet)
        gains.append(agc.gain)
    # Held while the loud frame is in the last 25 frames, then up 5 % a frame, to x50.
    assert gains[20] == pytest.approx(FULL_SCALE / 0.5)
    assert gains[30] <= gains[29] * 1.05 + 1e-9
    assert gains[-1] == pytest.approx(50.0)
