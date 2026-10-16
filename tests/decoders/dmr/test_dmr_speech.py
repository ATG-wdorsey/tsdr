"""DMR speech comes out as speech, unclipped. The decoder's fixed +18 dB clipped 7 % of a
live talker's samples (a private recording, 2026-10-14) and 2.4 % of the speech here; the
speech gain now follows the reference decoders (DSD, mbelib's decode_ambe).

The signal is synthesized: AMBE+2 speech (tests/samples/ambe, public) coded as a radio
sends it (Golay C0 and C1, C1 scrambled, interleaved) in TS1's voice bursts, at 30 dB C/N."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.radio.decoders.dmr.constants import AMBE_RW, AMBE_RX, AMBE_RY, AMBE_RZ
from tsdr.radio.decoders.dmr.decoder import DMRDecoder
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.ambe._constants import GOLAY_GENERATOR
from tsdr.radio.vocoder.ambe.frame import demodulate_c1
from tsdr.radio.voice.detector import VoiceStream

from ._synth import RATE, noise, to_iq, voice_burst

AMB = Path(__file__).resolve().parents[2] / "samples" / "ambe" / "bmh_gasline.amb"
SECONDS = 8.0
FULL_SCALE = 32767 / 32768


def _amb_frames(path: Path) -> list[np.ndarray]:
    """The 49 voice bits of each frame of a `.amb` file (tests/samples/ambe/README.md)."""
    data = path.read_bytes()
    frames = []
    for pos in range(4, len(data) - 7, 8):
        bits = np.unpackbits(np.frombuffer(data[pos + 1 : pos + 8], dtype=np.uint8))
        frames.append(np.concatenate([bits[:48], bits[55:56]]).astype(np.int8))
    return frames


def _golay_23_12(data: int) -> list[int]:
    word, ecc, mask = data << 11, 0, 0x400000
    for i in range(12):
        if word & mask:
            ecc ^= int(GOLAY_GENERATOR[i])
        mask >>= 1
    return [((word | ecc) >> i) & 1 for i in range(23)]


def _frame_dibits(ambe_d: np.ndarray) -> list[int]:
    """49 voice bits as the 36 dibits a radio sends: the inverse of the decoder's ecc_c0,
    demodulate_c1, pack_ambe_d and deinterleave."""
    fr = np.zeros((4, 24), dtype=np.int8)
    fr[0, 1:24] = _golay_23_12(int("".join(map(str, ambe_d[0:12])), 2))
    fr[0, 0] = int(fr[0, 1:24].sum()) & 1
    fr[1, 0:23] = _golay_23_12(int("".join(map(str, ambe_d[12:24])), 2))
    demodulate_c1(fr)  # the scrambling is its own inverse
    fr[2, 10::-1] = ambe_d[24:35]
    fr[3, 13::-1] = ambe_d[35:49]
    return [
        (int(fr[AMBE_RW[i], AMBE_RX[i]]) << 1) | int(fr[AMBE_RY[i], AMBE_RZ[i]]) for i in range(36)
    ]


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    rng = np.random.default_rng(3)
    frames = _amb_frames(AMB)
    stream: list[int] = []
    for k in range(int(SECONDS / 0.06)):  # a TS1 burst every 60 ms carries 3 frames
        burst = voice_burst(0, k % 6, rng)
        a, b, c = (_frame_dibits(f) for f in frames[3 * k : 3 * k + 3])
        burst[12:48], burst[48:66], burst[90:108], burst[108:144] = a, b[:18], b[18:], c
        stream += burst + voice_burst(1, (k + 3) % 6, rng)
    iq = to_iq(stream)
    iq += noise(len(iq), rng, 30.0)
    decoder = DMRDecoder(sample_rate=RATE)
    chunk = int(RATE * 0.02)
    audio = []
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)
        audio += decoder.get_audio()
        decoder.get_messages()
    return np.concatenate([b.samples for b in audio])


def test_dmr_speech_decodes_to_speech(speech: np.ndarray) -> None:
    assert len(speech) / 8000 > SECONDS - 1.0  # all but the decoder's lock-in
    p = np.array([c.probability for _, c in VoiceStream().feed(speech.astype(np.float32))])
    assert np.mean(p >= 0.5) > 0.8


def test_loud_speech_is_not_clipped(speech: np.ndarray) -> None:
    assert np.max(np.abs(speech)) <= FULL_SCALE + 1e-6
    assert np.mean(np.abs(speech) >= 0.999) < 0.005  # 2.4 % with the fixed +18 dB
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
