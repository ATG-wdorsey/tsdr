"""IMBE vocoder (P25 Phase 1): public speech coded as IMBE frames, decoded through the frame's error
correction and checked frame by frame against the parameters mbelib decodes (tests/samples/imbe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.radio.vocoder.imbe import ImbeDecoder
from tsdr.radio.vocoder.imbe.decoder import encode_frame

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "tests" / "samples" / "imbe" / "bmh_gasline.imbe"
MBELIB_PARAMS = ROOT / "tests" / "samples" / "imbe" / "bmh_gasline_mbelib.npz"


@pytest.fixture(scope="module")
def frames() -> list[np.ndarray]:
    raw = np.frombuffer(VECTORS.read_bytes(), dtype=np.uint8).reshape(-1, 11)
    return [encode_frame(v) for v in np.unpackbits(raw, axis=1)[:, :88]]


def _check(decoder: ImbeDecoder, frames: list[np.ndarray], errors: list[int]) -> None:
    ref = np.load(MBELIB_PARAMS)
    assert len(frames) == len(ref["L"])
    for i, frame in enumerate(frames):
        audio = decoder.decode_frame_float(frame)
        assert decoder.last_errors == errors[i], i
        assert audio.shape == (160,) and np.all(np.isfinite(audio))
        # The parameters as they stood before the spectral enhancement.
        mp, n = decoder.prev_mp, int(ref["L"][i])
        assert n == mp.L and mp.repeat == ref["repeat"][i], i
        assert mp.w0 == pytest.approx(float(ref["w0"][i]), rel=1e-6), i
        np.testing.assert_array_equal(mp.Vl[1 : n + 1], ref["vl"][i, 1 : n + 1], err_msg=str(i))
        np.testing.assert_allclose(
            mp.Ml[1 : n + 1], ref["ml"][i, 1 : n + 1], rtol=1e-5, err_msg=str(i)
        )


def test_decoded_parameters_match_mbelib(frames: list[np.ndarray]) -> None:
    _check(ImbeDecoder(), frames, [0] * len(frames))


def test_errors_within_the_codes_reach_are_corrected(frames: list[np.ndarray]) -> None:
    # Two of the Golay words with up to 2 errors each and one Hamming word with 1: corrected, and
    # counted as mbelib counts them (data bits only for Golay, any correction for Hamming), at most
    # 5, so no frame is repeated.
    rng = np.random.default_rng(3)
    damaged, counts = [], []
    for frame in frames:
        fr = frame.copy()
        count = 0
        for i in rng.choice(4, 2, replace=False):
            for j in rng.choice(23, int(rng.integers(0, 3)), replace=False):
                fr[i, j] ^= 1
                count += int(j >= 11)
        if rng.random() < 0.5:
            fr[4 + int(rng.integers(0, 3)), int(rng.integers(0, 15))] ^= 1
            count += 1
        damaged.append(fr)
        counts.append(count)
    _check(ImbeDecoder(), damaged, counts)


def test_soft_decisions_correct_what_hard_decoding_cannot(frames: list[np.ndarray]) -> None:
    # Four errors in c0, on its least reliable bits: beyond the Golay code's hard correction (it
    # miscorrects, and the wrong c0 seeds a wrong scrambling sequence for the rest of the frame),
    # within the soft decoder's reach. The frame then decodes as if it had come clean.
    clean = frames[100]
    damaged = clean.copy()
    reliability = np.full((8, 23), 2.0)
    for j in (2, 9, 15, 20):
        damaged[0, j] ^= 1
        reliability[0, j] = 0.2
    expected = ImbeDecoder().decode_frame_float(clean)
    assert not np.array_equal(ImbeDecoder().decode_frame_float(damaged), expected)
    np.testing.assert_array_equal(ImbeDecoder().decode_frame_float(damaged, reliability), expected)
