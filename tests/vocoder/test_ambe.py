"""AMBE+2 vocoder tests.

API contract, FEC primitives, an end-to-end WAV smoke test, and the decoded
parameters checked frame by frame against mbelib on a speech recording
(tests/samples/ambe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdr.radio.vocoder.ambe import decoder as ambe_decoder
from tsdr.radio.vocoder.ambe._constants import GOLAY_GENERATOR
from tsdr.radio.vocoder.ambe.decoder import (
    AmbePlus2Decoder,
    _pack_voice_bits,
    _unpack_voice_bits,
)
from tsdr.radio.vocoder.ambe.fec import golay_23_12, hamming_15_11
from tsdr.radio.vocoder.ambe.params_decode import extract_bit_fields

ROOT = Path(__file__).resolve().parents[2]
AMB_SOURCE = ROOT / "tests" / "samples" / "ambe" / "bmh_gasline.amb"
MBELIB_PARAMS = ROOT / "tests" / "samples" / "ambe" / "bmh_gasline_mbelib.npz"


# API contract


def test_decode_frame_shape_dtype():
    """9 bytes in, (160,) int16 out."""
    dec = AmbePlus2Decoder()
    out = dec.decode_frame(bytes(9))
    assert out.shape == (160,)
    assert out.dtype == np.int16


def test_decode_frame_bfi_is_silent():
    dec = AmbePlus2Decoder()
    out = dec.decode_frame(bytes(9), bfi=True)
    assert np.all(out == 0)


def test_voice_bits_round_trip():
    rng = np.random.default_rng(0)
    bits = rng.bytes(9)
    fr = _unpack_voice_bits(bits)
    assert fr.shape == (4, 24)
    assert fr[1, 23] == 0
    assert np.all(fr[2, 11:] == 0)
    assert np.all(fr[3, 14:] == 0)
    back = _pack_voice_bits(fr)
    assert back == bits


# FEC primitives


def test_golay_23_12_no_error_roundtrip():
    """An ECC-consistent codeword should decode to itself with zero errors."""
    data = 0b101010110011
    ecc = 0
    mask = 0x400000
    word = data << 11
    for i in range(12):
        if word & mask:
            ecc ^= int(GOLAY_GENERATOR[i])
        mask >>= 1
    code_int = word | ecc
    bits = np.array([(code_int >> i) & 1 for i in range(23)], dtype=np.int8)

    out, errs = golay_23_12(bits)
    assert errs == 0
    np.testing.assert_array_equal(out, bits)


def test_golay_23_12_single_bit_correction():
    """Single-bit errors in the data portion should be corrected."""
    data = 0xABC
    ecc = 0
    mask = 0x400000
    word = data << 11
    for i in range(12):
        if word & mask:
            ecc ^= int(GOLAY_GENERATOR[i])
        mask >>= 1
    code_int = word | ecc

    corrupt = np.array([(code_int >> i) & 1 for i in range(23)], dtype=np.int8)
    corrupt[15] ^= 1

    out, errs = golay_23_12(corrupt)
    assert errs == 1
    clean = np.array([(code_int >> i) & 1 for i in range(23)], dtype=np.int8)
    np.testing.assert_array_equal(out[11:23], clean[11:23])


def test_hamming_15_11_no_error():
    """A zero Hamming codeword decodes with errs==0 and unchanged output."""
    bits = np.zeros(15, dtype=np.int8)
    out, errs = hamming_15_11(bits)
    assert errs == 0
    np.testing.assert_array_equal(out, bits)


def test_extract_bit_fields_silence():
    """Silence frame (b0 = 124) is recognised."""
    ambe_d = np.zeros(49, dtype=np.int8)
    ambe_d[0] = ambe_d[1] = ambe_d[2] = ambe_d[3] = 1
    ambe_d[37] = 1  # b0 bits 6..0 = 1111100 = 124
    b0, *_ = extract_bit_fields(ambe_d)
    assert b0 == 124


# End-to-end


def _amb_records(path: Path) -> list[tuple[int, np.ndarray]]:
    data = path.read_bytes()
    assert data[:4] == b".amb"
    records = []
    pos = 4
    while pos + 8 <= len(data):
        errs2 = data[pos]
        bits = np.zeros(49, dtype=np.int8)
        for i in range(6):
            for j in range(8):
                bits[i * 8 + j] = (data[pos + 1 + i] >> (7 - j)) & 1
        bits[48] = data[pos + 7] & 1
        pos += 8
        records.append((errs2, bits))
    return records


def test_decoded_parameters_match_mbelib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pitch, harmonics, voicing and amplitudes, frame by frame, as mbelib decodes them
    (the decoder's own path, state carried between frames)."""
    ref = np.load(MBELIB_PARAMS)
    seen: list[tuple[int, float, int, int, np.ndarray]] = []
    decode = ambe_decoder.decode_ambe2450_parms

    def snapshot(ambe_d, cur_mp, prev_mp):  # type: ignore[no-untyped-def]
        bad = decode(ambe_d, cur_mp, prev_mp)
        n = cur_mp.L
        seen.append(
            (bad, float(cur_mp.w0), n, int(cur_mp.Vl[1 : n + 1].sum()), cur_mp.Ml[1 : n + 1].copy())
        )
        return bad

    monkeypatch.setattr(ambe_decoder, "decode_ambe2450_parms", snapshot)
    dec = AmbePlus2Decoder()
    for errs2, bits in _amb_records(AMB_SOURCE):
        dec.decode_ambe_d(bits, errs2=errs2)

    assert len(seen) == len(ref["L"])
    for i, (bad, w0, n, voiced, ml) in enumerate(seen):
        assert bad == ref["bad"][i], i
        assert w0 == pytest.approx(float(ref["w0"][i]), rel=1e-5), i
        assert ref["L"][i] == n, i
        assert voiced == ref["voiced"][i], i
        np.testing.assert_allclose(
            ml, ref["ml"][i, 1 : n + 1], rtol=5e-3, atol=1e-6, err_msg=str(i)
        )
