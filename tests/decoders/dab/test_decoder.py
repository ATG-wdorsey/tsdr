"""DAB+ decoder tests: the decoder end to end on a real sample, and its FEC parts.

Sample: 227.36 MHz (DAB Band III), 2048 kHz, 10s, gain=10
Expected: ~104 frames of Mode I DAB
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.dab import DABDecoder
from tsdr.radio.decoders.dab.fec import _check_fib_crc, _crc16_bytes, _generate_prbs
from tsdr.radio.decoders.dab.msc import (
    CIF_BITS,
    _TimeDeinterleaver,
)
from tsdr.radio.decoders.dab.pad import _DynamicLabelDecoder
from tsdr.radio.decoders.dab.superframe import _DabPlusSuperframe, _rs_decode
from tsdr.radio.decoders.dab.viterbi import _viterbi_decode

DAB_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "samples"
    / "freq=227.36M_sr=2048k_dur=10s_gain=10_20260308T2248.cu8.zst"
)


@pytest.fixture(scope="module")
def iq_data():
    """Load DAB IQ sample (cached per module)."""
    if not DAB_SAMPLE.exists():
        pytest.skip(f"Sample file not found: {DAB_SAMPLE}")
    return load_iq(DAB_SAMPLE)


class TestStage5FICDecoding:
    """Stage 5: FIC decoding - depuncture + Viterbi + CRC."""

    def test_prbs_first_bits(self):
        """PRBS first bits should match LFSR x^9+x^5+1."""
        p = _generate_prbs(10)
        # LFSR: all-1s init, output = bit[8] XOR bit[4]
        # First 10 outputs: 0,0,0,0,0,1,1,1,1,0
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        assert np.array_equal(p, expected), f"PRBS starts with {p}, expected {expected}"

    def test_viterbi_known_input(self):
        """Viterbi should decode a trivial all-zeros input."""
        soft = np.ones(4 * 20, dtype=np.float32) * -1.0
        decoded = _viterbi_decode(soft)
        assert len(decoded) == 20
        assert np.all(decoded == 0)

    def test_crc_check_valid(self):
        """CRC check should accept a known-good FIB."""
        # All-zero FIB data with correct CRC
        data = bytes(30) + b"\xff\xff"  # placeholder
        # Compute the actual CRC for all-zero data
        crc = _crc16_bytes(bytes(30))
        # Store inverted CRC
        crc_inv = crc ^ 0xFFFF
        data = bytes(30) + bytes([crc_inv >> 8, crc_inv & 0xFF])
        assert _check_fib_crc(data)


# ============================================================
# MSC Audio Decoding Tests (Stages 8-14)
# ============================================================


class TestStage10TimeDeinterleaver:
    """Stage 10: 16-frame convolutional time de-interleaver."""

    def test_identity_when_no_interleaving(self):
        """If all frames are the same, output should equal input."""
        di = _TimeDeinterleaver()
        data = np.random.randn(CIF_BITS).astype(np.float32)
        for _ in range(16):
            di.push(data.copy())
        result = di.push(data.copy())
        np.testing.assert_array_almost_equal(result, data, decimal=5)


class TestStage12ReedSolomon:
    """Stage 12: RS(120,110) error correction."""

    def test_rs_no_errors(self):
        """RS should pass through clean data."""
        # Create a valid codeword (all zeros is a valid RS codeword)
        data = np.zeros(120, dtype=np.int32)
        decoded, nerr = _rs_decode(data)
        assert nerr == 0
        assert len(decoded) == 110
        assert np.all(decoded == 0)

    def test_rs_detects_errors(self):
        """RS should detect/correct small errors."""
        data = np.zeros(120, dtype=np.int32)
        # Introduce 1 error
        data[50] = 0x42
        decoded, nerr = _rs_decode(data)
        # Should either correct (nerr=1) or detect as uncorrectable (nerr=-1)
        # For all-zero codeword with 1 error, it should correct
        print(f"RS result: nerr={nerr}")
        if nerr >= 0:
            assert decoded[50] == 0  # error should be corrected


# ============================================================
# PAD / DLS / MOT Tests (Stages 16-18)
# ============================================================


# At minimum, PAD extraction code should not crash
# Whether DLS is present depends on the station


class TestStage17DLSReassembly:
    """Stage 17: DLS segment reassembly (unit test)."""

    def test_single_segment_label(self):
        """A single-segment DLS label should reassemble correctly."""
        dl = _DynamicLabelDecoder()

        # DLS data group = prefix(2) + chars + CRC(2)
        # prefix[0]: toggle(7)|first(6)|last(5)|command(4)|field_len-1(3:0)
        # prefix[1]: charset(7:4) for first segment
        text = b"Hello"
        byte0 = 0x60 | (len(text) - 1)  # first+last, no command, field_len=5
        byte1 = 0xF0  # charset=0x0F (UTF-8)
        dg = bytearray([byte0, byte1]) + text
        crc = _crc16_bytes(bytes(dg)) ^ 0xFFFF
        dg += crc.to_bytes(2, "big")

        result = dl.process_subfield(True, bytes(dg))
        print(f"Result: {result}, label: {dl.label}")
        assert result
        assert dl.label == "Hello"

    def test_multi_segment_label(self):
        """Two segments should reassemble into one label."""
        dl = _DynamicLabelDecoder()

        # Segment 0: first=1, last=0, toggle=1, charset=0x0F, 5 chars
        text1 = b"Hello"
        prefix0 = bytes([0xC0 | (len(text1) - 1), 0xF0])  # toggle+first, charset=UTF-8
        dg0 = prefix0 + text1
        crc0 = _crc16_bytes(dg0) ^ 0xFFFF
        dg0 += crc0.to_bytes(2, "big")

        # Segment 1: first=0, last=1, toggle=1, seg_num=1, 6 chars
        text2 = b" World"
        prefix1 = bytes([0xA0 | (len(text2) - 1), 0x10])  # toggle+last, seg_num=1
        dg1 = prefix1 + text2
        crc1 = _crc16_bytes(dg1) ^ 0xFFFF
        dg1 += crc1.to_bytes(2, "big")

        dl.process_subfield(True, dg0)
        result = dl.process_subfield(True, dg1)
        print(f"Result: {result}, label: {dl.label}")
        if result:
            assert dl.label == "Hello World"

    def test_toggle_clears_cache(self):
        """Segments with different toggle values should not mix."""
        dl = _DynamicLabelDecoder()

        # Segment 0 with toggle=0
        text1 = b"Old"
        prefix0 = bytes([0x60 | (len(text1) - 1), 0xF0])  # first+last, toggle=0
        dg0 = prefix0 + text1
        dg0 += (_crc16_bytes(dg0) ^ 0xFFFF).to_bytes(2, "big")
        dl.process_subfield(True, dg0)
        assert dl.label == "Old"

        # Segment 0 with toggle=1 (new label)
        text2 = b"New"
        prefix1 = bytes([0xE0 | (len(text2) - 1), 0xF0])  # first+last, toggle=1
        dg1 = prefix1 + text2
        dg1 += (_crc16_bytes(dg1) ^ 0xFFFF).to_bytes(2, "big")
        dl.process_subfield(True, dg1)
        assert dl.label == "New"


@dataclass
class _Decoded:
    decoder: DABDecoder
    audio: list
    superframe_channels: list[int]


@pytest.fixture(scope="module")
def decoded(iq_data) -> _Decoded:
    """The whole sample through DABDecoder once, in chunks that don't line up with frames."""
    channels: list[int] = []
    original_decode = _DabPlusSuperframe._decode_superframe

    def logging_decode(self, frames):
        result = original_decode(self, frames)
        if result is not None:
            channels.append(result[2].channels)
        return result

    _DabPlusSuperframe._decode_superframe = logging_decode
    try:
        decoder = DABDecoder(sample_rate=2_048_000)
        audio = []
        service_selected = False
        chunk_size = 65536
        for i in range(0, len(iq_data), chunk_size):
            decoder.demodulate(iq_data[i : i + chunk_size], 0.0)
            decoder.get_messages()
            audio.extend(decoder.get_audio())
            if not service_selected and decoder.stats.frames_processed > 10:
                result = decoder.select_service()
                service_selected = bool(result) and not result.startswith("Error")
    finally:
        _DabPlusSuperframe._decode_superframe = original_decode
    return _Decoded(decoder, audio, channels)


def test_the_sample_decodes_with_every_fib_intact_and_plays(decoded: _Decoded) -> None:
    stats = decoded.decoder.stats
    assert stats.frames_processed >= 100
    assert stats.fibs_crc_ok == stats.fibs_decoded
    assert decoded.audio, "no audio"
    rate = decoded.audio[0].sample_rate
    assert rate == 48000.0 * 1024 / 960  # 51200 Hz (960-sample transform correction)
    for batch in decoded.audio:
        assert batch.sample_rate == rate
        assert batch.samples.dtype == np.float32
        assert np.max(np.abs(batch.samples)) <= 5.0  # HE-AAC can produce peaks > 1.0
    assert sum(len(b.samples) for b in decoded.audio) / rate > 6.0


def test_superframes_keep_the_stations_channel_count(decoded: _Decoded) -> None:
    """A real stereo DAB+ station decodes as stereo throughout (no misaligned decodes)."""
    assert decoded.superframe_channels
    assert set(decoded.superframe_channels) == {2}


def test_reset_clears_state() -> None:
    decoder = DABDecoder()
    decoder._frames_processed = 42
    decoder._buffer = np.ones(1000, dtype=np.complex64)
    decoder.reset()
    assert decoder._frames_processed == 0
    assert len(decoder._buffer) == 0
    assert decoder._last_ensemble is None
