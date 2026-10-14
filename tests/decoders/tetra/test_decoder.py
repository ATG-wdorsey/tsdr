import functools
from pathlib import Path

import numpy as np
import pytest

from tsdr.core.sdr.io import load_iq
from tsdr.radio.decoders.tetra.burst import (
    extract_sync_burst,
)
from tsdr.radio.decoders.tetra.channel import (
    decode_block,
)
from tsdr.radio.decoders.tetra.decoder import TETRADecoder
from tsdr.radio.decoders.tetra.demod import TetraDemod
from tsdr.radio.decoders.tetra.mac import SB1Info, parse_sb1
from tsdr.radio.decoders.tetra.scramble import SCRAMB_INIT
from tsdr.radio.decoders.tetra.sync import (
    SyncDetector,
)

# TETRA convolutional code parameters (K=5, rate 1/4)
# Register convention: input bit at MSB (bit K-1), same as DAB.
#   G1 = 1+D+D^4:         taps at bit4(input), bit3(D), bit0(D^4) -> 0o31
#   G2 = 1+D^2+D^3+D^4:   taps at bit4, bit2, bit1, bit0          -> 0o27
#   G3 = 1+D+D^2+D^4:     taps at bit4, bit3, bit2, bit0          -> 0o35
#   G4 = 1+D+D^3+D^4:     taps at bit4, bit3, bit1, bit0          -> 0o33


# Checkpoint 3: π/4-DQPSK demodulator

TETRA_SAMPLE = "tests/samples/freq=427.031M_sr=240k_dur=5s_gain=30_20260423T2014.cu8.zst"
SAMPLE_RATE = 240_000

# Phase change mapping for π/4-DQPSK modulator (consistent with demod soft bit extraction)
# With MSB=-im/norm, LSB=-re/norm:
#   Phase π/4 -> dibit 00, Phase -π/4 -> dibit 10
#   Phase 3π/4 -> dibit 01, Phase -3π/4 -> dibit 11


@functools.lru_cache(maxsize=1)
def _load_tetra_iq() -> np.ndarray:
    """Load TETRA IQ sample with caching (avoids 16x reload)."""
    sample_path = Path(TETRA_SAMPLE)
    if not sample_path.exists():
        pytest.skip(f"Sample not found: {TETRA_SAMPLE}")
    return load_iq(sample_path)


class TestCheckpoint7Bootstrap:
    """Checkpoint 7: SB1 bootstrap (BSCH decode)."""

    def test_real_sample_bootstrap(self):
        """Decode SB1 from real sample, extract MCC/MNC/CC."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        n_samples = int(10.0 * sample_rate)

        demod = TetraDemod(sample_rate)
        symbols = demod.process_symbols(iq[:n_samples])
        detector = SyncDetector()
        results = detector.process(symbols)

        sync_bursts = [r for r in results if r.burst_type == "sync"]

        mccs, mncs, ccs = [], [], []
        for sb_result in sync_bursts:
            sb = extract_sync_burst(sb_result.soft_bits)
            type1, crc_ok = decode_block(sb.sb1, "SB1", SCRAMB_INIT)
            if crc_ok:
                info = parse_sb1(type1)
                mccs.append(info.mcc)
                mncs.append(info.mnc)
                ccs.append(info.colour_code)

        assert len(mccs) > 0, "No valid SB1 decodes"
        print(f"MCC={mccs[0]}, MNC={mncs[0]}, CC={ccs[0]}")

        # All decodes should be consistent
        assert len(set(mccs)) == 1, f"Inconsistent MCC: {set(mccs)}"
        assert len(set(mncs)) == 1, f"Inconsistent MNC: {set(mncs)}"
        assert len(set(ccs)) == 1, f"Inconsistent CC: {set(ccs)}"

        # Valid ranges
        assert 0 <= mccs[0] <= 1023
        assert 0 <= mncs[0] <= 16383


class TestCheckpoint10Integration:
    """Checkpoint 10: Decoder integration."""

    def test_streaming_chunks(self):
        """Feed sample in 0.1s chunks, verify messages produced within 2 seconds."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        chunk_size = int(0.1 * sample_rate)

        decoder = TETRADecoder(sample_rate)
        all_messages = []
        first_msg_chunk = None

        for i in range(50):  # 5 seconds
            start = i * chunk_size
            end = start + chunk_size
            if end > len(iq):
                break
            decoder.demodulate(iq[start:end], i * 0.1)
            msgs = decoder.get_messages()
            if msgs and first_msg_chunk is None:
                first_msg_chunk = i
            all_messages.extend(msgs)

        assert len(all_messages) > 0, "No messages produced"
        assert first_msg_chunk is not None
        first_msg_time = first_msg_chunk * 0.1
        print(f"First message at {first_msg_time:.1f}s, total messages: {len(all_messages)}")
        assert first_msg_time < 2.0, f"First message at {first_msg_time:.1f}s (>2s)"

    def test_chunk_size_independence(self):
        """Same MCC/MNC/CC extracted regardless of chunk size."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        duration = 5.0
        n_samples = int(duration * sample_rate)

        results = {}
        for chunk_dur in [0.05, 0.1, 0.5]:
            chunk_size = int(chunk_dur * sample_rate)
            decoder = TETRADecoder(sample_rate)
            for i in range(n_samples // chunk_size):
                start = i * chunk_size
                decoder.demodulate(iq[start : start + chunk_size], i * chunk_dur)

            msgs = []
            # Drain any remaining
            final_msgs = decoder.get_messages()
            for m in final_msgs:
                if isinstance(m.data, SB1Info):
                    msgs.append(m)

            if msgs:
                sb1 = msgs[0].data
                assert isinstance(sb1, SB1Info)
                results[chunk_dur] = (sb1.mcc, sb1.mnc, sb1.colour_code)

        assert len(results) >= 2, f"Only {len(results)} chunk sizes produced results"
        values = list(results.values())
        for v in values[1:]:
            assert v == values[0], f"Inconsistent results: {results}"

    def test_reset(self):
        """Reset and re-lock."""

        iq = _load_tetra_iq()
        sample_rate = SAMPLE_RATE
        chunk_size = int(0.5 * sample_rate)

        decoder = TETRADecoder(sample_rate)

        # First pass
        decoder.demodulate(iq[: chunk_size * 4], 0.0)
        msgs1 = decoder.get_messages()
        assert len(msgs1) > 0, "No messages before reset"

        # Reset
        decoder.reset()
        assert decoder._state.network is None

        # Second pass with different data
        decoder.demodulate(iq[chunk_size * 4 : chunk_size * 8], 2.0)
        msgs2 = decoder.get_messages()
        assert len(msgs2) > 0, "No messages after reset"
