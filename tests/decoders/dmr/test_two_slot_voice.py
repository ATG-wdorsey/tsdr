"""A repeater carrying voice on both timeslots: every voice burst of each slot is decoded.

A slot's voice superframe is 6 bursts: A has the voice sync, B-F embedded signalling.
With both slots in voice, up to 10 bursts in a row have no sync. The decoder used to drop
its lock whenever the next burst was a sync-less burst of a slot it had not yet seen in a
call, and a lock drop cleared both slots' calls: joined mid-call, or after one missed
sync, both slots stayed silent in 5 of the 6 ways their superframes can align. A missed
A sync on one slot also needs 11 bursts in a row by timing on that slot.
The signal here is synthesized: valid CACH and syncs, 4FSK at 4800 baud, from IQ.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tsdr.radio.decoders.dmr import decoder as dmr
from tsdr.radio.decoders.dmr.constants import (
    BURST_DIBITS,
    CACH_INTERLEAVE,
    SYMBOL_RATE,
    SYNC_BS_VOICE,
    SYNC_START,
)
from tsdr.radio.decoders.dmr.decoder import DMRDecoder
from tsdr.radio.decoders.dmr.fec import _QR_16_7_6_WORDS, hamming_7_4_decode

RATE = 48_000.0
_LEVEL = {1: 3.0, 0: 1.0, 2: -1.0, 3: -3.0}  # dibit -> 4FSK symbol (ETSI TS 102 361-1)


def _hamming_codeword(data: tuple[int, int, int, int]) -> list[int]:
    for parity in itertools.product((0, 1), repeat=3):
        word = np.array([*data, *parity], dtype=np.uint8)
        if hamming_7_4_decode(word.copy()) and list(word[:4]) == list(data):
            check = word.copy()
            hamming_7_4_decode(check)
            if np.array_equal(check, word):
                return list(word)
    raise AssertionError("no codeword")


def _cach(slot: int) -> list[int]:
    plain = np.zeros(24, dtype=np.uint8)
    plain[0:7] = _hamming_codeword((1, slot, 0, 0))  # AT, TC (the slot), LCSS
    plain[7:14] = _hamming_codeword((0, 0, 0, 0))
    plain[14:21] = _hamming_codeword((0, 0, 0, 0))
    bits = [int(plain[CACH_INTERLEAVE[i]]) for i in range(24)]
    return [(bits[2 * k] << 1) | bits[2 * k + 1] for k in range(12)]


def _emb(color_code: int) -> list[int]:
    """The EMB's 8 dibits: color code, PI 0, LCSS 0, QR(16,7,6) parity."""
    word = int(_QR_16_7_6_WORDS[color_code << 3])
    bits = [(word >> (15 - i)) & 1 for i in range(16)]
    return [(bits[2 * k] << 1) | bits[2 * k + 1] for k in range(8)]


def _voice_burst(
    slot: int, letter: int, rng: np.random.Generator, *, bad_sync: bool = False
) -> list[int]:
    burst = list(rng.integers(0, 4, BURST_DIBITS))
    burst[0:12] = _cach(slot)
    if letter == 0:
        sync = list(SYNC_BS_VOICE)
        if bad_sync:  # 3 of 24 sync symbols wrong: one more than the sync search allows
            for i in (3, 10, 17):
                sync[i] = 1 if sync[i] == 3 else 3
        burst[SYNC_START : SYNC_START + 24] = sync
    else:  # EMB around the embedded signalling (anything but a sync)
        emb = _emb(1)
        burst[SYNC_START : SYNC_START + 24] = [*emb[:4], *[0, 2] * 8, *emb[4:]]
    return burst


def _iq(dibits: list[int], offset_hz: float = 0.0) -> np.ndarray:
    sps = int(RATE / SYMBOL_RATE)
    freq = np.repeat([_LEVEL[d] * dmr.DEVIATION / 3.0 for d in dibits], sps)
    freq = np.convolve(freq, np.hanning(sps) / np.hanning(sps).sum(), mode="same")
    phase = 2 * np.pi * np.cumsum(freq + offset_hz) / RATE
    return (0.5 * np.exp(1j * phase)).astype(np.complex64)


@pytest.mark.parametrize("phase", range(6))
def test_both_slots_decoded_after_a_late_entry_and_a_missed_sync(phase: int) -> None:
    """`phase`: slot 1's place in its superframe when slot 0 starts one (its A follows
    slot 0's by 1, 11, 9, 7, 5 or 3 bursts). The decoder joins both calls mid-superframe;
    slot 0's A sync in superframe 8 is missed."""
    rng = np.random.default_rng(10 + phase)
    superframes, missed, counted_from = 14, 8 * 6, 4 * 6
    stream: list[int] = []
    for k in range(3, 3 + superframes * 6):  # join at slot 0's burst D
        stream += _voice_burst(0, k % 6, rng, bad_sync=k == missed)
        stream += _voice_burst(1, (k + phase) % 6, rng)
    iq = _iq(stream)

    decoder = DMRDecoder(sample_rate=RATE)
    burst_s = BURST_DIBITS / SYMBOL_RATE
    window = (1000.0 + 2 * counted_from * burst_s, 1000.0 + 2 * (superframes - 1) * 6 * burst_s)
    decoded: dict[int, int] = {0: 0, 1: 0}
    process = decoder._process_voice_burst

    def count(burst, sync_type, timeslot, timestamp):
        ok = process(burst, sync_type, timeslot, timestamp)
        if decoder._ts_state[timeslot].confirmed and window[0] <= timestamp < window[1]:
            decoded[timeslot] += 1
        return ok

    decoder._process_voice_burst = count  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)

    # Every burst of superframes 4-12 of each slot reaches the vocoder, the missed A's too.
    sent = (superframes - 1 - 4) * 6
    assert decoded == {0: sent, 1: sent}


@pytest.mark.parametrize("offset_hz", [-500.0, 500.0])
def test_a_repeater_off_frequency_is_read_without_errors(offset_hz: float) -> None:
    """Repeaters are often a few hundred Hz off (1 ppm at 440 MHz), which moves every
    level. The timing loop and the sync search decided by sign around zero, and the
    slicer's thresholds sat off the midpoints: 250 Hz off, this signal was not received
    at all, and the 443.7 MHz recording (450 Hz off) had bit errors in most AMBE frames,
    7 % of them replaced."""
    rng = np.random.default_rng(7)
    sent = [_voice_burst(slot, letter, rng) for letter in [*range(6)] * 6 for slot in (0, 1)]
    iq = _iq([d for burst in sent for d in burst], offset_hz)

    decoder = DMRDecoder(sample_rate=RATE)
    received: list[np.ndarray] = []
    process = decoder._process_voice_burst

    def keep(burst, sync_type, timeslot, timestamp):
        if timestamp >= 1001.0:  # past the start: offset tracker, timing, both calls joined
            received.append(burst.copy())
        return process(burst, sync_type, timeslot, timestamp)

    decoder._process_voice_burst = keep  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)

    # CACH and voice payload (around the sync/EMB field), dibit for dibit.
    def payload(burst) -> tuple[int, ...]:
        return tuple(int(d) for d in [*burst[:SYNC_START], *burst[SYNC_START + 24 :]])

    burst_s = BURST_DIBITS / SYMBOL_RATE
    assert len(received) >= len(sent) - int(1.0 / burst_s) - 2
    assert {payload(b) for b in received} <= {payload(b) for b in sent}


def test_a_lost_signal_is_not_decoded_as_voice() -> None:
    """Bursts B-F are followed by timing; when the signal goes mid-call (a fade, no
    terminator), noise was decoded as speech for up to 11 bursts (both slots' calls
    followed). Their EMB, a color code under a QR(16,7,6) code, tells noise apart."""
    rng = np.random.default_rng(3)
    stream: list[int] = []
    for k in range(4 * 6):  # slot 1's A follows slot 0's by 7 bursts
        stream += _voice_burst(0, k % 6, rng)
        stream += _voice_burst(1, (k + 3) % 6, rng)
    signal = _iq(stream)
    noise = (
        0.5 * (rng.standard_normal(int(RATE)) + 1j * rng.standard_normal(int(RATE))) / np.sqrt(2)
    )
    iq = np.concatenate([signal, noise.astype(np.complex64)])
    lost_at = 1000.0 + len(signal) / RATE

    decoder = DMRDecoder(sample_rate=RATE)
    after: list[int] = []
    process = decoder._process_voice_burst

    def count(burst, sync_type, timeslot, timestamp):
        ok = process(burst, sync_type, timeslot, timestamp)
        if decoder._ts_state[timeslot].confirmed and timestamp >= lost_at:
            after.append(timeslot)
        return ok

    decoder._process_voice_burst = count  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)

    assert len(after) <= 1, after


def test_the_timeslot_heard_is_chosen() -> None:
    """auto plays the first call; "1" / "2" one timeslot; "both" TS1 left, TS2 right,
    in step, without waiting for a slot that carries no voice."""
    decoder = DMRDecoder(sample_rate=RATE, audio_slot="both")
    now = 1000.0
    for slot, level in ((0, 0.25), (1, 0.5)):
        decoder._ts_state[slot].last_voice_ts = now
        decoder._pcm_per_ts[slot] = [np.full(480, level, dtype=np.float32)]
    decoder._pcm_per_ts[1].append(np.full(160, 0.5, dtype=np.float32))  # TS2 a frame ahead
    decoder._flush_selected_audio(now)
    (both,) = decoder.get_audio()
    assert both.stereo and both.samples.shape == (480, 2)
    assert np.all(both.samples[:, 0] == 0.25) and np.all(both.samples[:, 1] == 0.5)

    decoder._ts_state[1].last_voice_ts = now - 1.0  # TS2's call is over: not waited for
    decoder._pcm_per_ts[0] = [np.full(320, 0.25, dtype=np.float32)]
    decoder._flush_selected_audio(now)
    (alone,) = decoder.get_audio()
    assert alone.samples.shape == (320, 2)
    assert np.all(alone.samples[:160, 1] == 0.5) and np.all(alone.samples[160:, 1] == 0.0)

    decoder.set_audio_slot("2")
    for slot, level in ((0, 0.25), (1, 0.5)):
        decoder._pcm_per_ts[slot] = [np.full(480, level, dtype=np.float32)]
    decoder._flush_selected_audio(now)
    (one,) = decoder.get_audio()
    assert not one.stereo and np.all(one.samples == 0.5)
