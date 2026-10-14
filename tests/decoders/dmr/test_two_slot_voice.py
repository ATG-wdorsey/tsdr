"""A repeater carrying voice on both timeslots: every voice burst of each slot is decoded.

A slot's voice superframe is 6 bursts: A has the voice sync, B-F embedded signalling.
With both slots in voice, 10 bursts in a row have no sync; the decoder counted them
across both slots, dropped lock after 5 and lost the rest of each superframe until the
next sync: about a third of one slot's speech and half of the other's (choppy audio).
The signal here is synthesized: valid CACH and syncs, 4FSK at 4800 baud, from IQ.
"""

from __future__ import annotations

import itertools

import numpy as np

from tsdr.radio.decoders.dmr import decoder as dmr
from tsdr.radio.decoders.dmr.constants import (
    BURST_DIBITS,
    CACH_INTERLEAVE,
    SYMBOL_RATE,
    SYNC_BS_VOICE,
    SYNC_START,
)
from tsdr.radio.decoders.dmr.decoder import DMRDecoder
from tsdr.radio.decoders.dmr.fec import hamming_7_4_decode

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


def _voice_burst(slot: int, letter: int, rng: np.random.Generator) -> list[int]:
    burst = list(rng.integers(0, 4, BURST_DIBITS))
    burst[0:12] = _cach(slot)
    if letter == 0:
        burst[SYNC_START : SYNC_START + 24] = list(SYNC_BS_VOICE)
    else:  # embedded signalling: anything but a sync
        burst[SYNC_START : SYNC_START + 24] = [0, 2] * 12
    return burst


def _iq(dibits: list[int]) -> np.ndarray:
    sps = int(RATE / SYMBOL_RATE)
    freq = np.repeat([_LEVEL[d] * dmr.DEVIATION / 3.0 for d in dibits], sps)
    freq = np.convolve(freq, np.hanning(sps) / np.hanning(sps).sum(), mode="same")
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    return (0.5 * np.exp(1j * phase)).astype(np.complex64)


def test_voice_on_both_slots_is_decoded_in_full() -> None:
    rng = np.random.default_rng(1)
    superframes = 4
    lead = [d for _ in range(2) for d in _voice_burst(0, 0, rng)]  # sync to lock on first
    stream = list(lead)
    for _ in range(superframes):
        for letter in range(6):
            for slot in (0, 1):
                stream += _voice_burst(slot, letter, rng)
    iq = _iq(stream)

    decoder = DMRDecoder(sample_rate=RATE)
    voice: dict[int, int] = {0: 0, 1: 0}
    process = decoder._process_voice_burst

    def count(burst, sync_type, timeslot, timestamp):
        voice[timeslot] += 1
        return process(burst, sync_type, timeslot, timestamp)

    decoder._process_voice_burst = count  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)

    # Every voice burst after the lead-in, on both slots (the first may be missed while locking).
    sent = 6 * superframes
    assert voice[0] >= sent
    assert voice[1] >= sent - 1
