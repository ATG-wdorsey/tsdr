"""Synthesized DMR for the tests: a repeater's voice bursts (valid CACH, syncs and EMBs) as
RRC-shaped 4FSK at 4800 baud, as IQ, and receiver noise."""

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
from tsdr.radio.decoders.dmr.fec import _QR_16_7_6_WORDS, hamming_7_4_decode
from tsdr.radio.dsp import rrc_taps

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


def voice_burst(
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


def to_iq(dibits: list[int], offset_hz: float = 0.0) -> np.ndarray:
    """4FSK as DMR transmitters send it: each symbol a root-raised-cosine frequency pulse
    (alpha 0.2), scaled so the receiver's matching filter lands it on its level."""
    sps = int(RATE / SYMBOL_RATE)
    impulses = np.zeros(len(dibits) * sps)
    impulses[sps // 2 :: sps] = [_LEVEL[d] * dmr.DEVIATION / 3.0 for d in dibits]
    pulse = rrc_taps(sps, 0.2, 8)
    freq = np.convolve(impulses, pulse * (sps / pulse.sum()), mode="same")
    phase = 2 * np.pi * np.cumsum(freq + offset_hz) / RATE
    return (0.5 * np.exp(1j * phase)).astype(np.complex64)


def noise(n: int, rng: np.random.Generator, cn_db: float) -> np.ndarray:
    """Receiver noise for a 0.5-amplitude carrier at `cn_db` C/N in DMR's 12.5 kHz."""
    power = 0.25 / 10 ** (cn_db / 10) * (RATE / 12_500.0)
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return (noise * np.sqrt(power / 2)).astype(np.complex64)
