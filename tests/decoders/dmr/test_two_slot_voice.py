"""A repeater carrying voice on both timeslots: every voice burst of each slot is decoded.

A slot's voice superframe is 6 bursts: A has the voice sync, B-F embedded signalling.
With both slots in voice, up to 10 bursts in a row have no sync. The decoder used to drop
its lock whenever the next burst was a sync-less burst of a slot it had not yet seen in a
call, and a lock drop cleared both slots' calls: joined mid-call, or after one missed
sync, both slots stayed silent in 5 of the 6 ways their superframes can align. A missed
A sync on one slot also needs 11 bursts in a row by timing on that slot.
The signal here is synthesized (_synth): valid CACH, syncs and EMBs, RRC-shaped 4FSK, as IQ.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdr.core.demod_spec import DemodSpec
from tsdr.radio.decoders.dmr.constants import BURST_DIBITS, SYMBOL_RATE, SYNC_START
from tsdr.radio.decoders.dmr.decoder import DMRDecoder
from tsdr.radio.registry import make_demodulator

from ._synth import RATE, noise, to_iq, voice_burst


@pytest.mark.parametrize("phase", range(6))
def test_both_slots_decoded_after_a_late_entry_and_a_missed_sync(phase: int) -> None:
    """`phase`: slot 1's place in its superframe when slot 0 starts one (its A follows
    slot 0's by 1, 11, 9, 7, 5 or 3 bursts). The decoder joins both calls mid-superframe;
    slot 0's A sync in superframe 8 is missed."""
    rng = np.random.default_rng(10 + phase)
    superframes, missed, counted_from = 14, 8 * 6, 4 * 6
    stream: list[int] = []
    for k in range(3, 3 + superframes * 6):  # join at slot 0's burst D
        stream += voice_burst(0, k % 6, rng, bad_sync=k == missed)
        stream += voice_burst(1, (k + phase) % 6, rng)
    iq = to_iq(stream)

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
    sent = [voice_burst(slot, letter, rng) for letter in [*range(6)] * 6 for slot in (0, 1)]
    iq = to_iq([d for burst in sent for d in burst], offset_hz)

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
        stream += voice_burst(0, k % 6, rng)
        stream += voice_burst(1, (k + 3) % 6, rng)
    signal = to_iq(stream)
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
    decoder = make_demodulator(DemodSpec(mode="DMR", dmr_slot="both"), RATE)  # as tsdr builds it
    assert isinstance(decoder, DMRDecoder)
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


def _voice_heard(iq: np.ndarray, t_from: float) -> dict[int, int]:
    """Voice bursts per slot that reach the vocoder from capture time `t_from` (the IQ
    begins at 1000 s, in 20 ms chunks)."""
    decoder = DMRDecoder(sample_rate=RATE)
    heard = {0: 0, 1: 0}
    process = decoder._process_voice_burst

    def count(burst, sync_type, timeslot, timestamp):
        ok = process(burst, sync_type, timeslot, timestamp)
        if decoder._ts_state[timeslot].confirmed and timestamp >= t_from:
            heard[timeslot] += 1
        return ok

    decoder._process_voice_burst = count  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)
    return heard


def test_a_second_repeater_on_the_frequency_is_heard() -> None:
    """Two repeaters on one frequency, color codes 1 and 2. The color code a burst taken
    by timing must carry was the one heard most since the last retune: after 3.6 s of
    repeater A, B's bursts B-F were refused and its calls were lost. It is the locked
    signal's own."""
    rng = np.random.default_rng(5)
    a: list[int] = []
    b: list[int] = []
    for k in range(10 * 6):
        a += voice_burst(0, k % 6, rng) + voice_burst(1, (k + 3) % 6, rng)
    for k in range(6 * 6):
        b += voice_burst(0, k % 6, rng, color_code=2)
        b += voice_burst(1, (k + 3) % 6, rng, color_code=2)
    first, gap = to_iq(a), noise(int(0.5 * RATE), rng, 0.0)
    b_starts = 1000.0 + (len(first) + len(gap)) / RATE
    heard = _voice_heard(np.concatenate([first, gap, to_iq(b)]), b_starts + 1.0)
    burst_s = BURST_DIBITS / SYMBOL_RATE
    per_slot = int((len(b) / BURST_DIBITS * burst_s - 1.0) / (2 * burst_s))  # B's after 1 s
    assert min(heard.values()) >= per_slot - 2, (heard, per_slot)


def test_a_radio_in_direct_mode_is_heard() -> None:
    """A radio transmitting direct (simplex, or heard on a repeater's input) sends the MS
    syncs, nothing where the CACH would be (here its carrier stays on with random symbols
    there) and nothing in the other slot. The lock wanted a clean CACH to acquire, dropped
    on one contradicting the slot alternation, and dropped after 12 bursts without a sync
    (the radio's A syncs are 11 apart): 3-80 % of its voice was heard."""
    rng = np.random.default_rng(11)
    stream: list[int] = []
    for k in range(3, 3 + 12 * 6):  # joined mid-call
        stream += voice_burst(0, k % 6, rng, mobile=True)
        stream += list(rng.integers(0, 4, BURST_DIBITS))
    heard = _voice_heard(to_iq(stream), 1001.0)
    burst_s = BURST_DIBITS / SYMBOL_RATE
    per_slot = int((len(stream) / BURST_DIBITS * burst_s - 1.0) / (2 * burst_s))
    assert heard[0] >= per_slot - 2 and heard[1] == 0, (heard, per_slot)


def test_both_calls_go_on_when_both_slots_miss_their_a_sync() -> None:
    """A fade that takes both slots' A syncs in one superframe: 22 bursts in a row carry no
    sync, and the lock was dropped after 12, cutting 3-8 bursts from each call."""
    rng = np.random.default_rng(13)
    stream: list[int] = []
    for k in range(14 * 6):  # slot 1's A three bursts after slot 0's
        stream += voice_burst(0, k % 6, rng, bad_sync=k == 8 * 6)
        stream += voice_burst(1, (k + 3) % 6, rng, bad_sync=k == 8 * 6 + 3)
    heard = _voice_heard(to_iq(stream), 1001.0)
    burst_s = BURST_DIBITS / SYMBOL_RATE
    per_slot = int((len(stream) / BURST_DIBITS * burst_s - 1.0) / (2 * burst_s))
    assert min(heard.values()) >= per_slot - 2, (heard, per_slot)


def test_a_transmitter_at_low_deviation_is_read() -> None:
    """At 75 % deviation the outer symbols sat 0.08 above the slicer's fixed 2/3 threshold,
    and noise at 15 dB C/N pushed some of them over. The levels are measured from each
    sync."""
    rng = np.random.default_rng(17)
    sent = [voice_burst(slot, letter, rng) for letter in [*range(6)] * 8 for slot in (0, 1)]
    clean = to_iq([d for burst in sent for d in burst], deviation=0.75)
    iq = clean + noise(len(clean), rng, 15.0)

    decoder = DMRDecoder(sample_rate=RATE)
    received: list[np.ndarray] = []
    process = decoder._process_voice_burst

    def keep(burst, sync_type, timeslot, timestamp):
        if timestamp >= 1001.0:
            received.append(burst.copy())
        return process(burst, sync_type, timeslot, timestamp)

    decoder._process_voice_burst = keep  # type: ignore[method-assign]
    chunk = int(RATE * 0.02)
    for i in range(0, len(iq), chunk):
        decoder.demodulate(iq[i : i + chunk], 1000.0 + i / RATE)

    # Each received burst against the sent one it is nearest to, outside the sync/EMB field.
    payload = np.r_[0:SYNC_START, SYNC_START + 24 : BURST_DIBITS]
    sent_payloads = np.array(sent)[:, payload]
    errors = [int(np.min(np.count_nonzero(sent_payloads != b[payload], axis=1))) for b in received]
    assert len(received) >= 40
    assert sum(errors) / (len(received) * len(payload)) < 0.002
