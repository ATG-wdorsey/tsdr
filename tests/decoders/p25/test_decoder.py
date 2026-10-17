"""P25 Phase 1 control channel: the FEC and CRCs against reference values, and the decoder end to
end on synthesized C4FM and CQPSK control channels of a made-up system (TSBKs and MBTs)."""

import numpy as np
import pytest

from tsdr.radio.decoders.p25 import fec, tsbk
from tsdr.radio.decoders.p25.decoder import P25Decoder, P25Site, P25Snapshot
from tsdr.radio.decoders.p25.phase1 import Nid, Phase1Framer, Tsbk

from . import _synth as synth

NAC, WACN, SYSID, RFSS, SITE = 0x2B3, 0x12345, 0x2A5, 2, 7
CYCLES = 6


def test_crc_check_values() -> None:
    assert fec.crc16(b"123456789") == 0xCE3C  # CRC-16/GSM: the TSBK and PDU header CRC
    assert fec.crc32(b"123456789") == 0x765E7680  # CRC-32/CKSUM parameters: the PDU packet CRC


def test_nid_decodes_through_errors() -> None:
    rng = np.random.default_rng(1)
    for _ in range(300):
        nac, duid = int(rng.integers(0, 4096)), int(rng.choice([0, 3, 5, 7, 10, 12, 15]))
        errors = int(rng.integers(0, fec.NID_MAX_ERRORS + 1))
        # bit 0 is the parity bit: a wrong one counts as one more error
        flips = rng.choice(np.arange(64), errors, replace=False)
        word = fec.nid_encode(nac, duid) ^ int(sum(1 << int(b) for b in flips))
        assert fec.nid_decode(word) == (nac, duid, errors)


def test_soft_block_decoding_corrects_noise() -> None:
    rng = np.random.default_rng(2)
    levels = np.array([1.0, 3.0, -1.0, -3.0])  # dibit -> symbol
    for _ in range(100):
        block = rng.integers(0, 256, 12, dtype=np.uint8).tobytes()
        symbols = levels[fec.block_encode(block)] + rng.normal(0, 0.55, fec.BLOCK_DIBITS)
        # A phase step wraps at +-180 degrees: noise carries +3 over to -3 and back.
        wrapped = (symbols + 4.0) % 8.0 - 4.0
        assert fec.block_decode_soft(wrapped)[0] == block


def test_tsdus_decode_whatever_the_nac() -> None:
    # For these NACs a TSDU's NID, with its status symbol, is within 6 bits of the inverted frame
    # sync; a sync seen inside the NID must not end the frame, nor displace the NID when the channel
    # had another NAC (a new system taking over the channel).
    nacs = [0x956, 0x957, 0xC53, 0xD57, 0xD73, 0xDD5, 0xF57, 0x2B3]
    block = synth.tsbk(tsbk.NET_STS_BCST, bytes(8), last=True)
    levels = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)
    for status in range(4):
        dibits = [d for nac in nacs for d in synth.frame(nac, fec.DUID_TSDU, [block], status)]
        for channel_nac in (None, 0x111):
            framer = Phase1Framer()
            framer.expected_nac = channel_nac
            events = framer.process(levels[np.array(dibits)])
            assert [e.nac for e in events if isinstance(e, Tsbk)] == nacs


def test_a_sync_inside_a_nid_leaves_the_better_frame() -> None:
    levels = np.array([1.0, 3.0, -1.0, -3.0], dtype=np.float32)
    block = synth.tsbk(tsbk.NET_STS_BCST, bytes(range(8)), last=True)
    # A new system whose TSDU NIDs hold an inverted sync, received with 6 or 11 bit errors after
    # it, on a channel that had another NAC.
    for errors in (6, 11):
        frame = synth.frame(0x956, fec.DUID_TSDU, [block], 1)
        for i in range(32 - errors, 32):
            frame[24 + i + (i >= 11)] ^= 1  # NID dibit i (a status symbol follows the 11th)
        framer = Phase1Framer()
        framer.expected_nac = NAC
        events = framer.process(levels[np.array(frame * 20)])
        assert [(e.nac, e.errors) for e in events if isinstance(e, Nid)] == [(0x956, errors)] * 20
        assert sum(isinstance(e, Tsbk) for e in events) == 20
    # A false sync just before a frame (padding and the sync's first 11 dibits inverted) whose NID
    # decodes, as another NAC's, with as few errors as the real NID's 5.
    nac = 0x2D5
    real = synth.frame(nac, fec.DUID_TSDU, [block], 0)
    for pos, mask in ((26, 2), (27, 2), (30, 2), (32, 1), (37, 1)):
        real[pos] ^= mask
    stream = synth.frame(nac, fec.DUID_TSDU, [block], 0) * 3 + [0] * 5
    stream += [d ^ 2 for d in real[:11]] + real
    events = Phase1Framer().process(levels[np.array(stream)])
    assert [e for e in events if isinstance(e, Nid)][-1] == Nid(nac, fec.DUID_TSDU, 5)
    assert sum(isinstance(e, Tsbk) for e in events) == 4
    # A new system repeating one broadcast: the sync inside its clean NID starts a candidate that
    # decodes only through the retry for the old NAC (as a TDU with 16 errors).
    broadcast = synth.tsbk(
        tsbk.NET_STS_BCST,
        synth.pack((0, 8), (WACN, 20), (SYSID, 12), (0x1005, 16), (0x70, 8)),
        last=True,
    )
    framer = Phase1Framer()
    framer.expected_nac = 0xE56
    frame = synth.frame(0x956, fec.DUID_TSDU, [broadcast], 1)
    events = framer.process(levels[np.array(frame * 20)])
    assert [e.nac for e in events if isinstance(e, Nid)] == [0x956] * 20
    assert sum(isinstance(e, Tsbk) for e in events) == 20
    # A false sync before each frame (padding and the sync's first 11 dibits inverted). Before a
    # short TDU, the false NID (a TSDU, 11 errors) waits for a block that the next frame's sync cuts
    # off; before a TSDU of NAC 956, the real NID holds a sync of its own, a second candidate.
    tdu = synth.frame(0x11F, fec.DUID_TDU, [], 0)
    tsdu = synth.frame(0x956, fec.DUID_TSDU, [block], 1)
    for frame, nid, blocks in (
        (tdu, Nid(0x11F, fec.DUID_TDU, 0), 0),
        (tsdu, Nid(0x956, fec.DUID_TSDU, 0), 20),
    ):
        framer = Phase1Framer()
        framer.expected_nac = nid.nac
        stream = ([0] * 5 + [d ^ 2 for d in frame[:11]] + frame) * 20 + [0] * 200
        events = framer.process(levels[np.array(stream)])
        assert [e for e in events if isinstance(e, Nid)] == [nid] * 20
        assert sum(isinstance(e, Tsbk) for e in events) == blocks


def _control_channel() -> list[int]:
    return [d for frame in _control_frames() for d in frame]


def _control_frames() -> list[list[int]]:
    p, t = synth.pack, synth.tsbk
    # Identifiers: 1 = 450 MHz FDMA, 12.5 kHz (IDEN_UP_VU, +5 MHz); 3 = 851.0125 MHz TDMA, two
    # slots, 12.5 kHz (IDEN_UP_TDMA); 2 = 851.00625 MHz FDMA, 6.25 kHz (IDEN_UP, -45 MHz), sent
    # after the grant on it, so that call's frequency becomes known only in the next cycle.
    idens = [
        t(
            tsbk.IDEN_UP_VU,
            p((1, 4), (5, 4), (1, 1), (400, 13), (100, 10), (90_000_000, 32)),
            last=False,
        ),
        t(
            tsbk.IDEN_UP_TDMA,
            p((3, 4), (3, 4), (0, 1), (3600, 13), (100, 10), (170_202_500, 32)),
            last=True,
        ),
    ]
    status = [
        t(
            tsbk.NET_STS_BCST,
            p((0, 8), (WACN, 20), (SYSID, 12), (0x1005, 16), (0x70, 8)),
            last=False,
        ),
        t(
            tsbk.RFSS_STS_BCST,
            p((0, 8), (0, 3), (1, 1), (SYSID, 12), (RFSS, 8), (SITE, 8), (0x1005, 16), (0x70, 8)),
            last=True,
        ),
    ]
    # 2002 is updated before and after its MBT grant; 3003 only ever appears as an update's second
    # group, on the TDMA identifier, so its flags stay unknown.
    pairs = p((0x1014, 16), (2002, 16), (0x3015, 16), (3003, 16))
    update, update_last = (t(tsbk.GRP_V_CH_GRANT_UPDT, pairs, last=last) for last in (False, True))
    grant_1001 = t(
        tsbk.GRP_V_CH_GRANT, p((0x04, 8), (0x100A, 16), (1001, 16), (0x101, 24)), last=False
    )
    explicit = t(  # emergency; downlink 450.625, uplink 451.25 MHz
        tsbk.GRP_V_CH_GRANT_UPDT_EXP,
        p((0x80, 8), (0, 8), (0x1032, 16), (0x1064, 16), (4004, 16)),
        last=False,
    )
    grant_800 = t(
        tsbk.GRP_V_CH_GRANT, p((0x00, 8), (0x2011, 16), (6006, 16), (0x606, 24)), last=True
    )
    motorola = t(
        tsbk.GRP_V_CH_GRANT,
        p((0x00, 8), (0x100A, 16), (5005, 16), (0x505, 24)),
        last=False,
        mfid=0x90,
    )
    iden_800 = t(
        tsbk.IDEN_UP, p((2, 4), (100, 9), (0, 1), (180, 8), (50, 10), (170_201_250, 32)), last=True
    )
    mbt_grant = synth.mbt(  # encrypted; downlink 450.25, uplink 451.5 MHz
        tsbk.GRP_V_CH_GRANT,
        p((0x202, 24)),
        bytes([0x44, 0]),
        p((0, 16), (0x1014, 16), (0x1078, 16), (2002, 16)),
    )
    mbt_adj = synth.mbt(  # flags: valid and active
        tsbk.ADJ_STS_BCST,
        p((0, 8), (3, 4), (SYSID, 12)),
        bytes([RFSS, 9]),
        p((0x1028, 16), (0x108C, 16), (0x70, 8)),
    )
    # A lead-in of system broadcasts lets the receiver settle before the identifiers and calls.
    frames = [synth.frame(NAC, fec.DUID_TSDU, status) for _ in range(4)]
    for _ in range(CYCLES):
        frames.append(synth.frame(NAC, fec.DUID_TSDU, status))
        frames.append(synth.frame(NAC, fec.DUID_TSDU, idens))
        frames.append(synth.frame(NAC, fec.DUID_TSDU, [grant_1001, update_last]))
        frames.append(synth.frame(NAC, fec.DUID_PDU, mbt_grant))
        frames.append(synth.frame(NAC, fec.DUID_TSDU, [update, explicit, grant_800]))
        frames.append(synth.frame(NAC, fec.DUID_TSDU, [motorola, iden_800]))
        frames.append(synth.frame(NAC, fec.DUID_PDU, mbt_adj))
    return frames


@pytest.fixture(scope="module", params=["c4fm", "cqpsk"])
def decoded(request) -> tuple[P25Decoder, list[str], P25Snapshot]:
    modulate = synth.c4fm if request.param == "c4fm" else synth.cqpsk
    iq = synth.receive(
        modulate(_control_channel()), np.random.default_rng(3), snr_db=20, offset_hz=400
    )
    # Built for another rate first, as after a sample rate change in a running pipeline.
    decoder = P25Decoder(sample_rate=4 * synth.RATE)
    decoder.set_sample_rate(synth.RATE)
    texts, snapshots = [], []
    for start in range(0, len(iq), 12_000):
        decoder.demodulate(iq[start : start + 12_000], start / synth.RATE)
        for m in decoder.get_messages():
            if isinstance(m.data, P25Snapshot):
                snapshots.append(m.data)
            elif m.text:
                texts.append(m.text)
    return decoder, texts, snapshots[-1]


def test_system_and_site(decoded) -> None:
    _, _, snap = decoded
    assert (snap.nac, snap.wacn, snap.sysid, snap.rfss, snap.site) == (NAC, WACN, SYSID, RFSS, SITE)
    # From the MBT (downlink first), resolved with the TSBK's identifier.
    assert snap.adjacent == (P25Site(SYSID, RFSS, 9, 450_500_000, 3),)


def test_every_block_after_the_first_cycle(decoded) -> None:
    framer = decoded[0]._framer  # the last snapshot may be up to half a second old
    assert framer.blocks_ok >= 11 * (CYCLES - 1) and framer.mbts_ok >= 2 * (CYCLES - 1)
    assert framer.blocks_bad == 0 and framer.pdus_bad == 0


def test_calls_and_their_reports(decoded) -> None:
    _, texts, snap = decoded
    assert texts == [
        "TG 1001 from 257 on 450.1250 MHz",  # FDMA: no slot; priority 4 is not a flag
        "TG 2002 on 450.2500 MHz ENC?",  # an update before the grant: flags unknown
        "TG 3003 on 851.1375 MHz slot 1 ENC?",  # TDMA identifier, an update's second group
        "TG 2002 from 514 on 450.2500 MHz ENC",  # the MBT grant brings the flags
        "TG 4004 on 450.6250 MHz EMERG",  # explicit update: the downlink is the first channel
        "TG 6006 from 1542 on channel 0x2011",  # its identifier not announced yet
        "TG 6006 from 1542 on 851.1125 MHz",  # the next cycle: the 800 MHz identifier (IDEN_UP)
    ]  # and nothing from the manufacturer's (MFID 0x90) grant
    calls = {c.group: c for c in snap.calls}
    assert calls.keys() == {1001, 2002, 3003, 4004, 6006}
    # Updates after the grant (the last message for 2002) keep its flags.
    assert (calls[2002].svcopts, calls[2002].source, calls[2002].frequency_hz) == (
        0x44,
        0x202,
        450_250_000,
    )
    assert (calls[3003].frequency_hz, calls[3003].slot, calls[3003].svcopts) == (
        851_137_500,
        1,
        None,
    )


def _decode(iq: np.ndarray) -> tuple[P25Decoder, list[str], list[P25Snapshot]]:
    decoder = P25Decoder(sample_rate=synth.RATE)
    texts, snapshots = [], []
    for start in range(0, len(iq), 12_000):
        decoder.demodulate(iq[start : start + 12_000], start / synth.RATE)
        for m in decoder.get_messages():
            if isinstance(m.data, P25Snapshot):
                snapshots.append(m.data)
            elif m.text:
                texts.append(m.text)
    return decoder, texts, snapshots


def test_finds_a_receiver_far_off_frequency() -> None:
    # The offset tracking alone measures only within +-600 Hz; the coarse search finds +1200.
    iq = synth.receive(
        synth.cqpsk(_control_channel()), np.random.default_rng(6), snr_db=20, offset_hz=1700
    )
    decoder, _, snapshots = _decode(iq)
    assert decoder._framer.blocks_ok >= 11 * (CYCLES - 2)
    assert abs(snapshots[-1].frequency_offset_hz - 1700) < 30


def test_a_later_message_overrides_an_earlier_one() -> None:
    # An MBT grant, then an update moving the call: however the IQ is cut into batches, the update
    # has the last word.
    p, t = synth.pack, synth.tsbk
    status = t(
        tsbk.NET_STS_BCST, p((0, 8), (WACN, 20), (SYSID, 12), (0x1005, 16), (0x70, 8)), last=True
    )
    iden = t(
        tsbk.IDEN_UP_VU,
        p((1, 4), (5, 4), (1, 1), (400, 13), (100, 10), (90_000_000, 32)),
        last=True,
    )
    grant = synth.mbt(
        tsbk.GRP_V_CH_GRANT,
        p((0x101, 24)),
        bytes([0x44, 0]),
        p((0, 16), (0x1010, 16), (0x1080, 16), (1001, 16)),
    )
    update = t(
        tsbk.GRP_V_CH_GRANT_UPDT, p((0x1020, 16), (1001, 16), (0x1020, 16), (1001, 16)), last=True
    )
    frames = [synth.frame(NAC, fec.DUID_TSDU, [status])] * 5
    frames += [synth.frame(NAC, fec.DUID_TSDU, [iden])] * 3
    frames += [synth.frame(NAC, fec.DUID_PDU, grant), synth.frame(NAC, fec.DUID_TSDU, [update])]
    iq = synth.cqpsk([d for f in frames for d in f] + [0] * 200)
    for lead in (0, 1000, 6000, 11_000):
        _, texts, snapshots = _decode(np.concatenate([np.zeros(lead, np.complex64), iq]))
        assert [(c.group, c.channel) for c in snapshots[-1].calls] == [(1001, 0x1020)]
        assert texts == [
            "TG 1001 from 257 on 450.2000 MHz ENC",
            "TG 1001 from 257 on 450.4000 MHz ENC",
        ]


def test_another_system_on_the_channel_stays_out() -> None:
    p, t = synth.pack, synth.tsbk
    other_nac, other_sys = 0x111, 0x123
    other = [
        t(
            tsbk.NET_STS_BCST,
            p((0, 8), (WACN, 20), (other_sys, 12), (0x1005, 16), (0x70, 8)),
            last=False,
        ),
        t(  # its identifier 1 is at 460 MHz
            tsbk.IDEN_UP_VU,
            p((1, 4), (5, 4), (1, 1), (400, 13), (100, 10), (92_000_000, 32)),
            last=False,
        ),
        t(tsbk.GRP_V_CH_GRANT, p((0x04, 8), (0x100A, 16), (7007, 16), (0x707, 24)), last=True),
    ]
    theirs = synth.frame(other_nac, fec.DUID_TSDU, other)
    # A frame of the other system after every fourth of ours, then the other system alone.
    ours = _control_frames()
    mixed = [d for i, f in enumerate(ours) for d in f + (theirs if i % 4 == 3 else [])]
    takeover = len(mixed) / synth.SYMBOL_RATE
    iq = synth.receive(
        synth.cqpsk(mixed + theirs * 40), np.random.default_rng(7), snr_db=25, offset_hz=0
    )
    _, texts, snapshots = _decode(iq)
    before = [snap for snap in snapshots if snap.time < takeover][-1]
    assert (before.nac, before.sysid) == (NAC, SYSID)
    assert 7007 not in {c.group for c in before.calls}
    assert {c.frequency_hz for c in before.calls if c.group == 1001} == {450_125_000}
    after = snapshots[-1]
    assert (after.nac, after.sysid) == (other_nac, other_sys)
    assert [(c.group, c.frequency_hz) for c in after.calls] == [(7007, 460_125_000)]
    # Its calls are never resolved with our plan (450 MHz).
    theirs_reported = [text for text in texts if text.startswith("TG 7007")]
    assert theirs_reported[-1] == "TG 7007 from 1799 on 460.1250 MHz"
    assert not any("450." in text for text in theirs_reported)
