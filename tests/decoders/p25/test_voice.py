"""P25 Phase 1 voice: the voice-frame codes against reference vectors (DSD's encoders), and the decoder
end to end on synthesized voice calls carrying public speech (tests/samples/imbe) as C4FM."""

from pathlib import Path

import numpy as np
import pytest

from tsdr.radio.decoders.p25 import fec, voice
from tsdr.radio.decoders.p25 import voice_fec as vf
from tsdr.radio.decoders.p25.decoder import P25Decoder, P25Snapshot
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.imbe import ImbeDecoder
from tsdr.radio.vocoder.imbe.decoder import encode_frame

from . import _synth as synth

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "tests" / "samples" / "imbe" / "bmh_gasline.imbe"
NAC, TALKGROUP, SOURCE = 0x2B3, 1810, 0x123456


def test_codes_match_reference_vectors() -> None:
    # Parity words from DSD's encoders (an oracle outside the repository): Hamming(10,6,3),
    # Golay(24,12) and (18,6), and the three Reed-Solomon codes.
    assert [vf.hamming_10_6_encode(d) & 0xF for d in (1, 42, 63)] == [12, 6, 0]
    assert [vf.golay_24_12_encode(d) & 0xFFF for d in (1, 1445, 4095)] == [0x8EB, 0x6E4, 0xFFF]
    assert [vf.golay_18_6_encode(d) & 0xFFF for d in (1, 42)] == [0x8EB, 0x235]
    assert vf.rs_encode(list(range(1, 13)), 12) == [26, 33, 46, 33, 12, 58, 60, 23, 17, 40, 1, 58]
    assert vf.rs_encode(list(range(1, 17)), 8) == [12, 60, 47, 40, 48, 12, 57, 52]
    assert vf.rs_encode(list(range(1, 21)), 16) == [
        58,
        1,
        38,
        1,
        2,
        60,
        45,
        8,
        41,
        38,
        7,
        11,
        41,
        49,
        40,
        10,
    ]


def test_reed_solomon_corrects_to_its_limit() -> None:
    rng = np.random.default_rng(4)
    for n, k in ((24, 12), (24, 16), (36, 20)):
        for errors in range((n - k) // 2 + 1):
            data = [int(v) for v in rng.integers(0, 64, k)]
            word = data + vf.rs_encode(data, n - k)
            for pos in rng.choice(n, errors, replace=False):
                word[pos] ^= int(rng.integers(1, 64))
            assert vf.rs_decode(word, k) == (data, errors)


@pytest.fixture(scope="module")
def speech() -> list[np.ndarray]:
    raw = np.frombuffer(VECTORS.read_bytes(), dtype=np.uint8).reshape(-1, 11)
    return [encode_frame(v) for v in np.unpackbits(raw, axis=1)[:36, :88]]


def _call(speech: list[np.ndarray], algid: int) -> list[int]:
    """A group call: HDU, four LDUs (LDU1 with link control, LDU2 with the encryption sync), TDU,
    with idle frame time before and after."""
    lc = bytes([voice.LCO_GROUP_VOICE, 0, 0x40 if algid != voice.ALGID_CLEAR else 0, 0])
    lc += TALKGROUP.to_bytes(2, "big") + SOURCE.to_bytes(3, "big")
    es = bytes(9) + bytes([algid]) + (0 if algid == voice.ALGID_CLEAR else 7).to_bytes(2, "big")
    dibits = synth.raw_frame(NAC, fec.DUID_TDU, []) * 10
    dibits += synth.raw_frame(NAC, fec.DUID_HDU, synth.hdu(bytes(9), 0, algid, 0, TALKGROUP))
    for k in range(4):
        second = k % 2 == 1
        info = es if second else lc
        dibits += synth.raw_frame(
            NAC,
            fec.DUID_LDU2 if second else fec.DUID_LDU1,
            synth.ldu(speech[9 * k : 9 * k + 9], info, second),
        )
    dibits += synth.raw_frame(NAC, fec.DUID_TDU, []) * 10
    return dibits


def _decode(iq: np.ndarray) -> tuple[np.ndarray, list[P25Snapshot], list[str]]:
    decoder = P25Decoder(sample_rate=synth.RATE)
    audio, snapshots, texts = [], [], []
    for start in range(0, len(iq), 12_000):
        decoder.demodulate(iq[start : start + 12_000], start / synth.RATE)
        audio += [b.samples for b in decoder.get_audio()]
        for m in decoder.get_messages():
            if isinstance(m.data, P25Snapshot):
                snapshots.append(m.data)
            elif m.text:
                texts.append(m.text)
    return (np.concatenate(audio) if audio else np.zeros(0)), snapshots, texts


def test_a_clear_call_plays_its_speech(speech: list[np.ndarray]) -> None:
    iq = synth.receive(
        synth.c4fm(_call(speech, voice.ALGID_CLEAR)),
        np.random.default_rng(5),
        snr_db=25,
        offset_hz=200,
    )
    audio, snapshots, texts = _decode(iq)
    # The same frames through the vocoder and the level control directly.
    imbe = ImbeDecoder()
    agc = VoiceAGC(max_gain=50.0, start_gain=25.0, peak=30000 / 32768)
    expected = np.concatenate([agc.process(imbe.decode_frame_float(f) / 32768.0) for f in speech])
    np.testing.assert_allclose(audio, expected, atol=1e-6)
    heard = [s.voice for s in snapshots if s.voice is not None]
    assert heard and heard[-1].group == TALKGROUP and heard[-1].source == SOURCE
    assert not heard[-1].encrypted
    assert snapshots[-1].voice is None  # the terminator ended it
    assert any(f"Voice TG {TALKGROUP}" in t for t in texts)


def test_an_encrypted_call_is_muted(speech: list[np.ndarray]) -> None:
    iq = synth.receive(
        synth.c4fm(_call(speech, 0x84)), np.random.default_rng(6), snr_db=25, offset_hz=0
    )
    audio, snapshots, _ = _decode(iq)
    assert len(audio) == 0
    heard = [s.voice for s in snapshots if s.voice is not None]
    assert heard and heard[-1].encrypted and heard[-1].algid == 0x84


def _lc(flags: int, group: int = TALKGROUP, lcf: int = voice.LCO_GROUP_VOICE) -> bytes:
    return bytes([lcf, 0, flags, 0]) + group.to_bytes(2, "big") + SOURCE.to_bytes(3, "big")


def _es(algid: int) -> bytes:
    return bytes(9) + bytes([algid]) + bytes(2)


def _samples(dibits: list[int]) -> int:
    idle = synth.raw_frame(NAC, fec.DUID_TDU, []) * 10
    iq = synth.receive(
        synth.c4fm(idle + dibits + idle), np.random.default_rng(7), snr_db=40, offset_hz=150
    )
    audio, _, _ = _decode(iq)
    return len(audio)


def test_a_header_starts_a_new_call(speech: list[np.ndarray]) -> None:
    # An encrypted transmission whose terminator was lost, then a clear one whose first LDU1 was
    # lost too: its header and its LDU2 say clear, so its speech plays (it inherited the old call's
    # encryption flag).
    ldu1 = synth.ldu(speech[:9], _lc(voice.SVC_ENCRYPTED), second=False)
    ldu2 = synth.ldu(speech[:9], _es(voice.ALGID_CLEAR), second=True)
    dibits = synth.raw_frame(NAC, fec.DUID_LDU1, ldu1)
    dibits += synth.raw_frame(NAC, fec.DUID_HDU, synth.hdu(bytes(9), 0, voice.ALGID_CLEAR, 0, 2710))
    dibits += [0] * 864  # an LDU1's time without a frame to find
    dibits += synth.raw_frame(NAC, fec.DUID_LDU2, ldu2)
    assert _samples(dibits) == 9 * 160


def test_unknown_service_options_keep_a_call_muted(speech: list[np.ndarray]) -> None:
    # An LDU1 flags encryption, the LDU2's encryption sync can't be corrected, and the next LDU1's
    # link control is protected or a manufacturer's: what is known (encrypted) stands (unknown
    # options were taken as clear).
    rng = np.random.default_rng(91)
    bad_es = synth.ldu(speech[:9], _es(0x84), second=True)
    for start in voice._WORD_STARTS:
        for k in range(4):
            word = synth._dibits(vf.hamming_10_6_encode(int(rng.integers(0, 64))), 5)
            bad_es[start + 5 * k : start + 5 * k + 5] = word
    assert voice.parse_ldu(np.array(bad_es), second=True).sync is None
    first = synth.ldu(speech[:9], _lc(voice.SVC_ENCRYPTED), second=False)
    for unknown in (_lc(0, lcf=0x80), bytes([0, 0x90]) + _lc(0)[2:]):
        dibits = synth.raw_frame(NAC, fec.DUID_LDU1, first)
        dibits += synth.raw_frame(NAC, fec.DUID_LDU2, bad_es)
        dibits += synth.raw_frame(NAC, fec.DUID_LDU1, synth.ldu(speech[:9], unknown, second=False))
        assert _samples(dibits) == 0


def _call_dibits(
    nac: int, speech: list[np.ndarray], ldus: int, group: int = TALKGROUP
) -> list[int]:
    dibits = synth.raw_frame(nac, fec.DUID_HDU, synth.hdu(bytes(9), 0, voice.ALGID_CLEAR, 0, group))
    for k in range(ldus):
        second = k % 2 == 1
        info = _es(voice.ALGID_CLEAR) if second else _lc(0, group)
        duid = fec.DUID_LDU2 if second else fec.DUID_LDU1
        dibits += synth.raw_frame(nac, duid, synth.ldu(speech[:9], info, second))
    return dibits + synth.raw_frame(nac, fec.DUID_TDU, [])


def test_a_call_after_a_quiet_channel_decodes_from_its_start(speech: list[np.ndarray]) -> None:
    # A voice channel that drops its carrier between calls: 4.2 s of noise alone must not move the
    # coarse frequency search off the shift that decoded, nor blur the frequency tracking (the next
    # call lost its first LDUs).
    first = synth.c4fm(_call_dibits(NAC, speech, 4))
    second = synth.c4fm(_call_dibits(NAC, speech, 6))
    quiet = np.zeros(int(4.2 * synth.RATE), dtype=np.complex64)
    iq = synth.receive(
        np.concatenate([first, quiet, second]), np.random.default_rng(9), snr_db=20, offset_hz=150
    )
    decoder = P25Decoder(sample_rate=synth.RATE)
    for start in range(0, len(iq), 12_000):
        decoder.demodulate(iq[start : start + 12_000], start / synth.RATE)
    assert decoder._voice_frames == 4 + 6


def test_calls_on_the_edge_between_two_shifts(speech: list[np.ndarray]) -> None:
    # 600 Hz below the channel center the fourth-power estimate sits on the end of its range: read
    # plainly it jumps between the two ends and a call lost most of its frames (review 31: 5 of 30).
    # The same transmitter keying up again after a pause finds the side it was read on.
    first = synth.c4fm(_call_dibits(NAC, speech, 30))
    second = synth.c4fm(_call_dibits(NAC, speech, 6))
    quiet = np.zeros(int(1.0 * synth.RATE), dtype=np.complex64)
    iq = synth.receive(
        np.concatenate([first, quiet, second]), np.random.default_rng(3), snr_db=30, offset_hz=-600
    )
    decoder = P25Decoder(sample_rate=synth.RATE)
    frames = []
    for start in range(0, len(iq), 12_000):
        decoder.demodulate(iq[start : start + 12_000], start / synth.RATE)
        frames.append(decoder._voice_frames)
    assert frames[(len(first) + len(quiet)) // 12_000] >= 25  # of 30
    assert frames[-1] - frames[(len(first) + len(quiet)) // 12_000] >= 5  # of 6


@pytest.mark.parametrize(
    ("call_hz", "reply_hz", "gap_s"),
    [
        (580, -100, 1.0),
        (580, -300, 0.15),
        (580, 800, 1.0),
        (-580, -800, 0.15),
        (0, 900, 1.0),
        (900, -900, 1.0),
        (1000, -1000, 1.0),
    ],
)
def test_a_reply_from_another_radio(
    speech: list[np.ndarray], call_hz: int, reply_hz: int, gap_s: float
) -> None:
    # A call, then a short reply from a radio elsewhere on the channel. The fourth-power estimate
    # reads +900 Hz as -300: the reply must be read on its own branch (reviews 32-34: -100 Hz read as
    # +1100, -300 as +900, +800 as -400, -900 after +900 as +1500, each reply lost whole). After +1000
    # the shift follows the call to +1200, and a reply at -1000 is 2.2 kHz from it, beyond what the
    # timing follows: the shift must find it (review 34).
    first = synth.c4fm(_call_dibits(NAC, speech, 8))
    second = synth.c4fm(_call_dibits(NAC, speech, 2))
    start = len(first) + int(gap_s * synth.RATE)
    n = np.arange(start + len(second) + 72_000)
    wanted = np.zeros(len(n), dtype=np.complex64)
    wanted[: len(first)] = first * np.exp(2j * np.pi * call_hz / synth.RATE * n[: len(first)])
    wanted[start : start + len(second)] = second * np.exp(
        2j * np.pi * reply_hz / synth.RATE * n[start : start + len(second)]
    )
    iq = synth.receive(wanted, np.random.default_rng(1), snr_db=30, offset_hz=0)
    decoder = P25Decoder(sample_rate=synth.RATE)
    frames = []
    for k in range(0, len(iq), 12_000):
        decoder.demodulate(iq[k : k + 12_000], k / synth.RATE)
        frames.append(decoder._voice_frames)
    assert frames[start // 12_000] >= 7  # of 8
    assert frames[-1] - frames[start // 12_000] == 2


def test_a_reply_beside_a_strong_neighbor(speech: list[np.ndarray]) -> None:
    # As above, +900 Hz then -900, beside a C4FM neighbor 50 dB stronger 12.5 kHz below the reply,
    # in 10 ms chunks. The neighbor's skirt fills the valley beside the reply to within 4-5 dB of its
    # top, so the reply 2.1 kHz from the shift that followed the call was located only now and then,
    # and the shift never went to it (review 35).
    first = synth.c4fm(_call_dibits(NAC, speech, 8))
    second = synth.c4fm(_call_dibits(NAC, speech, 4))
    start = len(first) + int(synth.RATE)
    n = np.arange(start + len(second) + 72_000)
    wanted = np.zeros(len(n), dtype=np.complex64)
    wanted[: len(first)] = first * np.exp(2j * np.pi * 900 / synth.RATE * n[: len(first)])
    wanted[start : start + len(second)] = second * np.exp(
        -2j * np.pi * 900 / synth.RATE * n[start : start + len(second)]
    )
    rng = np.random.default_rng(1)
    iq = synth.receive(wanted, rng, snr_db=30, offset_hz=0)
    neighbor = synth.c4fm(list(rng.integers(0, 4, len(n) // synth.SPS + 1)))[: len(n)]
    iq += 10 ** (50 / 20) * neighbor * np.exp(-2j * np.pi * 13_400 / synth.RATE * n)
    decoder = P25Decoder(sample_rate=synth.RATE)
    frames = []
    for k in range(0, len(iq), 2_400):
        decoder.demodulate(iq[k : k + 2_400], k / synth.RATE)
        frames.append(decoder._voice_frames)
    assert frames[-1] - frames[start // 2_400] >= 3  # of 4


def test_a_reply_across_the_whole_range(speech: list[np.ndarray]) -> None:
    # A call at -2400 Hz, then a reply at +2400: both within the range the search covers, but the
    # reply is 4.8 kHz from the shift the call left. Looked for around that shift rather than the
    # dial, it was never located, and the whole reply was lost (review 36).
    first = synth.c4fm(_call_dibits(NAC, speech, 8))
    second = synth.c4fm(_call_dibits(NAC, speech, 4))
    start = len(first) + int(synth.RATE)
    n = np.arange(start + len(second) + 72_000)
    wanted = np.zeros(len(n), dtype=np.complex64)
    wanted[: len(first)] = first * np.exp(-2j * np.pi * 2400 / synth.RATE * n[: len(first)])
    wanted[start : start + len(second)] = second * np.exp(
        2j * np.pi * 2400 / synth.RATE * n[start : start + len(second)]
    )
    iq = synth.receive(wanted, np.random.default_rng(1), snr_db=30, offset_hz=0)
    decoder = P25Decoder(sample_rate=synth.RATE)
    frames = []
    for k in range(0, len(iq), 12_000):
        decoder.demodulate(iq[k : k + 12_000], k / synth.RATE)
        frames.append(decoder._voice_frames)
    assert frames[start // 12_000] >= 6  # of 8
    assert frames[-1] - frames[start // 12_000] >= 3  # of 4


def test_a_fresh_call_beyond_the_edge(speech: list[np.ndarray]) -> None:
    # A fresh decoder, a call 720 Hz off at 12 dB. Its first frame sync comes while the timing pulls
    # in and the offset estimate crosses the edge between two branches: decided across that jump, it
    # matched a branch the wrong way (-1700 Hz), and the call waited for the search's next shift. The
    # shift may follow the call (+1200), never away from it.
    call = synth.c4fm(_call_dibits(NAC, speech, 20))
    start = 72_000
    n = np.arange(start + len(call) + 72_000)
    wanted = np.zeros(len(n), dtype=np.complex64)
    wanted[start : start + len(call)] = call * np.exp(
        2j * np.pi * 720 / synth.RATE * n[start : start + len(call)]
    )
    iq = synth.receive(wanted, np.random.default_rng(2), snr_db=12, offset_hz=0)
    decoder = P25Decoder(sample_rate=synth.RATE)
    for k in range(0, len(iq), 12_000):
        decoder.demodulate(iq[k : k + 12_000], k / synth.RATE)
    assert decoder._voice_frames == 20
    assert decoder._demod.coarse_hz >= 0


def test_a_transmission_on_another_nac_is_heard(speech: list[np.ndarray]) -> None:
    # A radio on another NAC (the default 0x293) on a channel whose NAC has locked to 0x2B3.
    dibits = _call_dibits(NAC, speech, 12) + _call_dibits(0x293, speech, 8, group=2001)
    assert _samples(dibits) == (12 + 8) * 9 * 160
