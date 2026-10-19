"""P25 trunking (M4a): the decoder follows granted calls on a second channel while it keeps decoding the control channel,
on a synthesized site in one capture (`_site.py`): the follow choices, one call at a time, the encryption rule (a followed
talkgroup's encrypted call is looked at until its algorithm and key are known, then skipped), the capture's edges."""

import numpy as np
import pytest

from tsdr.radio.decoders.p25 import talkgroups
from tsdr.radio.decoders.p25.decoder import P25Decoder, P25Snapshot
from tsdr.radio.vocoder.agc import VoiceAGC
from tsdr.radio.vocoder.imbe import ImbeDecoder
from tsdr.radio.vocoder.imbe.decoder import encode_frame

from . import _site as site
from .test_voice import VECTORS

AES_256, KEY = 0x84, 0x0007
# Channels within 200 kHz of the control channel (number 200) are inside the 480 kS/s capture. A talkgroup's calls
# are more than _ACTIVE_S apart, so each is a call of its own.
SCENARIO = [
    # followed: plays
    site.Call(1001, 190, 2.0, ldus=8),
    # while 1001 plays: one call at a time
    site.Call(4004, 214, 2.5, ldus=4),
    # flagged: looked at, its type learned, left
    site.Call(2002, 210, 5.5, ldus=6, algid=AES_256, kid=KEY),
    # not followed
    site.Call(3003, 186, 7.0, ldus=4),
    # flagged, type known: skipped
    site.Call(2002, 210, 10.0, ldus=6, algid=AES_256, kid=KEY),
    # 750 kHz away: outside the capture
    site.Call(5005, site.CC_NUMBER + 60, 10.2, voice=False),
    # the same talkgroup in the clear: plays
    site.Call(2002, 210, 14.5, ldus=4),
    # its grant missed: found by an update
    site.Call(6006, 195, 17.0, ldus=8, grant=False, header=False),
    # followed: plays
    site.Call(8008, 205, 21.0, ldus=4),
    # in 8008's hang time the site gives its channel to a talkgroup not followed: not to be played
    site.Call(3003, 205, 22.4, ldus=4),
]


@pytest.fixture(scope="module")
def speech() -> list[np.ndarray]:
    raw = np.frombuffer(VECTORS.read_bytes(), dtype=np.uint8).reshape(-1, 11)
    return [encode_frame(v) for v in np.unpackbits(raw, axis=1)[:36, :88]]


@pytest.fixture(scope="module")
def followed(speech):
    talkgroups.clear()
    talkgroups.table_for(site.WACN, site.SYSID).set_follow(3003, False)
    s = site.Site(SCENARIO, duration=26.0, error_hz=1500.0)
    iq = s.capture(speech)
    decoder = P25Decoder(s.rate)
    decoder.set_tuned_frequency(site.CC_HZ)
    decoder.set_capture_center(site.CC_HZ)
    audio: list[tuple[float, np.ndarray]] = []
    snapshots: list[P25Snapshot] = []
    for start in range(0, len(iq), 24_000):
        decoder.demodulate(iq[start : start + 24_000], start / s.rate)
        audio += [(start / s.rate, b.samples) for b in decoder.get_audio()]
        snapshots += [m.data for m in decoder.get_messages() if isinstance(m.data, P25Snapshot)]
    return audio, snapshots


def _audio(audio, start: float, end: float) -> np.ndarray:
    parts = [a for t, a in audio if start <= t < end]
    return np.concatenate(parts) if parts else np.zeros(0)


def _log(snapshots, group: int, after: float):
    return [e for e in snapshots[-1].log if e.group == group and e.start >= after - 0.5]


def test_a_granted_call_plays_its_speech(followed, speech) -> None:
    audio, snapshots = followed
    imbe = ImbeDecoder()
    agc = VoiceAGC(max_gain=50.0, start_gain=25.0, peak=30000 / 32768)
    frames = [speech[j % len(speech)] for j in range(8 * 9)]
    expected = np.concatenate([agc.process(imbe.decode_frame_float(f) / 32768.0) for f in frames])
    np.testing.assert_allclose(_audio(audio, 2.0, 4.5), expected, atol=1e-6)
    [entry] = _log(snapshots, 1001, 2.0)
    assert (
        entry.outcome == "played" and entry.ldus == 8 and entry.frequency_hz == site.frequency(190)
    )


def test_one_call_at_a_time(followed) -> None:
    _, snapshots = followed
    [entry] = _log(snapshots, 4004, 2.5)
    assert entry.outcome == "busy" and entry.ldus == 0


def test_an_encrypted_talkgroup_is_looked_at_once(followed) -> None:
    audio, snapshots = followed
    assert len(_audio(audio, 5.5, 7.0)) == 0
    first, second, clear = sorted(_log(snapshots, 2002, 5.5), key=lambda e: e.start)
    assert first.outcome == "learned" and (first.algid, first.kid) == (AES_256, KEY)
    assert (
        first.ldus <= 2
    )  # left once the header or the first LDU2 gave the type, long before its six LDUs end
    assert second.outcome == "encrypted" and second.ldus == 0
    assert clear.outcome == "played" and clear.ldus == 4
    assert len(_audio(audio, 14.5, 16.5)) >= 4 * 9 * 160
    [tg] = [t for t in snapshots[-1].talkgroups if t.group == 2002]
    assert (tg.algid, tg.kid) == (AES_256, KEY)


def test_a_talkgroup_not_followed_is_listed_only(followed) -> None:
    audio, snapshots = followed
    entry = min(_log(snapshots, 3003, 7.0), key=lambda e: e.start)
    assert entry.outcome == "not_followed"
    assert len(_audio(audio, 7.0, 8.9)) == 0
    assert any(c.call.group == 3003 and not c.followed for s in snapshots for c in s.channels)


def test_another_call_on_the_channel_in_the_hang_time(followed) -> None:
    audio, snapshots = followed
    [entry] = _log(snapshots, 8008, 21.0)
    assert entry.outcome == "played" and entry.ldus == 4
    [other] = _log(snapshots, 3003, 22.4)
    assert other.outcome == "not_followed" and other.ldus == 0
    assert len(_audio(audio, 22.45, 26.0)) == 0


def test_a_call_outside_the_capture_is_listed_only(followed) -> None:
    _, snapshots = followed
    [entry] = _log(snapshots, 5005, 10.2)
    assert entry.outcome == "outside"
    assert any(c.call.group == 5005 and c.inside is False for s in snapshots for c in s.channels)
    assert snapshots[-1].site_span == (site.frequency(186), site.frequency(site.CC_NUMBER + 60))


def test_a_call_found_by_its_update(followed) -> None:
    audio, snapshots = followed
    [entry] = _log(snapshots, 6006, 17.0)
    assert entry.outcome == "played" and entry.ldus >= 4
    assert len(_audio(audio, 17.0, 20.0)) >= 4 * 9 * 160


def test_the_control_channel_keeps_decoding(followed) -> None:
    _, snapshots = followed
    last = snapshots[-1]
    assert last.tsbk_pct >= 99.0
    # About 36 blocks a second for 26 s, the calls' time included.
    assert last.tsbks >= 26 * 30


def test_turning_a_talkgroup_off_stops_its_call(speech) -> None:
    # Review 38, F1: the follow choice was checked only when a call was taken; review 39, F2: a
    # call stopped so was never taken again.
    talkgroups.clear()
    s = site.Site([site.Call(1001, 190, 1.0, ldus=30)], duration=7.5, error_hz=-700.0, seed=4)
    iq = s.capture(speech)
    decoder = P25Decoder(s.rate)
    decoder.set_tuned_frequency(site.CC_HZ)
    decoder.set_capture_center(site.CC_HZ)
    audio: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(iq), 24_000):
        t = start / s.rate
        if 3.5 <= t < 4.5:
            talkgroups.table_for(site.WACN, site.SYSID).set_follow(1001, False)
        elif t >= 4.5:  # and back on during the same call (review 39, F2): it is taken again
            talkgroups.table_for(site.WACN, site.SYSID).set_follow(1001, True)
        decoder.demodulate(iq[start : start + 24_000], t)
        audio += [(t, b.samples) for b in decoder.get_audio()]
    assert len(_audio(audio, 1.0, 3.5)) >= 9 * 9 * 160  # it played until then
    assert len(_audio(audio, 3.5, 4.5)) == 0
    assert len(_audio(audio, 4.5, 7.5)) >= 6 * 9 * 160


@pytest.mark.parametrize(
    ("number", "start"),
    [
        (190, 7.0),  # the same channel, after a pause (review 41, R41-1)
        (192, 4.2),  # another channel, a second after the encrypted call ends (review 42, R42-1)
    ],
)
def test_a_clear_call_after_an_encrypted_one_found_by_updates(speech, number, start) -> None:
    # A new call seen only through updates (its grant missed) took the last call's encrypted flag,
    # and with the talkgroup's encryption learned, it was skipped.
    talkgroups.clear()
    calls = [
        site.Call(1001, 190, 2.0, ldus=4, algid=AES_256, kid=KEY),
        site.Call(1001, number, start, ldus=8, grant=False),
    ]
    s = site.Site(calls, duration=start + 4.0, error_hz=700.0, seed=6)
    iq = s.capture(speech)
    decoder = P25Decoder(s.rate)
    decoder.set_tuned_frequency(site.CC_HZ)
    decoder.set_capture_center(site.CC_HZ)
    audio: list[tuple[float, np.ndarray]] = []
    for i in range(0, len(iq), 24_000):
        decoder.demodulate(iq[i : i + 24_000], i / s.rate)
        audio += [(i / s.rate, b.samples) for b in decoder.get_audio()]
    assert len(_audio(audio, 2.0, start)) == 0  # the encrypted call: muted, its type learned
    assert len(_audio(audio, start, start + 4.0)) >= 6 * 9 * 160
