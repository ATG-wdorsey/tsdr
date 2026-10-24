"""P25 following's logic without DSP: the call records' identity, the one outcome rule, the follower's phases, and
what a talkgroup learns. A sequence checker outside the suite drives longer sequences; these pin the rules most easily
broken."""

from tsdr.radio.decoders.p25 import systems
from tsdr.radio.decoders.p25.calls import DONE, PAUSED, CallRecord, CallTracker
from tsdr.radio.decoders.p25.decoder import P25Voice
from tsdr.radio.decoders.p25.follower import HANG, WAIT, Follower
from tsdr.radio.decoders.p25.voice_channel import VoiceReport

SYS = (0xBEE00, 0x2A1)
CH_A, CH_B = 0x1010, 0x1020
AES = 0x84


def _report(voices=(), frames=0, on_air=None) -> VoiceReport:
    return VoiceReport(audio=(), texts=(), frames=frames, voices=tuple(voices), on_air=on_air)


def _voice(group, algid=None, kid=None) -> P25Voice:
    return P25Voice(
        nac=0x1A5,
        group=group,
        source=None,
        target=None,
        algid=algid,
        flagged=False,
        emergency=False,
        started=0.0,
        last_heard=0.0,
        kid=kid,
    )


def test_a_new_call_inherits_nothing_and_a_closed_one_is_never_continued() -> None:
    # Reviews 41-42: a clear call found by updates took an encrypted call's flags. Rule C1/C2.
    t = CallTracker()
    enc, new = t.message(SYS, 1001, CH_A, 0x44, 7, 10.0, None)
    assert new and enc.flagged
    other, new = t.message(SYS, 1001, CH_B, None, None, 10.5, None)  # its update on another channel
    assert new and other.svcopts is None and other.source is None
    # back: a new call again, not `enc`
    again, new = t.message(SYS, 1001, CH_A, None, None, 10.8, None)
    assert new and again is not enc and again.svcopts is None
    # Time steps back: the closed call stays closed when time catches up (C3, C4).
    assert [c.cid for c in t.close_inactive(9.0, None)] == [enc.cid, other.cid, again.cid]
    later, new = t.message(SYS, 1001, CH_A, None, None, 10.9, None)
    assert new and later.cid not in (enc.cid, other.cid, again.cid)


def test_the_outcome_rule() -> None:
    def call(**kw) -> CallRecord:
        c = CallRecord(1, SYS, 1001, CH_A, 0.0, 0.0)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    assert call(learned=(AES, 7), audible=True).outcome == "learned"
    # a clear key: the flag was wrong
    assert call(learned=(0x80, 0), audible=True).outcome == "played"
    # D1 (Q146): muted, then its talkgroup turned off
    assert call(ldus=3, state=PAUSED, reason="unfollowed", taken=1.0).outcome == "muted"
    assert call(state=PAUSED, reason="unfollowed", taken=1.0).outcome == "not_followed"
    assert call(state=DONE, reason="no_voice", taken=1.0, verdict_at_take="busy").outcome == "busy"
    assert call(state=DONE, reason="no_voice", taken=1.0).outcome == "no_voice"
    assert call(verdict="encrypted").outcome == "encrypted"
    assert call().outcome == "unknown"


def test_a_call_taken_again_that_had_voice_waits_out_a_hang_time() -> None:
    # D2 (Q145): a call paused (its talkgroup off) and taken again had voice: no voice within VOICE_WAIT_S of the
    # take ends its wait in HANG, not DONE(no_voice), so its next transmission without a grant is still heard.
    f = Follower()
    c = CallRecord(1, SYS, 1001, CH_A, 0.0, 0.0)
    f.take(c, 0.0)
    f.voice(_report([_voice(1001)], frames=2, on_air=_voice(1001)), 0.2, True, False, True)
    assert f.voice(_report(), 0.4, False, False, True).stop == "unfollowed"
    f.stop("unfollowed")
    f.take(c, 1.0)
    for now in (1.5, 2.05):
        decision = f.voice(_report(), now, True, False, True)
    assert decision.stop is None and f.phase == HANG
    # its hang time from the take
    assert f.voice(_report(), 3.05, True, False, True).stop == "ended"
    fresh = CallRecord(2, SYS, 2002, CH_B, 0.0, 0.0)
    f2 = Follower()
    f2.take(fresh, 0.0)
    assert f2.phase == WAIT and f2.voice(_report(), 1.05, True, False, True).stop == "no_voice"


def test_an_algorithm_is_learned_only_once_its_transmission_names_the_talkgroup() -> None:
    # The sequence grid's finding: a call taken while its channel still carries another talkgroup's transmission
    # receives that one's LDU2 before any link control; its algorithm isn't the followed talkgroup's.
    f = Follower()
    c = CallRecord(1, SYS, 1001, CH_A, 0.0, 0.0)
    f.take(c, 0.0)
    unnamed = _voice(None, AES, 9)
    decision = f.voice(_report([unnamed], frames=1, on_air=unnamed), 0.2, True, False, True)
    assert decision.learn is None and decision.stop is None
    other = _voice(2002, AES, 9)  # its link control: another talkgroup's
    assert (
        f.voice(_report([other], frames=1, on_air=other), 0.4, True, False, True).stop
        == "other_call"
    )
    g = Follower()
    g.take(CallRecord(2, SYS, 1001, CH_A, 0.0, 0.0), 0.0)
    own = _voice(1001, AES, 9)
    decision = g.voice(_report([own], frames=1, on_air=own), 0.2, True, False, True)
    assert (decision.learn, decision.stop) == ((AES, 9), "learned")


def test_a_rescan_is_cleared_only_by_a_call_taken_after_it() -> None:
    # D9: a pending look is cleared by a call taken while it was pending that adds or refreshes a key; a second
    # rescan during that call stays pending.
    systems.clear()
    s = systems.system_for(*SYS)
    s.learn(1001, AES, 7, 1.0, True, None)
    s.rescan(1001)
    look = s.look(1001)
    assert s.pending(1001) and not s.skips_flagged(1001)
    s.rescan(1001)  # again, during the call taken under the first
    s.learn(1001, AES, 7, 2.0, True, look)
    assert s.pending(1001)
    s.learn(1001, AES, 7, 3.0, True, s.look(1001))
    assert not s.pending(1001) and s.skips_flagged(1001)
    assert [k[:2] for k in s.get(1001).keys] == [(AES, 7)]
