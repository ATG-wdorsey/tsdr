from tsdr.radio.decoders.tetra import trackers
from tsdr.radio.decoders.tetra.mac import (
    CmceEvent,
    SB1Info,
)
from tsdr.radio.decoders.tetra.state import (
    SYNC_STATE_LOCKED,
    SYNC_STATE_UNLOCKED,
    SYNC_STATE_UNLOCKING,
    TetraState,
)


def _make_sb1(timeslot: int = 0, mcc: int = 204, mnc: int = 500, cc: int = 2) -> SB1Info:
    return SB1Info(
        system_code=0,
        colour_code=cc,
        timeslot=timeslot,
        frame_number=1,
        multiframe_number=1,
        sharing_mode=0,
        ts_reserved=0,
        u_plane_dtx=0,
        frame18_ext=0,
        mcc=mcc,
        mnc=mnc,
        scramble_init=0xDEADBEEF,
    )


# TDMA counter


# network lock


# AACH slot usage


# CMCE atomic call state


def test_record_cmce_d_setup_creates_active_call():
    state = TetraState()
    cmce = CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0)
    trackers.record_cmce(state, cmce, ts=5.0)
    assert 1234 in state.active_calls
    call = state.active_calls[1234]
    assert call.encryption_algo == "clear"


def test_record_cmce_d_release_removes_call():
    state = TetraState()
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0), ts=0.0
    )
    trackers.record_cmce(state, CmceEvent(msg_type="D-RELEASE", call_id=1234), ts=5.0)
    assert 1234 not in state.active_calls


def test_record_cmce_encryption_cannot_desync():
    """A D-CONNECT that flips encryption updates the existing call atomically."""
    state = TetraState()
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-SETUP", call_id=1234, encryption_type=0), ts=0.0
    )
    trackers.record_cmce(
        state, CmceEvent(msg_type="D-CONNECT", call_id=1234, encryption_type=1), ts=1.0
    )
    call = state.active_calls[1234]
    assert call.encryption_algo == "TEA1"


# channel allocation / carrier role


# sync watchdog


def test_watchdog_trips_at_exact_threshold():
    state = TetraState()
    # Put state in LOCKED so failures escalate to UNLOCKING first.
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    assert state.quality.sync_state == SYNC_STATE_LOCKED

    tripped = [trackers.record_sync_failure(state, max_failures=3) for _ in range(3)]
    assert tripped == [False, False, True]
    assert state.quality.sync_state == SYNC_STATE_UNLOCKED


def test_watchdog_unlocking_state_on_first_failure():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    trackers.record_sync_failure(state, max_failures=5)
    assert state.quality.sync_state == SYNC_STATE_UNLOCKING


def test_watchdog_recovery_resets_counter():
    state = TetraState()
    trackers.record_sb1(state, _make_sb1(), ts=0.0)
    trackers.record_sync_failure(state, max_failures=10)
    trackers.record_sync_failure(state, max_failures=10)
    assert state.quality.consecutive_sb1_failures == 2
    trackers.record_sync_recovery(state)
    assert state.quality.consecutive_sb1_failures == 0


# quality windowing
