"""Post-decimation demods clamp inherited channel_bandwidth to Nyquist.

Regression: switching WFM (200 kHz default) to AM/NFM/SSB/CW carried the
200 kHz value into the new demodulator, which crashed `firwin` because
its channel filter runs at decimated_rate ≈ audio_rate (Nyquist 24 kHz).
"""

from __future__ import annotations

import pytest

from tsdr.radio.demodulators.am import AMDemodulator
from tsdr.radio.demodulators.cw import CWDemodulator
from tsdr.radio.demodulators.nfm import NarrowbandFMDemodulator
from tsdr.radio.demodulators.ssb import SSBDemodulator
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator

SAMPLE_RATE = 2_400_000.0
WIDE_BW = 200_000.0
EXPECTED_CLAMP = 48_000.0 * 0.95


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: NarrowbandFMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: SSBDemodulator(mode="USB", sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: SSBDemodulator(mode="LSB", sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
        lambda: CWDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=WIDE_BW),
    ],
    ids=["AM", "NFM", "USB", "LSB", "CW"],
)
def test_post_decim_demods_clamp_inherited_wide_bandwidth(factory):
    demod = factory()
    assert demod.channel_bandwidth == pytest.approx(EXPECTED_CLAMP)


def test_set_channel_bandwidth_clamps_too_wide_value():
    demod = AMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=10_000)
    demod.set_channel_bandwidth(100_000)
    assert demod.channel_bandwidth == pytest.approx(EXPECTED_CLAMP)


def test_wfm_unaffected_by_wide_bandwidth():
    # WFM filters at the native sample rate -- 200 kHz must work without clamping.
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, channel_bandwidth=200_000)
    assert demod.channel_bandwidth == pytest.approx(200_000)
