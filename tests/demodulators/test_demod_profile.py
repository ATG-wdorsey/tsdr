"""DemodProfile (structural, spec-derived) and DemodStatus (dynamic) tests.

`demod_profile()` is the single source of a demod's structural fields, derived
from the mode + config without building an instance. `status()` carries only
the runtime fields.
"""

from tsdr.radio.registry import DEMODULATORS, demod_profile


def test_every_registered_mode_has_a_profile() -> None:
    for mode in DEMODULATORS:
        profile = demod_profile(mode)
        assert profile.label, f"{mode} has empty label"
        assert profile.modulation, f"{mode} has empty modulation"
        assert profile.channel_bandwidth > 0
