"""Device persistence must round-trip every device type through devices.toml.
KiwiSDR was dropped on restart because `_build_params` had no case for it (and
`PersistedDevice` had no password/user fields), so added Kiwis vanished.
"""

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.devices import PersistedDevice
from tsdr.core.preferences import (
    _build_device_config,
    _build_params,
    _build_persisted_device,
    _persist_params,
    _restore_audio_pipeline,
    load_preferences,
    restore_tuning_state,
    save_tuning_state,
)
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.engine import SDREngine
from tsdr.core.tuning_state import TuningState
from tsdr.devices import KiwiSDRParams, MockParams


def test_kiwisdr_params_round_trip():
    params = KiwiSDRParams(host="h.example.com", port=8073, password="sec", user="me")
    persisted = PersistedDevice(id="kiwi-x", type="kiwisdr", **_persist_params(params))
    assert (persisted.password, persisted.user) == ("sec", "me")
    assert _build_params(persisted) == params


def test_kiwisdr_restore_defaults_when_auth_absent():
    persisted = PersistedDevice(id="kiwi-x", type="kiwisdr", host="h", port=8073)
    assert _build_params(persisted) == KiwiSDRParams(host="h", port=8073, password="", user="tsdr")


def test_unknown_device_type_still_skipped():
    assert _build_params(PersistedDevice(id="x", type="bogus")) is None


def test_fft_geometry_restores_per_device():
    persisted = PersistedDevice(id="d", type="mock", fft_size=4096, fft_window="blackman")
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.fft_size, config.fft_window) == (4096, "blackman")


def test_spectrum_view_restores_per_device():
    persisted = PersistedDevice(id="d", type="mock", spectrum_center=7.1e6, spectrum_span=100e3)
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.spectrum_center, config.spectrum_span) == (7.1e6, 100e3)


def test_spectrum_view_absent_restores_none():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert (config.spectrum_center, config.spectrum_span) == (None, None)


def test_tuned_frequency_round_trips():
    persisted = PersistedDevice(id="d", type="mock", tuned_frequency=7.2e6, center_frequency=7e6)
    config = _build_device_config(persisted)
    assert config is not None
    assert (config.tuned_frequency, config.center_frequency) == (7.2e6, 7e6)


def test_legacy_prefs_without_tuned_frequency_fall_back_to_center():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert config.tuned_frequency == 1e6


def test_tuning_mode_round_trips():
    persisted = PersistedDevice(id="d", type="mock", tuning_mode="free")
    config = _build_device_config(persisted)
    assert config is not None
    assert config.tuning_mode == "free"


def test_tuning_mode_absent_defaults_to_center():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert config.tuning_mode == "center"


def test_lo_offset_round_trips():
    persisted = PersistedDevice(id="d", type="mock", lo_offset="above", lo_offset_hz=30e3)
    config = _build_device_config(persisted)
    assert config is not None
    assert config.lo_offset == "above"
    assert config.lo_offset_hz == 30e3


def test_lo_offset_absent_defaults_to_auto():
    config = _build_device_config(PersistedDevice(id="d", type="mock", center_frequency=1e6))
    assert config is not None
    assert config.lo_offset == "auto"
    assert config.lo_offset_hz is None


def test_each_mode_keeps_its_own_tuning_step_across_a_restart() -> None:
    # Switching to NFM put the step back to 12.5 kHz (auto) every time; the owner's 5 kHz
    # had to be set again after each SSB or memory detour.
    state = TuningState()
    state.set_step("NFM", 5_000)
    state.set_step("USB", 100)
    save_tuning_state(state)
    restored = TuningState()
    restore_tuning_state(restored, load_preferences())
    assert restored.step_for("NFM") == 5_000
    assert restored.step_for("USB") == 100
    assert restored.step_for("WFM") is None  # never chosen: the mode's own (auto) step
    restored.set_step("NFM", None)  # the ladder's "auto" forgets the choice
    assert restored.step_for("NFM") is None


def test_the_squelch_kept_while_demodulation_is_off_survives_a_restart() -> None:
    """The squelch belongs to the receiver: turning demodulation off keeps it for the next
    mode, but quitting then lost it (it was saved only with a demodulator)."""
    engine = SDREngine()
    engine.add_device("m", "mock", MockParams(), DeviceConfig())
    engine.set_audio_demod("m", DemodSpec(mode="NFM"))
    engine.update_squelch("m", "audio", enabled=True, threshold_db=-78.0)
    engine.remove_pipeline("m", "audio")
    saved = _build_persisted_device(engine.devices["m"])

    restarted = SDREngine()
    restarted.add_device("m", "mock", MockParams(), DeviceConfig())
    _restore_audio_pipeline(restarted, saved)
    restarted.set_audio_demod("m", DemodSpec(mode="NFM"))
    audio = restarted.devices["m"].config.pipelines["audio"]
    assert (audio.squelch_enabled, audio.squelch_threshold_db) == (True, -78.0)
