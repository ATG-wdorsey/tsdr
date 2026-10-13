"""Stub tests for the SoapySDR wrapper's stream-format selection and read path.

No hardware: a fake device object mimics the slice of the SoapySDR API that
SoapySDRDevice touches. Verifies the CF32 path preserves weak signals (the
flatline bug), the CU8/CS8 fallback for drivers without CF32, and graceful
degradation when a driver exposes no gains / no AGC.
"""

import numpy as np
import pytest

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.devices import soapy
from tsdr.devices.soapy import (
    _HAS_SOAPY,
    SoapySDRDevice,
    _has_static_analog_filter,
    _is_zero_if,
)

pytestmark = pytest.mark.skipif(not _HAS_SOAPY, reason="SoapySDR bindings not installed")


class FakeSR:
    def __init__(self, ret: int):
        self.ret = ret


class FakeDevice:
    """Minimal SoapySDR.Device stand-in for the calls read_samples/select use."""

    def __init__(self, *, formats=("CF32",), gains=(), has_agc=False, fill=None):
        self._formats = formats
        self._gains = gains
        self._has_agc = has_agc
        self._fill = fill
        self.calls: list[tuple] = []

    def getStreamFormats(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return list(self._formats)

    def listGains(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return list(self._gains)

    def hasGainMode(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self._has_agc

    def setGainMode(self, direction, channel, enable):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("setGainMode", enable))

    def setGain(self, direction, channel, *name_value):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("setGain", *name_value))

    def readStream(self, stream, buffers, numElems, timeoutUs):  # noqa: N802, N803 (SoapySDR API names)
        if self._fill is not None:
            self._fill(buffers[0], numElems)
        return FakeSR(numElems)

    def setupStream(self, direction, fmt):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("setupStream", fmt))
        return object()

    def activateStream(self, stream):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("activateStream",))

    def deactivateStream(self, stream):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("deactivateStream",))

    def closeStream(self, stream):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("closeStream",))

    def setSampleRate(self, direction, channel, rate):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("setSampleRate", rate))
        self._rate = rate

    def getSampleRate(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self._rate

    def setFrequency(self, direction, channel, freq):  # noqa: N802 (SoapySDR API name)
        self.calls.append(("setFrequency", freq))

    def getHardwareKey(self):  # noqa: N802 (SoapySDR API name)
        return "Fake"


def _device(**kw) -> SoapySDRDevice:
    dev = SoapySDRDevice(driver="x", serial="", antenna="", device_args="")
    dev._device = FakeDevice(**kw)
    dev._stream = object()
    return dev


def _read(dev: SoapySDRDevice) -> np.ndarray:
    n = 1024
    raw = dev.read_samples(n * dev.get_sample_format().bytes_per_sample)
    return SamplesBatch(raw_samples=raw, sample_format=dev.get_sample_format()).to_iq_array()


def test_cf32_preserves_weak_signal():
    """A 0.001-amplitude signal survives CF32 (would quantize to 0 under CS8)."""

    def fill(chunk, n):
        assert chunk.dtype == np.complex64
        chunk[:n] = np.complex64(0.001 + 0.001j)

    dev = _device(formats=("CF32", "CS16", "CS8"), fill=fill)
    assert dev._select_stream_format() == "CF32"
    assert dev.get_sample_format() == SampleFormat.COMPLEX64
    iq = _read(dev)
    assert len(iq) == 1024
    assert np.mean(np.abs(iq)) == pytest.approx(0.001414, abs=1e-5)


def test_falls_back_to_cs8_when_no_cf32():
    """A driver advertising only CS8 selects the int8 path and round-trips."""

    def fill(chunk, n):
        assert chunk.dtype == np.int8
        chunk[: n * 2] = 64

    dev = _device(formats=("CS8",), fill=fill)
    assert dev._select_stream_format() == "CS8"
    assert dev.get_sample_format() == SampleFormat.SINT8_IQ
    iq = _read(dev)
    assert len(iq) == 1024
    assert iq[0].real == pytest.approx(64 / 127, abs=1e-3)


def test_no_supported_format_raises():
    dev = _device(formats=("CS16",))
    with pytest.raises(DeviceError):
        dev._select_stream_format()


def test_auto_gain_guarded_by_has_gain_mode():
    """Driver without AGC: set_auto_gain must not call setGainMode or raise."""
    dev = _device(has_agc=False)
    dev.set_auto_gain(True)
    assert all(c[0] != "setGainMode" for c in dev._device.calls)

    dev2 = _device(has_agc=True)
    dev2.set_auto_gain(True)
    assert ("setGainMode", True) in dev2._device.calls


class FakeRange:
    def __init__(self, lo: float, hi: float, step: float = 0.0):
        self._lo, self._hi, self._step = lo, hi, step

    def minimum(self) -> float:
        return self._lo

    def maximum(self) -> float:
        return self._hi

    def step(self) -> float:
        return self._step


class FakeOpenableDevice(FakeDevice):
    """Adds the calls SoapySDRDevice.open() makes, with per-driver capabilities."""

    def __init__(self, *, freq_ranges, rate_ranges, gain_range, hw="fake", **kw):
        super().__init__(**kw)
        self._freq_ranges = freq_ranges
        self._rate_ranges = rate_ranges
        self._gain_range = gain_range
        self._hw = hw

    def setupStream(self, direction, fmt):  # noqa: N802 (SoapySDR API name)
        return object()

    def activateStream(self, stream):  # noqa: N802 (SoapySDR API name)
        pass

    def getSettingInfo(self):  # noqa: N802 (SoapySDR API name)
        return []

    def getHardwareKey(self):  # noqa: N802 (SoapySDR API name)
        return self._hw

    def getDriverKey(self):  # noqa: N802 (SoapySDR API name)
        return self._hw

    def getGainRange(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self._gain_range

    def getFrequencyRange(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        if isinstance(self._freq_ranges, Exception):
            raise self._freq_ranges
        return self._freq_ranges

    def getSampleRateRange(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        if isinstance(self._rate_ranges, Exception):
            raise self._rate_ranges
        return self._rate_ranges


def _open_with(monkeypatch, fake: FakeOpenableDevice) -> SoapySDRDevice:
    monkeypatch.setattr(soapy._SoapySDR, "Device", lambda args: fake)
    dev = SoapySDRDevice(driver="x", serial="", antenna="", device_args="")
    dev.open()
    return dev


def test_open_compiles_the_native_reader_before_the_stream_starts(monkeypatch):
    """A driver may stall if its first read comes late (SoapyAirspy stops for
    good when nothing reads ~0.2 s after a rate change), so the reader's ~0.5 s
    compile must not wait for the first read. The stream itself starts with
    that read (see test_the_stream_starts_with_the_first_read_after_tuning)."""
    order: list = []
    fake = FakeOpenableDevice(
        freq_ranges=[FakeRange(24e6, 1.8e9)],
        rate_ranges=[FakeRange(10e6, 10e6)],
        gain_range=FakeRange(0.0, 45.0, 1.0),
    )
    fake.activateStream = lambda stream: order.append("activate")  # type: ignore[method-assign]
    monkeypatch.setattr(soapy, "_native_read_stream", lambda: order.append("compile"))
    _open_with(monkeypatch, fake)
    assert order == ["compile"]


def test_a_reader_that_fails_to_compile_falls_back_to_python_reads(monkeypatch, caplog):
    def fail(read_stream):
        raise soapy.NumbaError("no")

    monkeypatch.setattr(soapy, "compile_read_loop", fail)
    soapy._native_read_stream.cache_clear()
    try:
        with caplog.at_level("WARNING"):
            assert soapy._native_read_stream() is None
    finally:
        soapy._native_read_stream.cache_clear()
    assert any("compile_failed" in r.getMessage() for r in caplog.records)


def test_open_reports_discrete_sample_rates_and_band(monkeypatch):
    """Airspy R2: two fixed rates, 24 MHz-1.8 GHz, 0-45 dB gain."""
    fake = FakeOpenableDevice(
        freq_ranges=[FakeRange(24e6, 1.8e9)],
        rate_ranges=[FakeRange(10e6, 10e6), FakeRange(2.5e6, 2.5e6)],
        gain_range=FakeRange(0.0, 45.0, 1.0),
        hw="AIRSPY",
    )
    caps = _open_with(monkeypatch, fake).capabilities
    assert caps.frequency_range == (24e6, 1.8e9)
    assert caps.sample_rates == (2.5e6, 10e6)
    assert caps.gain_range == (0.0, 45.0)
    assert caps.gain_step == 1.0


def test_open_leaves_continuous_sample_rates_unconstrained(monkeypatch):
    """PlutoSDR: any rate in 65.1 kS/s-61.44 MS/s, so no discrete list."""
    fake = FakeOpenableDevice(
        freq_ranges=[FakeRange(70e6, 6e9)],
        rate_ranges=[FakeRange(65105.0, 61.44e6)],
        gain_range=FakeRange(-1.0, 73.0, 0.5),
        hw="PlutoSDR",
    )
    caps = _open_with(monkeypatch, fake).capabilities
    assert caps.frequency_range == (70e6, 6e9)
    assert caps.sample_rates is None
    assert caps.gain_range == (-1.0, 73.0)
    assert caps.gain_step == 0.5


def test_open_spans_disjoint_frequency_ranges(monkeypatch):
    fake = FakeOpenableDevice(
        freq_ranges=[FakeRange(52e6, 1.1e9), FakeRange(1.25e9, 2.2e9)],
        rate_ranges=[FakeRange(1e6, 3.2e6)],
        gain_range=FakeRange(0.0, 50.0),
    )
    caps = _open_with(monkeypatch, fake).capabilities
    assert caps.frequency_range == (52e6, 2.2e9)
    assert caps.gain_step == 1.0  # driver reports no step


def test_open_tolerates_drivers_without_range_queries(monkeypatch):
    fake = FakeOpenableDevice(
        freq_ranges=RuntimeError("not supported"),
        rate_ranges=RuntimeError("not supported"),
        gain_range=FakeRange(0.0, 30.0),
    )
    caps = _open_with(monkeypatch, fake).capabilities
    assert caps.frequency_range is None
    assert caps.sample_rates is None


def test_read_error_names_the_soapy_code():
    dev = _device(fill=lambda chunk, n: None)
    dev._device.readStream = lambda *a, **k: FakeSR(-1)
    with pytest.raises(DeviceError, match="TIMEOUT"):
        dev.read_samples(1024 * 8)


def test_overflow_is_counted_and_the_read_completes(caplog):
    """SOAPY_SDR_OVERFLOW means the radio dropped samples; count it, keep reading."""
    results = iter([soapy._SoapySDR.SOAPY_SDR_OVERFLOW, 1024])
    dev = _device(fill=lambda chunk, n: None)
    dev._device.readStream = lambda *a, **k: FakeSR(next(results))
    with caplog.at_level("WARNING"):
        raw = dev.read_samples(1024 * 8)
    assert len(raw) == 1024 * 8
    assert dev.overflow_count == 1
    assert any("soapy_overflow" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("driver_key", "hardware_key", "expected"),
    [
        ("PlutoSDR", "PlutoSDR", True),
        ("HackRF", "HackRF One", True),
        ("lime", "LimeSDR-USB", True),
        ("uhd", "B200", True),
        ("bladeRF", "bladeRF 2.0", True),
        ("Airspy", "AirSpy", False),
        ("airspyhf", "AirspyHF", False),
        ("RTLSDR", "Generic RTL2832U", False),
    ],
)
def test_zero_if_detection(driver_key: str, hardware_key: str, expected: bool) -> None:
    assert _is_zero_if(driver_key, hardware_key, "") is expected


@pytest.mark.parametrize(
    ("driver_key", "hardware_key", "expected"),
    [
        ("PlutoSDR", "ADALM-PLUTO", True),
        ("HackRF", "HackRF One", False),
        ("Airspy", "Airspy", False),
        ("RTLSDR", "Generic RTL2832U", False),
    ],
)
def test_static_analog_filter_detection(driver_key: str, hardware_key: str, expected: bool) -> None:
    assert _has_static_analog_filter(driver_key, hardware_key, "") is expected


class FakeFilterDevice(FakeDevice):
    def __init__(self, ranges):
        super().__init__()
        self._ranges = ranges
        self.bandwidth = 18e6

    def getBandwidthRange(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self._ranges

    def getBandwidth(self, direction, channel):  # noqa: N802 (SoapySDR API name)
        return self.bandwidth

    def setBandwidth(self, direction, channel, bw):  # noqa: N802 (SoapySDR API name)
        self.bandwidth = bw


def test_analog_filter_probe_and_set() -> None:
    dev = SoapySDRDevice(driver="plutosdr", serial="", antenna="", device_args="")
    dev._device = FakeFilterDevice([FakeRange(0.2e6, 0.2e6), FakeRange(10e6, 10e6)])
    assert dev._probe_analog_filter() is True
    assert dev.set_analog_bandwidth(1.6e6) == 1.6e6
    dev._device = FakeFilterDevice([])  # the Airspy reports no range
    assert dev._probe_analog_filter() is False


def _openable() -> FakeOpenableDevice:
    fake = FakeOpenableDevice(
        freq_ranges=[FakeRange(70e6, 6e9)],
        rate_ranges=[FakeRange(0.52e6, 61.44e6)],
        gain_range=FakeRange(0.0, 73.0, 1.0),
        hw="PlutoSDR",
    )
    fake.activateStream = lambda stream: fake.calls.append(("activateStream",))  # type: ignore[method-assign]
    return fake


def test_the_stream_starts_with_the_first_read_after_tuning(monkeypatch):
    """Activated at open, before the rate and gain were set, a PlutoSDR
    streamed ~0.4 s of near-full-scale junk (-1.8 dBFS) once reads began:
    a burst of noise at every start. Activated by the first read, after the
    I/O worker has tuned the radio, it starts clean."""
    fake = _openable()
    dev = _open_with(monkeypatch, fake)
    dev.set_frequency(146e6)
    dev.set_sample_rate(10e6)
    dev.set_gain(35.0)
    assert ("activateStream",) not in fake.calls
    dev.read_samples(1024 * dev.get_sample_format().bytes_per_sample)
    names = [c[0] for c in fake.calls]
    assert names.index("activateStream") > names.index("setSampleRate")
    assert names.index("activateStream") > names.index("setGain")
    dev.read_samples(1024 * dev.get_sample_format().bytes_per_sample)
    assert [c[0] for c in fake.calls].count("activateStream") == 1
    dev.close()
    assert [c[0] for c in fake.calls][-2:] == ["deactivateStream", "closeStream"]


def test_a_stream_never_read_is_closed_without_deactivating(monkeypatch):
    fake = _openable()
    dev = _open_with(monkeypatch, fake)
    dev.close()
    names = [c[0] for c in fake.calls]
    assert "deactivateStream" not in names and names[-1] == "closeStream"
