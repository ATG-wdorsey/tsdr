"""Stub tests for the SoapySDR wrapper's stream-format selection and read path.

No hardware: a fake device object mimics the slice of the SoapySDR API that
SoapySDRDevice touches. Verifies the CF32 path preserves weak signals (the
flatline bug), the CU8/CS8 fallback for drivers without CF32, and graceful
degradation when a driver exposes no gains / no AGC.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tsdr.core.sdr.exceptions import DeviceError
from tsdr.core.sdr.samples_batch import SampleFormat, SamplesBatch
from tsdr.devices import soapy as soapy_module
from tsdr.devices.soapy import _HAS_SOAPY, SoapySDRDevice

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


def test_the_stream_starts_with_the_first_read_after_tuning(monkeypatch):
    """Activated at open, before the rate and gain were set, a PlutoSDR
    streamed ~0.4 s of near-full-scale junk (-1.8 dBFS) once reads began:
    a burst of noise at every start. Activated by the first read, after the
    I/O worker has tuned the radio, it starts clean."""
    fake = FakeDevice()
    module = SimpleNamespace(
        Device=lambda args: fake,
        SoapySDRKwargs=dict,
        SOAPY_SDR_RX=0,
        SOAPY_SDR_CF32="CF32",
    )
    monkeypatch.setattr(soapy_module, "_SoapySDR", module)
    monkeypatch.setattr(soapy_module, "_HAS_SOAPY", True)
    dev = SoapySDRDevice(driver="plutosdr", serial="", antenna="", device_args="")
    dev.open()
    dev.set_frequency(146e6)
    dev.set_sample_rate(10e6)
    dev.set_gain(35.0)
    assert ("activateStream",) not in fake.calls
    dev.read_samples(1024 * dev.get_sample_format().bytes_per_sample)
    names = [c[0] for c in fake.calls]
    assert names.index("activateStream") > names.index("setSampleRate")
    assert names.index("activateStream") > names.index("setGain")
    dev.read_samples(1024 * dev.get_sample_format().bytes_per_sample)
    assert names.count("activateStream") == 1 == [c[0] for c in fake.calls].count("activateStream")
    dev.close()
    assert [c[0] for c in fake.calls][-2:] == ["deactivateStream", "closeStream"]


def test_a_stream_never_read_is_closed_without_deactivating(monkeypatch):
    fake = FakeDevice()
    module = SimpleNamespace(
        Device=lambda args: fake, SoapySDRKwargs=dict, SOAPY_SDR_RX=0, SOAPY_SDR_CF32="CF32"
    )
    monkeypatch.setattr(soapy_module, "_SoapySDR", module)
    monkeypatch.setattr(soapy_module, "_HAS_SOAPY", True)
    dev = SoapySDRDevice(driver="plutosdr", serial="", antenna="", device_args="")
    dev.open()
    dev.close()
    names = [c[0] for c in fake.calls]
    assert "deactivateStream" not in names and names[-1] == "closeStream"
