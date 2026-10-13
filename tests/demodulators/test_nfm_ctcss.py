"""NFM filters CTCSS (sub-audible tones) from its audio, as receivers do."""

import numpy as np
import pytest

from tsdr.radio.demodulators.nfm import CTCSS_CUTOFF_HZ, CTCSS_ORDER, NarrowbandFMDemodulator
from tsdr.radio.dsp import butter_highpass_sections

RATE = 1e6


def _tone_levels(demod: NarrowbandFMDemodulator) -> tuple[float, float]:
    """dB levels of 131.8 Hz and 1 kHz in the audio of a carrier FM-modulated by both, 2.5 kHz deviation each."""
    t = np.arange(int(RATE * 2.0)) / RATE
    phase = (
        2
        * np.pi
        * 2500
        * (np.sin(2 * np.pi * 131.8 * t) / 131.8 + np.sin(2 * np.pi * 1000 * t) / 1000)
    )
    iq = np.exp(1j * phase).astype(np.complex64)
    out = []
    for i in range(0, len(iq), 50_000):
        demod.demodulate(iq[i : i + 50_000], 0.0)
        out.extend(b.samples for b in demod.get_audio())
    audio = np.concatenate(out)[int(demod.decimated_rate) :]  # after the filters settle
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    f = np.fft.rfftfreq(len(audio), 1 / demod.decimated_rate)
    level = lambda hz: 20 * np.log10(spec[np.abs(f - hz) < 3].max())  # noqa: E731
    return level(131.8), level(1000.0)


def test_the_ctcss_tone_is_filtered_and_voice_is_not() -> None:
    ctcss, voice = _tone_levels(NarrowbandFMDemodulator(sample_rate=RATE, channel_bandwidth=12_500))
    unfiltered = NarrowbandFMDemodulator(sample_rate=RATE, channel_bandwidth=12_500)
    unfiltered._ctcss = []
    ctcss_before, voice_before = _tone_levels(unfiltered)
    assert ctcss_before - ctcss > 50  # 131.8 Hz: 57 dB down
    assert ctcss < voice - 40
    assert abs(voice - voice_before) < 0.2


def test_the_sections_are_a_butterworth_high_pass() -> None:
    signal = pytest.importorskip("scipy.signal")
    fs = 50_000.0
    sections = butter_highpass_sections(CTCSS_ORDER, CTCSS_CUTOFF_HZ, fs)
    f = np.array([67.0, 131.8, 250.0, 300.0, 1000.0, 3000.0])
    h = np.ones(len(f), dtype=complex)
    for b, a in sections:
        h *= signal.freqz(b, a, worN=f, fs=fs)[1]
    ref = signal.sosfreqz(
        signal.butter(CTCSS_ORDER, CTCSS_CUTOFF_HZ, "highpass", fs=fs, output="sos"), worN=f, fs=fs
    )[1]
    np.testing.assert_allclose(np.abs(h), np.abs(ref), rtol=1e-6, atol=1e-9)
    db = 20 * np.log10(np.abs(h))
    assert db[1] < -56 and db[2] < -12 and db[4] > -0.15
