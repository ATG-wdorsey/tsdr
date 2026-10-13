import numpy as np

from tsdr.core.sdr.io import load_iq
from tsdr.radio.demodulators.wfm import WidebandFMDemodulator
from tsdr.radio.dsp import firwin

SAMPLE_RATE = 250_000
AUDIO_RATE = 48000


def _make_fm_signal(
    duration: float,
    *,
    pilot: bool = True,
    stereo_tone_hz: float = 1000.0,
) -> np.ndarray:
    """Synthesize a baseband FM IQ signal with optional stereo pilot and subcarrier.

    Args:
        duration: Signal duration in seconds
        pilot: Whether to include 19 kHz pilot tone
        stereo_tone_hz: Frequency of the L-R test tone (Hz)
    """
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    deviation = 75000.0

    # Baseband audio: L+R mono tone at 1 kHz
    mono = 0.5 * np.sin(2 * np.pi * stereo_tone_hz * t)

    # Composite baseband
    composite = mono

    if pilot:
        # 19 kHz pilot at 10% amplitude (per FM broadcast spec)
        pilot_signal = 0.1 * np.sin(2 * np.pi * 19000 * t)
        # L-R on 38 kHz DSB-SC subcarrier (different tone for L vs R)
        lr_diff = 0.3 * np.sin(2 * np.pi * stereo_tone_hz * 1.5 * t)
        subcarrier = lr_diff * np.sin(2 * np.pi * 38000 * t)
        composite = mono + pilot_signal + subcarrier

    # FM modulate: integrate composite to get phase
    phase = 2 * np.pi * deviation * np.cumsum(composite) / SAMPLE_RATE
    return np.exp(1j * phase).astype(np.complex64)


def test_stereo_detected_with_pilot():
    """Stereo is detected when a 19 kHz pilot tone is present."""
    iq = _make_fm_signal(0.5, pilot=True)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    # Process in chunks to simulate real streaming
    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)

    assert demod.stereo_detected is True


def test_mono_when_no_pilot():
    """Stereo is NOT detected when there is no pilot tone."""
    iq = _make_fm_signal(0.5, pilot=False)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)

    assert demod.stereo_detected is False


def test_output_shape_is_stereo():
    """Output is always (N, 2) regardless of stereo detection."""
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    iq = _make_fm_signal(0.1, pilot=False)
    demod.demodulate(iq, 0.0)
    batches = demod.get_audio()

    assert len(batches) == 1
    audio = batches[0].samples
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert audio.dtype == np.float32


def test_empty_input_returns_no_audio():
    """Empty input produces no audio batches."""
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)
    demod.demodulate(np.array([], dtype=np.complex64), 0.0)
    assert demod.get_audio() == []


def test_stereo_on_real_data():
    """Stereo detection works on real FM broadcast recording."""
    iq = load_iq("tests/samples/freq=98.9M_sr=240k_dur=5s_gain=28_20260423T1733.cu8.zst")
    demod = WidebandFMDemodulator(sample_rate=240_000, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, min(len(iq), chunk_size * 5), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)
        batches = demod.get_audio()
        for batch in batches:
            assert batch.samples.ndim == 2
            assert batch.samples.shape[1] == 2

    assert demod.stereo_detected is True


def test_reset_clears_stereo_state():
    """Reset clears stereo detection and PLL state."""
    iq = _make_fm_signal(0.5, pilot=True)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, audio_rate=AUDIO_RATE, rds_enabled=False)

    chunk_size = 65536
    for i in range(0, len(iq), chunk_size):
        demod.demodulate(iq[i : i + chunk_size], 0.0)
    assert demod.stereo_detected is True

    demod.reset()
    assert demod.stereo_detected is False
    assert demod.pilot_snr_ema == 0.0
    assert demod.pll_phase == 0.0


def _broadcast(
    left: np.ndarray, right: np.ndarray, rate: float, *, pilot: bool = True, seed: int = 0
) -> np.ndarray:
    """FM stereo as broadcast: 90 % program, (L+R)/2 plus (L-R)/2 on a 38 kHz
    subcarrier in phase with the pilot's second harmonic, and a 9 % pilot;
    40 dB carrier-to-noise in 200 kHz."""
    t = np.arange(len(left)) / rate
    composite = 0.9 * (left + right) / 2
    if pilot:
        composite += 0.9 * (left - right) / 2 * np.sin(2 * np.pi * 38_000 * t)
        composite += 0.09 * np.sin(2 * np.pi * 19_000 * t)
    iq = np.exp(1j * 2 * np.pi * 75_000 * np.cumsum(composite) / rate)
    noise_sigma = np.sqrt(rate / 200e3 * 1e-4 / 2)
    rng = np.random.default_rng(seed)
    iq += noise_sigma * (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t)))
    return iq.astype(np.complex64)


def _tone_levels(
    demod: WidebandFMDemodulator, iq: np.ndarray, rate: float, tone_hz: float = 1000.0
) -> tuple[float, float]:
    """Amplitude of the tone in the left and right outputs, after 0.7 s for the
    pilot lock to settle."""
    out = []
    step = int(rate * 0.02)
    for i in range(0, len(iq), step):
        demod.demodulate(iq[i : i + step], 0.0)
        out.extend(b.samples for b in demod.get_audio())
    audio = np.concatenate(out)[int(demod.output_sample_rate * 0.7) :]
    ref = np.exp(-2j * np.pi * tone_hz * np.arange(len(audio)) / demod.output_sample_rate)
    return 2 * abs(np.mean(audio[:, 0] * ref)), 2 * abs(np.mean(audio[:, 1] * ref))


def _db(x: float, ref: float) -> float:
    return 20 * np.log10(x / ref)


def test_a_stereo_station_plays_at_a_mono_stations_level():
    t = np.arange(int(SAMPLE_RATE * 1.5)) / SAMPLE_RATE
    centered = 0.5 * np.sin(2 * np.pi * 1000 * t)
    mono = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, rds_enabled=False)
    mono_level, _ = _tone_levels(
        mono, _broadcast(centered, centered, SAMPLE_RATE, pilot=False), SAMPLE_RATE
    )
    stereo = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, rds_enabled=False)
    left, right = _tone_levels(stereo, _broadcast(centered, centered, SAMPLE_RATE), SAMPLE_RATE)
    assert stereo.stereo_detected and not mono.stereo_detected
    assert abs(_db(left, mono_level)) < 0.5 and abs(_db(right, mono_level)) < 0.5


def _pilot_readings(demod: WidebandFMDemodulator, iq: np.ndarray, rate: float):
    """(pilot SNR, stereo) after each 20 ms chunk, from 0.8 s on."""
    readings = []
    step = int(rate * 0.02)
    for i in range(0, len(iq), step):
        demod.demodulate(iq[i : i + step], 0.0)
        demod.get_audio()
        if i >= 0.8 * rate:
            readings.append((demod.pilot_snr_ema, demod.stereo_detected))
    return readings


def test_the_pilot_reading_follows_the_signal_not_the_filters():
    t = np.arange(int(SAMPLE_RATE * 1.5)) / SAMPLE_RATE
    centered = 0.3 * np.sin(2 * np.pi * 1000 * t)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, rds_enabled=False)
    readings = _pilot_readings(demod, _broadcast(centered, centered, SAMPLE_RATE), SAMPLE_RATE)
    assert min(snr for snr, _ in readings) > 30  # 40 dB CNR, 9 % pilot

    rng = np.random.default_rng(1)
    noise = (rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))).astype(np.complex64)
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, rds_enabled=False)
    readings = _pilot_readings(demod, noise, SAMPLE_RATE)
    assert max(snr for snr, _ in readings) < 3 and not any(st for _, st in readings)


def test_bright_treble_keeps_stereo():
    # Pre-emphasized 12-15 kHz program (cymbals) at 20 % RMS deviation, 1-3 s
    t = np.arange(int(SAMPLE_RATE * 4)) / SAMPLE_RATE
    rng = np.random.default_rng(3)
    band = firwin(401, [12_000, 15_000], pass_zero=False, fs=SAMPLE_RATE)
    treble = np.convolve(rng.normal(size=len(t)), band, "same")
    treble *= 0.2 / np.sqrt(np.mean(treble**2))
    program = 0.1 * np.sin(2 * np.pi * 1000 * t) + treble * ((t > 1) & (t < 3))
    demod = WidebandFMDemodulator(sample_rate=SAMPLE_RATE, rds_enabled=False)
    readings = _pilot_readings(demod, _broadcast(program, program, SAMPLE_RATE), SAMPLE_RATE)
    assert all(stereo for _, stereo in readings)
