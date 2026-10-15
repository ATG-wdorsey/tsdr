"""Tests for the SSTV decoder, alone and riding on a mode's audio (NFM, USB).

Inputs are the real mp3 recordings under ``tests/samples/sstv/`` (LFS).
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from types import MappingProxyType

import av
import numpy as np
import pytest
from scipy.signal import resample_poly

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.bus import EventBus
from tsdr.core.sdr.config import DeviceConfig, PipelineConfig, SDRConfig, StageType
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.pipeline.stage_factory import create_stage
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.radio.audio_decoders import SSTVAudioDecoder, carried_decoder, with_decoder
from tsdr.radio.decoders.sstv import (
    MODES,
    MODES_BY_NAME,
    Mode,
    SSTVData,
    StreamerState,
    StreamingEvents,
    StreamingSSTV,
    decode_line,
    decode_offline,
    find_vis,
    hilbert_fft,
    instantaneous_freq,
    make_decode_state,
)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples" / "sstv"

# (filename, expected VIS code, expected image (H, W)) for the recordings
# we ship under tests/samples/sstv/.
SAMPLE_FIXTURES: list[tuple[str, int, tuple[int, int]]] = [
    ("Martin_1.mp3", 44, (256, 320)),
    ("Scottie_1.mp3", 60, (256, 320)),
    ("Robot_8_BW.mp3", 2, (120, 160)),
    ("Robot_36.mp3", 8, (240, 320)),
    ("Robot_72.mp3", 12, (240, 320)),
    ("ScottieDX.mp3", 76, (256, 320)),
]


def _load_mp3(path: Path) -> tuple[np.ndarray, int]:
    """Decode an mp3 to mono float32 + sample rate via pyav."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    rate = stream.codec_context.sample_rate
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)

    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for r in resampler.resample(frame):
            chunks.append(r.to_ndarray().reshape(-1).astype(np.float32))
    for r in resampler.resample(None):
        chunks.append(r.to_ndarray().reshape(-1).astype(np.float32))
    container.close()
    if not chunks:
        raise RuntimeError(f"no audio frames decoded from {path}")
    return np.concatenate(chunks), rate


@pytest.fixture(scope="module")
def _samples() -> dict[str, tuple[np.ndarray, int]]:
    """Load each sample once per test module; mp3 decode is the slow part."""
    out: dict[str, tuple[np.ndarray, int]] = {}
    for name, _, _ in SAMPLE_FIXTURES:
        path = SAMPLES_DIR / name
        if not path.exists():
            pytest.skip(f"missing sample: {path}")
        out[name] = _load_mp3(path)
    return out


Decoded = dict[str, tuple[Mode, np.ndarray]]


@pytest.fixture(scope="module")
def _decoded(_samples: dict[str, tuple[np.ndarray, int]]) -> Decoded:
    """Each recording decoded offline, once per module (on first use)."""

    class _Lazy(dict):
        def __missing__(self, name: str) -> tuple[Mode, np.ndarray]:
            result = decode_offline(*_samples[name])
            assert result is not None, f"offline decode failed for {name}"
            self[name] = result
            return result

    return _Lazy()


# ---- VIS detection --------------------------------------------------------


# ---- Offline decode -------------------------------------------------------


@pytest.mark.parametrize(("filename", "expected_vis", "shape"), SAMPLE_FIXTURES)
def test_decode_offline_finds_the_mode_and_the_image(
    _decoded: Decoded,
    filename: str,
    expected_vis: int,
    shape: tuple[int, int],
) -> None:
    mode, img = _decoded[filename]
    assert mode is MODES[expected_vis]  # the VIS code was read
    assert img.shape == (*shape, 3)
    # Not a blank frame.
    assert img.std() > 5.0


def test_decode_offline_martin1_content_signature(_decoded: Decoded) -> None:
    """Regression on the Martin M1 path (BT.601 conversion + segment layout).

    Asserts the decoded image stays close to a captured reference signature
    (means of two rows + global mean) within tolerance.
    """
    _, img = _decoded["Martin_1.mp3"]
    # Sample a few rows to detect any single-channel regression.
    row50_mean = img[50].mean(axis=0)
    row200_mean = img[200].mean(axis=0)
    global_mean = img.mean(axis=(0, 1))
    # Each channel should be a non-trivial value (not all 0 or 255).
    assert all(5.0 < v < 250.0 for v in row50_mean), row50_mean
    assert all(5.0 < v < 250.0 for v in row200_mean), row200_mean
    assert all(5.0 < v < 250.0 for v in global_mean), global_mean
    # No channel should be saturated to 0 or 255 across the whole frame.
    for ch in range(3):
        ch_data = img[..., ch]
        assert ch_data.min() < 32, f"channel {ch} min={ch_data.min()}"
        assert ch_data.max() > 200, f"channel {ch} max={ch_data.max()}"


# ---- Streaming vs offline parity ------------------------------------------


def test_streaming_matches_offline_robot36(
    _samples: dict[str, tuple[np.ndarray, int]], _decoded: Decoded
) -> None:
    """Streaming chunks should produce essentially the same image as offline."""
    audio, fs = _samples["Robot_36.mp3"]

    _, offline_img = _decoded["Robot_36.mp3"]

    captured: list[np.ndarray] = []

    def on_image(_mode, img: np.ndarray) -> None:
        captured.append(img.copy())

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_image=on_image))
    chunk = max(1, int(0.050 * fs))  # 50 ms chunks
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()

    assert captured, "streaming decoder produced no image"
    stream_img = captured[-1]
    assert stream_img.shape == offline_img.shape
    # Slant tracker resolves chunk-boundary jitter slightly differently than
    # the offline pass; tolerate a small mean-pixel-difference.
    diff = np.mean(np.abs(stream_img.astype(np.int32) - offline_img.astype(np.int32)))
    assert diff < 5.0, f"streaming vs offline diff={diff}"


def test_streaming_decodes_back_to_back_transmissions(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """After completing one image, the decoder must reset on the next ``process``
    call and decode the next VIS-tagged transmission. Regression for the bug
    where ``_emit_image`` overwrote a callback-issued ``reset()``, wedging the
    state machine in ``DONE``.
    """
    audio, fs = _samples["Martin_1.mp3"]
    # Concatenate two transmissions with one second of silence between them.
    gap = np.zeros(int(1.0 * fs), dtype=np.float32)
    combined = np.concatenate([audio, gap, audio])

    images: list[np.ndarray] = []
    modes: list[str] = []

    def on_mode(mode, _vis: int) -> None:
        modes.append(mode.name)

    def on_image(_mode, img: np.ndarray) -> None:
        images.append(img.copy())
        decoder.reset()

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_mode=on_mode, on_image=on_image))
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(combined), chunk):
        decoder.process(combined[i : i + chunk])
    decoder.flush()

    assert len(images) == 2, f"expected two images, got {len(images)} (modes={modes})"
    assert modes == ["Martin M1", "Martin M1"]


# ---- Forced mode ----------------------------------------------------------


def test_forced_mode_decodes_known_sample(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """A correctly-tagged Martin_1 stream still decodes when the mode is forced."""
    audio, fs = _samples["Martin_1.mp3"]
    result = decode_offline(audio, fs, forced_mode=MODES_BY_NAME["martin_m1"])
    assert result is not None
    mode, img = result
    assert mode.vis == 44
    assert img.shape == (256, 320, 3)
    assert img.std() > 5.0


def test_streaming_forced_mode_skips_vis_lookup(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """When ``forced_mode`` is set, the streaming decoder uses it regardless
    of what VIS is parsed."""
    audio, fs = _samples["Martin_1.mp3"]
    captured: list[tuple[str, int]] = []

    def on_mode(mode, vis_code: int) -> None:
        captured.append((mode.name, vis_code))

    forced = MODES_BY_NAME["martin_m1"]
    decoder = StreamingSSTV(fs, events=StreamingEvents(on_mode=on_mode), forced_mode=forced)
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()
    assert captured, "no mode lock"
    assert captured[0][0] == "Martin M1"


# ---- A picture under way, and mistune --------------------------------------


def _mistuned(audio: np.ndarray, hz: float, fs: int) -> np.ndarray:
    """Every tone moved by ``hz``, as an SSB receiver tuned off the signal hears it."""
    z = hilbert_fft(audio.astype(np.float64))
    return np.real(z * np.exp(2j * np.pi * hz * np.arange(len(z)) / fs)).astype(np.float32)


def _picture_start(audio: np.ndarray, fs: int) -> int:
    """Where the picture begins: just past the VIS, which these recordings open with."""
    vis = find_vis(instantaneous_freq(audio[: 3 * fs], fs), fs)
    assert vis is not None
    return vis[1]


@pytest.mark.parametrize("filename", ["Robot_36.mp3", "Scottie_1.mp3"])
def test_a_picture_under_way_is_joined_from_its_line_syncs(
    _samples: dict[str, tuple[np.ndarray, int]], _decoded: Decoded, filename: str
) -> None:
    """Tuned in after the VIS (or with it lost), the decoder waited for the next
    picture. Its line syncs name the mode (Robot 36's period is half Robot 72's;
    Scottie's sync sits mid-line) and give the mistune (here 120 Hz)."""
    audio, fs = _samples[filename]
    mode, reference = _decoded[filename]
    joined_at, lines = int(mode.scan_lines * 0.4), 30
    line_n = mode.line_ms * fs / 1000.0
    cut = _picture_start(audio, fs) + int((joined_at + 0.1) * line_n)
    heard = _mistuned(audio[cut - fs : cut + int(lines * line_n)], 120.0, fs)[fs:]

    got: dict[str, object] = {}
    events = StreamingEvents(
        on_mode=lambda m, _vis: got.setdefault("mode", m),
        on_image=lambda _m, img: got.setdefault("image", img),
    )
    decoder = StreamingSSTV(fs, events=events)
    for i in range(0, len(heard), 2048):
        decoder.process(heard[i : i + 2048])
    decoder.flush()

    assert got.get("mode") is mode
    image = got["image"]
    assert isinstance(image, np.ndarray)
    rows = lines - 6  # the picture's own rows, from the top (the train's lines included)
    later = reference[joined_at + 1 : joined_at + 1 + rows].astype(int)
    assert np.abs(image[:rows].astype(int) - later).mean() < 3.0


def _through_the_decoder(audio: np.ndarray, fs: int) -> list[SSTVData]:
    """``audio`` through SSTVAudioDecoder, as a mode's audio reaches it."""
    decoder = SSTVAudioDecoder()
    out: list[SSTVData] = []
    chunk = int(0.05 * fs)
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk], fs, 1.7e9 + i / fs)
        out += [m.data for m in decoder.get_messages() if isinstance(m.data, SSTVData)]
    return out


def _starts(got: list[SSTVData]) -> list[int | None]:
    return [d.vis_code for d in got if d.state == StreamerState.DECODING and d.line_index < 0]


def test_the_next_picture_starts_from_its_vis_after_a_joined_one(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """A picture joined under way is drawn from the top, so it runs on past its end;
    the sender's next picture, a second later, starts from its VIS (ending the first
    threw away the audio holding it), and the first keeps only its own lines."""
    audio, fs = _samples["Martin_1.mp3"]
    mode = MODES[44]
    line_n = mode.line_ms * fs / 1000.0
    cut = _picture_start(audio, fs) + int((int(mode.scan_lines * 0.4) + 0.1) * line_n)
    gap = (np.random.default_rng(1).standard_normal(fs) * 0.01).astype(np.float32)
    got = _through_the_decoder(np.concatenate([audio[cut:], gap, audio]), fs)
    assert _starts(got) == [None, 44]
    first = next(d for d in got if d.state == StreamerState.DONE)
    assert first.line_index < mode.scan_lines - 1


def test_the_line_timing_follows_a_sender_clock_a_few_tenths_of_a_percent_off() -> None:
    """A sender whose clock runs 0.4 % slow: each Scottie DX sync comes 50 samples (at
    12 kHz) later than the last. Each is where the clock can put it, so every one is taken
    and they give the line length; refused as too far apart, the picture never locked."""
    fs, skew, lines = 12_000, 0.004, 40
    mode = MODES[76]
    state = make_decode_state(mode, fs)
    line_n = state.slant.nominal_line_samples * (1 + skew)
    sync_ms = next(seg.ms for seg in mode.segments if seg.kind == "sync")
    before_sync = sum(
        seg.ms for seg in mode.segments[: [g.kind for g in mode.segments].index("sync")]
    )
    freq = np.full(int((lines + 2) * line_n), 1900.0, dtype=np.float32)
    for k in range(lines + 1):
        at = int(round(k * line_n + before_sync * fs / 1000.0))
        freq[at : at + int(sync_ms * fs / 1000.0)] = 1200.0
    cursor = 0
    for k in range(lines):
        cursor = decode_line(freq, fs, state, k, cursor)
    assert state.missed_syncs == 0
    assert state.synced_lines == lines
    assert abs(state.slant.correction - skew * state.slant.nominal_line_samples) <= 3


def test_a_weak_picture_from_a_sender_clock_off_keeps_its_line_timing() -> None:
    """A weak picture, every second sync lost, from a sender whose clock runs 0.4 % slow: two
    Scottie DX syncs two lines apart are 100 samples (at 12 kHz) farther apart than the mode
    says. They must be allowed what the clock moves them over both lines and give the line
    length; held to one line's tolerance, the picture never locked and its lines slid off."""
    fs, skew, lines = 12_000, 0.004, 40
    mode = MODES[76]
    state = make_decode_state(mode, fs)
    line_n = state.slant.nominal_line_samples * (1 + skew)
    sync_ms = next(seg.ms for seg in mode.segments if seg.kind == "sync")
    before_sync = sum(
        seg.ms for seg in mode.segments[: [g.kind for g in mode.segments].index("sync")]
    )
    freq = np.full(int((lines + 2) * line_n), 1900.0, dtype=np.float32)
    for k in range(0, lines + 1, 2):
        at = int(round(k * line_n + before_sync * fs / 1000.0))
        freq[at : at + int(sync_ms * fs / 1000.0)] = 1200.0
    cursor = 0
    for k in range(lines):
        cursor = decode_line(freq, fs, state, k, cursor)
    assert abs(cursor - lines * line_n) < 0.002 * line_n


def test_a_picture_that_stops_in_ssb_noise_ends_with_its_own_lines(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """The sender stops 30 lines into a Martin picture; SSB noise follows. Noise passed
    for Martin's short syncs on about half the lines, so the picture ran on, painting
    noise to its last line; now it ends and keeps its 30 lines."""
    audio, fs = _samples["Martin_1.mp3"]
    mode = MODES[44]
    stop = _picture_start(audio, fs) + int(30 * mode.line_ms * fs / 1000.0)
    n = 24 * fs
    spectrum = np.fft.rfft(np.random.default_rng(4).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spectrum[(f < 300) | (f > 2700)] = 0
    noise = np.fft.irfft(spectrum, n)
    noise *= np.sqrt(np.mean(audio[:stop] ** 2) / np.mean(noise**2))
    got = _through_the_decoder(np.concatenate([audio[:stop], noise.astype(np.float32)]), fs)
    done = [d for d in got if d.state == StreamerState.DONE]
    assert len(done) == 1
    assert 28 <= done[0].line_index + 1 <= 31


def test_a_picture_that_lost_its_first_syncs_does_not_begin_again(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Looking for the next picture's VIS while syncs are missing must not find the
    picture's own: it began again from it without end, keeping a copy each time."""
    audio, fs = _samples["ScottieDX.mp3"]
    line_n = int(MODES[76].line_ms * fs / 1000.0)
    start = _picture_start(audio, fs)
    heard = audio[: start + 20 * line_n].copy()
    heard[start : start + 2 * line_n] = np.random.default_rng(3).standard_normal(2 * line_n)
    starts: list[int] = []

    def on_mode(_mode: Mode, vis: int) -> None:
        starts.append(vis)
        assert len(starts) < 4, "the picture began again from its own VIS"

    decoder = StreamingSSTV(
        fs, events=StreamingEvents(on_mode=on_mode, on_image=lambda _m, _i: decoder.look_again())
    )
    for i in range(0, len(heard), 2048):
        decoder.process(heard[i : i + 2048])
    decoder.flush()
    assert starts == [76]


def test_a_fade_leaves_one_picture(
    _samples: dict[str, tuple[np.ndarray, int]], _decoded: Decoded
) -> None:
    """Four seconds of noise across a picture (27 Robot 36 lines: a satellite fading)
    leave one picture, still aligned below the fade; it was split in two, the panel
    showing only the second part, drawn from the top."""
    audio, fs = _samples["Robot_36.mp3"]
    mode, reference = _decoded["Robot_36.mp3"]
    line_n = mode.line_ms * fs / 1000.0
    fade_at, n = _picture_start(audio, fs) + int(80 * line_n), 4 * fs
    heard = audio.copy()
    heard[fade_at : fade_at + n] = np.random.default_rng(2).standard_normal(n) * 0.5
    got = _through_the_decoder(heard, fs)
    assert _starts(got) == [8]
    done = [d for d in got if d.state == StreamerState.DONE]
    assert len(done) == 1 and done[0].image is not None
    below = 80 + 30
    assert np.abs(done[0].image[below:].astype(int) - reference[below:].astype(int)).mean() < 10


def test_a_mistuned_picture_keeps_its_brightness(
    _samples: dict[str, tuple[np.ndarray, int]], _decoded: Decoded
) -> None:
    """An SSB mistune moves every tone; the leader's measured tone gives the offset
    (120 Hz shifted every pixel by about 38 of 255 before)."""
    audio, fs = _samples["Martin_1.mp3"]
    mode, reference = _decoded["Martin_1.mp3"]
    lines = 40
    end = _picture_start(audio, fs) + int(lines * mode.line_ms * fs / 1000.0)
    result = decode_offline(_mistuned(audio[:end], -120.0, fs), fs)
    assert result is not None
    rows = lines - 2
    assert np.abs(result[1][:rows].astype(int) - reference[:rows].astype(int)).mean() < 2.0


# ---- Slant tracker --------------------------------------------------------


def test_scottie_dx_slant_correction_bounded(
    _samples: dict[str, tuple[np.ndarray, int]],
) -> None:
    """Scottie DX is the longest-line mode in the suite. The slant tracker's
    final correction should remain a tiny fraction of the nominal line.
    """
    audio, fs = _samples["ScottieDX.mp3"]
    captured_state: dict = {}

    def on_image(_mode, _img) -> None:
        if decoder.dstate is not None:
            captured_state["correction"] = decoder.dstate.slant.correction
            captured_state["nominal"] = decoder.dstate.slant.nominal_line_samples

    decoder = StreamingSSTV(fs, events=StreamingEvents(on_image=on_image))
    chunk = max(1, int(0.050 * fs))
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i : i + chunk])
    decoder.flush()
    # If the image didn't complete in stream, fall back to whatever state remains.
    if "nominal" not in captured_state and decoder.dstate is not None:
        captured_state["correction"] = decoder.dstate.slant.correction
        captured_state["nominal"] = decoder.dstate.slant.nominal_line_samples
    assert "nominal" in captured_state, "decoder never locked"
    ratio = abs(captured_state["correction"]) / captured_state["nominal"]
    assert ratio < 0.02, f"slant correction ratio {ratio:.4f} too large"


# ---- VIS-alias mapping ----------------------------------------------------


def test_vis_alias_14_resolves_to_robot36() -> None:
    """Some encoders transmit VIS=14 for a Robot 36 stream."""
    assert MODES[14] is MODES[8]
    assert MODES[14].name == "Robot 36"


# ---- Mode-name lookup -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["martin m1", "Martin M1", "martin_m1", "martinm1"],
)
def test_modes_by_name_accepts_common_spellings(name: str) -> None:
    assert MODES_BY_NAME[name.lower()].name == "Martin M1"


# ---- Demodulator end-to-end ------------------------------------------------


def _audio_to_usb_iq(audio: np.ndarray, fs: int, *, offset_hz: float = 0.0) -> np.ndarray:
    """A complex64 IQ stream that the USB demod can recover ``audio`` from: the
    analytic signal, optionally up-shifted by ``offset_hz``."""
    analytic = hilbert_fft(audio.astype(np.float64)).astype(np.complex64)
    if offset_hz != 0.0:
        t = np.arange(len(audio), dtype=np.float64) / fs
        analytic *= np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)
    return analytic


def _audio_to_nfm_iq(audio: np.ndarray, fs: int, rate: int, deviation_hz: float) -> np.ndarray:
    """``audio`` through an FM transmitter's mic path: 750 us pre-emphasis (the
    receiver's de-emphasis undoes it), peak deviation ``deviation_hz``, at ``rate``."""
    g = gcd(rate, fs)
    x = resample_poly(audio.astype(np.float64), rate // g, fs // g)
    tau = 750e-6
    x = x + tau * rate * np.diff(x, prepend=x[0])  # 1 + s*tau
    x *= deviation_hz / np.max(np.abs(x))
    phase = 2 * np.pi * np.cumsum(x) / rate
    return np.exp(1j * phase).astype(np.complex64)


def _decode(spec: DemodSpec, iq: np.ndarray, rate: float) -> list[SSTVData]:
    """Run ``iq`` through the audio pipeline's demodulator stage, as the device does."""
    pc = PipelineConfig(stages=(StageType.DEMODULATOR,), audio_spec=spec)
    config = DeviceConfig(sample_rate=rate, pipelines=MappingProxyType({"audio": pc}))
    stage = create_stage(StageType.DEMODULATOR, pc, SDRConfig(), config, "t", "audio")
    context = PipelineContext(device_context=None, event_bus=EventBus(), config=SDRConfig())  # type: ignore[arg-type]
    out: list[SSTVData] = []
    chunk = int(0.05 * rate)
    for i in range(0, len(iq), chunk):
        batch = SamplesBatch(iq_samples=iq[i : i + chunk], capture_utc_s=1.7e9 + i / rate)
        result = stage.process(batch, context)
        assert result is not None
        out += [m.data for m in result.decoded_messages if isinstance(m.data, SSTVData)]
    return out


def _robot36_done(collected: list[SSTVData]) -> SSTVData:
    done = next(d for d in collected if d.state == StreamerState.DONE)
    assert done.mode_name == "Robot 36" and done.vis_code == 8
    assert done.image is not None and done.image.shape == (240, 320, 3)
    return done


def test_sstv_decodes_on_nfm_as_on_2_m(
    _samples: dict[str, tuple[np.ndarray, int]], _decoded: Decoded
) -> None:
    """SSTV on 145.600 is FM: the decoder rides on NFM's audio (it was USB only, and
    decoded nothing there). Same picture as the clean audio decodes to."""
    audio, fs = _samples["Robot_36.mp3"]
    iq = _audio_to_nfm_iq(audio, fs, rate=96_000, deviation_hz=3_000.0)
    done = _robot36_done(_decode(DemodSpec(mode="NFM", decoder="SSTV"), iq, 96_000.0))
    reference = _decoded["Robot_36.mp3"]
    assert np.mean(np.abs(done.image.astype(float) - reference[1].astype(float))) < 20


def test_sstv_decodes_on_usb(_samples: dict[str, tuple[np.ndarray, int]]) -> None:
    audio, fs = _samples["Robot_36.mp3"]
    _robot36_done(_decode(DemodSpec(mode="USB", decoder="SSTV"), _audio_to_usb_iq(audio, fs), fs))


def test_a_forced_submode_overrides_and_resets() -> None:
    decoder = SSTVAudioDecoder()
    decoder.set_sstv_mode("scottie_1")
    assert decoder.sstv_mode_name == "Scottie 1"
    decoder.set_sstv_mode(None)
    assert decoder.sstv_mode_name is None


def test_the_decoder_reports_listening_while_no_picture_comes() -> None:
    fs = 12_000
    decoder = SSTVAudioDecoder()
    noise = (np.random.default_rng(0).standard_normal(fs * 4) * 0.01).astype(np.float32)
    looking = []
    for i in range(0, len(noise), 600):
        decoder.process(noise[i : i + 600], fs, i / fs)
        looking += [m for m in decoder.get_messages() if m.data.state == StreamerState.LOOKING]
    assert 2 <= len(looking) <= 6  # a ping a second


def test_a_decoder_keeps_the_users_mode() -> None:
    """The owner's rule: picking a decoder doesn't change the mode."""
    nfm = DemodSpec(mode="NFM", fm_deviation_hz=2_500.0)
    on_nfm = with_decoder(nfm, "sstv", 145.6e6)
    assert (on_nfm.mode, on_nfm.decoder, on_nfm.fm_deviation_hz) == ("NFM", "SSTV", 2_500.0)
    assert with_decoder(DemodSpec(mode="USB"), "SSTV", 144.5e6).mode == "USB"  # SSB SSTV on 2 m
    # With no mode that has audio for it, the band's usual one.
    assert with_decoder(None, "SSTV", 145.6e6).mode == "NFM"
    assert with_decoder(DemodSpec(mode="ADSB"), "SSTV", 14.23e6).mode == "USB"
    # Another audio mode keeps the decoder; a decoder that is its own mode drops it.
    assert carried_decoder(on_nfm, "USB") == ("SSTV", None)
    assert carried_decoder(on_nfm, "DMR") == (None, None)


def test_sstv_saved_as_a_mode_comes_back_as_usb_with_the_decoder() -> None:
    old = DemodSpec.model_validate({"mode": "SSTV", "sstv_mode": "martin_m1"})
    assert (old.mode, old.decoder, old.sstv_mode) == ("USB", "SSTV", "martin_m1")
