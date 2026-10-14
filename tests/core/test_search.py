"""SEARCH: channel levels in squelch units, and the jump/hold decisions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tsdr.core.sdr.processing import compute_fft, fft_window, window_enbw_bins
from tsdr.core.sdr.search import (
    ATTACK_S,
    SearchConfig,
    SearchController,
    channel_powers_db,
    local_peaks,
    locate_levels,
    signal_peaks,
)
from tsdr.core.sdr.voice_squelch import Verdict
from tsdr.radio.dsp.squelch import iq_power_db

RATE = 10e6
N = 32768
CENTER = 146e6


def _spectrum(iq: np.ndarray) -> np.ndarray:
    window = fft_window("hanning", N)
    coherent_gain = float(np.sum(window)) / N
    return compute_fft(iq, window, 1.0 / (coherent_gain * N) ** 2)


def _levels(spectrum: np.ndarray, channels: np.ndarray, bandwidth: float) -> np.ndarray:
    return channel_powers_db(
        spectrum, CENTER, RATE, channels, bandwidth, window_enbw_bins("hanning", N)
    )


def test_a_tone_reads_its_power_like_the_squelch() -> None:
    t = np.arange(N) / RATE
    tone = (0.1 * np.exp(2j * np.pi * 1_000_123.0 * t)).astype(np.complex64)
    level = _levels(_spectrum(tone), np.array([CENTER + 1_000_000.0]), 12_500.0)[0]
    assert level == pytest.approx(iq_power_db(tone), abs=0.2)


def test_noise_reads_the_power_a_channel_filter_passes() -> None:
    rng = np.random.default_rng(0)
    sigma2 = 1e-4  # -40 dBFS over the whole capture
    noise = (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(sigma2 / 2)
    channels = CENTER + np.arange(-200, 200) * 12_500.0
    levels = _levels(_spectrum(noise.astype(np.complex64)), channels, 12_500.0)
    mean_db = 10 * math.log10(np.mean(10 ** (levels / 10)))
    assert mean_db == pytest.approx(10 * math.log10(sigma2 * 12_500.0 / RATE), abs=0.3)


def test_channels_outside_the_capture_read_minus_infinity() -> None:
    spectrum = np.full(N, -100.0, dtype=np.float32)
    levels = _levels(spectrum, np.array([CENTER + 6e6, CENTER]), 12_500.0)
    assert levels[0] == -np.inf
    assert np.isfinite(levels[1])


def test_channels_are_the_step_grid_within_the_range() -> None:
    config = SearchConfig(low_hz=144.0e6, high_hz=144.05e6, step_hz=12_500.0, trigger_db=-70.0)
    assert config.channels().tolist() == [144.0e6, 144.0125e6, 144.025e6, 144.0375e6, 144.05e6]


# --- decisions ---------------------------------------------------------------

STEP = 12_500.0
BW = 12_500.0
TRIGGER = -70.0
SQUELCH = -80.0
NOISE = -95.0
START = 145.0e6  # the dial when SEARCH starts, quiet


def _config(**changes) -> SearchConfig:
    base = {"low_hz": 144.0e6, "high_hz": 146.0e6, "step_hz": STEP, "trigger_db": TRIGGER}
    return SearchConfig(**(base | changes))


class _Run:
    """Drives a controller at 20 updates/s, applying each jump like the engine."""

    def __init__(
        self, config: SearchConfig | None = None, *, lo_hz: float = 0.0, lo_exclusion: float = 0.0
    ):
        self.config = config or _config()
        self.channels = self.config.channels()
        self.controller = SearchController()
        self.tuned = START
        self.now = 0.0
        self.signals: dict[float, float] = {}
        self.jumps: list[float] = []
        self.lo_hz = lo_hz
        self.lo_exclusion = lo_exclusion
        self.status = None
        self.voice: dict[float, Verdict] = {}  # VSQ's verdict per channel (none = VSQ off)
        self.squelch_open: bool | None = None  # the demodulator's; None = the level decides

    def level(self, f: float) -> float:
        return self.signals.get(round(f), NOISE)

    def tick(self, seconds: float = 0.05) -> None:
        end = self.now + seconds
        while self.now < end - 1e-9:
            levels = np.array([self.level(f) for f in self.channels])
            step = self.config.step_hz
            outer = (self.level(self.channels[0] - step), self.level(self.channels[-1] + step))
            target, self.status = self.controller.update(
                self.now,
                self.config,
                self.channels,
                levels,
                tuned_hz=self.tuned,
                current_db=self.level(self.tuned),
                hold_db=SQUELCH,
                squelch_open=self.squelch_open,
                bandwidth_hz=BW,
                lo_hz=self.lo_hz,
                lo_exclusion_hz=self.lo_exclusion,
                outer_locate_db=outer,
                voice=self.voice.get(self.tuned),
            )
            if target is not None:
                self.jumps.append(target)
                self.tuned = target
            self.now += 0.05


def test_jumps_only_after_the_signal_stays_above_the_trigger_for_the_attack() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(ATTACK_S - 0.01)
    assert run.jumps == []
    run.tick(0.1)
    assert run.jumps == [144.5e6]


def test_a_blip_shorter_than_the_attack_is_ignored() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.05)
    del run.signals[round(144.5e6)]
    run.tick(1.0)
    assert run.jumps == []


def test_the_strongest_eligible_channel_wins() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -65.0
    run.signals[round(145.5e6)] = -55.0
    run.tick(0.2)
    assert run.jumps == [145.5e6]


def test_holds_while_the_channel_is_above_the_squelch() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -50.0  # stronger, but we hold
    run.signals[round(144.5e6)] = -75.0  # below the trigger, above the squelch
    run.tick(5.0)
    assert run.jumps == [144.5e6]
    assert run.status.state == "holding"


def test_moves_on_only_after_the_hold_time_below_the_squelch() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    del run.signals[round(144.5e6)]  # transmission ends
    run.signals[round(145.5e6)] = -60.0
    run.tick(1.9)
    assert run.jumps == [144.5e6]
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6]


def test_a_reply_within_the_hold_time_keeps_the_channel() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    del run.signals[round(144.5e6)]
    run.signals[round(145.5e6)] = -60.0
    run.tick(1.5)
    run.signals[round(144.5e6)] = -62.0  # the reply
    run.tick(1.5)
    del run.signals[round(144.5e6)]
    run.tick(1.5)
    assert run.jumps == [144.5e6]


def test_stays_on_the_last_channel_until_a_new_hit() -> None:
    run = _Run()
    assert run.status is None
    run.tick(0.5)
    assert run.status.state == "searching"
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    del run.signals[round(144.5e6)]
    run.tick(5.0)
    assert run.tuned == 144.5e6
    assert run.status.state == "waiting"


def test_a_channel_active_at_start_is_held() -> None:
    run = _Run()
    run.signals[round(START)] = -60.0
    run.signals[round(144.5e6)] = -50.0
    run.tick(3.0)
    assert run.jumps == []


def test_locked_out_channels_and_their_neighbors_are_skipped() -> None:
    run = _Run(_config(lockouts=frozenset({144.5e6}), step_hz=5_000.0))
    run.channels = run.config.channels()
    for f in (144.495e6, 144.5e6, 144.505e6):  # one carrier seen by three 5 kHz channels
        run.signals[round(f)] = -60.0
    run.tick(1.0)
    assert run.jumps == []


def test_memory_lockouts_are_skipped() -> None:
    run = _Run(_config(memory_lockouts=frozenset({144.5e6})))
    run.signals[round(144.5e6)] = -60.0
    run.signals[round(145.5e6)] = -65.0
    run.tick(0.3)
    assert run.jumps == [145.5e6]


def test_locking_out_the_current_channel_releases_it() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -65.0
    run.config = _config(lockouts=frozenset({144.5e6}))
    run.tick(0.2)
    assert run.jumps == [144.5e6, 145.5e6]


def test_channels_beside_the_lo_are_skipped() -> None:
    run = _Run(lo_hz=145.25e6, lo_exclusion=50e3)
    run.signals[round(145.25e6)] = -60.0
    run.signals[round(145.2625e6)] = -60.0
    run.tick(1.0)
    assert run.jumps == []


def test_max_dwell_moves_on_and_skips_the_old_channel_until_it_drops() -> None:
    run = _Run(_config(max_dwell_s=3.0))
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -65.0
    run.tick(2.8)
    assert run.jumps == [144.5e6]
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6]
    del run.signals[round(145.5e6)]
    run.tick(3.0)  # 144.5 is still on: not taken back
    assert run.jumps == [144.5e6, 145.5e6]
    del run.signals[round(144.5e6)]
    run.tick(0.2)
    run.signals[round(144.5e6)] = -60.0  # a new transmission
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6, 144.5e6]


def test_not_voice_starts_the_hold_under_a_carrier_then_moves_on() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.signals[round(145.5e6)] = -65.0
    run.voice[144.5e6] = "listening"
    run.tick(3.0)
    assert run.jumps == [144.5e6]
    assert run.status.state == "holding" and run.status.voice == "listening"
    run.voice[144.5e6] = "not_voice"
    run.tick(1.9)  # the carrier is up, but the hold counts down: time to lock it out
    assert run.jumps == [144.5e6]
    assert run.status.voice == "not_voice" and 0.0 < run.status.hold_left_s < 0.2
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6]


def test_a_channel_left_as_not_voice_waits_until_its_signal_drops() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.voice[144.5e6] = "not_voice"
    run.tick(2.5)
    assert run.status.state == "waiting"  # nowhere else to go: it stays, muted
    run.signals[round(145.5e6)] = -65.0
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6]
    del run.signals[round(145.5e6)]
    run.tick(3.0)  # 144.5 is still on: not taken back
    assert run.jumps == [144.5e6, 145.5e6]
    del run.signals[round(144.5e6)]
    run.tick(0.2)
    run.signals[round(144.5e6)] = -60.0  # a new transmission
    run.tick(0.3)
    assert run.jumps == [144.5e6, 145.5e6, 144.5e6]


def test_a_voice_verdict_holds_like_the_squelch() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.signals[round(145.5e6)] = -65.0
    run.voice[144.5e6] = "voice"
    run.tick(10.0)
    assert run.jumps == [144.5e6]
    assert run.status.state == "holding"


def test_a_requested_jump_is_not_repeated_while_the_engine_applies_it() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    controller = run.controller
    levels = np.array([run.level(f) for f in run.channels])
    kwargs = {
        "tuned_hz": START,
        "current_db": NOISE,
        "hold_db": SQUELCH,
        "bandwidth_hz": BW,
        "lo_hz": 0.0,
        "lo_exclusion_hz": 0.0,
    }
    targets = [
        controller.update(t, run.config, run.channels, levels, **kwargs)[0]
        for t in (0.0, 0.1, 0.2, 0.3)
    ]
    assert targets == [None, 144.5e6, None, None]
    assert controller.update(0.7, run.config, run.channels, levels, **kwargs)[0] == 144.5e6


def test_the_hold_never_sits_above_the_trigger() -> None:
    """With the squelch raised above the trigger, a channel SEARCH jumped to
    must still count as open while it is above the trigger."""
    run = _Run()
    run.signals[round(144.5e6)] = -65.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -60.0
    controller, channels = run.controller, run.channels
    levels = np.array([run.level(f) for f in channels])
    target, status = controller.update(
        run.now,
        run.config,
        channels,
        levels,
        tuned_hz=run.tuned,
        current_db=-65.0,
        hold_db=-60.0,
        bandwidth_hz=BW,
        lo_hz=0.0,
        lo_exclusion_hz=0.0,
    )
    assert target is None
    assert status.state == "holding"


def test_a_channel_reached_by_a_jump_key_is_held_like_one_it_found() -> None:
    """`[` / `]` move the dial mid-run: the new channel gets a full hold even
    while quiet, so SEARCH doesn't jump straight back to the channel left."""
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.3)
    assert run.jumps == [144.5e6]
    run.tuned = 145.5e6  # a jump key, to a quiet channel; 144.5 MHz is still active
    run.tick(run.config.hold_s - 0.1)
    assert run.jumps == [144.5e6]
    assert run.status is not None and run.status.state == "holding"
    run.tick(0.3)
    assert run.jumps == [144.5e6, 144.5e6]


def test_a_locked_out_signals_edges_are_not_stopped_on() -> None:
    """A strong signal leaks into the channels beside it, well over the trigger
    (an NFM signal at -40 dBFS reads about -55 two 5 kHz steps away): with its
    center locked out, SEARCH landed on those edges. A separate signal nearby
    is still found."""
    run = _Run(_config(lockouts=frozenset({144.5e6})))
    run.signals |= {
        round(144.4875e6): -55.0,
        round(144.5e6): -40.0,  # locked out
        round(144.5125e6): -55.0,
        round(144.525e6): -68.0,
    }
    run.tick(1.0)
    assert run.jumps == []
    run.signals[round(144.55e6)] = -65.0  # a station of its own, beside the edge
    run.tick(0.5)
    assert run.jumps == [144.55e6]


def test_a_signal_past_an_edge_does_not_pull_search_to_the_edge() -> None:
    run = _Run()
    run.signals |= {round(143.9875e6): -40.0, round(144.0e6): -55.0}  # range starts at 144 MHz
    run.tick(1.0)
    assert run.jumps == []


def test_without_a_trigger_it_follows_the_squelch() -> None:
    """The squelch is the one control: SEARCH stops on channels 3 dB above it."""
    run = _Run(_config(trigger_db=None))
    run.signals[round(144.5e6)] = SQUELCH + 2.0  # opens the squelch, under the trigger
    run.signals[round(145.5e6)] = SQUELCH + 4.0
    run.tick(0.3)
    assert run.jumps == [145.5e6]


def test_a_window_outside_the_capture_reads_nothing_on_either_side() -> None:
    """Below the capture, a window once borrowed the first FFT bin."""
    flat = np.full(4096, -100.0, dtype=np.float32)
    outside = np.array([95.0e6 - 20e3, 105.0e6 + 20e3])
    assert np.all(np.isneginf(channel_powers_db(flat, 100.0e6, 10e6, outside, 12_500.0, 1.5)))


def test_an_edge_channel_stays_a_peak_beside_a_neighbor_outside_the_capture() -> None:
    """2.4 MS/s, 1 MHz steps: the grid point below the capture borrowed the
    strong low-edge interference and hid a real signal at the edge channel."""
    n, rate, center = 4096, 2.4e6, 146.5e6
    spectrum = np.full(n, -100.0, dtype=np.float32)
    spectrum[:8] = -20.0  # interference at the bottom of the capture
    spectrum[int((146.0e6 - center) / rate * n + n / 2)] = -30.0  # a signal at 146 MHz
    channels = np.array([146.0e6, 147.0e6])
    locate, outer = locate_levels(spectrum, center, rate, channels, 1e6, 12_500.0, 1.5)
    assert np.isneginf(outer[0])
    assert local_peaks(locate, outer)[0]


def test_a_window_narrower_than_a_bin_takes_the_nearest_bin() -> None:
    n, rate = 32768, 10e6
    bin_hz = rate / n
    spectrum = np.full(n, -100.0, dtype=np.float32)
    spectrum[n // 2] = -30.0  # the bin at the center frequency
    # A 200 Hz window centered a third of a bin above that bin's center covers no
    # bin center; it took the next bin up, which holds nothing.
    level = channel_powers_db(spectrum, 0.0, rate, np.array([bin_hz / 3]), 200.0, 1.0)
    assert level[0] == pytest.approx(-30.0, abs=0.01)


def test_a_run_of_identical_levels_peaks_at_its_middle() -> None:
    """Grid points finer than the FFT bins read alike: the peak was the run's
    last point, a consistent bias to one side."""
    assert list(np.flatnonzero(local_peaks(np.array([-90.0, -50, -50, -50, -90])))) == [2]
    assert list(np.flatnonzero(local_peaks(np.array([-90.0, -50, -60, -50, -50])))) == [1, 3]


def test_signal_peaks_take_each_signals_strongest_channel() -> None:
    channels = np.arange(10) * 5_000.0
    levels = np.array([-99, -70, -60, -75, -99, -99, -65, -99, -72, -71], dtype=float)
    assert signal_peaks(channels, levels, -80.0) == [10_000.0, 30_000.0, 45_000.0]


def test_signal_peaks_find_a_signal_beside_a_stronger_one() -> None:
    channels = np.arange(8) * 5_000.0
    levels = np.array([-99, -60, -40, -55, -50, -62, -99, -99], dtype=float)
    assert signal_peaks(channels, levels, -80.0) == [10_000.0, 20_000.0]


def test_signal_peaks_leave_out_the_signal_the_dial_is_on() -> None:
    channels = np.arange(10) * 5_000.0
    levels = np.array([-99, -70, -60, -75, -99, -99, -65, -99, -99, -99], dtype=float)
    # Dial at 15 kHz, off the strongest channel of its signal: that signal isn't a target.
    assert signal_peaks(channels, levels, -80.0, around_hz=15_000.0, half_width_hz=6_250.0) == [
        30_000.0
    ]


def test_the_locating_levels_pick_among_channels_that_read_alike() -> None:
    """Channels a step either side of a narrow signal read as high as its own
    (their wider windows hold it whole); the step-wide levels tell them apart."""
    config = _config(step_hz=5_000.0)
    channels = config.channels()
    levels = np.full(len(channels), NOISE)
    locate = np.full(len(channels), NOISE)
    k = int(np.searchsorted(channels, 145.5e6))
    levels[k - 1 : k + 2] = -60.0
    locate[k] = -60.0
    controller = SearchController()
    kwargs = {
        "tuned_hz": START,
        "current_db": NOISE,
        "hold_db": SQUELCH,
        "bandwidth_hz": BW,
        "lo_hz": 0.0,
        "lo_exclusion_hz": 0.0,
        "locate_db": locate,
    }
    targets = [
        controller.update(t, config, channels, levels, **kwargs)[0] for t in (0.0, 0.05, 0.1)
    ]
    assert targets[-1] == 145.5e6


def test_noise_at_the_squelch_does_not_hold_a_channel_whose_squelch_is_shut() -> None:
    # One unaveraged FFT frame per update reads a 10 kHz channel of noise with
    # about 1 dB of scatter: noise 0.75 dB under the squelch reads above it in a
    # quarter of the frames, which kept restarting the hold on an empty airband
    # channel. The demodulator's squelch (averaged, opening 3 dB higher) stays
    # shut there, and it decides.
    rng = np.random.default_rng(0)
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -60.0
    run.squelch_open = False
    for _ in range(int(4.0 / 0.05)):
        run.signals[round(144.5e6)] = SQUELCH - 0.75 + rng.normal(scale=1.06)
        run.tick(0.05)
    assert run.jumps == [144.5e6, 145.5e6]


def test_holds_while_the_squelch_is_open_even_if_a_frame_reads_low() -> None:
    run = _Run()
    run.signals[round(144.5e6)] = -60.0
    run.tick(0.2)
    run.signals[round(145.5e6)] = -50.0
    run.signals[round(144.5e6)] = NOISE  # the FFT frame misses it; the squelch hears it
    run.squelch_open = True
    run.tick(5.0)
    assert run.jumps == [144.5e6]
    run.squelch_open = False
    run.tick(2.2)
    assert run.jumps == [144.5e6, 145.5e6]
