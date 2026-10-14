"""Starting, steering and stopping SEARCH (core/sdr/search.py) on a device."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Literal

from tsdr.core.events.events import FFTUpdateEvent, MemoriesChangedEvent
from tsdr.core.landmarks import peak_frequencies
from tsdr.core.memories import get_memory_store
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.device_context import DeviceState, SDRDeviceContext
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.stages.search_stage import AudioChannel, audio_channel
from tsdr.core.sdr.processing import window_enbw_bins
from tsdr.core.sdr.search import (
    DEFAULT_HOLD_S,
    LOCKOUT_TAG,
    MAX_CHANNEL_BANDWIDTH_HZ,
    SearchConfig,
    channel_powers_db,
    locate_levels,
    signal_peaks,
    step_grid,
)
from tsdr.core.sdr.tune_policy import fitting_offset, lo_offset_for
from tsdr.core.tuning import STEP_LADDER, resolve_auto_step
from tsdr.core.tuning_state import get_tuning_state
from tsdr.radio.voice.detector import detector

logger = logging.getLogger(__name__)

_runs = itertools.count(1)
# SEARCH (and `[` / `]`) measure every grid channel each batch: a 10 Hz step
# over 9 MHz is 890,000 channels, 127 ms a batch against a 50 ms budget
# (i7-8700). The grid is never finer than this.
MIN_SEARCH_STEP_HZ = 1_000.0


@dataclass
class SearchSettings:
    """A device's SEARCH settings for this session: what was set explicitly
    (None = the default) and the session's lockouts."""

    low_hz: float | None = None  # None = the captured band's edge
    high_hz: float | None = None
    trigger_db: float | None = None  # None = follow the squelch (SearchConfig.trigger_for)
    hold_s: float = DEFAULT_HOLD_S
    max_dwell_s: float | None = None
    voice_squelch: bool = False
    lockouts: set[float] = field(default_factory=set)


_settings: dict[str, SearchSettings] = {}


def search_settings(device_id: str) -> SearchSettings:
    return _settings.setdefault(device_id, SearchSettings())


def search_limits(context: SDRDeviceContext, channel: AudioChannel | None) -> tuple[float, float]:
    """The widest range SEARCH (or `[` / `]`) can cover without moving the hardware."""
    config = context.config
    bandwidth = channel.bandwidth_hz if channel is not None else 0.0
    half = fitting_offset(config.sample_rate, bandwidth)
    low, high = config.center_frequency - half, config.center_frequency + half
    frequency_range = context.device.capabilities.frequency_range
    if frequency_range is not None:
        low, high = max(low, frequency_range[0]), min(high, frequency_range[1])
    return low, high


def _search_range(
    settings: SearchSettings, limit_low: float, limit_high: float
) -> tuple[float, float]:
    """The set edges that fit the captured band, else the band's own edges."""
    low, high = settings.low_hz, settings.high_hz
    low = low if low is not None and limit_low - 1.0 <= low <= limit_high else limit_low
    high = high if high is not None and limit_low <= high <= limit_high + 1.0 else limit_high
    if low >= high:
        return limit_low, limit_high
    return low, high


def _edge_in_use(edge: float | None, limit_low: float, limit_high: float) -> float | None:
    """A kept edge, if it lies within this capture (one set on another band doesn't)."""
    return edge if edge is not None and limit_low - 1.0 <= edge <= limit_high + 1.0 else None


def _check_edge_order(edges: SearchSettings, limit_low: float, limit_high: float) -> None:
    low = _edge_in_use(edges.low_hz, limit_low, limit_high)
    high = _edge_in_use(edges.high_hz, limit_low, limit_high)
    if low is not None and high is not None and low >= high:
        raise SDRException(
            f"The lower edge ({low / 1e6:.4f} MHz) must be below the upper edge "
            f"({high / 1e6:.4f} MHz)"
        )


def search_step_floor(config: DeviceConfig) -> float:
    """The finest grid SEARCH and `[` / `]` use: MIN_SEARCH_STEP_HZ, and at
    least one FFT bin (finer grid points read alike), rounded up to a step on
    the tuning ladder."""
    floor = max(MIN_SEARCH_STEP_HZ, config.sample_rate / config.fft_size)
    return min((float(s) for s in STEP_LADDER if s is not None and s >= floor), default=floor)


def search_step(context: SDRDeviceContext, mode: str) -> float:
    """The tuning step (`s`), or the mode's channel raster when it is on auto; see grid_step."""
    return grid_step(context, mode, get_tuning_state().step_for(mode))


def grid_step(context: SDRDeviceContext, mode: str, step: float | None) -> float:
    """SEARCH's grid for a tuning step (None = auto): no finer than search_step_floor, and
    never coarser than the channel. Channels are measured at the grid points, one channel
    wide, so a coarser grid leaves gaps: a station between grid points is never seen (a
    100 kHz step left a whole 2 m band blind). Then the mode's automatic step is used."""
    channel = audio_channel(context.config)
    if step is None or (channel is not None and step > channel.bandwidth_hz):
        step = resolve_auto_step(mode, context.config.tuned_frequency)
    return max(float(step), search_step_floor(context.config))


def jump_targets(
    context: SDRDeviceContext, fft: FFTUpdateEvent | None
) -> tuple[list[float], tuple[float, float]]:
    """Targets for `[` / `]` and the band they stay in: memories, and signals
    as SEARCH sees them (channels on the step grid at or above the squelch,
    one per signal, not the one the dial is on). Without a demodulated channel,
    spectrum peaks stand in for signals. While SEARCH runs, only signals within
    its range that it could stop on (not locked out). In free tuning, or while
    SEARCH runs, nothing at the LO: tuning there would move it."""
    config = context.config
    channel = audio_channel(config)
    search = config.search
    if search is not None:
        limits = (search.low_hz, search.high_hz)
        targets: list[float] = []
    else:
        limits = search_limits(context, channel)
        targets = [float(m.frequency) for m in get_memory_store().all()]
    targets += _signal_targets(context, channel, fft, limits)
    if channel is not None and (search is not None or config.tuning_mode == "free"):
        lo_exclusion = abs(
            lo_offset_for(
                setting=config.lo_offset,
                size_hz=config.lo_offset_hz,
                channel_bandwidth=channel.bandwidth_hz,
                sample_rate=config.sample_rate,
                caps=context.device.capabilities,
            )
        )
        targets = [f for f in targets if abs(f - config.center_frequency) >= lo_exclusion]
    return targets, limits


def _signal_targets(
    context: SDRDeviceContext,
    channel: AudioChannel | None,
    fft: FFTUpdateEvent | None,
    limits: tuple[float, float],
) -> list[float]:
    config = context.config
    if fft is None or len(fft.spectrum) == 0:
        return []
    if abs(fft.center_frequency - config.center_frequency) > fft.sample_rate / len(fft.spectrum):
        return []  # a frame from before a retune
    if channel is None or context.device.capabilities.provides_spectrum:
        return peak_frequencies(fft)
    search = config.search
    step = search.step_hz if search is not None else search_step(context, channel.mode)
    channels = step_grid(*limits, step)
    measured = channels + channel.passband_offset_hz
    enbw = window_enbw_bins(config.fft_window, len(fft.spectrum))
    levels = channel_powers_db(
        fft.spectrum, fft.center_frequency, fft.sample_rate, measured, channel.bandwidth_hz, enbw
    )
    locate, outer = locate_levels(
        fft.spectrum,
        fft.center_frequency,
        fft.sample_rate,
        measured,
        step,
        channel.bandwidth_hz,
        enbw,
    )
    signals = signal_peaks(
        channels,
        levels,
        channel.squelch_db,
        locate_db=locate,
        outer_db=outer,
        around_hz=config.tuned_frequency,
        half_width_hz=max(step, channel.bandwidth_hz) / 2,
    )
    if search is not None:
        signals = [f for f in signals if not search.is_locked_out(f, channel.bandwidth_hz)]
    return signals


def search_jump(device_id: str, frequency_hz: float) -> bool:
    """Move a running SEARCH's dial to `frequency_hz` (`[` / `]`): SEARCH keeps
    running and holds the channel as if it had found it. False if the move
    would retune the hardware, which SEARCH never does."""
    engine = get_engine()
    search = engine.get_device(device_id).config.search
    if search is None:
        return False
    return engine.update_device_config(
        device_id, from_search=True, while_search=search, tuned_frequency=frequency_hz
    )


def start_search(
    device_id: str,
    *,
    low_hz: float | None = None,
    high_hz: float | None = None,
    trigger_db: float | None = None,
    hold_s: float | None = None,
    max_dwell_s: float | None = None,
    step_hz: float | None = None,
    voice_squelch: bool | None = None,
    full_band: bool = False,
) -> SearchConfig:
    """Start (or restart) SEARCH. Settings not given are the ones set earlier
    this session; `max_dwell_s=0` means unlimited and `full_band` forgets the
    edges. Switches to free tuning and turns the squelch on: its threshold
    holds the channel."""
    engine = get_engine()
    context = engine.get_device(device_id)
    if context.state != DeviceState.RUNNING:
        raise SDRException(f"Start '{device_id}' before searching")
    if context.device.capabilities.provides_spectrum:
        raise SDRException("SEARCH needs the radio's own IQ; this device sends a finished spectrum")
    channel = audio_channel(context.config)
    if channel is None:
        raise SDRException("Choose a demodulator first (d, or the demod command)")
    if channel.bandwidth_hz > MAX_CHANNEL_BANDWIDTH_HZ:
        raise SDRException(
            f"SEARCH is for narrowband channels; {channel.mode} is "
            f"{channel.bandwidth_hz / 1e3:g} kHz wide"
        )

    for name, value in (
        ("lower edge", low_hz),
        ("upper edge", high_hz),
        ("trigger", trigger_db),
        ("hold", hold_s),
        ("dwell", max_dwell_s),
        ("step", step_hz),
    ):
        if value is not None and not math.isfinite(value):
            raise SDRException(f"The {name} must be a finite number")
    if step_hz is not None and step_hz <= 0:
        raise SDRException("The step must be positive")
    if (hold_s is not None and hold_s < 0) or (max_dwell_s is not None and max_dwell_s < 0):
        raise SDRException("Hold and dwell must not be negative")

    # Checked in full before any setting is kept: a refused command changes nothing.
    settings = search_settings(device_id)
    limit_low, limit_high = search_limits(context, channel)
    for edge in (low_hz, high_hz):
        if edge is not None and not limit_low - 1.0 <= edge <= limit_high + 1.0:
            raise SDRException(
                f"The search range must lie within {limit_low / 1e6:.4f}-"
                f"{limit_high / 1e6:.4f} MHz, the captured band"
            )
    edges = SearchSettings(
        low_hz=None if full_band else settings.low_hz,
        high_hz=None if full_band else settings.high_hz,
    )
    if low_hz is not None:
        edges.low_hz = low_hz
    if high_hz is not None:
        edges.high_hz = high_hz
    _check_edge_order(edges, limit_low, limit_high)
    low, high = _search_range(edges, limit_low, limit_high)

    trigger_given = trigger_db is not None
    if trigger_db is None:
        trigger_db = settings.trigger_db
    if trigger_db is not None and trigger_db <= channel.squelch_db:
        raise SDRException(
            f"The trigger ({trigger_db:.0f} dBFS) must be above the squelch "
            f"({channel.squelch_db:.0f} dBFS), which holds the channel"
        )
    hold = settings.hold_s if hold_s is None else hold_s
    if max_dwell_s is None:
        max_dwell_s = settings.max_dwell_s
    elif max_dwell_s == 0:
        max_dwell_s = None
    if voice_squelch is None:
        voice_squelch = settings.voice_squelch
    if voice_squelch:
        _load_detector()

    memory_lockouts = frozenset(
        float(m.frequency) for m in get_memory_store().find_by_tag(LOCKOUT_TAG)
    )
    config = SearchConfig(
        low_hz=low,
        high_hz=high,
        step_hz=grid_step(
            context,
            channel.mode,
            get_tuning_state().step_for(channel.mode) if step_hz is None else step_hz,
        ),
        trigger_db=trigger_db,
        hold_s=hold,
        max_dwell_s=max_dwell_s,
        voice_squelch=voice_squelch,
        lockouts=frozenset(settings.lockouts),
        memory_lockouts=memory_lockouts,
        run=next(_runs),
    )
    try:
        config.validate()
    except ValueError as e:
        raise SDRException(str(e)) from e

    if not channel.squelch_enabled:
        engine.update_squelch(device_id, "audio", enabled=True)
    engine.update_device_config(device_id, search=config, tuning_mode="free")
    settings.low_hz, settings.high_hz = edges.low_hz, edges.high_hz
    if trigger_given:
        settings.trigger_db = trigger_db
    settings.hold_s, settings.max_dwell_s = hold, max_dwell_s
    settings.voice_squelch = voice_squelch
    logger.info(
        "search_started device=%s low_hz=%.0f high_hz=%.0f step_hz=%.0f trigger_db=%s "
        "squelch_db=%.1f hold_s=%.1f",
        device_id,
        low,
        high,
        config.step_hz,
        "squelch" if trigger_db is None else f"{trigger_db:.1f}",
        channel.squelch_db,
        hold,
    )
    return config


def set_search_edge(device_id: str, edge: Literal["low", "high"]) -> float:
    """Put a SEARCH edge at the dial, for the running SEARCH or the next one.
    An edge on the wrong side of the other one is refused. Returns the frequency."""
    engine = get_engine()
    context = engine.get_device(device_id)
    frequency = context.config.tuned_frequency
    settings = search_settings(device_id)
    channel = audio_channel(context.config)
    limits = search_limits(context, channel)
    proposed = SearchSettings(low_hz=settings.low_hz, high_hz=settings.high_hz)
    if edge == "low":
        proposed.low_hz = frequency
    else:
        proposed.high_hz = frequency
    _check_edge_order(proposed, *limits)
    settings.low_hz, settings.high_hz = proposed.low_hz, proposed.high_hz
    search = context.config.search
    if search is not None and channel is not None:
        low, high = _search_range(settings, *limits)
        engine.update_device_config(
            device_id, while_search=search, search=replace(search, low_hz=low, high_hz=high)
        )
    logger.info("search_edge_set device=%s edge=%s frequency_hz=%.0f", device_id, edge, frequency)
    return frequency


def stop_search(device_id: str) -> bool:
    """Stop SEARCH; the dial stays where it is. False if it wasn't running."""
    engine = get_engine()
    if engine.get_device(device_id).config.search is None:
        return False
    engine.update_device_config(device_id, search=None)
    logger.info("search_stopped device=%s reason=user", device_id)
    return True


def lock_out_channel(device_id: str, *, permanent: bool = False) -> float:
    """Lock out the dial's channel: for this session, or as a memory tagged
    LOCKOUT_TAG. A running SEARCH moves on. Returns the frequency."""
    engine = get_engine()
    context = engine.get_device(device_id)
    frequency = context.config.tuned_frequency
    if permanent:
        audio = context.config.pipelines.get("audio")
        channel = audio_channel(context.config)
        if audio is None or audio.audio_spec is None or channel is None:
            raise SDRException("Choose a demodulator first (d, or the demod command)")
        store = get_memory_store()
        if not any(abs(m.frequency - frequency) < 1.0 for m in store.find_by_tag(LOCKOUT_TAG)):
            store.add(
                int(round(frequency)),
                f"Lockout {frequency / 1e6:.4f}",
                audio.audio_spec,
                int(channel.bandwidth_hz),
                tags=(LOCKOUT_TAG,),
            )
            engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
    else:
        settings = search_settings(device_id)
        settings.lockouts.add(frequency)
        search = context.config.search
        if search is not None:
            engine.update_device_config(
                device_id,
                while_search=search,
                search=replace(search, lockouts=frozenset(settings.lockouts)),
            )
    logger.info(
        "search_lockout device=%s frequency_hz=%.0f permanent=%s", device_id, frequency, permanent
    )
    return frequency


def clear_session_lockouts(device_id: str) -> int:
    """Drop this session's lockouts; memories tagged LOCKOUT_TAG stay. Returns how many."""
    engine = get_engine()
    settings = search_settings(device_id)
    count = len(settings.lockouts)
    settings.lockouts.clear()
    search = engine.get_device(device_id).config.search
    if search is not None and search.lockouts:
        engine.update_device_config(
            device_id, while_search=search, search=replace(search, lockouts=frozenset())
        )
    return count


@dataclass(frozen=True)
class Unlocked:
    """What `unlock` removed: the dial's lockouts, or else every session lockout."""

    frequency_hz: float | None  # the dial's channel; None = all session lockouts
    session: int  # session lockouts dropped
    saved: int  # lockout memories deleted


def unlock(device_id: str) -> Unlocked:
    """`ctrl+x`: unlock the dial's channel when it is locked out (a session
    lockout, or a memory tagged LOCKOUT_TAG, which is deleted: `X` there makes
    it again); otherwise drop every session lockout. A running SEARCH follows."""
    engine = get_engine()
    context = engine.get_device(device_id)
    config = context.config
    dial = config.tuned_frequency
    channel = audio_channel(config)
    if channel is None:
        half = 1.0
    else:
        step = (
            config.search.step_hz
            if config.search is not None
            else search_step(context, channel.mode)
        )
        half = max(step, channel.bandwidth_hz) / 2
    settings = search_settings(device_id)
    store = get_memory_store()
    here = {f for f in settings.lockouts if abs(f - dial) < half}
    saved = [m for m in store.find_by_tag(LOCKOUT_TAG) if abs(m.frequency - dial) < half]
    if here or saved:
        settings.lockouts -= here
        for memory in saved:
            store.remove(memory.id)
        if saved:
            engine.event_bus.publish(MemoriesChangedEvent(memories=tuple(store.all())))
        result = Unlocked(dial, len(here), len(saved))
    else:
        result = Unlocked(None, len(settings.lockouts), 0)
        settings.lockouts.clear()
    search = engine.get_device(device_id).config.search
    if search is not None and search.lockouts != frozenset(settings.lockouts):
        engine.update_device_config(
            device_id,
            while_search=search,
            search=replace(search, lockouts=frozenset(settings.lockouts)),
        )
    logger.info(
        "search_unlock device=%s frequency_hz=%s session=%d saved=%d",
        device_id,
        "all" if result.frequency_hz is None else f"{result.frequency_hz:.0f}",
        result.session,
        result.saved,
    )
    return result


def set_voice_squelch(device_id: str, on: bool | None = None) -> bool:
    """Turn VSQ on or off (None: toggle) in the running SEARCH; later runs
    this session keep it. Returns the new state."""
    engine = get_engine()
    search = engine.get_device(device_id).config.search
    if search is None:
        raise SDRException("VSQ works while SEARCH runs: start it with /")
    on = not search.voice_squelch if on is None else on
    if on:
        _load_detector()
    engine.update_device_config(
        device_id, while_search=search, search=replace(search, voice_squelch=on)
    )
    search_settings(device_id).voice_squelch = on
    return on


def _load_detector() -> None:
    """Load the voice detector here (once; a few ms), not in the pipeline thread."""
    try:
        detector()
    except (OSError, ValueError) as e:
        raise SDRException(f"VSQ unavailable: {e}") from e


def set_search_step(device_id: str, step_hz: float) -> float:
    """Follow a tuning-step change (`s`) in a running SEARCH, within grid_step's
    limits. Returns the step SEARCH uses."""
    engine = get_engine()
    context = engine.get_device(device_id)
    config = context.config
    channel = audio_channel(config)
    step_hz = grid_step(context, channel.mode if channel is not None else "", step_hz)
    search = config.search
    if search is not None and search.step_hz != step_hz:
        engine.update_device_config(
            device_id, while_search=search, search=replace(search, step_hz=step_hz)
        )
    return step_hz
