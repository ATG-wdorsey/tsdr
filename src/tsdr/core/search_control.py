"""Starting, steering and stopping SEARCH (core/sdr/search.py) on a device."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace

from tsdr.core.events.events import MemoriesChangedEvent
from tsdr.core.memories import get_memory_store
from tsdr.core.sdr.device_context import DeviceState, SDRDeviceContext
from tsdr.core.sdr.engine import get_engine
from tsdr.core.sdr.exceptions import SDRException
from tsdr.core.sdr.pipeline.stages.search_stage import AudioChannel, audio_channel
from tsdr.core.sdr.search import (
    DEFAULT_HOLD_S,
    LOCKOUT_TAG,
    MAX_CHANNEL_BANDWIDTH_HZ,
    SearchConfig,
)
from tsdr.core.sdr.tune_policy import fitting_offset
from tsdr.core.tuning import resolve_auto_step
from tsdr.core.tuning_state import get_tuning_state

logger = logging.getLogger(__name__)

# Without a trigger given, SEARCH stops on channels this far above the squelch.
DEFAULT_TRIGGER_ABOVE_SQUELCH_DB = 6.0

_runs = itertools.count(1)


@dataclass
class SearchSettings:
    """A device's last SEARCH settings and this session's lockouts."""

    low_hz: float | None = None
    high_hz: float | None = None
    trigger_db: float | None = None
    hold_s: float = DEFAULT_HOLD_S
    max_dwell_s: float | None = None
    lockouts: set[float] = field(default_factory=set)


_settings: dict[str, SearchSettings] = {}


def search_settings(device_id: str) -> SearchSettings:
    return _settings.setdefault(device_id, SearchSettings())


def search_limits(context: SDRDeviceContext, channel: AudioChannel) -> tuple[float, float]:
    """The widest range SEARCH can cover without moving the hardware."""
    config = context.config
    half = fitting_offset(config.sample_rate, channel.bandwidth_hz)
    low, high = config.center_frequency - half, config.center_frequency + half
    frequency_range = context.device.capabilities.frequency_range
    if frequency_range is not None:
        low, high = max(low, frequency_range[0]), min(high, frequency_range[1])
    return low, high


def search_step(context: SDRDeviceContext, mode: str) -> float:
    """The tuning step (`s`), or the mode's channel raster when it is on auto."""
    step = get_tuning_state().step
    return step if step is not None else resolve_auto_step(mode, context.config.tuned_frequency)


def start_search(
    device_id: str,
    *,
    low_hz: float | None = None,
    high_hz: float | None = None,
    trigger_db: float | None = None,
    hold_s: float | None = None,
    max_dwell_s: float | None = None,
    step_hz: float | None = None,
) -> SearchConfig:
    """Start (or restart) SEARCH. Settings not given are the device's last
    ones; `max_dwell_s=0` means unlimited. Switches to free tuning and turns
    the squelch on: its threshold holds the channel."""
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

    settings = search_settings(device_id)
    limit_low, limit_high = search_limits(context, channel)
    if (
        low_hz is None
        and high_hz is None
        and settings.low_hz is not None
        and settings.high_hz is not None
        and limit_low <= settings.low_hz
        and settings.high_hz <= limit_high
    ):
        low_hz, high_hz = settings.low_hz, settings.high_hz
    low = limit_low if low_hz is None else low_hz
    high = limit_high if high_hz is None else high_hz
    if low < limit_low - 1.0 or high > limit_high + 1.0:
        raise SDRException(
            f"The search range must lie within {limit_low / 1e6:.4f}-{limit_high / 1e6:.4f} MHz, "
            "the captured band"
        )

    if trigger_db is None:
        trigger_db = settings.trigger_db
    if trigger_db is None:
        trigger_db = channel.squelch_db + DEFAULT_TRIGGER_ABOVE_SQUELCH_DB
    if trigger_db <= channel.squelch_db:
        raise SDRException(
            f"The trigger ({trigger_db:.0f} dBFS) must be above the squelch "
            f"({channel.squelch_db:.0f} dBFS), which holds the channel"
        )
    hold = settings.hold_s if hold_s is None else hold_s
    if max_dwell_s is None:
        max_dwell_s = settings.max_dwell_s
    elif max_dwell_s == 0:
        max_dwell_s = None

    memory_lockouts = frozenset(
        float(m.frequency) for m in get_memory_store().find_by_tag(LOCKOUT_TAG)
    )
    config = SearchConfig(
        low_hz=low,
        high_hz=high,
        step_hz=step_hz or search_step(context, channel.mode),
        trigger_db=trigger_db,
        hold_s=hold,
        max_dwell_s=max_dwell_s,
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
    settings.low_hz, settings.high_hz = low, high
    settings.trigger_db, settings.hold_s, settings.max_dwell_s = trigger_db, hold, max_dwell_s
    logger.info(
        "search_started device=%s low_hz=%.0f high_hz=%.0f step_hz=%.0f trigger_db=%.1f "
        "squelch_db=%.1f hold_s=%.1f",
        device_id,
        low,
        high,
        config.step_hz,
        trigger_db,
        channel.squelch_db,
        hold,
    )
    return config


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
                device_id, search=replace(search, lockouts=frozenset(settings.lockouts))
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
        engine.update_device_config(device_id, search=replace(search, lockouts=frozenset()))
    return count


def set_search_step(device_id: str, step_hz: float) -> None:
    """Follow a tuning-step change (`s`) in a running SEARCH."""
    engine = get_engine()
    search = engine.get_device(device_id).config.search
    if search is not None and search.step_hz != step_hz:
        engine.update_device_config(device_id, search=replace(search, step_hz=step_hz))
