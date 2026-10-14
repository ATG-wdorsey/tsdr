"""Runs SEARCH (core/sdr/search.py) on each batch's unaveraged spectrum."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tsdr.core import clock_sync
from tsdr.core.events.events import SearchStatusEvent, SearchTuneEvent
from tsdr.core.sdr.config import DeviceConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.processing import window_enbw_bins
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.sdr.search import (
    SearchController,
    SearchStatus,
    channel_powers_db,
    locate_levels,
)
from tsdr.core.sdr.tune_policy import lo_offset_for
from tsdr.core.tracing import traced
from tsdr.radio.registry import demod_profile

logger = logging.getLogger(__name__)

# Status events between state changes, for the level and hold countdown.
_STATUS_INTERVAL_S = 0.25


@dataclass(frozen=True)
class AudioChannel:
    """The demodulated channel SEARCH steers."""

    mode: str
    bandwidth_hz: float
    # Where the passband sits relative to the dial: +bw/2 for upper sideband.
    passband_offset_hz: float
    squelch_db: float
    squelch_enabled: bool


def audio_channel(config: DeviceConfig) -> AudioChannel | None:
    """The demodulated channel, or None without a demodulator."""
    audio = config.pipelines.get("audio")
    if audio is None or audio.audio_spec is None:
        return None
    mode = audio.audio_spec.mode
    # The profile's width: the setting for audio modes, a decoder's own fixed
    # width (ADS-B, DAB, RTTY) whatever the setting says.
    profile = demod_profile(mode, config.channel_bandwidth)
    bandwidth = float(profile.channel_bandwidth)
    offset = {"upper": bandwidth / 2, "lower": -bandwidth / 2}.get(profile.sideband or "", 0.0)
    return AudioChannel(
        mode=mode,
        bandwidth_hz=bandwidth,
        passband_offset_hz=offset,
        squelch_db=audio.squelch_threshold_db,
        squelch_enabled=audio.squelch_enabled,
    )


class SearchStage:
    """Measures every channel in the search range, asks the engine to jump
    (SearchTuneEvent) and publishes SearchStatusEvent. Idle while
    `DeviceConfig.search` is None."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._controller = SearchController()
        self._run: int | None = None
        self._channels = np.empty(0)
        self._channels_key: tuple[float, float, float] | None = None
        self._last_status: SearchStatus | None = None
        self._status_at = 0.0

    def on_config_change(self, config: object) -> None:
        if isinstance(config, DeviceConfig):
            self._config = config

    def reset(self) -> None:
        self._controller = SearchController()
        self._last_status = None

    @traced("search")
    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        config = self._config
        search = config.search
        if search is None:
            self._run = None
            return data
        if search.run != self._run:
            self._controller = SearchController()
            self._run = search.run
            self._last_status = None
        spectrum = data.instant_spectrum
        channel = audio_channel(config)
        if spectrum is None or channel is None or len(spectrum) == 0:
            return data
        # A batch captured before a retune or rate change measures the wrong
        # band. Batches carry the rate the device reports (a PlutoSDR set to
        # 10 MS/s reports 9999998), so compare loosely.
        if (
            abs(data.center_frequency - config.center_frequency) > data.sample_rate / len(spectrum)
            or abs(data.sample_rate - config.sample_rate) > 1e-3 * config.sample_rate
        ):
            return data

        key = (search.low_hz, search.high_hz, search.step_hz)
        if key != self._channels_key:
            self._channels = search.channels()
            self._channels_key = key
        enbw = window_enbw_bins(config.fft_window, len(spectrum))
        tuned = config.tuned_frequency
        measured = np.append(self._channels, tuned) + channel.passband_offset_hz
        levels = channel_powers_db(
            spectrum, data.center_frequency, data.sample_rate, measured, channel.bandwidth_hz, enbw
        )
        locate, outer = locate_levels(
            spectrum,
            data.center_frequency,
            data.sample_rate,
            measured[:-1],
            search.step_hz,
            channel.bandwidth_hz,
            enbw,
        )
        lo_offset = lo_offset_for(
            setting=config.lo_offset,
            size_hz=config.lo_offset_hz,
            channel_bandwidth=channel.bandwidth_hz,
            sample_rate=config.sample_rate,
            caps=context.device_context.device.capabilities,
        )
        # Capture time (0.0 = not stamped, as in tests). Clock sync can step it;
        # a step only shortens or lengthens one hold or attack.
        now = data.capture_utc_s if data.capture_utc_s > 0.0 else clock_sync.now_utc_seconds()
        demod = context.device_context.demod_status
        squelch_open = demod.squelch_open if demod is not None else None
        target, status = self._controller.update(
            now,
            search,
            self._channels,
            levels[:-1],
            tuned_hz=tuned,
            current_db=float(levels[-1]),
            hold_db=channel.squelch_db,
            squelch_open=squelch_open,
            bandwidth_hz=channel.bandwidth_hz,
            lo_hz=data.center_frequency,
            lo_exclusion_hz=abs(lo_offset),
            locate_db=locate,
            outer_locate_db=outer,
        )
        device_id = context.device_context.device_id
        if target is not None:
            logger.info(
                "search_jump device=%s from_hz=%.0f to_hz=%.0f level_db=%.1f",
                device_id,
                tuned,
                target,
                float(levels[:-1][np.searchsorted(self._channels, target)]),
            )
            context.event_bus.publish(
                SearchTuneEvent(device_id=device_id, frequency_hz=target, run=search.run)
            )
        last = self._last_status
        if (
            last is None
            or (status.state, status.channel_hz) != (last.state, last.channel_hz)
            or now - self._status_at >= _STATUS_INTERVAL_S
        ):
            self._last_status = status
            self._status_at = now
            context.event_bus.publish(SearchStatusEvent(device_id=device_id, status=status))
        return data
