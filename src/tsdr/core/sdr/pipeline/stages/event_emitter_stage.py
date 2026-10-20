import math
import time

import numpy as np

from tsdr.core.events.events import (
    ConstellationUpdateEvent,
    DecoderOutputEvent,
    DemodStatusEvent,
    FFTUpdateEvent,
    StatsUpdateEvent,
)
from tsdr.core.sdr.config import SDRConfig
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.processing import BandFloor, channel_cn_db, compute_statistics
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import get_smoothed_stats, traced
from tsdr.devices.base import as_spectrum_source
from tsdr.radio.dsp._kernels import _iq_metrics_c64

# A channel measured across the capture: P25's 12.5 kHz.
P25_CHANNEL_HZ = 12_500.0


class RateLimiter:
    """Fixed-FPS gate for UI updates."""

    def __init__(self, target_fps: int):
        self.target_fps = target_fps
        self.min_interval = 1.0 / target_fps
        self.last_send_time: float | None = None

    def should_send(self) -> bool:
        now = time.perf_counter()
        if self.last_send_time is None or now - self.last_send_time >= self.min_interval:
            self.last_send_time = now
            return True
        return False

    def set_target_fps(self, fps: int) -> None:
        if fps <= 0:
            raise ValueError("FPS must be positive")

        self.target_fps = fps
        self.min_interval = 1.0 / fps


# The C/N meter's ballistics (the owner's peak detector): a higher reading is taken at once,
# a lower one approached as a leaky integrator with this time constant.
CN_RELEASE_S = 0.5


def cn_peak_hold(held: float | None, cn: float | None, dt: float) -> float | None:
    """The meter's C/N after a reading `cn`, `dt` seconds after the last."""
    if cn is None or held is None or cn >= held:
        return cn
    return held + (cn - held) * (1.0 - math.exp(-max(dt, 0.0) / CN_RELEASE_S))


class EventEmitterStage:
    """Terminal stage that publishes TUI-bound events.

    Publishes:
    - FFTUpdateEvent (rate-limited) when spectrum/frequencies are populated
    - StatsUpdateEvent (rate-limited) when spectrum is populated
    - DecoderOutputEvent when decoded_messages are present
    """

    def __init__(self, config: SDRConfig):
        self.fft_rate_limiter = RateLimiter(target_fps=config.update_rate_fps)
        self.stats_rate_limiter = RateLimiter(target_fps=config.update_rate_fps)
        self.constellation_rate_limiter = RateLimiter(target_fps=20)
        self._last_demod_status: DemodStatus | None = None
        self._floor = BandFloor()
        self._cn_held: float | None = None
        self._cn_time = 0.0
        # The channels across the capture (P25's calls): granted frequency -> (the call's serial, C/N held)
        self._channels_held: dict[int, tuple[int, float | None]] = {}

    @traced("event_emitter")
    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        dc = context.device_context
        device_id = dc.device_id

        self._emit_fft(data, context, device_id)
        self._emit_stats(data, context, device_id)
        self._emit_demod_status(data, context, device_id)
        self._emit_decoder_messages(data, context, device_id)
        self._emit_constellation(data, context, device_id)

        return data

    def _emit_fft(self, data: SamplesBatch, context: PipelineContext, device_id: str) -> None:
        # Spectrum-providing devices publish their wideband frames from the I/O
        # worker; the narrowband IQ FFT would fight them over the display. Stats
        # below stay IQ-grounded.
        if context.device_context.device.capabilities.provides_spectrum:
            return
        if data.spectrum is None or data.frequencies is None:
            return
        if not self.fft_rate_limiter.should_send():
            return

        context.event_bus.publish(
            FFTUpdateEvent(
                source_id=f"fft_{device_id}",
                device_id=device_id,
                spectrum=data.spectrum,
                frequencies=data.frequencies,
                center_frequency=data.center_frequency,
                sample_rate=data.sample_rate,
            )
        )

    def _emit_stats(self, data: SamplesBatch, context: PipelineContext, device_id: str) -> None:
        if data.spectrum is None:
            return
        if not self.stats_rate_limiter.should_send():
            return

        dc = context.device_context
        profile = dc.demod_profile if dc else None
        channel_bw = profile.channel_bandwidth if profile else None
        signal_stats = compute_statistics(data.spectrum, data.center_frequency, data.sample_rate)
        # The meter's C/N from the spectrum before the display's averaging (its power
        # averaged over instant_frames frames), the band's floor following it.
        measured = data.instant_spectrum if data.instant_spectrum is not None else data.spectrum
        frames = data.instant_frames if data.instant_spectrum is not None else 1
        dt = data.capture_utc_s - self._cn_time
        self._cn_time = data.capture_utc_s
        self._floor.update(measured, frames, data.center_frequency, data.sample_rate, dt)
        cn = None
        if channel_bw is not None:
            offset = dc.config.tuned_frequency - data.center_frequency if dc else 0.0
            cn = channel_cn_db(measured, data.sample_rate, offset, channel_bw, self._floor)
        self._cn_held = cn_peak_hold(self._cn_held, cn, dt)
        channel_snrs = self._channel_cns(measured, data, dc, dt) if dc else ()

        sample_queue = dc.sample_queue
        queue_size = sample_queue.qsize() if sample_queue else 0
        queue_capacity = sample_queue.maxsize if sample_queue else 0

        # IQ amplitude metrics (single-pass numba kernel)
        iq_rms = None
        iq_peak = None
        iq_clip_pct = None
        if data.iq_samples is not None and len(data.iq_samples) > 0:
            iq_rms, iq_peak, iq_clip_pct = _iq_metrics_c64(
                np.ascontiguousarray(data.iq_samples, dtype=np.complex64)
            )

        source = as_spectrum_source(dc.device)
        spectrum_view = source.spectrum_view_status() if source is not None else None

        context.event_bus.publish(
            StatsUpdateEvent(
                source_id=f"stats_{device_id}",
                device_id=device_id,
                center_frequency=data.center_frequency,
                sample_rate=data.sample_rate,
                rf_gain=data.rf_gain,
                samples_processed=dc.total_samples_read,
                samples_dropped=dc.dropped_samples,
                queue_size=queue_size,
                queue_capacity=queue_capacity,
                peak_power=signal_stats.peak_power,
                average_power=signal_stats.average_power,
                peak_frequency=signal_stats.peak_frequency,
                peak_bin=signal_stats.peak_bin,
                noise_floor=signal_stats.noise_floor,
                dynamic_range=signal_stats.dynamic_range,
                fft_size=dc.config.fft_size,
                fft_window=dc.config.fft_window,
                spectrum_bins=len(data.spectrum),
                demod_mode=dc.active_mode,
                channel_snr=self._cn_held,
                channel_snrs=channel_snrs,
                stereo=dc.stereo,
                iq_rms=iq_rms,
                iq_peak=iq_peak,
                iq_clip_pct=iq_clip_pct,
                update_rate_fps=context.config.update_rate_fps,
                performance_stats=get_smoothed_stats(window_seconds=5.0),
                spectrum_view=spectrum_view,
            )
        )

    def _channel_cns(
        self, measured: np.ndarray, data: SamplesBatch, dc, dt: float
    ) -> tuple[tuple[int, float], ...]:
        """Each channel the demodulator decodes across the capture, measured as the meter measures the dial's (12.5
        kHz at its corrected position), with a peak hold of its own reset when its frequency's call changes."""
        positions = dc.channel_positions
        if not positions:
            self._channels_held = {}
            return ()
        held: dict[int, tuple[int, float | None]] = {}
        for cid, freq, position in positions:
            offset = dc.config.tuned_frequency + position - data.center_frequency
            cn = channel_cn_db(measured, data.sample_rate, offset, P25_CHANNEL_HZ, self._floor)
            old = self._channels_held.get(freq)
            before = old[1] if old is not None and old[0] == cid else None
            held[freq] = (cid, cn_peak_hold(before, cn, dt))
        self._channels_held = held
        return tuple((f, v) for f, (_, v) in held.items() if v is not None)

    def _emit_demod_status(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if data.demod_status is None:
            return
        if data.demod_status == self._last_demod_status:
            return
        self._last_demod_status = data.demod_status
        context.event_bus.publish(
            DemodStatusEvent(
                source_id=f"demod_status_{device_id}",
                device_id=device_id,
                demod_status=data.demod_status,
            )
        )

    def _emit_decoder_messages(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if not data.decoded_messages:
            return

        context.event_bus.publish(
            DecoderOutputEvent(
                source_id=f"demod_{device_id}",
                device_id=device_id,
                protocol=context.device_context.active_mode,
                messages=data.decoded_messages,
            )
        )

    def _emit_constellation(
        self, data: SamplesBatch, context: PipelineContext, device_id: str
    ) -> None:
        if data.constellation_points is None:
            return
        if not self.constellation_rate_limiter.should_send():
            return

        context.event_bus.publish(
            ConstellationUpdateEvent(
                source_id=f"constellation_{device_id}",
                device_id=device_id,
                points=data.constellation_points,
                modulation=data.constellation_modulation,
            )
        )

    def on_config_change(self, config) -> None:
        if isinstance(config, SDRConfig):
            self.fft_rate_limiter.set_target_fps(config.update_rate_fps)
            self.stats_rate_limiter.set_target_fps(config.update_rate_fps)

    def reset(self) -> None:
        self._last_demod_status = None
