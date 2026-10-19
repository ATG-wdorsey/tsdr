import logging
import math
import time
from dataclasses import replace

import numpy as np

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.datatypes import DemodStatus
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span
from tsdr.radio.audio_decoders import AudioDecoder, make_audio_decoder
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.registry import make_demodulator

logger = logging.getLogger(__name__)

# While the squelch stays open, its level is logged this often (a hiss that won't go away).
SQUELCH_HELD_LOG_S = 10.0


class DemodulatorStage:
    """Unified stage that runs any Demodulator (audio demod or protocol decoder).

    Calls demodulate(), attaches audio batches and decoded messages onto the
    SamplesBatch. The downstream DenoiserStage drains the audio to the queue.
    An audio decoder named by the spec (`DemodSpec.decoder`) reads the
    demodulator's audio from before its squelch and adds its messages.
    Pure data stage, no event publishing.
    """

    def __init__(
        self,
        demodulator: Demodulator,
        mode_name: str = "",
        pipeline_name: str = "",
        de_emphasis: float = 50e-6,
        spec: DemodSpec | None = None,
    ):
        self._demodulator = demodulator
        self._de_emphasis = de_emphasis
        demodulator.set_de_emphasis(de_emphasis)
        raw_name = mode_name or type(demodulator).LABEL
        self.mode_name = raw_name.upper().replace(" ", "")
        self._pipeline_name = pipeline_name
        self._last_freq: float | None = None
        self._last_sample_rate: float | None = None
        self._last_sstv_mode: str | None = None
        self._last_dmr_slot: str | None = spec.dmr_slot if spec is not None else None
        self._decoder: AudioDecoder | None = None
        self._decoder_name: str | None = None
        self._squelch_open: bool | None = None
        self._squelch_since = time.monotonic()
        self._squelch_logged = 0.0
        if spec is not None:
            self._follow_decoder(spec)

    @property
    def demodulator(self) -> Demodulator:
        return self._demodulator

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if data.iq_samples is None:
            return data

        # Pass the currently tuned frequency down so protocol decoders can
        # compare it against network-announced frequencies (e.g. TETRA MCCH),
        # and the capture's center, for those that decode other channels in it
        # (P25 following calls).
        if context.device_context is not None:
            self._demodulator.set_tuned_frequency(
                int(context.device_context.config.tuned_frequency)
            )
            self._demodulator.set_capture_center(
                int(data.capture_center_frequency or data.center_frequency)
            )

        with span("demodulate"):
            self._demodulator.demodulate(data.iq_samples, data.capture_utc_s)

        batches = tuple(self._demodulator.get_audio())

        # Cross-pipeline stereo flag (stats reads device_context)
        if batches and context.device_context is not None:
            context.device_context.stereo = batches[-1].stereo

        messages = self._demodulator.get_messages()
        status = self._demodulator.status()
        if self._decoder is not None:
            rate = float(getattr(self._demodulator, "decimated_rate", 0.0))
            with span("audio_decoder"):
                for audio in self._demodulator.get_tapped_audio():
                    self._decoder.process(audio, rate, data.capture_utc_s)
            messages = [*messages, *self._decoder.get_messages()]
            description = self._decoder.description()
            if description is not None:
                status = replace(status, description=description)
        self._log_squelch(status, data, context)
        changes: dict = {
            "stage_name": "demodulator",
            "demod_status": status,
            "audio_batches": batches,
        }
        if messages:
            changes["decoded_messages"] = tuple(messages)

        # Constellation points (gated on device config)
        if (
            context.device_context is not None
            and context.device_context.config.calculate_constellation
        ):
            result = self._demodulator.get_constellation()
            if result is not None:
                points, modulation = result
                changes["constellation_points"] = points
                changes["constellation_modulation"] = modulation

        return data.with_changes(**changes)

    def on_config_change(self, config) -> None:
        if isinstance(config, SDRConfig):
            self._de_emphasis = config.wfm_de_emphasis_us / 1e6
            self._demodulator.set_de_emphasis(self._de_emphasis)
            return
        if not isinstance(config, DeviceConfig):
            return
        pipeline_config = config.pipelines.get(self._pipeline_name)
        spec = pipeline_config.audio_spec if pipeline_config is not None else None

        if pipeline_config is not None and spec is not None:
            new_mode = spec.mode.upper()
            if new_mode != self.mode_name:
                # Build before assigning so a failed make_demodulator leaves the old demod intact.
                new_demod = make_demodulator(spec, config.sample_rate, config.channel_bandwidth)
                logger.info("demodulator_swapped old=%s new=%s", self.mode_name, new_mode)
                self._demodulator = new_demod
                self._demodulator.set_de_emphasis(self._de_emphasis)
                self.mode_name = new_mode
                self._last_sample_rate = config.sample_rate
                self._last_freq = config.tuned_frequency
                self._last_sstv_mode = spec.sstv_mode
                self._last_dmr_slot = spec.dmr_slot
                self._demodulator.set_squelch(
                    enabled=pipeline_config.squelch_enabled,
                    threshold_db=pipeline_config.squelch_threshold_db,
                    hang_ms=pipeline_config.squelch_hang_ms,
                )
                self._follow_decoder(spec, rebuild=True)
                return

        # Sample-rate first: bandwidth/deviation/squelch hooks below
        # build filters off decimated_rate, which the rebuild updates.
        if config.sample_rate != self._last_sample_rate:
            self._last_sample_rate = config.sample_rate
            logger.debug("demodulator_sample_rate_changed rate=%d", int(config.sample_rate))
            self._demodulator.set_sample_rate(config.sample_rate)
            self.reset()
        if config.channel_bandwidth is not None:
            self._demodulator.set_channel_bandwidth(config.channel_bandwidth)
        # Keyed on the dial, not center: a recenter keeps the baseband input
        # identical and resetting would only cause an audio dropout.
        if config.tuned_frequency != self._last_freq:
            logger.debug("demodulator_reset reason=frequency_changed")
            self._last_freq = config.tuned_frequency
            self.reset()
        if pipeline_config is None or spec is None:
            return
        self._demodulator.set_deviation(spec.fm_deviation_hz)
        self._follow_decoder(spec)
        if spec.sstv_mode != self._last_sstv_mode:
            self._last_sstv_mode = spec.sstv_mode
            if self._decoder is not None:
                self._decoder.set_sstv_mode(spec.sstv_mode)
        if spec.dmr_slot != self._last_dmr_slot:
            self._last_dmr_slot = spec.dmr_slot
            self._demodulator.set_audio_slot(spec.dmr_slot)
        self._demodulator.set_squelch(
            enabled=pipeline_config.squelch_enabled,
            threshold_db=pipeline_config.squelch_threshold_db,
            hang_ms=pipeline_config.squelch_hang_ms,
        )

    def _log_squelch(
        self, status: DemodStatus, data: SamplesBatch, context: PipelineContext
    ) -> None:
        """Log each squelch opening and closing, and its level every SQUELCH_HELD_LOG_S while
        it stays open, with what decides it (the in-channel level against the threshold) and
        the capture's peak (an overloaded radio makes in-channel noise of its own)."""
        is_open = status.squelch_open
        if is_open is None:
            self._squelch_open = None
            return
        now = time.monotonic()
        lasted = now - self._squelch_since  # how long the state before this one held
        if is_open == self._squelch_open:
            if not is_open or now - self._squelch_logged < SQUELCH_HELD_LOG_S:
                return
            state = "held_open"
        else:
            state = "opened" if is_open else "closed"
            self._squelch_open = is_open
            self._squelch_since = now
        self._squelch_logged = now
        device = context.device_context
        iq = data.iq_samples
        peak = float(np.max(np.abs(iq))) if iq is not None and len(iq) else 0.0
        logger.info(
            "squelch_change device=%s state=%s after_s=%.0f freq_hz=%.0f level_db=%.1f "
            "threshold_db=%.1f adc_peak_dbfs=%.1f",
            device.device_id if device is not None else None,
            state,
            lasted,
            device.config.tuned_frequency if device is not None else 0.0,
            status.squelch_level_db if status.squelch_level_db is not None else math.nan,
            status.squelch_threshold_db if status.squelch_threshold_db is not None else math.nan,
            20 * math.log10(peak + 1e-12),
        )

    def _follow_decoder(self, spec: DemodSpec, *, rebuild: bool = False) -> None:
        """Attach, swap or drop the audio decoder the spec names; a new demodulator
        (`rebuild`) starts a fresh one, since its audio starts afresh too."""
        name = spec.decoder.upper() if spec.decoder else None
        if name == self._decoder_name and not rebuild:
            return
        self._decoder_name = name
        self._decoder = make_audio_decoder(name, spec.sstv_mode) if name else None
        self._last_sstv_mode = spec.sstv_mode
        self._demodulator.tap_audio = self._decoder is not None
        logger.info("audio_decoder_changed decoder=%s mode=%s", name, self.mode_name)

    def reset(self) -> None:
        self._demodulator.reset()
        if self._decoder is not None:
            self._decoder.reset()
