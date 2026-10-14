import logging
from dataclasses import replace

from tsdr.core.demod_spec import DemodSpec
from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.core.tracing import span
from tsdr.radio.audio_decoders import AudioDecoder, make_audio_decoder
from tsdr.radio.demodulators import Demodulator
from tsdr.radio.registry import make_demodulator

logger = logging.getLogger(__name__)


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
        self._decoder: AudioDecoder | None = None
        self._decoder_name: str | None = None
        if spec is not None:
            self._follow_decoder(spec)

    @property
    def demodulator(self) -> Demodulator:
        return self._demodulator

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        if data.iq_samples is None:
            return data

        # Pass the currently tuned frequency down so protocol decoders can
        # compare it against network-announced frequencies (e.g. TETRA MCCH).
        if context.device_context is not None:
            self._demodulator.set_tuned_frequency(
                int(context.device_context.config.tuned_frequency)
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
        if spec.fm_deviation_hz is not None:
            self._demodulator.set_deviation(spec.fm_deviation_hz)
        self._follow_decoder(spec)
        if spec.sstv_mode != self._last_sstv_mode:
            self._last_sstv_mode = spec.sstv_mode
            if self._decoder is not None:
                self._decoder.set_sstv_mode(spec.sstv_mode)
        self._demodulator.set_squelch(
            enabled=pipeline_config.squelch_enabled,
            threshold_db=pipeline_config.squelch_threshold_db,
            hang_ms=pipeline_config.squelch_hang_ms,
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
