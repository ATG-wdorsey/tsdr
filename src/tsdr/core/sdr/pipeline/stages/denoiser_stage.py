import logging
import queue
from dataclasses import replace

import numpy as np

from tsdr.core.sdr.config import DeviceConfig, SDRConfig
from tsdr.core.sdr.pipeline.pipeline import PipelineContext
from tsdr.core.sdr.samples_batch import SamplesBatch
from tsdr.radio.dsp.denoise import AudioDenoiser
from tsdr.radio.dsp.rnnoise import rnnoise_available

logger = logging.getLogger(__name__)


class DenoiserStage:
    """Audio output stage: drains the batch's audio to the audio queue, applying
    RNNoise speech denoising when the global ``SDRConfig.denoise`` flag is on (not
    to audio decoded from a digital codec, ``AudioBatch.codec``: DMR, TETRA, DAB).

    Always present in the audio pipeline (after DEMODULATOR); the global flag
    only flips it between denoise and passthrough, so toggling never changes the
    pipeline shape.

    While SEARCH runs with VSQ on, it also feeds the audio, before RNNoise, to
    the device's VSQ (core/sdr/voice_squelch.py), and mutes the output while
    VSQ calls the transmission not voice.
    """

    def __init__(self, enabled: bool = False):
        self._denoiser: AudioDenoiser | None = None
        self._denoise_warned = False
        self._vsq_key: tuple[int, float] | None = None
        self._last_sample_rate: float | None = None
        self._last_freq: float | None = None
        self._set_denoise(enabled)

    def process(self, data: SamplesBatch, context: PipelineContext) -> SamplesBatch | None:
        vsq = context.device_context.voice_squelch if context.device_context is not None else None
        if vsq is not None:
            vsq.set_target(self._vsq_key)
        squelch_open = data.demod_status.squelch_open if data.demod_status is not None else None
        for batch in data.audio_batches:
            if vsq is not None and batch.samples.ndim == 1:  # not WFM's two channels
                vsq.feed(batch.samples, batch.sample_rate, squelch_open)
            if context.audio_queue is None:
                continue
            if self._denoiser is not None and not batch.codec:
                batch = self._denoiser.process(batch)
                if len(batch.samples) == 0:
                    continue
            if vsq is not None and vsq.muted:
                batch = replace(batch, samples=np.zeros_like(batch.samples))
            try:
                context.audio_queue.put(batch, block=False)
            except queue.Full:
                pass
        return data.with_changes(audio_batches=())

    def on_config_change(self, config) -> None:
        if isinstance(config, SDRConfig):
            self._set_denoise(config.denoise)
            return
        if not isinstance(config, DeviceConfig):
            return
        search = config.search
        self._vsq_key = (
            (search.run, config.tuned_frequency)
            if search is not None and search.voice_squelch
            else None
        )
        if (
            config.sample_rate != self._last_sample_rate
            or config.tuned_frequency != self._last_freq
        ):
            # A retune or sample-rate change makes the carried tail stale.
            self._last_sample_rate = config.sample_rate
            self._last_freq = config.tuned_frequency
            if self._denoiser is not None:
                self._denoiser.reset()

    def _set_denoise(self, want: bool) -> None:
        if want and self._denoiser is None:
            if rnnoise_available():
                self._denoiser = AudioDenoiser()
                logger.info("denoise_enabled")
            elif not self._denoise_warned:
                self._denoise_warned = True
                logger.warning("denoise_unavailable reason=pyrnnoise_not_installed")
        elif not want and self._denoiser is not None:
            self._denoiser.close()
            self._denoiser = None
            logger.info("denoise_disabled")

    def reset(self) -> None:
        if self._denoiser is not None:
            self._denoiser.reset()

    def close(self) -> None:
        if self._denoiser is not None:
            self._denoiser.close()
            self._denoiser = None
