"""VSQ, the voice squelch for SEARCH (`SearchConfig.voice_squelch`, `v`).

DenoiserStage feeds it the dial channel's demodulated audio before RNNoise and
mutes its output on a "not voice" verdict. VSQ resamples the audio to 8 kHz and
the voice detector (radio/voice, about 1.3 ms per window) scores the last 2 s
of the transmission (the squelch open on one dial channel of one SEARCH run)
every 0.5 s from 1 s. A window the detector calls speech makes it "voice".
Without one by DECIDE_S it is "not voice", unless the last second was a quiet
carrier (below QUIET_DB: someone keyed up and has not spoken yet), which gets
until LATE_S. SearchController then treats the squelch as closed: the hold
counts down with the audio muted (time to lock the channel out), and SEARCH
moves on. Speech reopens it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tsdr.core.sdr.buffers import CircularBuffer
from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler, make_rational_resampler
from tsdr.radio.voice.detector import Classification, VoiceStream, detector
from tsdr.radio.voice.frontend import FS

logger = logging.getLogger(__name__)

# On the owner's 63 labeled transmissions (and 47 more picked where the detector
# and a large classifier disagreed), voice transmissions show speech within 3 s
# of key-up but for those that start with a quiet carrier; those spoke by 5 s.
# A not-voice transmission that quiet is a kerchunk or a tail, over by then.
DECIDE_S = 3.0
LATE_S = 5.0
QUIET_DB = -42.0  # dBFS of the demodulated audio over the last second

Verdict = Literal["listening", "voice", "not_voice"]


@dataclass(frozen=True)
class VsqStatus:
    verdict: Verdict
    carrier_s: float  # of the transmission on the dial
    last: Classification | None  # the latest window's


class VoiceSquelch:
    """VSQ for one device, in its pipeline thread: DenoiserStage sets the
    target and feeds audio, SearchStage reads the verdict."""

    def __init__(self, device_id: str) -> None:
        self._device_id = device_id
        self._key: tuple[int, float] | None = None  # (SEARCH run, dial Hz) while on
        self._resampler: StreamingPolyphaseResampler | None = None
        self._rate = 0.0
        self._stream: VoiceStream | None = None  # made on the first feed: the model loads earlier
        self._recent = CircularBuffer(FS, dtype=np.float32)  # the last second, for QUIET_DB
        self._verdict: Verdict = "listening"
        self._carrier = 0  # samples at FS
        self._deadline: float | None = None  # when no speech means not voice (set at DECIDE_S)
        self._last: Classification | None = None

    @property
    def status(self) -> VsqStatus | None:
        if self._key is None:
            return None
        return VsqStatus(self._verdict, self._carrier / FS, self._last)

    @property
    def muted(self) -> bool:
        return self._key is not None and self._verdict == "not_voice"

    def verdict_for(self, run: int, tuned_hz: float) -> Verdict | None:
        """The verdict on the dial's transmission; None unless VSQ is on for
        exactly this run and channel."""
        return self._verdict if self._key == (run, tuned_hz) else None

    def set_target(self, key: tuple[int, float] | None) -> None:
        """(SEARCH run, dial Hz) while VSQ is on, else None."""
        if key == self._key:
            return
        if key is None:
            logger.info("vsq_off device=%s", self._device_id)
        elif self._key is None:
            logger.info("vsq_on device=%s", self._device_id)
        self._key = key
        self._last = None
        self._start_transmission()

    def feed(self, samples: np.ndarray, sample_rate: float, squelch_open: bool | None) -> None:
        """Mono demodulated audio of the dial's channel, before the denoiser."""
        if self._key is None:
            return
        if squelch_open is False:
            if self._carrier:
                self._start_transmission()
            return
        if self._resampler is None or sample_rate != self._rate:
            self._resampler = make_rational_resampler(FS, sample_rate)
            self._rate = sample_rate
        if self._stream is None:
            self._stream = VoiceStream(detector())
        x = self._resampler.process(np.asarray(samples, dtype=np.float32).reshape(-1, 1))[:, 0]
        self._recent.append(x)
        self._carrier += len(x)
        for t, result in self._stream.feed(x):
            self._last = result
            verdict = self._verdict
            if result.is_speech:
                verdict = "voice"
            elif verdict == "listening":
                if self._deadline is None and t >= DECIDE_S:
                    self._deadline = LATE_S if self._quiet() else DECIDE_S
                if self._deadline is not None and t >= self._deadline:
                    verdict = "not_voice"
            if verdict != self._verdict:
                self._verdict = verdict
                logger.info(
                    "vsq_verdict device=%s channel_hz=%.0f verdict=%s after_s=%.1f p=%.2f",
                    self._device_id,
                    self._key[1],
                    verdict,
                    t,
                    result.probability,
                )

    def _quiet(self) -> bool:
        a = self._recent.get_window(min(self._recent.size, FS))
        return len(a) > 0 and float(10 * np.log10(np.mean(a * a) + 1e-12)) < QUIET_DB

    def _start_transmission(self) -> None:
        self._verdict = "listening"
        self._carrier = 0
        self._deadline = None
        self._recent.clear()
        if self._stream is not None:
            self._stream.reset()
        if self._resampler is not None:
            self._resampler.reset()
