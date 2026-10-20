"""The second channel a P25 decoder follows calls on (M4 of plan_p25.md): the capture shifted to the granted channel, a
decoder of its own (its voice at its dial, never following), and what it heard each chunk as one immutable report.

The shift centers the channel with the receiver's error the control channel measures, so the inner decoder's search
starts from 0 whatever an earlier call's transmitter needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from tsdr.core.events.events import DecodedMessage
from tsdr.core.sdr.datatypes import AudioBatch
from tsdr.radio.dsp._kernels import apply_freq_shift_c64

if TYPE_CHECKING:
    from tsdr.radio.decoders.p25.decoder import P25Decoder, P25Voice


@dataclass(frozen=True)
class VoiceReport:
    """What the second channel heard in one chunk."""

    audio: tuple[AudioBatch, ...]
    texts: tuple[DecodedMessage, ...]
    frames: int  # voice frames (LDUs) decoded
    # Every transmission heard, in order, each as it stood last (the ended ones first).
    voices: tuple[P25Voice, ...]
    on_air: P25Voice | None  # the transmission under way at the chunk's end


class VoiceChannel:
    def __init__(self, decoder: P25Decoder) -> None:
        self.decoder = decoder
        self.frequency_hz: int | None = None
        self.shift_hz = 0.0
        self._phase = 0.0

    def start(self, frequency_hz: int, shift_hz: float, mute: bool) -> None:
        """A new call: the decoder reset, its search from 0, muted until its algorithm shows when `mute` (its grant's
        flag)."""
        d = self.decoder
        d._good_coarse = 0.0
        d.reset()
        d._expect_encrypted = mute
        self.frequency_hz, self.shift_hz, self._phase = frequency_hz, shift_hz, 0.0

    def stop(self) -> None:
        self.frequency_hz = None

    def set_mute_flag(self, on: bool) -> None:
        """The call's grant flags encryption (or no longer): muted until its algorithm shows (D5)."""
        d = self.decoder
        d._expect_encrypted = on
        if d._voice is not None and d._voice.granted_encrypted != on:
            d._voice = replace(d._voice, granted_encrypted=on)

    @property
    def muted_by_flag(self) -> bool:
        return self.decoder._expect_encrypted

    def set_sample_rate(self, rate: float) -> None:
        self.decoder.set_sample_rate(rate)

    def feed(self, iq: np.ndarray, now: float) -> VoiceReport:
        d = self.decoder
        shifted, self._phase = apply_freq_shift_c64(iq, self.shift_hz, d.sample_rate, self._phase)
        frames = d._voice_frames
        d.demodulate(shifted, now)  # it starts its chunk's list of ended transmissions afresh
        ended = tuple(v for _, v in d._chunk_ended)
        voices = ended + ((d._voice,) if d._voice is not None else ())
        return VoiceReport(
            audio=tuple(d.get_audio()),
            texts=tuple(m for m in d.get_messages() if m.text),
            frames=d._voice_frames - frames,
            voices=voices,
            on_air=d._voice,
        )
