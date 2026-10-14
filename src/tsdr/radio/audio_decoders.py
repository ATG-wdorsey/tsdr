"""Decoders that run on a mode's demodulated audio instead of replacing the mode.

A receiver feeding a decoder program: the user keeps the mode they tuned with
(NFM for SSTV on 2 m, USB on HF), its bandwidth, squelch and audio, and the
decoder reads that mode's audio before the squelch (`Demodulator.tap_audio`).
`DemodSpec.decoder` names one; `AUDIO_DECODERS` maps names to classes, and
`CARRIER_MODES` are the modes that offer their audio.
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from tsdr.core.clock_sync import now_utc_seconds
from tsdr.core.demod_spec import DemodSpec
from tsdr.core.events.events import DecodedMessage
from tsdr.radio.decoders.sstv import (
    MODES_BY_NAME,
    Mode,
    SSTVData,
    StreamerState,
    StreamingEvents,
    StreamingSSTV,
)
from tsdr.radio.dsp import AGC, DCBlocker, StreamingDecimFilter, firwin

# Modes whose audio a decoder can take: what a radio's speaker would carry.
CARRIER_MODES = ("NFM", "USB", "LSB", "AM")


class AudioDecoder:
    """Base for decoders fed with a mode's audio (any rate; they resample as needed)."""

    LABEL: ClassVar[str] = ""
    MESSAGE_TYPE: ClassVar[str] = "text"

    def process(self, audio: np.ndarray, rate: float, capture_utc_s: float) -> None:
        raise NotImplementedError

    def get_messages(self) -> list[DecodedMessage]:
        return []

    def description(self) -> str | None:
        """Shown under the mode on the tuner (e.g. the SSTV submode being received)."""
        return None

    def reset(self) -> None:
        pass

    def set_sstv_mode(self, name: str | None) -> None:  # noqa: B027
        pass


class SSTVAudioDecoder(AudioDecoder):
    """SSTV pictures from a mode's audio: VIS, sync and lines from the tone frequency.

    The audio is brought down to about 11 kHz by a whole factor (the tones are
    1100-2300 Hz), its DC removed and its level held by an AGC, then streamed into
    `StreamingSSTV`; images surface as `DecodedMessage(data=SSTVData)`.
    """

    LABEL = "SSTV"
    MESSAGE_TYPE = "sstv"
    DECODE_RATE = 11_025
    DC_BLOCKER_CUTOFF = 100.0

    # Min interval between per-line snapshots emitted to the UI. Fast modes
    # (Robot 8 BW: 30 lines/s) get coalesced so the message bus doesn't see
    # 30 KB×30 copies/s.
    _LINE_EMIT_MIN_INTERVAL_S = 0.1
    _LOOKING_EMIT_MIN_INTERVAL_S = 1.0

    def __init__(self, sstv_mode: str | None = None) -> None:
        self.sstv_mode_name: str | None = None
        self._rate: float | None = None
        self._pending: list[DecodedMessage] = []
        self._capture_utc_s = 0.0
        self._decoder: StreamingSSTV | None = None
        if sstv_mode is not None:
            self.set_sstv_mode(sstv_mode)

    def _build(self, rate: float) -> None:
        self._rate = rate
        q = max(1, int(rate // self.DECODE_RATE))
        fs = rate / q
        self._decim = (
            StreamingDecimFilter(firwin(8 * q + 1, 0.45 * fs, fs=rate), q) if q > 1 else None
        )
        self._dc_blocker = DCBlocker(fs, cutoff_hz=self.DC_BLOCKER_CUTOFF)
        self._agc = AGC(fs, attack_ms=5.0, decay_ms=200.0, setpoint=0.5)
        forced = MODES_BY_NAME[self.sstv_mode_name.lower()] if self.sstv_mode_name else None
        self._decoder = StreamingSSTV(
            int(round(fs)),
            events=StreamingEvents(
                on_mode=self._on_mode, on_line=self._on_line, on_image=self._on_image
            ),
            forced_mode=forced,
        )
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0

    def process(self, audio: np.ndarray, rate: float, capture_utc_s: float) -> None:
        if len(audio) == 0:
            return
        if self._decoder is None or rate != self._rate:
            self._build(rate)
        self._capture_utc_s = capture_utc_s
        x = np.asarray(audio, dtype=np.float32)
        if self._decim is not None:
            x = self._decim.process(x)
            if len(x) == 0:
                return
        x = self._dc_blocker.process(x)
        x = self._agc.process(x)
        np.clip(x, -1.0, 1.0, out=x)
        assert self._decoder is not None
        self._decoder.process(x)
        self._maybe_emit_looking()

    def get_messages(self) -> list[DecodedMessage]:
        out, self._pending = self._pending, []
        return out

    def description(self) -> str | None:
        mode = self._decoder.mode if self._decoder is not None else None
        return mode.name if mode is not None else None

    def reset(self) -> None:
        self._pending.clear()
        if self._rate is not None:
            self._build(self._rate)

    def set_sstv_mode(self, name: str | None) -> None:
        """Force a submode (skips the VIS code) or clear the override; drops any
        picture in progress (the new mode applies from the next VIS)."""
        if name is None:
            self.sstv_mode_name = None
        else:
            mode = MODES_BY_NAME.get(name.lower())
            if mode is None:
                raise ValueError(f"unknown SSTV mode: {name}")
            self.sstv_mode_name = mode.name
        self.reset()

    def _push(self, text: str, data: SSTVData) -> None:
        ts = self._capture_utc_s if self._capture_utc_s > 0 else now_utc_seconds()
        self._pending.append(DecodedMessage(text=text, timestamp=ts, data=data))

    def _make_data(
        self,
        state: StreamerState,
        mode: Mode | None,
        *,
        line_index: int,
        image: np.ndarray | None,
        error: str | None = None,
    ) -> SSTVData:
        assert self._decoder is not None
        return SSTVData(
            state=state,
            vis_code=self._decoder.vis_code,
            mode_name=mode.name if mode is not None else None,
            line_index=line_index,
            total_lines=mode.scan_lines if mode is not None else 0,
            image_width=mode.width if mode is not None else 0,
            image_height=mode.height if mode is not None else 0,
            image=image,
            forced_mode=self.sstv_mode_name is not None,
            error=error,
        )

    def _maybe_emit_looking(self) -> None:
        """A bare LOOKING-state ping while scanning, rate-limited on the audio
        sample clock (right for live audio and faster-than-real-time replay)."""
        assert self._decoder is not None
        if self._decoder.state != StreamerState.LOOKING:
            return
        if self._decoder.samples_processed < self._next_looking_emit_sample:
            return
        self._next_looking_emit_sample = self._decoder.samples_processed + int(
            self._LOOKING_EMIT_MIN_INTERVAL_S * self._decoder.fs
        )
        data = self._make_data(StreamerState.LOOKING, None, line_index=-1, image=None)
        self._push("SSTV listening", data)

    def _on_mode(self, mode: Mode, _vis_code: int) -> None:
        self._last_line_emit_mono = 0.0
        data = self._make_data(StreamerState.DECODING, mode, line_index=-1, image=None)
        self._push(f"SSTV {mode.name} (start)", data)

    def _on_line(self, mode: Mode, idx: int, _row: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_line_emit_mono < self._LINE_EMIT_MIN_INTERVAL_S:
            return
        self._last_line_emit_mono = now
        assert self._decoder is not None
        dstate = self._decoder.dstate
        snapshot = dstate.image.copy() if dstate is not None else None
        data = self._make_data(StreamerState.DECODING, mode, line_index=idx, image=snapshot)
        self._push(f"SSTV {mode.name} L{idx + 1}/{mode.scan_lines}", data)

    def _on_image(self, mode: Mode, img: np.ndarray) -> None:
        data = self._make_data(
            StreamerState.DONE, mode, line_index=mode.scan_lines - 1, image=img.copy()
        )
        self._push(f"SSTV {mode.name} (done)", data)
        assert self._decoder is not None
        self._decoder.reset()
        self._last_line_emit_mono = 0.0
        self._next_looking_emit_sample = 0


AUDIO_DECODERS: dict[str, type[AudioDecoder]] = {"SSTV": SSTVAudioDecoder}


def make_audio_decoder(name: str, sstv_mode: str | None = None) -> AudioDecoder:
    cls = AUDIO_DECODERS[name.upper()]
    if cls is SSTVAudioDecoder:
        return SSTVAudioDecoder(sstv_mode=sstv_mode)
    return cls()


def with_decoder(current: DemodSpec | None, decoder: str, freq_hz: float) -> DemodSpec:
    """`current` with `decoder` riding on it: the user's mode and settings stay. With no
    mode that has audio for it (none, or a decoder that is its own mode), the band's
    usual one: USB below 30 MHz, NFM above."""
    name = decoder.upper()
    if current is not None and current.mode.upper() in CARRIER_MODES:
        updated: DemodSpec = current.model_copy(update={"decoder": name})
        return updated
    return DemodSpec(mode="USB" if freq_hz < 30e6 else "NFM", decoder=name)


def carried_decoder(current: DemodSpec | None, mode: str) -> tuple[str | None, str | None]:
    """(decoder, sstv_mode) a change to `mode` keeps: a decoder rides on to another
    mode that has audio for it, and is dropped otherwise."""
    if current is None or current.decoder is None or mode.upper() not in CARRIER_MODES:
        return None, None
    return current.decoder, current.sstv_mode
