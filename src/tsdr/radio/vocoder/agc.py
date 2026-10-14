"""Speech level for a vocoder's output, set as the reference decoders set it (DSD, and
mbelib's decode_ambe harness): the gain aims the loudest peak of the last 25 frames
(half a second of AMBE+2) at full scale; it falls at once for a louder frame and rises
at most 5 % a frame, up to a ceiling for quiet talkers. A fixed gain clips loud talkers
or leaves quiet ones faint (DMR's +18 dB clipped 7 % of a 443.700 MHz repeater's samples).
"""

from __future__ import annotations

from collections import deque

import numpy as np

_RISE_PER_FRAME = 0.05


class VoiceAGC:
    """Frames in, frames out, with peaks at most `peak` (full scale = 1.0)."""

    def __init__(
        self, max_gain: float, start_gain: float, history: int = 25, peak: float = 32767 / 32768
    ) -> None:
        self.max_gain = max_gain
        self.start_gain = start_gain
        self.peak = peak
        self.gain = start_gain
        self._peaks: deque[float] = deque(maxlen=history)

    def process(self, frame: np.ndarray) -> np.ndarray:
        self._peaks.append(float(np.max(np.abs(frame))) if len(frame) else 0.0)
        loudest = max(self._peaks)
        target = self.peak / loudest if loudest > 0.0 else self.max_gain
        if target < self.gain:
            self.gain = target
        else:
            self.gain += min(min(target, self.max_gain) - self.gain, _RISE_PER_FRAME * self.gain)
        return (frame * self.gain).astype(np.float32)

    def reset(self) -> None:
        self.gain = self.start_gain
        self._peaks.clear()
