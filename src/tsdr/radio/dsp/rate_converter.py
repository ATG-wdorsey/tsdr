"""Complex IQ from a radio's capture rate to the exact rate a decoder is written for.

ADS-B is decoded at 2.4 MS/s and DAB at 2.048 MS/s, while radios capture at their
own rates (a Pluto at 10 MS/s, an Airspy at 2.5 or 10 MS/s). The conversion is a
`MultistageDecimator` by a whole factor, to the lowest rate at or above the target,
then a rational polyphase resampler to the target itself: the decimator's early
stages do the bulk of the reduction cheaply, and the resampler only bridges the
last step (2.5 to 2.4 MS/s is 24/25).
"""

from __future__ import annotations

import numpy as np

from tsdr.radio.dsp._kernels import StreamingPolyphaseResampler, make_rational_resampler
from tsdr.radio.dsp.decimator import MultistageDecimator, smooth_decimation

# Rates within this fraction of each other are the same rate (a radio's reported rate is
# often a few Hz off what was asked for: a Pluto asked for 10 MS/s reports 9999998).
_SAME_RATE = 1e-4

# Rejection of what would fold onto the passband. The decoders work on strong,
# narrow-in-time signals; 60 dB keeps the cascade short.
_ATTEN_DB = 60.0


class IQRateConverter:
    """Convert complex64 IQ from `source_rate` to `target_rate`, keeping `passband_hz`
    (either side of 0) free of aliases. Streaming: state carries across calls.

    A source at or within a hair of the target passes through untouched; a slower
    source cannot be converted (`too_slow`) and also passes through.
    """

    def __init__(self, source_rate: float, target_rate: float, passband_hz: float) -> None:
        self.source_rate = float(source_rate)
        self.target_rate = float(target_rate)
        self.too_slow = source_rate < target_rate * (1 - _SAME_RATE)
        self._decim: MultistageDecimator | None = None
        self._resampler: StreamingPolyphaseResampler | None = None
        if self.too_slow or _same(source_rate, target_rate):
            return
        decimation = smooth_decimation(source_rate / target_rate)
        if decimation > 1:
            self._decim = MultistageDecimator(
                source_rate, decimation, passband_hz, atten_db=_ATTEN_DB
            )
        mid_rate = source_rate / decimation
        if not _same(mid_rate, target_rate):
            self._resampler = make_rational_resampler(target_rate, mid_rate)

    @property
    def passthrough(self) -> bool:
        return self._decim is None and self._resampler is None

    def process(self, iq: np.ndarray) -> np.ndarray:
        if self._decim is not None:
            iq = self._decim.process(iq)
        if self._resampler is None:
            return iq
        return self._resampler.process_iq(iq)

    def reset(self) -> None:
        if self._decim is not None:
            self._decim.reset()
        if self._resampler is not None:
            self._resampler.reset()


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= b * _SAME_RATE
