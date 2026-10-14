"""The voice detector's network, numpy only: a small CNN on a normalized log-mel
window (32 bands x 124 frames) -> logit of P(speech).

Three 3x3 convolution + ReLU + 2x2 max-pool stages and a fourth convolution
(16-32-48-64 channels; batch norm folded into each), the bands averaged out,
the maximum and the mean over time into a 32-unit dense layer and the output.
About 50 k weights; 10.2 M multiply-accumulates per window. A convolution is
an im2col (stride tricks over the padded input) and one matrix product.
"""

from pathlib import Path

import numpy as np


class VoiceNet:
    def __init__(self, path: Path) -> None:
        with np.load(path) as m:
            n_conv = int(m["n_conv"])
            self._w = [m[f"w{i}"].astype(np.float32) for i in range(n_conv)]  # (C_out, C_in * 9)
            self._b = [m[f"b{i}"].astype(np.float32) for i in range(n_conv)]
            self._d1w = m["d1w"].astype(np.float32)
            self._d1b = m["d1b"].astype(np.float32)
            self._d2w = m["d2w"].astype(np.float32)
            self._d2b = float(m["d2b"])

    @staticmethod
    def _conv(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        c, h, t = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp, (c, 3, 3, h, t), (s[0], s[1], s[2], s[1], s[2])
        ).reshape(c * 9, h * t)
        y: np.ndarray = w @ cols + b[:, None]
        np.maximum(y, 0.0, out=y)
        return y.reshape(-1, h, t)

    @staticmethod
    def _pool(x: np.ndarray) -> np.ndarray:
        h2, t2 = x.shape[1] // 2 * 2, x.shape[2] // 2 * 2
        a: np.ndarray = np.maximum(x[:, 0:h2:2, 0:t2:2], x[:, 1:h2:2, 0:t2:2])
        np.maximum(a, np.maximum(x[:, 0:h2:2, 1:t2:2], x[:, 1:h2:2, 1:t2:2]), out=a)
        return a

    def logit(self, win: np.ndarray) -> float:
        """win: (bands, frames) normalized log-mel."""
        x = win[None].astype(np.float32)
        last = len(self._w) - 1
        for i, (w, b) in enumerate(zip(self._w, self._b, strict=True)):
            x = self._conv(x, w, b)
            if i < last:
                x = self._pool(x)
        h = x.mean(axis=1)
        h = np.concatenate([h.max(axis=1), h.mean(axis=1)])
        h = np.maximum(self._d1w @ h + self._d1b, 0.0)
        return float(self._d2w @ h + self._d2b)

    def prob(self, win: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-self.logit(win))))
