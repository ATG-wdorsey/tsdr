"""Log-mel front end of the voice detector.

Audio at 8 kHz; frames of 256 samples (32 ms, Hann) every 128 (16 ms); power
spectrum -> 32 mel bands (HTK scale, 100-3800 Hz, triangular, area-normalized)
-> log10. A window of WIN_FRAMES frames (2 s) is normalized by subtracting its
mean over time and bands, so the detector does not depend on the audio level
(NFM audio follows the deviation setting and the transmitter). The model was
trained on exactly these computations: voice.npz carries the constants and the
detector refuses a model made for others.
"""

import numpy as np

FS = 8000
FRAME, HOP = 256, 128
N_MELS = 32
F_LO, F_HI = 100.0, 3800.0
WIN_S = 2.0
WIN_FRAMES = int(round((WIN_S * FS - FRAME) / HOP)) + 1  # 124


def _mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_inv(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_matrix() -> np.ndarray:
    """(N_MELS, FRAME // 2 + 1) triangular filters, each summing to 1."""
    f = np.fft.rfftfreq(FRAME, 1 / FS)
    edges = _mel_inv(np.linspace(_mel(np.array(F_LO)), _mel(np.array(F_HI)), N_MELS + 2))
    m = np.zeros((N_MELS, len(f)))
    for i in range(N_MELS):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        m[i] = np.maximum(0.0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))
        m[i] /= m[i].sum() + 1e-12
    return m.astype(np.float32)


_MEL = mel_matrix()
_WIN = np.hanning(FRAME).astype(np.float32)


def logmel(a8: np.ndarray) -> np.ndarray:
    """(frames, N_MELS) log10 mel power of audio at FS, frames starting every HOP."""
    a8 = np.ascontiguousarray(a8, dtype=np.float32)
    if len(a8) < FRAME:
        return np.zeros((0, N_MELS), np.float32)
    k = 1 + (len(a8) - FRAME) // HOP
    fr = np.lib.stride_tricks.as_strided(a8, (k, FRAME), (a8.strides[0] * HOP, a8.strides[0]))
    p = np.abs(np.fft.rfft(fr * _WIN, axis=1)) ** 2
    out: np.ndarray = np.log10(p.astype(np.float32) @ _MEL.T + 1e-10)
    return out


def normalize(win: np.ndarray) -> np.ndarray:
    """A window of log-mel frames, level-normalized."""
    return win - win.mean()
