"""Voice detector for demodulated audio at 8 kHz: P(speech) of the last 2 s,
every 0.5 s of a transmission.

The model (voice.npz, frontend.py + net.py) was trained in the author's vdetect
project on real over-the-air audio demodulated by tsdr (2 m, 70 cm, VHF/UHF
land mobile, NOAA, airband AM), labeled by a large audio classifier (PANNs
Cnn14) and checked against the owner's listening; weak-signal copies of real
transmissions (receiver noise added to their IQ) and synthetic data, tones,
digital voice, SSTV, key-up bursts, quiet open microphones and AM noise as not
voice. On the owner's labels it separates voice from not voice by 3 s after
key-up (AUC 0.994; only speech that starts later is missed; 0.974 and 0.996 on
two sets of hard cases the owner labeled) at about 3 ms of CPU per second of
audio.

VoiceStream frames a transmission's audio into log-mel frames as it arrives
(aligned to the transmission's first sample, as in training) and scores the
last WIN_S of frames every HOP_S from FIRST_S on; before WIN_S of audio, the
frames so far are repeated to fill the window, as in training.
"""

import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from tsdr.radio.voice import frontend
from tsdr.radio.voice.frontend import FRAME, HOP, N_MELS, WIN_FRAMES, logmel, normalize
from tsdr.radio.voice.net import VoiceNet

MODEL = Path(__file__).parent / "voice.npz"
THRESHOLD = 0.5  # P(speech) at or above which a window is speech
FIRST_S = 1.0
HOP_S = 0.5

_CONSTANTS = ("fs", "frame", "hop", "n_mels", "f_lo", "f_hi", "win_frames", "norm_floor")


@dataclass(frozen=True)
class Classification:
    probability: float  # P(speech) of the window

    @property
    def is_speech(self) -> bool:
        return self.probability >= THRESHOLD

    @property
    def label(self) -> str:
        return "speech" if self.is_speech else "not speech"


def load_model(path: Path = MODEL) -> VoiceNet:
    """The network of `path`; ValueError if it was trained on another front end, or the
    file is damaged or lacks an array (a truncated copy, another export's layout)."""
    try:
        with np.load(path) as m:
            for name in _CONSTANTS:
                if name not in m.files or float(m[name]) != float(getattr(frontend, name.upper())):
                    raise ValueError(f"{path} was trained on another front end ({name})")
        return VoiceNet(path)
    except (zipfile.BadZipFile, EOFError, KeyError) as e:
        raise ValueError(f"{path} is damaged or incomplete: {e}") from e


@cache
def detector() -> VoiceNet:
    return load_model()


class VoiceStream:
    """Scores one transmission's audio at frontend.FS."""

    def __init__(self, net: VoiceNet | None = None) -> None:
        self._net = net or detector()
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, np.float32)  # samples not yet in a complete frame
        self._frames = np.zeros((0, N_MELS), np.float32)  # the last WIN_FRAMES frames
        self._n_frames = 0  # frames since the transmission started
        self._next_s = FIRST_S

    def feed(self, a8: np.ndarray) -> list[tuple[float, Classification]]:
        """Audio at frontend.FS; returns (seconds since the start, classification)
        for each window that became due."""
        self._pending = np.concatenate([self._pending, np.asarray(a8, np.float32)])
        out: list[tuple[float, Classification]] = []
        while True:
            due = int(round(self._next_s * frontend.FS / HOP))  # frames the next window needs
            need = due - self._n_frames
            if need > 0:
                n_avail = (
                    (len(self._pending) - FRAME) // HOP + 1 if len(self._pending) >= FRAME else 0
                )
                k = min(need, n_avail)
                if k > 0:
                    new = logmel(self._pending[: (k - 1) * HOP + FRAME])
                    self._pending = self._pending[k * HOP :]
                    self._frames = np.concatenate([self._frames, new])[-WIN_FRAMES:]
                    self._n_frames += k
                if self._n_frames < due:
                    return out
            win = self._frames
            if len(win) < WIN_FRAMES:
                win = np.resize(win, (WIN_FRAMES, N_MELS))  # cyclic fill, as in training
            out.append((self._next_s, Classification(self._net.prob(normalize(win).T))))
            self._next_s += HOP_S
