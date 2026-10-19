"""A synthesized P25 Phase 1 trunked site in one capture, for the tests and the M4 acceptance grid: a control channel
sending its system broadcasts, channel identifier and grants on a schedule, and voice channels carrying the granted
calls (HDU, LDUs, terminators), with a receiver error common to all, other signals and noise.

The capture is the decoder's input: the dial (the control channel's frequency) at 0 Hz, `rate` samples a second,
the hardware's center `center_offset_hz` from the dial. A channel at frequency f lies at (f - dial) plus the receiver
error, which scales with frequency (`error_hz` at the control channel).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal

from tsdr.radio.decoders.p25 import fec, tsbk, voice

from . import _synth as synth

NAC, WACN, SYSID, RFSS, SITE = 0x1A5, 0xBEE00, 0x2A1, 1, 3
IDEN = 1
BASE_HZ = 851_000_000
SPACING_HZ = 12_500
CC_NUMBER = 200  # 853.5 MHz
SOURCE = 0x00F00D  # a made-up unit
# From a grant to its call's first voice frame (real sites: 0.1-0.3 s).
VOICE_DELAY_S = 0.15
UPDATE_EVERY_S = 0.5  # real sites: about 0.4 s (the reference capture)
HANGTIME_TDULCS = 3


def channel(number: int) -> int:
    """The 16-bit channel of channel number `number` on the site's identifier."""
    return IDEN << 12 | number


def frequency(number: int) -> int:
    return BASE_HZ + SPACING_HZ * number


CC_HZ = frequency(CC_NUMBER)


@dataclass(frozen=True)
class Call:
    """A call: granted at `start` on channel number `number`, then `ldus` LDUs of voice."""

    group: int
    number: int
    start: float
    ldus: int = 6
    algid: int = voice.ALGID_CLEAR
    kid: int = 0
    # The grant's encrypted flag: None follows the algorithm (flagged when encrypted); False leaves it off (G5 rule 2).
    flagged: bool | None = None
    emergency: bool = False
    grant: bool = True  # False: the grant is missed; the call is seen only in updates
    mbt: bool = False  # the grant as an MBT rather than a TSBK
    header: bool = True  # False: the call's voice begins with an LDU1 (no HDU)
    voice: bool = (
        True  # False: granted, but nothing on the voice channel (e.g. outside the capture)
    )
    cn_db: float | None = None  # its channel's C/N; None: the site's

    @property
    def encrypted(self) -> bool:
        return self.algid != voice.ALGID_CLEAR

    @property
    def svcopts(self) -> int:
        flagged = self.encrypted if self.flagged is None else self.flagged
        return (0x40 if flagged else 0) | (0x80 if self.emergency else 0) | 0x04

    @property
    def voice_start(self) -> float:
        return self.start + VOICE_DELAY_S

    @property
    def end(self) -> float:
        """When its voice ends (the terminators included)."""
        return self.voice_start + _duration(self)


@dataclass(frozen=True)
class Neighbor:
    """Another signal (C4FM of random dibits, another NAC): at `offset_hz` from the dial, `cn_db` C/N."""

    offset_hz: float
    cn_db: float
    modulation: str = "c4fm"


@dataclass
class Site:
    calls: list[Call]
    duration: float
    rate: float = 480_000.0
    error_hz: float = 0.0
    cn_db: float = 30.0
    modulation: str = "c4fm"
    center_offset_hz: float = 0.0
    neighbors: list[Neighbor] = field(default_factory=list)
    seed: int = 1

    def capture(self, speech: list[np.ndarray]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        up = round(self.rate / synth.RATE)
        assert up * synth.RATE == self.rate, "the capture rate must be a multiple of 240 kS/s"
        n = int(self.duration * self.rate)
        out = np.zeros(n, dtype=np.complex64)
        streams = [(CC_HZ, [(0.0, self._control_dibits())], self.cn_db)]
        by_number: dict[int, list[Call]] = {}
        for call in self.calls:
            if call.voice:
                by_number.setdefault(call.number, []).append(call)
        for number, calls in sorted(by_number.items()):
            calls.sort(key=lambda c: c.start)
            segments = [(c.voice_start, _voice_dibits(c, speech)) for c in calls]
            streams.append((frequency(number), segments, calls[0].cn_db or self.cn_db))
        for f, segments, cn in streams:
            offset = (f - CC_HZ) + self.error_hz * f / CC_HZ
            self._add(out, segments, offset, cn, up)
        for nb in self.neighbors:
            dibits = [
                int(d) for d in rng.integers(0, 4, int(self.duration * synth.SYMBOL_RATE) + 1)
            ]
            self._add(out, [(0.0, dibits)], nb.offset_hz, nb.cn_db, up, nb.modulation)
        # Noise: 1 in 12.5 kHz, so a channel's power is its C/N.
        sigma = np.sqrt(self.rate / 12_500 / 2)
        out += (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex64)
        return out / np.float32(np.sqrt(self.rate / 12_500))

    def _add(self, out, segments, offset_hz, cn_db, up, modulation=None) -> None:
        modulate = _c4fm if (modulation or self.modulation) == "c4fm" else _cqpsk
        amplitude = np.float32(10 ** (cn_db / 20))
        for start, dibits in segments:
            x = modulate(dibits)
            if up > 1:
                x = scipy.signal.resample_poly(x, up, 1).astype(np.complex64)
            i0 = int(round(start * self.rate))
            x = x[: max(0, len(out) - i0)]
            t = (i0 + np.arange(len(x))) / self.rate
            out[i0 : i0 + len(x)] += (
                amplitude * x * np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)
            )

    def _control_dibits(self) -> list[int]:
        """The control channel's frames to the end: grants when due (twice), updates every UPDATE_EVERY_S while a
        call lasts, and the system broadcasts in turn between them; three blocks a frame, MBTs in frames of their
        own."""
        p, t = synth.pack, synth.tsbk
        broadcasts = [
            t(
                tsbk.NET_STS_BCST,
                p((0, 8), (WACN, 20), (SYSID, 12), (channel(CC_NUMBER), 16), (0x70, 8)),
                last=False,
            ),
            t(
                tsbk.RFSS_STS_BCST,
                p(
                    (0, 8),
                    (0, 3),
                    (1, 1),
                    (SYSID, 12),
                    (RFSS, 8),
                    (SITE, 8),
                    (channel(CC_NUMBER), 16),
                    (0x70, 8),
                ),
                last=False,
            ),
            t(
                tsbk.IDEN_UP,
                p(
                    (IDEN, 4),
                    (100, 9),
                    (0, 1),
                    (180, 8),
                    (SPACING_HZ // 125, 10),
                    (BASE_HZ // 5, 32),
                ),
                last=False,
            ),
        ]
        grants_left = {i: 2 for i, c in enumerate(self.calls) if c.grant}
        last_update = dict.fromkeys(range(len(self.calls)), -1e9)
        dibits: list[int] = []
        turn = 0
        while len(dibits) / synth.SYMBOL_RATE < self.duration:
            now = len(dibits) / synth.SYMBOL_RATE
            blocks: list[bytes] = []
            for i, c in enumerate(self.calls):
                if grants_left.get(i) and c.start <= now and len(blocks) < 3:
                    grants_left[i] -= 1
                    last_update[i] = now
                    if c.mbt:
                        dibits += synth.frame(NAC, fec.DUID_PDU, _mbt_grant(c))
                        continue
                    blocks.append(
                        t(
                            tsbk.GRP_V_CH_GRANT,
                            p((c.svcopts, 8), (channel(c.number), 16), (c.group, 16), (SOURCE, 24)),
                            last=False,
                        )
                    )
            due = [
                i
                for i, c in enumerate(self.calls)
                if c.start <= now <= c.end and now - last_update[i] >= UPDATE_EVERY_S
            ]
            while due and len(blocks) < 3:
                a = due.pop(0)
                b = due.pop(0) if due else a
                last_update[a] = last_update[b] = now
                ca, cb = self.calls[a], self.calls[b]
                blocks.append(
                    t(
                        tsbk.GRP_V_CH_GRANT_UPDT,
                        p(
                            (channel(ca.number), 16),
                            (ca.group, 16),
                            (channel(cb.number), 16),
                            (cb.group, 16),
                        ),
                        last=False,
                    )
                )
            while len(blocks) < 3:
                blocks.append(broadcasts[turn % len(broadcasts)])
                turn += 1
            blocks[-1] = bytes([blocks[-1][0] | 0x80]) + blocks[-1][1:10]
            blocks[-1] += fec.crc16(blocks[-1]).to_bytes(2, "big")
            dibits += synth.frame(NAC, fec.DUID_TSDU, blocks)
        return dibits


def _mbt_grant(c: Call) -> list[bytes]:
    p = synth.pack
    return synth.mbt(
        tsbk.GRP_V_CH_GRANT,
        p((SOURCE, 24)),
        bytes([c.svcopts, 0]),
        p((0, 16), (channel(c.number), 16), (channel(c.number), 16), (c.group, 16)),
    )


def _lc(c: Call) -> bytes:
    flags = (0x40 if c.encrypted else 0) | (0x80 if c.emergency else 0)
    return (
        bytes([voice.LCO_GROUP_VOICE, 0, flags, 0])
        + c.group.to_bytes(2, "big")
        + SOURCE.to_bytes(3, "big")
    )


def _es(c: Call) -> bytes:
    return bytes(9) + bytes([c.algid]) + c.kid.to_bytes(2, "big")


def _voice_dibits(c: Call, speech: list[np.ndarray]) -> list[int]:
    out: list[int] = []
    if c.header:
        out += synth.raw_frame(NAC, fec.DUID_HDU, synth.hdu(bytes(9), 0, c.algid, c.kid, c.group))
    for k in range(c.ldus):
        second = k % 2 == 1
        frames = [speech[(9 * k + j) % len(speech)] for j in range(9)]
        out += synth.raw_frame(
            NAC,
            fec.DUID_LDU2 if second else fec.DUID_LDU1,
            synth.ldu(frames, _es(c) if second else _lc(c), second),
        )
    lc = (
        bytes([voice.LCO_GROUP_VOICE, 0, 0, 0])
        + c.group.to_bytes(2, "big")
        + SOURCE.to_bytes(3, "big")
    )
    for _ in range(HANGTIME_TDULCS):
        out += synth.raw_frame(NAC, fec.DUID_TDULC, synth.tdulc(lc))
    return out


def _frame_s(duid: int, data: int) -> float:
    return len(synth.raw_frame(NAC, duid, [0] * data)) / synth.SYMBOL_RATE


_HDU_S = _frame_s(fec.DUID_HDU, len(synth.hdu(bytes(9), 0, voice.ALGID_CLEAR, 0, 1)))
_LDU_S = _frame_s(fec.DUID_LDU1, 784)
_TDULC_S = _frame_s(fec.DUID_TDULC, len(synth.tdulc(bytes(9))))


def _duration(c: Call) -> float:
    return (_HDU_S if c.header else 0.0) + c.ldus * _LDU_S + HANGTIME_TDULCS * _TDULC_S


def _c4fm(dibits: list[int]) -> np.ndarray:
    """synth.c4fm with an FFT convolution (the same pulse), fast enough for minutes of signal."""
    impulses = np.zeros(len(dibits) * synth.SPS)
    impulses[:: synth.SPS] = [synth._LEVEL[d] * np.pi / 4 for d in dibits]
    phase = np.cumsum(scipy.signal.oaconvolve(impulses, _PULSE, mode="same"))
    return np.exp(1j * phase).astype(np.complex64)


def _cqpsk(dibits: list[int]) -> np.ndarray:
    steps = np.array([synth._LEVEL[d] * np.pi / 4 for d in dibits])
    impulses = np.zeros(len(dibits) * synth.SPS, dtype=np.complex128)
    impulses[:: synth.SPS] = np.exp(1j * np.cumsum(steps))
    return scipy.signal.oaconvolve(impulses, _RRC, mode="same").astype(np.complex64)


_PULSE = synth._c4fm_pulse()
_RRC = synth.rrc_taps(synth.SPS, 0.2, 8) * np.sqrt(synth.SPS)
