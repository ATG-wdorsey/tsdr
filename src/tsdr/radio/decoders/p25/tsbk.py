"""P25 trunking messages, standard set (MFID 0): single-block (TSBK) and multi-block (MBT); the
site's channel plan.

A TSBK is 12 bytes: last-block flag, protect flag and opcode (6 bits); manufacturer ID; 8 bytes of
arguments; CRC. An MBT in the alternate format puts the opcode and a few fields in its PDU header
and the rest in its data blocks; explicit messages name a downlink and an uplink channel apiece. A
channel is a 16-bit number: a 4-bit identifier, whose IDEN_UP message gives the base frequency,
spacing and transmit offset, and a 12-bit channel number. On a TDMA identifier the number also picks
the slot.

Reference: TIA-102.AABC (trunking control channel messages). Field positions checked against OP25's
decoding of the same blocks from local control channels (ICORRS 800 MHz, PA-STARNet VHF).
"""

import functools
from dataclasses import dataclass

GRP_V_CH_GRANT = 0x00
GRP_V_CH_GRANT_UPDT = 0x02
GRP_V_CH_GRANT_UPDT_EXP = 0x03
UU_V_CH_GRANT = 0x04
SNDCP_DATA_CH_ANN = 0x16
GRP_AFF_RSP = 0x28
LOC_REG_RSP = 0x2B
U_REG_RSP = 0x2C
U_DE_REG_ACK = 0x2F
IDEN_UP_TDMA = 0x33
IDEN_UP_VU = 0x34
SCCB = 0x39
RFSS_STS_BCST = 0x3A
NET_STS_BCST = 0x3B
ADJ_STS_BCST = 0x3C
IDEN_UP = 0x3D

# Service options (grants): emergency, encrypted, duplex, packet mode; the low 3 bits are the
# priority.
SVC_EMERGENCY = 0x80
SVC_ENCRYPTED = 0x40


@dataclass(frozen=True)
class Iden:
    """One channel identifier: base frequency, spacing and transmit offset (Hz); slots per carrier
    (1 = FDMA)."""

    iden: int
    base_hz: int
    spacing_hz: int
    tx_offset_hz: int
    slots: int = 1
    bandwidth_hz: int | None = None


@dataclass(frozen=True)
class Message:
    """A decoded TSBK: its opcode, name and fields (channels as 16-bit numbers, resolved by
    `ChannelPlan`)."""

    opcode: int
    name: str
    fields: dict[str, int]


def _bits(args: int, start: int, width: int) -> int:
    """`width` bits of the 64-bit argument field, `start` counted from its most significant bit."""
    return (args >> (64 - start - width)) & ((1 << width) - 1)


def _signed_offset(sign: int, magnitude: int, unit: int) -> int:
    """A transmit offset: the sign bit set means a positive offset."""
    return magnitude * unit if sign else -magnitude * unit


def parse(data: bytes) -> Message | Iden | None:
    """The message in a 12-byte TSBK (CRC already checked), or None for one this module doesn't
    decode: a manufacturer's own (MFID not 0), or a protected one (P bit set: its arguments are
    encrypted)."""
    opcode = data[0] & 0x3F
    if data[1] != 0x00 or data[0] & 0x40:
        return None
    a = int.from_bytes(data[2:10], "big")
    if opcode == GRP_V_CH_GRANT:
        return Message(
            opcode,
            "grp_v_ch_grant",
            {
                "svcopts": _bits(a, 0, 8),
                "channel": _bits(a, 8, 16),
                "group": _bits(a, 24, 16),
                "source": _bits(a, 40, 24),
            },
        )
    if opcode == GRP_V_CH_GRANT_UPDT:
        return Message(
            opcode,
            "grp_v_ch_grant_updt",
            {
                "channel1": _bits(a, 0, 16),
                "group1": _bits(a, 16, 16),
                "channel2": _bits(a, 32, 16),
                "group2": _bits(a, 48, 16),
            },
        )
    if opcode == GRP_V_CH_GRANT_UPDT_EXP:
        # Explicit channels: the downlink first, as in the MBT messages (NET_STS_BCST's first is the
        # control channel).
        return Message(
            opcode,
            "grp_v_ch_grant_updt_exp",
            {
                "svcopts": _bits(a, 0, 8),
                "channel": _bits(a, 16, 16),
                "channel_up": _bits(a, 32, 16),
                "group": _bits(a, 48, 16),
            },
        )
    if opcode == UU_V_CH_GRANT:
        return Message(
            opcode,
            "uu_v_ch_grant",
            {"channel": _bits(a, 0, 16), "target": _bits(a, 16, 24), "source": _bits(a, 40, 24)},
        )
    if opcode == SNDCP_DATA_CH_ANN:
        return Message(
            opcode,
            "sndcp_data_ch_ann",
            {"channel1": _bits(a, 16, 16), "channel2": _bits(a, 32, 16)},
        )
    if opcode == GRP_AFF_RSP:
        return Message(
            opcode,
            "grp_aff_rsp",
            {
                "gav": _bits(a, 6, 2),
                "announcement_group": _bits(a, 8, 16),
                "group": _bits(a, 24, 16),
                "target": _bits(a, 40, 24),
            },
        )
    if opcode == LOC_REG_RSP:
        return Message(
            opcode,
            "loc_reg_rsp",
            {
                "rv": _bits(a, 6, 2),
                "group": _bits(a, 8, 16),
                "rfss": _bits(a, 24, 8),
                "site": _bits(a, 32, 8),
                "target": _bits(a, 40, 24),
            },
        )
    if opcode == U_REG_RSP:
        return Message(
            opcode,
            "u_reg_rsp",
            {
                "rv": _bits(a, 2, 2),
                "sysid": _bits(a, 4, 12),
                "source_id": _bits(a, 16, 24),
                "source": _bits(a, 40, 24),
            },
        )
    if opcode == U_DE_REG_ACK:
        return Message(
            opcode,
            "u_de_reg_ack",
            {"wacn": _bits(a, 8, 20), "sysid": _bits(a, 28, 12), "source_id": _bits(a, 40, 24)},
        )
    if opcode == SCCB:
        return Message(
            opcode,
            "sccb",
            {
                "rfss": _bits(a, 0, 8),
                "site": _bits(a, 8, 8),
                "channel1": _bits(a, 16, 16),
                "service1": _bits(a, 32, 8),
                "channel2": _bits(a, 40, 16),
                "service2": _bits(a, 56, 8),
            },
        )
    if opcode == RFSS_STS_BCST:
        return Message(
            opcode,
            "rfss_sts_bcst",
            {
                "lra": _bits(a, 0, 8),
                "active": _bits(a, 11, 1),
                "sysid": _bits(a, 12, 12),
                "rfss": _bits(a, 24, 8),
                "site": _bits(a, 32, 8),
                "channel": _bits(a, 40, 16),
                "service": _bits(a, 56, 8),
            },
        )
    if opcode == NET_STS_BCST:
        return Message(
            opcode,
            "net_sts_bcst",
            {
                "lra": _bits(a, 0, 8),
                "wacn": _bits(a, 8, 20),
                "sysid": _bits(a, 28, 12),
                "channel": _bits(a, 40, 16),
                "service": _bits(a, 56, 8),
            },
        )
    if opcode == ADJ_STS_BCST:
        return Message(
            opcode,
            "adj_sts_bcst",
            {
                "lra": _bits(a, 0, 8),
                "flags": _bits(a, 8, 4),
                "sysid": _bits(a, 12, 12),
                "rfss": _bits(a, 24, 8),
                "site": _bits(a, 32, 8),
                "channel": _bits(a, 40, 16),
                "service": _bits(a, 56, 8),
            },
        )
    if opcode == IDEN_UP:
        iden = _bits(a, 0, 4)
        spacing = _bits(a, 22, 10) * 125
        return Iden(
            iden,
            base_hz=_bits(a, 32, 32) * 5,
            spacing_hz=spacing,
            tx_offset_hz=_signed_offset(_bits(a, 13, 1), _bits(a, 14, 8), 250_000),
            bandwidth_hz=_bits(a, 4, 9) * 125,
        )
    if opcode == IDEN_UP_VU:
        spacing = _bits(a, 22, 10) * 125
        return Iden(
            _bits(a, 0, 4),
            base_hz=_bits(a, 32, 32) * 5,
            spacing_hz=spacing,
            tx_offset_hz=_signed_offset(_bits(a, 8, 1), _bits(a, 9, 13), spacing),
            bandwidth_hz={4: 6250, 5: 12500}.get(_bits(a, 4, 4)),
        )
    if opcode == IDEN_UP_TDMA:
        spacing = _bits(a, 22, 10) * 125
        channel_type = _bits(a, 4, 4)
        return Iden(
            _bits(a, 0, 4),
            base_hz=_bits(a, 32, 32) * 5,
            spacing_hz=spacing,
            tx_offset_hz=_signed_offset(_bits(a, 8, 1), _bits(a, 9, 13), spacing),
            slots=_TDMA_SLOTS.get(channel_type, 1),
        )
    return None


def _u16(data: bytes, i: int) -> int:
    return int.from_bytes(data[i : i + 2], "big")


def parse_mbt(header: bytes, data: bytes) -> Message | None:
    """The message in an alternate-format MBT (header and packet CRCs already checked), or None for
    one this module doesn't decode. `data` is the data blocks without the packet CRC."""
    if header[0] & 0x1F != 0x17 or header[2] != 0x00:
        return None
    opcode = header[7] & 0x3F
    address = int.from_bytes(header[3:6], "big")
    sysid = address & 0xFFF
    u16 = functools.partial(_u16, data)
    if opcode == GRP_V_CH_GRANT and len(data) >= 8:
        return Message(
            opcode,
            "mbt_grp_v_ch_grant",
            {
                "svcopts": header[8],
                "source": address,
                "channel": u16(2),
                "channel_up": u16(4),
                "group": u16(6),
            },
        )
    if opcode == RFSS_STS_BCST and len(data) >= 7:
        return Message(
            opcode,
            "mbt_rfss_sts_bcst",
            {
                "lra": header[3],
                "sysid": sysid,
                "rfss": data[0],
                "site": data[1],
                "channel": u16(2),
                "channel_up": u16(4),
                "service": data[6],
            },
        )
    if opcode == NET_STS_BCST and len(data) >= 8:
        return Message(
            opcode,
            "mbt_net_sts_bcst",
            {
                "lra": header[3],
                "sysid": sysid,
                "wacn": int.from_bytes(data[0:3], "big") >> 4,
                "channel": u16(3),
                "channel_up": u16(5),
                "service": data[7],
            },
        )
    if opcode == ADJ_STS_BCST and len(data) >= 5:
        return Message(
            opcode,
            "mbt_adj_sts_bcst",
            {
                "lra": header[3],
                "flags": (address >> 12) & 0xF,
                "sysid": sysid,
                "rfss": header[8],
                "site": header[9],
                "channel": u16(0),
                "channel_up": u16(2),
                "service": data[4],
            },
        )
    return None


# IDEN_UP_TDMA's channel type: access type and slots per carrier (TIA-102.BBAC).
_TDMA_SLOTS = {0: 1, 1: 1, 2: 1, 3: 2, 4: 4, 5: 2}


class ChannelPlan:
    """The identifiers a site has announced, and the frequency and slot of a channel number."""

    def __init__(self) -> None:
        self.idens: dict[int, Iden] = {}

    def update(self, iden: Iden) -> None:
        self.idens[iden.iden] = iden

    def resolve(self, channel: int) -> tuple[int, int | None] | None:
        """(downlink frequency in Hz, TDMA slot or None on an FDMA channel) of a 16-bit channel, or
        None while its identifier is unknown."""
        iden = self.idens.get(channel >> 12)
        if iden is None:
            return None
        number = channel & 0xFFF
        slot = number % iden.slots if iden.slots > 1 else None
        return iden.base_hz + iden.spacing_hz * (number // iden.slots), slot
