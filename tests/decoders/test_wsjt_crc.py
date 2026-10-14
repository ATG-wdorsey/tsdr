"""CRC-14 regression vectors for the WSJT-X CRC-14 (polynomial 0x2757)."""

from tsdr.radio.decoders.wsjt.crc import add_crc, compute_crc, verify_crc


def _hx(s: str) -> bytes:
    return bytes.fromhex(s)


def test_compute_crc_raw_known_vectors() -> None:
    msg = _hx("deadbeef0123456789abcdef")
    assert compute_crc(msg, 82) == 0x15FF
    assert compute_crc(msg, 91) == 0x3917
    assert compute_crc(msg, 77) == 0x2F83


def test_verify_crc_roundtrip() -> None:
    for payload_hex in (
        "00000000000000000000",
        "aa55aa55aa55aa55aa50",
        "fffffffffffffffffff8",
        "00010203040506070808",
        "123456789abcdef01358",
    ):
        a91 = add_crc(_hx(payload_hex))
        ok, _ = verify_crc(a91)
        assert ok, f"verify failed for {payload_hex}"


def test_verify_crc_rejects_corrupted() -> None:
    a91 = bytearray(add_crc(_hx("aa55aa55aa55aa55aa50")))
    a91[0] ^= 0x80  # flip MSB of payload
    ok, _ = verify_crc(bytes(a91))
    assert not ok
