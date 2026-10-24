"""The P25 panel's state: the cursor holds a row's key, and a key acts on the target taken when it was pressed. Logic
only, no rendering; a sequence checker outside the suite drives it through longer sequences."""

from tsdr.radio.decoders.p25 import systems
from tsdr.radio.decoders.p25.calls import P25Call
from tsdr.radio.decoders.p25.decoder import P25Channel, P25Snapshot
from tsdr.tui.widgets.p25_panel import P25PanelState

A, B = (0xBEE00, 0x2A1), (0xA4321, 0x152)


def _snapshot(system, groups, now=100.0) -> P25Snapshot:
    channels = tuple(
        P25Channel(
            P25Call(
                g,
                0x1000 + g % 100,
                None,
                851_000_000 + 12_500 * (g % 100),
                None,
                0x04,
                now,
                now - i,
                cid=g,
            ),
            False,
            True,
        )
        for i, g in enumerate(groups)
    )
    return P25Snapshot(
        locked=True,
        nac=0x1A5,
        wacn=system[0],
        sysid=system[1],
        rfss=1,
        site=1,
        tsbk_pct=100.0,
        tsbks=0,
        mbts=0,
        frequency_offset_hz=0.0,
        calls=(),
        adjacent=(),
        time=now,
        channels=channels,
    )


def test_the_cursor_holds_its_rows_key() -> None:
    # Reviews 38 F2, 39 F1: rows come and go and reorder under the cursor.
    systems.clear()
    p = P25PanelState()
    p.update(_snapshot(A, [1001, 2002, 3003]))
    p.key("down")
    p.update(_snapshot(A, [4004, 3003, 2002, 1001]))  # reordered, a row added above it
    assert p.key("space") == "P25: TG 2002 not followed"
    assert not systems.system_for(*A).follows(2002) and systems.system_for(*A).follows(1001)
    # its row gone: the next action key only takes the row at its place
    p.update(_snapshot(A, [4004, 3003, 1001]))
    assert p.key("space").startswith("P25: that row has gone")
    assert systems.system_for(*A).follows(1001)
    assert p.key("space") == "P25: TG 1001 not followed"


def test_a_confirmation_acts_on_the_system_captured_at_its_key() -> None:
    # D3: `n`, then a retune to another system, then `y`: the first system's talkgroups go, not the second's.
    systems.clear()
    p = P25PanelState()
    p.update(_snapshot(A, [1001]))
    p.key("n", now=0.0)
    p.update(_snapshot(B, [1001]))
    p.key("y", now=1.0)
    assert not systems.system_for(*A).follows(1001) and systems.system_for(*B).follows(1001)
    p.key("a", now=2.0)
    # any other key cancels and does nothing else (R44-8)
    assert p.key("down", now=2.5) == "P25: cancelled"
    assert p.pending is None and systems.system_for(*B).follow_new


def test_tab_re_anchors_the_cursor_on_the_row_shown() -> None:
    # D4 (Q146): the cursor's row gone while the panel hadn't the keys; Tab makes the row shown its key.
    systems.clear()
    p = P25PanelState()
    p.update(_snapshot(A, [1001, 2002]))
    p.key("down")
    p.update(_snapshot(A, [1001, 3003]))
    p.anchor()
    assert p.key("space") == "P25: TG 3003 not followed"
