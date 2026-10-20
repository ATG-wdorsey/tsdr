from tsdr.tui.model import (
    DEFAULT_LAYOUT,
    UIModel,
)


def test_initial_layout_falls_back_on_invalid_active() -> None:
    """active that isn't in panels gets coerced to None, not rejected wholesale."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": ["stats"], "active": "ghost"},
                "bottom": {"panels": [], "active": ""},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.right.active is None


def test_initial_layout_filters_unknown_panel_ids() -> None:
    """Panel ids not in PANELS (typos, prior-schema ids like 'rds') are dropped;
    `active` gets coerced to None when its target was filtered."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": ["decoder-output"], "active": ""},
                "right": {"panels": ["stats", "performance", "ghost"], "active": "ghost"},
                "bottom": {"panels": ["demod"], "active": "demod"},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    assert m.layout.right.panels == ("stats", "performance", "keys")
    assert m.layout.right.active is None  # 'ghost' got filtered, so active is invalid


def test_initial_layout_augments_missing_panels() -> None:
    """Saved layout from a prior schema (e.g. only the old per-decoder ids in
    bottom) gets missing PANELS appended to their default edges. The user's
    own customizations to other edges are preserved."""
    prefs = {
        "ui": {
            "layout": {
                "left": {"panels": [], "active": ""},
                "right": {"panels": ["stats"], "active": ""},
                "bottom": {"panels": ["rds", "dab", "adsb"], "active": "rds"},
                "hotkeys": [],
            }
        }
    }
    m = UIModel.initial(prefs)
    # rds/dab/adsb filtered → bottom empty post-filter; directory (default-bottom)
    # augmented onto it.
    assert m.layout.bottom.panels == ("directory",)
    # demod, decoder-output and memories absent from saved → appended to their
    # default edge (left).
    assert m.layout.left.panels == ("demod", "decoder-output", "memories")
    # performance and keys were missing from saved right → appended.
    assert m.layout.right.panels == ("stats", "performance", "keys")


def test_initial_layout_falls_back_on_garbage() -> None:
    m = UIModel.initial({"ui": {"layout": "not a dict"}})
    assert m.layout == DEFAULT_LAYOUT


def test_initial_ignores_unknown_top_level_keys() -> None:
    m = UIModel.initial({"ui": {"unknown_key": 2.0}, "engine": {"audio_volume": 0.5}})
    assert m == UIModel()
