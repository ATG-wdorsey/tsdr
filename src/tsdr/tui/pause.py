"""Pause (Space): while the focused radio is paused, keys and mouse actions that operate
the radio are ignored, so a stray key or click can't change what was paused. Space
resumes; the console, the panels and the display's own keys still work."""

from tsdr.core.sdr.engine import get_engine

PAUSED_HINT = "Paused: Space resumes"


def focused_paused() -> bool:
    device = get_engine().get_focused_device()
    return device is not None and device.paused
