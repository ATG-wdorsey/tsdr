import importlib

__version__ = "0.3.0"

# A wheel built past the release tag says how far (tsdr/_build.py, from hatch_build.py):
# 0.2.0+106.ga767567. A source checkout or a release wheel shows the version alone.
try:
    __version__ += getattr(importlib.import_module("tsdr._build"), "LOCAL", "")
except ImportError:
    pass
