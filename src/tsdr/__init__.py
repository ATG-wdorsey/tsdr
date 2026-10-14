import os
import platform
import sys

# OpenBLAS (numpy's BLAS) keeps a worker per core that spins after each matrix
# product. tsdr's only products on the stream (the voice detector's small
# matrices) run as fast on one thread, and the spinning workers take the cores
# the pipeline and audio threads need. Read once, when numpy loads (numba
# imports it below); a user's own setting wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

if len(sys.argv) <= 1:
    print("Starting TSDR (this may take a moment on first run)...", file=sys.stderr, flush=True)

try:
    # Preflight: a missing native runtime (e.g. the MSVC redistributable on a
    # fresh Windows install) makes numba's compiled extensions fail to load.
    # Surface a clear, actionable message instead of a raw DLL-load traceback.
    import numba  # noqa: F401, E402
except ImportError as exc:
    _lines = [f"tsdr: numba failed to import ({exc})."]
    if platform.system() == "Windows":
        _lines += [
            "",
            "This usually means the Microsoft Visual C++ runtime is missing.",
            "Install it, then restart your terminal:",
            "",
            "    winget install --id Microsoft.VCRedist.2015+.x64 -e",
        ]
    print("\n".join(_lines), file=sys.stderr)
    sys.exit(1)

from tsdr.tui import TSDRApp  # noqa: E402

__version__ = "0.1.0"
__all__ = ["__version__", "TSDRApp"]
