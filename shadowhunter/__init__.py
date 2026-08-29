"""Shadow Hunter - hybrid DRL + CNN building-height estimation.

Layout follows a strict MVT split:

    models/     domain + machine learning (no UI, no HTTP)
    views/      FastAPI controllers and the five desktop/web front-ends
    templates/  design tokens rendered into QSS / CSS / theme dicts
    services/   the transport that connects views to the API
"""

# --------------------------------------------------------------------------- #
# Import-order guard (Windows).
#
# torch, OpenCV and pandas each ship their own OpenMP/BLAS runtime. On Windows,
# importing torch *before* pandas can corrupt the heap and kill the process
# with 0xC0000374 - no traceback, no message, and Stable-Baselines3 pulls in
# pandas, so `import stable_baselines3` after torch is enough to trigger it.
#
# Importing numpy and cv2 first pins a consistent runtime and the conflict
# disappears. Two cheap imports here save an unreproducible crash later, so
# every entry point that touches this package is protected by construction.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - environment guard, not logic
    import numpy as _numpy  # noqa: F401
    import cv2 as _cv2      # noqa: F401
except Exception:           # a missing dependency is reported properly later
    pass

__version__ = "1.0.0"
