"""Test-session bootstrap.

Import order is load-bearing on this machine. Torch ships its own copies of
the Intel OpenMP and MKL runtimes; when it is the first of the scientific
stack to load, a later ``pandas`` / ``numpy`` import binds against a second
copy and the process dies with a Windows access violation - after every
assertion has already passed, during interpreter teardown, which makes it
read like a flaky test rather than a DLL conflict.

Pulling OpenCV and NumPy in first pins the shared runtime before torch can
claim it. The application entry points do the same thing; this file is only
here because pytest imports its plugins before it imports our package, so the
package's own ordering never gets a chance to apply.
"""
from __future__ import annotations

import cv2  # noqa: F401  - must precede torch, see module docstring
import numpy  # noqa: F401

import pytest


@pytest.fixture(scope="session", autouse=True)
def _quiet_third_party_warnings():
    """Keep the report readable without hiding our own warnings."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.*")
    yield
