"""Installed console entry (``shadowhunter`` / ``python -m shadowhunter``)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from run import main as _main
    raise SystemExit(_main())
