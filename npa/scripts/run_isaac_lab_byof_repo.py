#!/usr/bin/env python3
"""Compatibility shim — use ``run_byof_repo.py`` instead."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_byof_repo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
