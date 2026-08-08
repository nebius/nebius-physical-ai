"""Helpers for embedding dependency-light modules in the remote agent backend."""

from __future__ import annotations

import re
from pathlib import Path


def embedded_module_source(path: Path) -> str:
    """Read a module and remove declarations invalid inside a concatenated file."""

    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    return re.sub(r"^from __future__ import annotations\s*\n", "", raw)
