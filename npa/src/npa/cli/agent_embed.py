"""Source transformations used while rendering the embedded agent backend."""

from __future__ import annotations

import re
from pathlib import Path


def without_embedded_standalone_block(source: str) -> str:
    """Remove direct-test sentinels before a runtime module is embedded."""
    start = "# NPA_EMBED_STANDALONE_START"
    end = "# NPA_EMBED_STANDALONE_END"
    before, marker, remainder = source.partition(start)
    if not marker:
        return source
    _standalone, closing, after = remainder.partition(end)
    if not closing:
        raise ValueError("embedded module has an unterminated standalone block")
    return before.rstrip() + "\n" + after.lstrip("\n")


def embedded_python_source(path: Path, *, strip_standalone: bool = False) -> str:
    """Read a module and remove declarations invalid in an embedded body."""
    source = path.read_text(encoding="utf-8")
    source = re.sub(r'^""".*?"""\s*\n', "", source, count=1, flags=re.DOTALL)
    source = re.sub(r"^from __future__ import annotations\s*\n", "", source)
    if strip_standalone:
        source = without_embedded_standalone_block(source)
    return source
