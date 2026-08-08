"""Source transformations used while rendering the embedded agent backend."""

from __future__ import annotations


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
