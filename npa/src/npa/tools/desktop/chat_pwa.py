"""Serve public install metadata and generated icons without caching chat data."""

from functools import lru_cache
import json
import struct
import zlib


def _chunk(kind, data):
    return (
        struct.pack("!I", len(data))
        + kind
        + data
        + struct.pack("!I", zlib.crc32(kind + data))
    )


@lru_cache(maxsize=3)
def _icon(size):
    # A terminal chevron and cursor remain legible at small Home Screen sizes.
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            across, down = x / size, y / size
            chevron = (
                0.28 < down < 0.65 and abs(across - (0.48 - abs(down - 0.465))) < 0.028
            )
            cursor = 0.48 < across < 0.72 and 0.60 < down < 0.66
            rows.extend((235, 242, 239) if chevron or cursor else (23, 42, 38))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack("!2I5B", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(rows)))
        + _chunk(b"IEND", b"")
    )


def public_asset(path):
    """Return installation-only assets that are safe to fetch before sign-in.

    Args:
        path: URL path, without a query string.
    Returns:
        A (body, MIME type) pair, or None for a protected or unknown route.
    Raises:
        None.
    """
    if path == "/chat/manifest.webmanifest":
        manifest = {
            "id": "./",
            "name": "Codex",
            "short_name": "Codex",
            "start_url": "./",
            "scope": "./",
            "display": "standalone",
            "background_color": "#171717",
            "theme_color": "#171717",
            "icons": [
                {
                    "src": f"./icon-{size}.png",
                    "sizes": f"{size}x{size}",
                    "type": "image/png",
                    "purpose": "any maskable",
                }
                for size in (192, 512)
            ],
        }
        return json.dumps(manifest).encode(), "application/manifest+json"
    for size in (180, 192, 512):
        if path == f"/chat/icon-{size}.png":
            return _icon(size), "image/png"
    return None
