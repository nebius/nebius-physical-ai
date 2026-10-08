"""Build portable HTML experiences from verified camera and spatial-scan artifacts."""

import json
from pathlib import Path

from .demo_assets import CSS, HTML, JAVASCRIPT


def write_demo(root: Path, result):
    """Write a self-contained report shell with relative, downloadable real assets.

    Args: Bundle directory and validated GPU result.
    Returns: Path to index.html.
    Raises: OSError if writing the report fails.
    """
    payload = json.dumps(result, allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__STYLE__", CSS).replace("__REPORT__", payload)
    html = html.replace("__SCRIPT__", JAVASCRIPT)
    path = root / "index.html"
    path.write_text(html)
    return path
