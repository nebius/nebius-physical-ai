"""Render compact, offline HTML previews from measured workflow output media."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image


def image_preview(image: Image.Image, *, width: int = 480) -> str:
    """Encode a display thumbnail, preserving the separate original artifact.

    Args:
        image: Decoded source image.
        width: Maximum preview width in pixels.

    Returns:
        An embedded JPEG data URL, requiring no network requests.

    Raises:
        OSError: The source image cannot be decoded or encoded.
    """
    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((width, width), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_preview(
    path: Path,
    *,
    title: str,
    summary: str,
    metrics: dict,
    groups: list,
    details: dict | None = None,
    allow_empty_media: bool = False,
) -> None:
    """Write independent image timelines with inert embedded data and escaped text.

    Args:
        path: Local HTML destination.
        title: Human-readable report title.
        summary: Scope and limitations of the preview.
        metrics: Measured scalar values safe for a portable report.
        groups: Timeline dictionaries containing title, note, and frames.
        details: Optional measured summary displayed as inert, escaped JSON.
        allow_empty_media: Explicitly allow a report describing missing or failed media.

    Returns:
        None.

    Raises:
        ValueError: A group contains no frames or JSON contains nonfinite values.
        OSError: The destination cannot be written.
    """
    if (not groups and not allow_empty_media) or any(
        not group["frames"] for group in groups
    ):
        raise ValueError("preview requires actual media in every timeline")
    body = _document(title, summary, metrics, groups, details)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _document(title, summary, metrics, groups, details):
    encoded = json.dumps(groups, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    values = "".join(
        f"<div><dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd></div>"
        for key, value in metrics.items()
    )
    replacements = {
        "TITLE": escape(title),
        "SUMMARY": escape(summary),
        "METRICS": values,
        "DATA": encoded,
        "DETAILS": _details(details),
    }
    return re.sub(
        r"@@(TITLE|SUMMARY|METRICS|DATA|DETAILS)@@",
        lambda match: replacements[match[1]],
        _DOCUMENT,
    )


def _details(value):
    if value is None:
        return ""
    document = escape(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return (
        '<details class="evidence-panel" id="measured-evidence"><summary>Measured evidence · original values</summary><pre>'
        + document
        + "</pre></details>"
    )


_DOCUMENT = Path(__file__).with_name("preview.html").read_text(encoding="utf-8")
