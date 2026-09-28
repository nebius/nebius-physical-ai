"""Present real NuRec capture, reconstruction and novel-view outputs as offline HTML."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
import yaml

from npa.workflows.preview_html import image_preview, write_preview


def _images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _group(title: str, images: list[Path], note: str) -> dict:
    count = min(len(images), 32)
    indices = [
        round(index * (len(images) - 1) / max(1, count - 1)) for index in range(count)
    ]
    frames = []
    for index in indices:
        with Image.open(images[index]) as image:
            frames.append(
                {
                    "label": f"Image {index + 1} of {len(images)}",
                    "images": [{"label": title, "data": image_preview(image)}],
                }
            )
    return {"title": title, "note": note, "frames": frames}


def _quality(path: Path) -> dict:
    if not path.is_file():
        return {"NRE quality metrics": "unavailable"}
    metrics = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = {}
    pending = [("", metrics)]
    while pending:
        prefix, value = pending.pop()
        if isinstance(value, dict):
            pending.extend(
                (f"{prefix}/{key}".strip("/"), item) for key, item in value.items()
            )
        elif prefix.lower() in {"test/psnr", "test/ssim", "test/lpips"}:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                values[prefix] = round(value, 6)
    return values or {"NRE quality metrics": "unavailable"}


def _timelines(capture, novel, validation):
    groups = [
        _group(
            "Input capture",
            capture,
            "Photographic source frames exported from the selected NCore capture.",
        ),
        _group(
            "Novel views",
            novel,
            "NRE renders from a nonzero rig offset. These views are not paired with source image indices.",
        ),
    ]
    if validation:
        groups.append(
            _group(
                "Reconstruction validation",
                validation,
                "NRE's validation renders. Quality measurements below come from NRE validation, not from the novel-view slideshow.",
            )
        )
    return groups


def write_nurec_preview(root: Path, output: Path) -> dict:
    """Build independent visual timelines from actual NuRec outputs.

    Args:
        root: Materialized canonical run directory.
        output: Local destination for the self-contained HTML preview.

    Returns:
        Actual image counts and presentation sampling scope.

    Raises:
        ValueError: Capture frames or novel renders are missing.
        OSError: An output image or metrics document cannot be read.
        yaml.YAMLError: NRE's metrics document is malformed.
    """
    capture = _images(root / "input")
    novel = _images(root / "novel_views")
    validation = _images(root / "reconstruction" / "val")
    if not capture or not novel:
        raise ValueError("NuRec preview requires actual capture and novel-view images")
    counts = {
        "capture images": len(capture),
        "novel views": len(novel),
        "validation images": len(validation),
    }
    write_preview(
        output,
        title="Neural reconstruction",
        metrics={**counts, **_quality(root / "reconstruction" / "metrics.yaml")},
        summary="Real photographs reconstructed into a renderable Gaussian scene. Each timeline samples up to 32 actual output images; playback is a slideshow, not a new video. Gaussian appearance alone does not establish collision geometry or navigation readiness.",
        groups=_timelines(capture, novel, validation),
    )
    return {"image_counts": counts, "maximum_preview_images_per_group": 32}
