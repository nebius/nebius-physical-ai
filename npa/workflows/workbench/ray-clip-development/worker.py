# Render and crop deterministic RGB inputs delivered through Ray working_dir.
"""Customer image preprocessing, independent of the Workbench inference component."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

# A source edit changes the model input while retaining the image and model pins.
CROP_POLICY = "left"


def source_hash() -> str:
    """Identify the preprocessing source imported by this worker.

    Args:
        None.
    Returns:
        SHA-256 of this module's source bytes.
    Raises:
        OSError: The imported source file cannot be read.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _draw_shapes(drawing, random_source, side: int) -> None:
    """Preserve the seeded shape sequence used by the original example."""
    for _ in range(14):
        horizontal = side + random_source.randrange(100)
        vertical = random_source.randrange(224)
        color = tuple(random_source.randrange(256) for _ in range(3))
        width = random_source.randrange(8, 28)
        height = random_source.randrange(8, 28)
        bounds = (horizontal, vertical, horizontal + width, vertical + height)
        drawing.ellipse(bounds, fill=color)


def render_record(record_id: int) -> bytes:
    """Render a deterministic, asymmetric synthetic perception input.

    Args:
        record_id: Seed and visible label for one image.
    Returns:
        Encoded RGB PNG bytes; no downloaded task data is used.
    Raises:
        OSError: Pillow cannot encode the generated image.
    """
    import random

    from PIL import Image, ImageDraw

    random_source = random.Random(record_id)
    image = Image.new("RGB", (256, 256))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, 127, 255), fill=(170 + record_id % 70, 35, 20))
    drawing.rectangle((128, 0, 255, 255), fill=(20, 45, 170 + record_id % 70))
    for side in (0, 128):
        _draw_shapes(drawing, random_source, side)
        drawing.text((side + 4, 236), f"sample {record_id}", fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def preprocess_image(raw: bytes) -> bytes:
    """Select the configured image half and resize it for CLIP.

    Args:
        raw: Encoded input image bytes.
    Returns:
        Encoded 224-by-224 PNG bytes for the selected crop.
    Raises:
        ValueError: The source selects an unsupported crop policy.
        OSError: Pillow cannot decode or encode the image.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        if CROP_POLICY not in {"left", "right"}:
            raise ValueError(f"Unsupported crop policy: {CROP_POLICY}")
        left = 0
        if CROP_POLICY == "right":
            left = image.width // 2
        bounds = (left, 0, left + image.width // 2, image.height)
        image = image.crop(bounds).resize((224, 224))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


def _current_node_id() -> str | None:
    """Allow local preprocessing tests without discovering or starting Ray."""
    import sys

    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return None
    return ray.get_runtime_context().get_node_id()


def _preprocess_record(record_id: int) -> dict:
    """Keep input and crop hashes beside the bytes submitted for inference."""
    raw = render_record(record_id)
    processed = preprocess_image(raw)
    return {
        "record_id": record_id,
        "image_bytes": processed,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "processed_sha256": hashlib.sha256(processed).hexdigest(),
    }


def preprocess_shard(record_ids: list[int]) -> dict:
    """Prepare one Ray task's image batch with source and input provenance.

    Args:
        record_ids: Deterministic image seeds assigned to this task.
    Returns:
        Image rows, imported-source identity, placement and elapsed time.
    Raises:
        ValueError: The configured crop policy is unsupported.
        OSError: A source file or generated image cannot be read or encoded.
    """
    import os
    import time

    started = time.perf_counter()
    rows = [_preprocess_record(record_id) for record_id in record_ids]
    return {
        "rows": rows,
        "source_sha256": source_hash(),
        "source_path": str(Path(__file__).resolve()),
        "pid": os.getpid(),
        "node_id": _current_node_id(),
        "crop_policy": CROP_POLICY,
        "preprocess_seconds": time.perf_counter() - started,
    }
