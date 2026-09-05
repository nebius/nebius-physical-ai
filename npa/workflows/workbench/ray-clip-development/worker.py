"""Customer application source shipped by Ray, independent of the image's UDF."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

# The reproduction changes this source line and submits a new Ray working_dir.
CROP_POLICY = "left"


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def render_record(record_id: int) -> bytes:
    """Render deterministic, asymmetric synthetic perception inputs; no downloads."""
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(record_id)
    image = Image.new("RGB", (256, 256))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 127, 255), fill=(170 + record_id % 70, 35, 20))
    draw.rectangle((128, 0, 255, 255), fill=(20, 45, 170 + record_id % 70))
    for side in (0, 128):
        for _ in range(14):
            x, y = side + rng.randrange(100), rng.randrange(224)
            color = tuple(rng.randrange(256) for _ in range(3))
            draw.ellipse((x, y, x + rng.randrange(8, 28), y + rng.randrange(8, 28)), fill=color)
        draw.text((side + 4, 236), f"sample {record_id}", fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def preprocess_image(raw: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        if CROP_POLICY not in {"left", "right"}:
            raise ValueError(f"Unsupported crop policy: {CROP_POLICY}")
        left = 0 if CROP_POLICY == "left" else image.width // 2
        image = image.crop((left, 0, left + image.width // 2, image.height)).resize((224, 224))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


def preprocess_shard(record_ids: list[int]) -> dict:
    import os
    import sys
    import time

    started = time.perf_counter()
    rows = []
    for record_id in record_ids:
        raw = render_record(record_id)
        processed = preprocess_image(raw)
        rows.append({
            "record_id": record_id,
            "image_bytes": processed,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "processed_sha256": hashlib.sha256(processed).hexdigest(),
        })
    ray = sys.modules.get("ray")
    return {
        "rows": rows,
        "source_sha256": source_hash(),
        "source_path": str(Path(__file__).resolve()),
        "pid": os.getpid(),
        "node_id": ray.get_runtime_context().get_node_id() if ray is not None and ray.is_initialized() else None,
        "crop_policy": CROP_POLICY,
        "preprocess_seconds": time.perf_counter() - started,
    }
