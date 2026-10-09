"""Keep source and generated frame identity visible in every judge image."""

from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image, ImageDraw, ImageFont

from npa.workflows.video_sweep.vision import text_block


def paired_frames(before, source_metadata, after, generated_metadata):
    """Present ordered, pixel-labelled comparisons and exact image hashes.

    Args:
        before: Sampled source JPEG content blocks.
        source_metadata: Decoded source frame indices and timestamps.
        after: Sampled generated JPEG content blocks.
        generated_metadata: Decoded generated frame indices and timestamps.
    Returns:
        Interleaved labels/images and a path-free presentation receipt.
    Raises:
        ValueError: Samples or their timeline metadata cannot be paired.
    """
    content, frames = [], []
    samples = _sample_pairs(before, source_metadata, after, generated_metadata)
    for ordinal, sample in enumerate(samples, start=1):
        labels, encoded, evidence = _comparison_pair(sample)
        content.extend(_content_pair(ordinal, labels, encoded))
        frames.append(evidence)
    return content, {"layout": "source-left-generated-right-v1", "frames": frames}


def _sample_pairs(before, source_metadata, after, generated_metadata):
    sequences = (
        before,
        source_metadata["sample_indices"],
        source_metadata["sample_times"],
        after,
        generated_metadata["sample_indices"],
        generated_metadata["sample_times"],
    )
    if len(before) < 2 or any(len(values) != len(before) for values in sequences):
        raise ValueError("Review requires matching frame and timeline sample counts")
    return zip(*sequences, strict=True)


def _comparison_pair(sample):
    source, source_index, source_time, generated, generated_index, generated_time = (
        sample
    )
    labels = (
        _label("SOURCE", source_index, source_time),
        _label("GENERATED", generated_index, generated_time),
    )
    encoded, size = _pair_image(source, generated, labels)
    evidence = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "width": size[0],
        "height": size[1],
        "source_index": source_index,
        "source_time": source_time,
        "generated_index": generated_index,
        "generated_time": generated_time,
    }
    return labels, encoded, evidence


def _content_pair(ordinal, labels, encoded):
    return (
        text_block(f"Pair {ordinal}: left {labels[0]}; right {labels[1]}."),
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
            },
        },
    )


def _label(role, index, timestamp):
    time = "unknown" if timestamp is None else f"{timestamp:.3f}s"
    return f"{role} | frame {index} | {time}"


def _pair_image(source, generated, labels):
    images = []
    prefix = "data:image/jpeg;base64,"
    for block in (source, generated):
        uri = block["image_url"]["url"]
        if not uri.startswith(prefix):
            raise ValueError("Review frames must be embedded JPEG samples")
        raw = base64.b64decode(uri[len(prefix) :], validate=True)
        with Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB")
        image.thumbnail((512, 384))
        images.append(image)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (1024, height + 44), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    for column, (image, label) in enumerate(zip(images, labels, strict=True)):
        position = (
            column * 512 + (512 - image.width) // 2,
            44 + (height - image.height) // 2,
        )
        canvas.paste(image, position)
        draw.text((column * 512 + 8, 10), label, font=font, fill="white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), canvas.size
