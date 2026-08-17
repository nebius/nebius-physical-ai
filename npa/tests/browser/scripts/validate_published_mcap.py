"""Validate the public Foxglove MCAP bytes used by live browser acceptance."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import jsonschema
from mcap.reader import make_reader
from PIL import Image


CAMERA_TOPICS = ("/camera", "/camera/side", "/camera/workspace")
SCENE_TOPIC = "/robot/diagnostic_scene"


def _arrays_without_items(node: object, path: str = "$") -> list[str]:
    if not isinstance(node, dict):
        return []
    missing = [path] if node.get("type") == "array" and "items" not in node else []
    for key, value in node.items():
        missing.extend(_arrays_without_items(value, f"{path}.{key}"))
    return missing


def validate(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    camera_times: dict[str, list[int]] = {topic: [] for topic in CAMERA_TOPICS}
    camera_hashes: dict[str, list[str]] = {topic: [] for topic in CAMERA_TOPICS}
    scene_messages: list[dict] = []
    scene_schema: dict = {}

    with path.open("rb") as handle:
        for schema, channel, message in make_reader(handle).iter_messages():
            topic = channel.topic
            payload = json.loads(message.data)
            if topic in camera_times:
                encoded = base64.b64decode(payload["data"], validate=True)
                with Image.open(io.BytesIO(encoded)) as image:
                    pixels = image.convert("RGB").tobytes()
                camera_times[topic].append(message.log_time)
                camera_hashes[topic].append(hashlib.sha256(pixels).hexdigest())
            elif topic == SCENE_TOPIC:
                scene_messages.append(payload)
                if not scene_schema:
                    scene_schema = json.loads(schema.data)

    validator = jsonschema.Draft7Validator(scene_schema)
    scene_errors = [
        f"message[{index}] {error.json_path}: {error.message}"
        for index, payload in enumerate(scene_messages)
        for error in validator.iter_errors(payload)
    ]
    counts = {topic: len(values) for topic, values in camera_hashes.items()}
    synchronized = len({tuple(values) for values in camera_times.values()}) == 1
    distinct_triplets = sum(
        len(set(hashes)) == len(CAMERA_TOPICS)
        for hashes in zip(*(camera_hashes[topic] for topic in CAMERA_TOPICS))
    )
    return {
        "sha256": digest,
        "camera_counts": counts,
        "camera_unique_frames": {
            topic: len(set(values)) for topic, values in camera_hashes.items()
        },
        "synchronized": synchronized,
        "distinct_aligned_triplets": distinct_triplets,
        "scene_message_count": len(scene_messages),
        "scene_validation_errors": scene_errors,
        "schema_arrays_without_items": _arrays_without_items(scene_schema),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_published_mcap.py PATH")
    print(json.dumps(validate(Path(sys.argv[1])), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
