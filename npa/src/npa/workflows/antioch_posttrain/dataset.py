"""Convert native frames into disjoint carton-cycle training and evaluation splits."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from npa.workflows.antioch_warehouse.evidence import verify_evidence
from .artifacts import file_hash, write_json

CLASSES = ("conveying", "pickup", "transfer", "placement", "retract")
PHASE_LABELS = {
    "CONVEYING": "conveying", "ACCUMULATION_STOP": "conveying",
    "APPROACH": "pickup", "LOWER": "pickup", "GRIP_DWELL": "pickup", "LIFT": "pickup",
    "TRANSFER": "transfer", "LOWER_TO_PALLET": "placement", "SETTLING": "placement",
    "RETRACT": "retract",
}
SPLIT_CYCLES = {"train": (0, 1, 2, 3), "validation": (4,), "test": (5,)}


def frame_records(frames: list[dict], stride: int) -> list[dict]:
    """Derive labels from recorded states and keep complete cycles in each split.

    Args:
        frames: Ordered native frame telemetry.
        stride: Positive frame sampling interval.
    Returns:
        Sampled records with explicit phase, cycle, and split.
    Raises:
        ValueError: Telemetry is invalid, unordered, or has unsupported states.
    """
    if stride < 1 or not frames:
        raise ValueError("Frames and positive sampling stride are required")
    records = []
    previous_time = -1.0
    for index, frame in enumerate(frames):
        phase, timestamp = frame["phase"], frame["sim_s"]
        if frame["frame"] != index or not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time:
            raise ValueError("Frame telemetry must be sequential with increasing finite time")
        previous_time = timestamp
        if phase == "BATCH_COMPLETE":
            if index != len(frames) - 1 or frame["placed"] != 6:
                raise ValueError("Batch completion must follow all six placements")
            continue
        if phase not in PHASE_LABELS:
            raise ValueError("Unknown warehouse phase")
        cycle = frame["placed"] - int(phase == "RETRACT")
        if type(cycle) is not int or not 0 <= cycle < 6:
            raise ValueError("Invalid carton cycle")
        if index % stride == 0:
            split = next(name for name, cycles in SPLIT_CYCLES.items() if cycle in cycles)
            records.append(dict(frame, cycle=cycle, split=split, label=CLASSES.index(PHASE_LABELS[phase])))
    return records


def _decode(video: Path, frames: list[dict], records: list[dict]) -> np.ndarray:
    import av

    selected = {record["frame"] for record in records}
    images = []
    decoded = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if stream.width != 1280 or stream.height != 720 or float(stream.average_rate) != 30:
            raise ValueError("Expected the native 1280x720, 30 fps recording")
        for index, frame in enumerate(container.decode(stream)):
            decoded += 1
            if index in selected:
                images.append(frame.reformat(width=224, height=224, format="rgb24").to_ndarray())
    if decoded != len(frames) or len(images) != len(records):
        raise ValueError("Video frames and telemetry disagree")
    return np.stack(images)


def _check_splits(records: list[dict]) -> dict:
    counts = {}
    for split, cycles in SPLIT_CYCLES.items():
        rows = [row for row in records if row["split"] == split]
        histogram = Counter(row["label"] for row in rows)
        if set(histogram) != set(range(len(CLASSES))):
            raise ValueError("Every split must contain all operation classes")
        if {row["cycle"] for row in rows} != set(cycles):
            raise ValueError("A split is missing its expected cycles")
        counts[split] = {CLASSES[key]: value for key, value in sorted(histogram.items())}
    return counts


def prepare_dataset(source: Path, output: Path, stride: int) -> None:
    """Decode authentic video, validate physics evidence, and seal cycle splits.

    Args:
        source: Verified source bundle with video, frame telemetry, and evidence.
        output: New dataset directory.
        stride: Native frame sampling interval.
    Returns:
        None.
    Raises:
        ValueError: Source evidence, video alignment, or splits are invalid.
        OSError: Artifact access fails.
    """
    physics = verify_evidence(source / "evidence")
    if not physics["all_passed"]:
        raise ValueError("Native warehouse evidence failed acceptance checks")
    frames = json.loads((source / "video-frames.json").read_text())
    records = frame_records(frames, stride)
    counts = _check_splits(records)
    images = _decode(source / "warehouse-native-raw.mp4", frames, records)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "frames.npz", images=images)
    write_json(output / "records.json", records)
    write_json(output / "dataset.json", {
        "schema": "npa.antioch-posttrain.dataset.v1", "classes": CLASSES,
        "split_cycles": SPLIT_CYCLES, "class_counts": counts, "samples": len(records),
        "native_frames": len(frames), "sampling_stride": stride, "image_size": [224, 224],
        "source_checksums_sha256": file_hash(source / "checksums.json"),
        "video_sha256": file_hash(source / "warehouse-native-raw.mp4"),
        "telemetry_sha256": file_hash(source / "video-frames.json"),
        "physics_checks": physics["checks"],
        "scope": "Held-out carton cycles from one scene and one run; no unseen-scene or robot-control claim.",
    })
