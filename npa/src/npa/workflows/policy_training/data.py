"""Curate LeRobot episode references with FiftyOne and split by source lineage."""

from __future__ import annotations

import io
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from npa.workbench.dataset.storage import read_bytes_uri, read_json_uri, write_json_uri
from .contracts import digest, probability
from .diagnostics import _cleanup


def _episodes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != "npa.policy.episodes.v1":
        raise ValueError("expected npa.policy.episodes.v1")
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("episode manifest is empty")
    identities = set()
    for episode in episodes:
        identity = (episode["dataset_uri"], episode["episode_index"])
        if identity in identities or not episode["group_id"]:
            raise ValueError("duplicate episode or missing source-lineage group")
        identities.add(identity)
        index = episode["episode_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        if episode["source"] not in {"real", "synthetic"}:
            raise ValueError("episode source must be real or synthetic")
    return episodes


def _quality(episode: dict[str, Any], preview: Path) -> tuple[float, float]:
    preview.write_bytes(read_bytes_uri(episode["preview_uri"]))
    with Image.open(io.BytesIO(preview.read_bytes())) as image:
        gray = np.asarray(image.convert("L"), dtype=float) / 255
    if min(gray.shape) < 3:
        raise ValueError("preview must be at least three pixels per axis")
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * center
    )
    return float(gray.mean()), float(laplacian.var())


def _sample(fo: Any, episode: dict[str, Any], preview: Path) -> Any:
    brightness, sharpness = _quality(episode, preview)
    detections = episode["detections"]
    if not isinstance(detections, list):
        raise ValueError("detections must contain reviewed detector labels")
    if not all(isinstance(label, str) and label for label in detections):
        raise ValueError("detection labels must be nonempty strings")
    review = episode.get("inhouse_keep")
    if review is not None and not isinstance(review, bool):
        raise ValueError("inhouse_keep must be a boolean review result when provided")
    return fo.Sample(
        filepath=str(preview),
        episode_key=digest(episode),
        brightness=brightness,
        sharpness=sharpness,
        detection_count=len(detections),
        inhouse_keep=review,
        detections=fo.Classifications(
            classifications=[fo.Classification(label=label) for label in detections]
        ),
    )


def _validate_sources(episodes: list[dict[str, Any]]) -> None:
    for root in {episode["dataset_uri"] for episode in episodes}:
        info = read_json_uri(root.rstrip("/") + "/meta/info.json")
        if info.get("codebase_version") != "v3.0":
            raise ValueError("curation requires LeRobot v3.0 source metadata")
        total = info.get("total_episodes", 0)
        if any(
            e["episode_index"] >= total for e in episodes if e["dataset_uri"] == root
        ):
            raise ValueError("episode reference exceeds source episode count")


def curate(input_uri: str, output_uri: str, policy_uri: str) -> None:
    """Apply real FiftyOne quality filters to episode preview frames.

    Args:
        input_uri: Real and optional synthetic episode-reference manifest.
        output_uri: Selected episodes and comparison report location.
        policy_uri: Operator-selected brightness, sharpness and detection thresholds.
    Returns:
        None.
    Raises:
        ValueError: Inputs, policy, or selection are invalid.
        ImportError: FiftyOne is unavailable; no fallback is used.
    """
    import fiftyone as fo

    manifest = read_json_uri(input_uri)
    episodes = _episodes(manifest)
    _validate_sources(episodes)
    policy = read_json_uri(policy_uri)
    expression = _filter(fo, policy)
    dataset = fo.Dataset()
    try:
        with tempfile.TemporaryDirectory(prefix="policy-curation-") as directory:
            for index, episode in enumerate(episodes):
                dataset.add_sample(
                    _sample(fo, episode, Path(directory) / f"{index}.png")
                )
            selected = set(dataset.match(expression).values("episode_key"))
            _write_curated(
                episodes, selected, policy, output_uri, _measurements(dataset)
            )
    finally:
        _cleanup(dataset.delete, output_uri)


def _measurements(dataset):
    values = dataset.values(
        ["episode_key", "brightness", "sharpness", "detection_count"]
    )
    return {
        key: {"brightness": bright, "sharpness": sharp, "detections": count}
        for key, bright, sharp, count in zip(*values, strict=True)
    }


def _filter(fo: Any, policy: dict[str, Any]) -> Any:
    low = probability(policy["min_brightness"])
    high = probability(policy["max_brightness"])
    sharpness = policy["min_sharpness"]
    detections = policy["min_detections"]
    if low > high or not math.isfinite(sharpness) or sharpness < 0:
        raise ValueError("invalid brightness or sharpness thresholds")
    if (
        isinstance(detections, bool)
        or not isinstance(detections, int)
        or detections < 0
    ):
        raise ValueError("min_detections must be a nonnegative integer")
    field = fo.ViewField
    return (
        (field("brightness") >= low)
        & (field("brightness") <= high)
        & (field("sharpness") >= sharpness)
        & (field("detection_count") >= detections)
    )


def _write_curated(episodes, selected, policy, output_uri, measurements=None):
    selection = policy.get("selection", "quality-pass")
    if selection not in {"quality-pass", "all"}:
        raise ValueError("selection must be quality-pass or all")
    kept = [e for e in episodes if selection == "all" or digest(e) in selected]
    if not kept:
        raise ValueError("FiftyOne curation selected no episodes")
    reviewed = [e for e in episodes if e.get("inhouse_keep") is not None]
    disagreement = sum((digest(e) in selected) != e["inhouse_keep"] for e in reviewed)
    write_json_uri(
        output_uri,
        {
            "schema": "npa.policy.episodes.v1",
            "episodes": kept,
            "engine": "fiftyone",
            "policy": policy,
            "input_count": len(episodes),
            "selected_count": len(kept),
            "inhouse_reviewed_count": len(reviewed),
            "inhouse_disagreement_count": disagreement if reviewed else None,
            "detection_source": "provided-labels",
            "quality_scope": "preview-frame",
            "selection": selection,
            "quality_pass_count": len(selected),
            "review": [
                {
                    "episode_sha256": digest(e),
                    "quality_pass": digest(e) in selected,
                    "inhouse_keep": e.get("inhouse_keep"),
                    "measurements": (measurements or {}).get(digest(e), {}),
                }
                for e in episodes
            ],
        },
    )


def split(input_uri: str, output_uri: str, seed: str) -> None:
    """Assign source-lineage groups to deterministic 90/5/5 partitions.

    Args:
        input_uri: Curated episode manifest.
        output_uri: Split index location; partition files are saved beside it.
        seed: Stable split seed.
    Returns:
        None.
    Raises:
        ValueError: Fewer than twenty independent source groups exist.
    """
    episodes = _episodes(read_json_uri(input_uri))
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["group_id"]].append(episode)
    if len(groups) < 20:
        raise ValueError("90/5/5 split requires at least twenty independent groups")
    ordered = sorted(groups, key=lambda key: digest([seed, key]))
    holdout = len(ordered) // 20
    partitions = {
        "train": ordered[2 * holdout :],
        "holdout_1": ordered[:holdout],
        "holdout_2": ordered[holdout : 2 * holdout],
    }
    index = {"schema": "npa.policy.split.v1", "seed": seed, "partitions": {}}
    for name, keys in partitions.items():
        payload = {
            "schema": "npa.policy.episodes.v1",
            "episodes": [item for key in keys for item in groups[key]],
        }
        uri = output_uri.rsplit("/", 1)[0] + f"/{name}.json"
        write_json_uri(uri, payload)
        index["partitions"][name] = {
            "uri": uri,
            "sha256": digest(payload),
            "groups": len(keys),
        }
    write_json_uri(output_uri, index)
