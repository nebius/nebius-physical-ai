"""Reuse completed candidates only after verifying their request and media bytes."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from npa.workflows.video_sweep import artifacts, vision


def generate_candidate(item, args, plan, generator) -> dict:
    """Persist each completed candidate independently of its worker partition.

    Args:
        item: Exact planned generation request.
        args: Stage arguments with the private artifact root.
        plan: Immutable plan for this run.
        generator: Real generation callable accepting item, args and plan.
    Returns:
        A newly generated or independently verified candidate.
    Raises:
        ValueError: A checkpoint or its referenced media differs from this request.
        OSError: Artifact retrieval or publication fails.
    """
    uri = args.root_uri + "/candidates/" + item["id"] + "/receipt.json"
    binding = {
        "schema": "npa.video_sweep.candidate.v1",
        "plan_sha256": artifacts.digest(plan),
        "item_sha256": artifacts.digest(item),
    }
    if artifacts.exists(uri):
        receipt = artifacts.read_json(uri)
        if any(receipt.get(key) != value for key, value in binding.items()):
            raise ValueError("Candidate checkpoint belongs to a different request")
        candidate = receipt["candidate"]
        _verify(candidate, item, args.root_uri, plan)
        return candidate
    candidate = generator(item, args, plan)
    _verify(candidate, item, args.root_uri, plan)
    artifacts.write_json(uri, {**binding, "candidate": candidate})
    return candidate


def _verify(candidate, item, root, plan):
    checksum = candidate.get("sha256", "")
    engine = plan.get("generator", "cosmos-transfer2.5")
    if (
        not isinstance(checksum, str)
        or not re.fullmatch("[a-f0-9]{64}", checksum)
        or candidate.get("id") != item["id"]
        or candidate.get("engine") != engine
        or type(candidate.get("seed")) is not int
        or candidate["seed"] != item["variant"]["seed"]
        or candidate.get("uri")
        != root + "/candidates/" + item["id"] + "/" + checksum + ".mp4"
    ):
        raise ValueError("Candidate checkpoint identity differs from its request")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.mp4"
        artifacts.download(candidate["uri"], path, checksum)
        _, metadata = vision.sample_video(path, plan["samples"])
    if metadata != candidate.get("video"):
        raise ValueError("Candidate checkpoint video metadata differs")
    if engine == "cosmos3-nano":
        _verify_native(candidate, item, root)


def _verify_native(candidate, item, root):
    evidence = artifacts.read_json(
        root + "/candidates/" + item["id"] + "/generation.json"
    )
    if (
        artifacts.digest(evidence) != candidate.get("generation_sha256")
        or evidence.get("source_sha256") != item["source"]["sha256"]
        or evidence["artifacts"]["edge"] != candidate["controls"]["edge"]
        or evidence["artifacts"]["reference"] != candidate["reference"]
        or any(
            evidence["artifacts"]["video"][key] != candidate[key]
            for key in ("uri", "sha256", "video")
        )
    ):
        raise ValueError("Cosmos3 generation evidence differs from the candidate")
    for control in (candidate["controls"]["edge"], candidate["reference"]):
        with tempfile.TemporaryDirectory() as directory:
            artifacts.download(
                control["uri"], Path(directory) / "control.mkv", control["sha256"]
            )
