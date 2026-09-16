"""Inspect original BEHAVIOR rollout bytes and assemble a reproducible submission."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

from .protocol import EVAL_DIRECTORY, file_digest


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_metrics(metrics: dict, case: dict) -> float:
    if not isinstance(metrics, dict):
        raise ValueError("Rollout metrics must be an object")
    for field in ("task", "instance_id", "rollout_id"):
        if (
            type(metrics.get(field)) is not type(case[field])
            or metrics[field] != case[field]
        ):
            raise ValueError(f"Rollout {field} does not match its prescribed case")
    if type(metrics.get("steps")) is not int or metrics["steps"] <= 0:
        raise ValueError("Rollout must contain a positive simulator step count")
    if type(metrics.get("success")) is not bool:
        raise ValueError("Rollout success must be a boolean")
    scores = metrics.get("q_score")
    score = scores.get("final") if isinstance(scores, dict) else None
    if not _finite_number(score) or not 0 <= score <= 1:
        raise ValueError("Rollout must contain a finite q_score.final in [0, 1]")
    for field in ("agent_distance", "normalized_agent_distance"):
        values = metrics.get(field, {})
        if not isinstance(values, dict):
            raise ValueError(f"Rollout {field} must be an object")
        for part in ("base", "left", "right"):
            value = values.get(part)
            valid = _finite_number(value)
            if field == "normalized_agent_distance" and value == float("inf"):
                valid = metrics["agent_distance"][part] == 0
            if not valid or value < 0:
                raise ValueError(f"Rollout is missing valid {field}.{part}")
    timing = metrics.get("time", {})
    if not isinstance(timing, dict):
        raise ValueError("Rollout time must be an object")
    for field in ("simulator_steps", "simulator_time", "normalized_time"):
        if not _finite_number(timing.get(field)) or timing[field] < 0:
            raise ValueError(f"Rollout is missing valid time.{field}")
    return float(score)


def inspect_rollout(output: Path, case: dict) -> dict:
    """Validate a rollout's original JSON and fully decode its required video.

    Args:
        output: Directory containing upstream json/ and videos/ outputs.
        case: Prescribed task, actual instance ID, and rollout ID.
    Returns:
        Separate validation evidence including original-byte hashes.
    Raises:
        ValueError: Metrics or video do not satisfy the artifact contract.
        OSError: Required files cannot be read.
        av.FFmpegError: Video decoding fails.
    """
    import av

    stem = f"{case['task']}_{case['instance_id']}_0"
    metrics = output / "json" / f"{stem}.json"
    video = output / "videos" / f"{stem}.mp4"
    score = _validate_metrics(json.loads(metrics.read_bytes()), case)
    with av.open(str(video)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    if frames == 0:
        raise ValueError("Rollout video contains no decoded frames")
    return {
        **case,
        "q_score": score,
        "video_frames": frames,
        "files": {str(p.relative_to(output)): file_digest(p) for p in (metrics, video)},
    }


def snapshot_evaluator(root: Path, output: Path) -> None:
    """Copy the exact official wrapper, robot, and evaluator sources for reproducibility.

    Args:
        root: Verified upstream checkout.
        output: New worker output directory.
    Returns:
        None.
    Raises:
        OSError: The snapshot cannot be written.
    """
    destination = output / "evaluator"
    shutil.copytree(
        root / EVAL_DIRECTORY,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(root / "OmniGibson/LICENSE", destination / "LICENSE")


def write_summary(output: Path, plan: dict, records: list[dict]) -> dict:
    """Summarize every planned case, counting missing report cases as zero.

    Args:
        output: Directory for separate summary.json evidence.
        plan: Frozen evaluation plan.
        records: Validated original rollouts, never a best-of selection.
    Returns:
        Artifact-validation summary; it is not an organizer certification.
    Raises:
        ValueError: A record is duplicated or outside the plan.
        OSError: The summary cannot be written.
    """
    expected = {(c["task"], c["instance_id"], c["rollout_id"]) for c in plan["cases"]}
    observed = [(c["task"], c["instance_id"], c["rollout_id"]) for c in records]
    if len(set(observed)) != len(observed) or not set(observed).issubset(expected):
        raise ValueError("Duplicate or unexpected rollout records")
    total_score = sum(record["q_score"] for record in records)
    summary = {
        "schema": "npa.behavior.artifacts.v1",
        "planned": len(expected),
        "completed": len(records),
        "complete": len(records) == len(expected),
        "eligible_for_reporting": plan["eligible_for_reporting"],
        "challenge_score": total_score / 1000
        if plan["eligible_for_reporting"]
        else None,
        "evaluated_mean_q": total_score / len(records) if records else None,
        "missing_challenge_rollouts": 1000 - len(records),
        "rollouts": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_submission(output: Path, plan: dict, records: list[dict]) -> Path:
    """Bundle unmodified metrics and evaluator code, retaining videos beside the ZIP.

    Args:
        output: Directory containing originals and the generated evaluation README.
        plan: Frozen reporting plan.
        records: Complete set of validated planned cases.
    Returns:
        Submission ZIP path; videos remain separate as required by the portal.
    Raises:
        ValueError: The run is development-only, incomplete, or changed after validation.
        OSError: Original files or the archive cannot be read or written.
    """
    if not plan["eligible_for_reporting"] or len(records) != len(plan["cases"]):
        raise ValueError("Only complete prescribed reporting selections can be bundled")
    for record in records:
        for relative, expected in record["files"].items():
            if file_digest(output / relative) != expected:
                raise ValueError("Original rollout bytes changed after validation")
    metadata_files = ("README.md", "policy.md", "plan.json", "summary.json")
    paths = [output / name for name in metadata_files]
    paths += sorted((output / "evaluator").rglob("*.py"))
    paths += [output / "evaluator/r1pro.yaml", output / "evaluator/LICENSE"]
    paths += [
        output / name
        for name in ("policy-server.py", "policy-provenance.json")
        if (output / name).is_file()
    ]
    paths += [
        output / relative
        for record in records
        for relative in record["files"]
        if relative.startswith("json/")
    ]
    target = output / "submission.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(output))
    return target
