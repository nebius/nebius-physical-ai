"""Recheck a retained native GPU film capture, without launching a duplicate job.

Run the RTX workflow with video_profile=film, retain its entire output directory,
then set NPA_INTEGRATION_E2E=1 and NPA_ARENA_FILM_RESULT to its result.json.
This verifies real MP4/PNG bytes against the measured capture and task evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.workbench.isaac_arena.acceptance import qualify_visual_acceptance
from npa.workbench.isaac_arena.video_evidence import verify_capture_evidence

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _verified_artifact_paths(result_path, result):
    paths = {}
    for artifact in result["artifacts"]:
        path = (result_path.parent / artifact["path"]).resolve()
        assert path.is_relative_to(result_path.parent)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == artifact["sha256"], artifact["path"]
        paths[digest] = path
    return paths


def test_native_film_capture_matches_successful_gpu_episode():
    selected = os.environ.get("NPA_ARENA_FILM_RESULT", "")
    if not selected:
        pytest.skip("NPA_ARENA_FILM_RESULT must name a retained live result.json")
    result_path = Path(selected).expanduser().resolve()
    result = json.loads(result_path.read_text())
    assert result["status"] == "ok"
    assert result["request"]["video_profile"] == "film"
    assert result["gpu"]["available"] is True
    paths = _verified_artifact_paths(result_path, result)
    video = next(a["video"] for a in result["artifacts"] if "video" in a)
    assert video["task_qualified"] is True
    assert (video["width"], video["height"]) == (3840, 2160)
    ground_truth = result["summary"]["simulator_ground_truth"]
    steps = ground_truth["task_motion"]["episode_length"]
    raw = paths[video["simulator_capture"]["source_mp4_sha256"]]
    verified = verify_capture_evidence(
        raw.parent,
        raw,
        task_motion=ground_truth["task_motion"],
        expected_steps=steps,
        expected_profile="film",
    )
    assert verified["physics_freeze"]["verified_capture_count"] == steps + 1
    accepted = qualify_visual_acceptance(
        environment=result["request"]["environment"],
        policy_type=result["request"]["policy_type"],
        evidence=result["input"],
        summary=result["summary"],
        ground_truth=ground_truth,
        capture=verified,
        video=video,
    )
    assert accepted["qualified"] is True
    assert accepted["actions"]["padding_steps"] == 0
    assert accepted["episode"]["native_success"] is True
