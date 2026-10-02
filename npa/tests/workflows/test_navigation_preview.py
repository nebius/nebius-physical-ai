"""Keep offline navigation previews inside the sealed, actually scored focal interval."""

import json

from PIL import Image
import pytest

from npa.workflows.navigation.artifacts import publish, write_json
from npa.workflows.navigation.preview import scored_preview_frames, scored_rollout_group


@pytest.fixture
def native_preview(tmp_path):
    source = tmp_path / "source"
    report = {"episodes": [{"steps": 2, "success": True, "goal_distance_m": 0.1}]}
    index = {
        "robot_index": 0,
        "renderer": "isaac-replicator-rgb",
        "frames": [
            {
                "step": step,
                "simulation_seconds": step * 0.1,
                "goal_distance_m": distance,
            }
            for step, distance in enumerate([2.0, 1.0, 0.1, 0.8, 1.5])
        ],
    }
    write_json(source / "evaluation.json", report)
    write_json(source / "rendered-rollout/frames.json", index)
    for row in index["frames"]:
        Image.new("RGB", (16, 12), (row["step"] * 40, 20, 60)).save(
            source / "rendered-rollout" / f"{row['step']:06d}.png"
        )
    destination = tmp_path / "sealed"
    publish(source, str(destination))
    return destination, report, index


def test_preview_excludes_later_observer_motion_and_embeds_real_pixels(native_preview):
    root, report, index = native_preview
    rows, focal = scored_preview_frames(report, index)
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert focal["goal_distance_m"] == 0.1
    group = scored_rollout_group(root, report, title="Scored focal episode")
    assert len(group["frames"]) == 3
    assert group["frames"][-1]["label"].startswith("Step 2")
    assert all(
        frame["images"][0]["data"].startswith("data:image/jpeg;base64,")
        for frame in group["frames"]
    )
    assert "Later observer frames are excluded" in group["note"]


@pytest.mark.parametrize("mutation", ["gap", "clock", "endpoint", "robot"])
def test_preview_refuses_inconsistent_scored_evidence(native_preview, mutation):
    _, report, index = native_preview
    if mutation == "gap":
        index["frames"].pop(1)
    elif mutation == "clock":
        index["frames"][1]["simulation_seconds"] = float("nan")
    elif mutation == "endpoint":
        report["episodes"][0]["goal_distance_m"] = 9.0
    else:
        index["robot_index"] = 1
    with pytest.raises(ValueError):
        scored_preview_frames(report, index)


def test_preview_refuses_changed_media_and_report(native_preview):
    root, report, _ = native_preview
    changed = json.loads(json.dumps(report))
    changed["episodes"][0]["success"] = False
    with pytest.raises(ValueError, match="sealed report"):
        scored_rollout_group(root, changed, title="Invalid")
    (root / "rendered-rollout/000001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        scored_rollout_group(root, report, title="Invalid")
