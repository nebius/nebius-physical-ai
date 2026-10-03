"""Contract tests for the real-media SwitchWorld workflow adapter."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from npa.orchestration.npa_workflow.spec import load_spec
from npa.workflows import switchworld


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/switchworld-lingbot-viewpoint-switch.yaml"


def _video(path: Path, filter_graph: str) -> None:
    """Encode a four-frame H.264 fixture with changing, decodable pixels."""

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            filter_graph,
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _controls() -> dict[str, object]:
    """Return a valid minimal native-control sidecar fixture."""

    return {
        "schema": "npa.switchworld.controls.v1",
        "prompt": "A robot moves between two camera viewpoints.",
        "view_ids": [1, 1, 2, 2],
        "switch_frame": 2,
        "camera_poses": [[0.0] * 6 for _ in range(4)],
        "intrinsics": [[900.0, 900.0, 32.0, 24.0] for _ in range(4)],
    }


def test_controls_require_an_actual_viewpoint_transition(tmp_path: Path) -> None:
    """Reject a sidecar that labels frames but does not contain a switch."""

    path = tmp_path / "controls.json"
    invalid = _controls()
    invalid["view_ids"] = [1, 1, 1, 1]
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(switchworld.SwitchWorldError, match="transition"):
        switchworld._load_controls(path, target_frames=4)


def test_native_condition_is_bound_to_the_real_control_sidecar(tmp_path: Path) -> None:
    """Reject a native tensor whose camera/view schedule differs from controls."""

    torch = pytest.importorskip("torch")
    views = [1] * 7 + [2] * 8
    controls = {
        "schema": "npa.switchworld.controls.v1",
        "prompt": "A robot moves between two camera viewpoints.",
        "view_ids": views,
        "switch_frame": 7,
        "camera_poses": [
            [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
            for _ in views
        ],
        "intrinsics": [[900.0, 900.0, 320.0, 176.0] for _ in views],
    }
    condition = {
        "meta": {"condition_schema": "physicalworld_lingbot_v3_causal_multihot"},
        "view_id": torch.tensor(views),
        "switch_mask": torch.tensor([0] * 7 + [1] + [0] * 7),
        "camera_c2w": torch.eye(4).repeat(15, 1, 1),
        "camera_intrinsics": torch.tensor(controls["intrinsics"]),
    }
    path = tmp_path / "condition.pt"
    torch.save(condition, path)

    switchworld._validate_native_condition(path, controls)

    controls["intrinsics"][0][0] = 901.0
    with pytest.raises(switchworld.SwitchWorldError, match="intrinsics"):
        switchworld._validate_native_condition(path, controls)


def test_pair_and_rrd_use_decoded_media_not_manifests(tmp_path: Path) -> None:
    """Pair real source frames and independently validate the emitted RRD."""

    baseline = tmp_path / "baseline.mp4"
    adapted = tmp_path / "adapted.mp4"
    target = tmp_path / "target.mp4"
    _video(baseline, "testsrc=size=64x48:rate=4")
    _video(adapted, "testsrc2=size=64x48:rate=4")
    _video(target, "smptebars=size=64x48:rate=4")

    paired = switchworld._pair_videos(baseline, adapted, tmp_path / "paired.mp4")
    rrd = switchworld._build_rrd(
        baseline,
        adapted,
        target,
        {"schema": "npa.switchworld.real_frame_metrics.v1"},
        tmp_path / "switchworld.rrd",
        "switchworld-fixture",
    )

    assert paired["frame_count"] == 4
    assert paired["width"] == 128
    assert paired["mean_temporal_abs_delta"] > 0
    assert rrd["frame_count"] == 4
    assert rrd["rerun_verify"] == "passed"
    assert rrd["rerun_inspection"] == "passed"


def test_workflow_has_five_connected_real_stages() -> None:
    """Require the workflow's substantive source, inference, metric, and viz path."""

    spec = load_spec(WORKFLOW)
    raw = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    states = raw["states"]

    assert raw["config"]["source_overlay"] is True
    assert len(states) == 5
    assert spec.initial == "prepare-real-case"
    assert states["prepare-real-case"]["next"] == "generate-lingbot-baseline"
    assert states["generate-lingbot-baseline"]["next"] == "generate-adapter-switch"
    assert states["generate-adapter-switch"]["next"] == "measure-real-frames"
    assert states["measure-real-frames"]["next"] == "emit-paired-artifacts"
    assert states["emit-paired-artifacts"]["terminal"] is True

    for state in states.values():
        command = " ".join(state["run"]["argv"])
        assert "npa.workflows.switchworld" in command
        assert "echo" not in command
        assert "mock" not in command.lower()

    assert str(states["generate-lingbot-baseline"]["inputs"][0]["uri"]).endswith(
        "prepared_uri}}"
    )
    assert (
        states["generate-lingbot-baseline"]["inputs"][0]["schema"]
        == "npa.switchworld.prepared_bundle.v1"
    )
    assert "baseline.mp4" in str(states["measure-real-frames"]["inputs"])
    assert "adapted.mp4" in str(states["emit-paired-artifacts"]["inputs"])
    assert "switchworld.rrd" in str(states["emit-paired-artifacts"]["outputs"])


def test_canonical_adapter_selection_is_hash_pinned() -> None:
    """Keep the known identical alias from silently replacing canonical provenance."""

    assert switchworld.CANONICAL_ADAPTER_REPOSITORY == "PencilHu/SwitchWorld"
    assert (
        switchworld.CANONICAL_ADAPTER_REVISION
        == "5a01361ae1f9c9b1cfe115bef1c4d0377d922c1f"
    )
    assert set(switchworld.ADAPTER_FILES) == {
        "joint_high_rank128.pt",
        "joint_low_rank128.pt",
    }
    for digest, size in switchworld.ADAPTER_FILES.values():
        assert len(digest) == 64
        assert size > 6_000_000_000
