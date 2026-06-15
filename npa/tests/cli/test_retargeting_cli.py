from __future__ import annotations

import json
import re

import joblib
import numpy as np
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


def test_retargeting_registered_under_sonic() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "--help"])

    assert result.exit_code == 0
    assert "retargeting" in result.output


def test_removed_tools_not_advertised_in_workbench_help() -> None:
    result = runner.invoke(app, ["workbench", "--help"])

    assert result.exit_code == 0
    for removed in ("sim2real", "retargeting", "trigger", "sim2real-envgen", "data"):
        assert not re.search(rf"│\s+{re.escape(removed)}\s+", result.output)


def test_retargeting_command_help() -> None:
    for command in ("run", "workflow", "status", "list"):
        result = runner.invoke(app, ["workbench", "sonic", "retargeting", command, "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output


def test_retargeting_run_writes_real_motion_lib_and_metadata(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_motion = source_dir / "walk.pkl"
    joblib.dump(
        {
            "walk": {
                "root_trans_offset": np.zeros((4, 3), dtype=np.float32),
                "pose_aa": np.zeros((4, 30, 3), dtype=np.float32),
                "dof": np.zeros((4, 29), dtype=np.float32),
                "root_rot": np.zeros((4, 4), dtype=np.float32),
                "fps": 30,
            }
        },
        source_motion,
    )
    output_dir = tmp_path / "retargeted"

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "retargeting",
            "run",
            "--input-path",
            str(source_dir),
            "--output-path",
            str(output_dir),
            "--source-format",
            "motion-lib",
            "--embodiment",
            "unitree-g1",
            "--frame-rate",
            "30",
            "--max-frames",
            "2",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "retargeted"
    assert payload["artifact_kind"] == "robot_motion_lib"
    assert payload["source_format"] == "motion-lib"
    assert payload["motion_count"] == 1
    written = output_dir / "walk.pkl"
    metadata = output_dir / "retargeting_result.json"
    assert written.exists()
    assert metadata.exists()
    copied = joblib.load(written)
    assert copied["walk"]["dof"].shape[0] == 2
    assert json.loads(metadata.read_text(encoding="utf-8"))["embodiment"] == "unitree-g1"


def test_retargeting_respects_env_dry_run(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "retargeted"
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "retargeting",
            "run",
            "--input-path",
            "s3://bucket/motions/source/",
            "--output-path",
            str(output_dir),
            "--source-format",
            "motion-lib",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["dry_run"] is True
    assert not output_dir.exists()


def test_retargeting_rejects_negative_frame_limit() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "sonic",
            "retargeting",
            "run",
            "--input-path",
            "s3://bucket/motions/source/",
            "--output-path",
            "s3://bucket/out/",
            "--max-frames",
            "-1",
        ],
    )

    assert result.exit_code == 1
    assert "--max-frames must be non-negative" in result.output


def test_retargeting_workflow_path() -> None:
    result = runner.invoke(app, ["workbench", "sonic", "retargeting", "workflow", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workflow"] == "npa/workflows/workbench/skypilot/retargeting.yaml"
    assert payload["image_env"] == "NPA_RETARGETING_IMAGE"
    assert payload["image"].endswith("/npa-retargeting:0.1.1")
