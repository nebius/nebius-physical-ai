from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCRIPT = REPO_ROOT / "npa" / "scripts" / "capture_isaac_lab_scene_frames.py"
WORKFLOW = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "isaac-franka-capture-reason.yaml"
EXAMPLE = REPO_ROOT / "npa" / "examples" / "isaac_franka_token_factory_reason.py"
SAMPLE_FRAMES = REPO_ROOT / "docs" / "assets" / "hackathon" / "isaac-franka-lift-cube"


def test_capture_isaac_lab_scene_frames_render_only() -> None:
    result = subprocess.run(
        [sys.executable, str(CAPTURE_SCRIPT), "--render-only", "-o", "s3://bucket/prefix/"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["task"] == "Isaac-Lift-Cube-Franka-v0"
    assert payload["output_path"] == "s3://bucket/prefix/"


def test_isaac_franka_capture_reason_workflow_references_token_factory_key() -> None:
    text = WORKFLOW.read_text()
    assert "Isaac-Lift-Cube-Franka-v0" in text
    assert "NEBIUS_TOKEN_FACTORY_KEY" in text
    assert "capture_isaac_lab_scene_frames.py" in text
    assert "token-factory reason" in text
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert len(docs) >= 3  # header + 2 stages


def test_isaac_franka_sdk_example_and_sample_frames_exist() -> None:
    assert EXAMPLE.is_file()
    assert (SAMPLE_FRAMES / "frame_00.png").is_file()
    assert len(list(SAMPLE_FRAMES.glob("frame_*.png"))) >= 4
