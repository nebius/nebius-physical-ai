"""The capture shim and the spec that replaced `isaac-franka-capture-reason.yaml`.

The implementation moved into `npa.workflows.isaac_capture` so a toolRef can run it in a pod with
no repo checkout; `npa/scripts/capture_isaac_lab_scene_frames.py` stays as a shim for the path
documented in `docs/hackathon-isaac-token-factory.md`. Behavioural tests for the module itself
live in `tests/workflows/test_isaac_capture.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCRIPT = REPO_ROOT / "npa" / "scripts" / "capture_isaac_lab_scene_frames.py"
SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "isaac-franka-capture-reason.yaml"
)
RETIRED_TEMPLATE = (
    REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "isaac-franka-capture-reason.yaml"
)
EXAMPLE = REPO_ROOT / "npa" / "examples" / "isaac_franka_token_factory_reason.py"
SAMPLE_FRAMES = REPO_ROOT / "docs" / "assets" / "hackathon" / "isaac-franka-lift-cube"


def test_the_shim_still_works_for_a_checkout() -> None:
    result = subprocess.run(
        [sys.executable, str(CAPTURE_SCRIPT), "--render-only", "-o", "s3://bucket/prefix/"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT / "npa",
        env={"PYTHONPATH": str(REPO_ROOT / "npa" / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["task"] == "Isaac-Lift-Cube-Franka-v0"
    assert payload["output_path"] == "s3://bucket/prefix/"


def test_the_spec_captures_on_a_gpu_then_reasons_on_cpu() -> None:
    """Live proof: job 283, both stages SUCCEEDED and the plan describes the actual scene.

    See EVIDENCE.md §R37 — the reasoner reported "a robot arm mounted on a black table ... a
    small, colorful cube", which is what the frames show.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="isaac-capture-test")
    steps = {step.state: step for step in plan.steps if step.argv}

    assert sorted(steps) == ["capture", "reason"]
    assert "accelerators" in spec.resources[steps["capture"].resources]
    assert "accelerators" not in spec.resources[steps["reason"].resources]

    capture = steps["capture"].argv
    reason = steps["reason"].argv
    assert capture[capture.index("--task") + 1] == "Isaac-Lift-Cube-Franka-v0"
    # The reasoner reads exactly where the capture wrote.
    assert reason[reason.index("--input-path") + 1] == capture[capture.index("--output-path") + 1]
    assert "NEBIUS_API_KEY" not in " ".join(reason)


def test_the_retired_template_is_gone() -> None:
    assert not RETIRED_TEMPLATE.exists(), "isaac-franka-capture-reason.yaml came back"


def test_isaac_franka_sdk_example_and_sample_frames_exist() -> None:
    assert EXAMPLE.is_file()
    assert (SAMPLE_FRAMES / "frame_00.png").is_file()
    assert len(list(SAMPLE_FRAMES.glob("frame_*.png"))) >= 4
