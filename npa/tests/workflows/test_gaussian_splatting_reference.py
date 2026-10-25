"""Check reference-workflow prerequisites and the standalone YAML handoff."""

import json
from pathlib import Path
import re
import subprocess

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/gaussian-splatting-reference.yaml"
HTML = ROOT / "docs/demos/lyra-splatting-overview.html"


@pytest.mark.parametrize(
    "image,message",
    [
        (None, "Provide an immutable compatible training image"),
        ("synthetic-training-runtime:latest", "Training image must be digest-pinned"),
    ],
)
def test_runtime_guard_fails_before_camera_or_storage_commands(image, message):
    spec = load_spec(SPEC)
    if image:
        spec.config["training_image"] = image
    command = build_plan(spec, run_id="unit-prerequisites").steps[0].shell
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("stride", [0, 1, -1, "invalid"])
def test_bad_evaluation_stride_fails_before_source_or_gpu_checks(stride):
    spec = load_spec(SPEC)
    spec.config["test_every"] = stride
    command = build_plan(spec, run_id="unit-stride").steps[1].shell
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Evaluation stride must be at least two" in result.stderr


def test_downloadable_yaml_matches_the_canonical_workflows():
    content = re.search(
        r'<script type="application/json" id="workflow-data">(.*?)</script>',
        HTML.read_text(),
        flags=re.S,
    )
    assert content
    embedded = json.loads(content.group(1))
    assert embedded == {
        "lyra": (ROOT / "workflows/testing/lyra-reconstruction.yaml").read_text(),
        "splat": SPEC.read_text(),
    }


def test_resolved_native_commands_have_valid_shell_syntax():
    plan = build_plan(load_spec(SPEC), run_id="unit-shell")
    for step in plan.steps:
        subprocess.run(["bash", "-n"], input=step.shell, text=True, check=True)
