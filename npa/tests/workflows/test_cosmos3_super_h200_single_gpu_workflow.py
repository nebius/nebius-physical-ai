from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
    secret_env_hints_for_plan,
)
from npa.orchestration.npa_workflow.submit import load_spec_for_submit


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    REPO_ROOT
    / "npa/workflows/workbench/npa-workflows/cosmos3-super-h200-single-gpu.yaml"
)


def test_h200_single_gpu_workflow_preserves_exact_isolated_contract() -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    resources = raw["resources"]["h200-single-gpu"]
    assert resources["accelerators"] == "H200:1"
    assert resources["cpus"] == 16
    assert resources["memory"] == "200Gi"
    assert raw["config"]["gpu_family"] == "H200"
    assert raw["config"]["topologies"] == "1x1"
    assert raw["config"]["attempts"] == "24"
    assert raw["config"]["suite"] == "h200-single-gpu"
    assert raw["states"]["benchmark"]["outputs"][0]["schema"] == (
        "npa.cosmos3-super.h200-single-gpu-validation.v1"
    )
    pod_spec = resources["kubernetes"]["pod_config"]["spec"]
    assert pod_spec["volumes"][0]["emptyDir"]["sizeLimit"] == "32Gi"


def test_h200_single_gpu_workflow_renders_tp1_command(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src")
    wrapper = "registry.example.invalid/operator/npa-cosmos3-super-benchmark@sha256:" + (
        "3" * 64
    )
    spec = load_spec_for_submit(
        SPEC_PATH, config_overrides={"runtime_image": wrapper}
    )
    plan = build_plan(spec, run_id="cosmos3-super-h200-single-test")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="cosmos3-super-h200-single-test",
        options=SkypilotRenderOptions(),
    )
    docs = [item for item in yaml.safe_load_all(rendered) if item]
    task = docs[1]
    assert task["resources"]["image_id"] == f"docker:{wrapper}"
    assert task["resources"]["accelerators"] == "H200:1"
    assert task["resources"]["cpus"] == "16+"
    assert task["resources"]["memory"] == "200+"
    assert "--topologies 1x1" in task["run"]
    assert "--suite h200-single-gpu" in task["run"]
    assert "--attempts 24" in task["run"]
    hints = secret_env_hints_for_plan(plan.steps)
    assert "HF_TOKEN" in hints
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE" in hints
