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
    / "npa/workflows/workbench/npa-workflows/cosmos3-super-h200-benchmark.yaml"
)


def test_h200_workflow_is_fixed_full_node_primary_sweep() -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert raw["resources"]["h200-node"]["accelerators"] == "H200:8"
    assert raw["resources"]["h200-node"]["image"] == "{{config.runtime_image}}"
    assert raw["config"]["gpu_family"] == "H200"
    assert raw["config"]["topologies"] == "1x8,2x4,4x2,8x1"
    assert raw["config"]["attempts"] == "24"
    assert raw["config"]["suite"] == "primary"
    assert raw["states"]["benchmark"]["toolRef"] == (
        "workbench.cosmos3.super_benchmark"
    )
    pod_spec = raw["resources"]["h200-node"]["kubernetes"]["pod_config"]["spec"]
    assert pod_spec["imagePullSecrets"] == [
        {"name": "{{config.image_pull_secret}}"}
    ]
    assert pod_spec["volumes"][0]["emptyDir"]["sizeLimit"] == "32Gi"
    assert pod_spec["containers"] == [
        {
            "name": "ray-node",
            "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
        }
    ]


def test_h200_workflow_renders_family_and_real_command(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src")
    wrapper = "registry.example.invalid/operator/npa-cosmos3-super-benchmark@sha256:" + (
        "2" * 64
    )
    spec = load_spec_for_submit(
        SPEC_PATH, config_overrides={"runtime_image": wrapper}
    )
    plan = build_plan(spec, run_id="cosmos3-super-h200-test")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="cosmos3-super-h200-test",
        options=SkypilotRenderOptions(),
    )
    docs = [item for item in yaml.safe_load_all(rendered) if item]
    assert docs[1]["resources"]["image_id"] == f"docker:{wrapper}"
    assert docs[1]["resources"]["accelerators"] == "H200:8"
    assert "npa workbench cosmos3 super-benchmark" in docs[1]["run"]
    assert "--gpu-family H200" in docs[1]["run"]
    assert "--attempts 24" in docs[1]["run"]
    assert "--suite primary" in docs[1]["run"]
    hints = secret_env_hints_for_plan(plan.steps)
    assert "HF_TOKEN" in hints
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE" in hints
