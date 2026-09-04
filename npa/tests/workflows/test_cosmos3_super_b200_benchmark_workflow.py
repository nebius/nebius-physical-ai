from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan
from npa.orchestration.npa_workflow.submit import load_spec_for_submit
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
    secret_env_hints_for_plan,
)
from npa.workbench.cosmos.super_benchmark import IMAGE, MODEL_REVISION, WORKLOAD


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    REPO_ROOT
    / "npa/workflows/workbench/npa-workflows/cosmos3-super-b200-benchmark.yaml"
)


def test_workflow_is_fixed_full_node_primary_sweep() -> None:
    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert raw["resources"]["b200-node"]["accelerators"] == "B200:8"
    assert raw["resources"]["b200-node"]["image"] == "{{config.runtime_image}}"
    assert raw["resources"]["b200-node"]["kubernetes"]["pod_config"]["spec"][
        "imagePullSecrets"
    ] == [{"name": "{{config.image_pull_secret}}"}]
    wrapper = REPO_ROOT / "npa/docker/workbench/cosmos3-super-benchmark/Dockerfile"
    assert IMAGE in wrapper.read_text(encoding="utf-8")
    assert raw["config"]["topologies"] == "1x8,2x4,4x2,8x1"
    assert raw["config"]["attempts"] == "24"
    assert raw["config"]["suite"] == "primary"
    assert raw["config"]["gpu_family"] == "B200"
    assert raw["states"]["benchmark"]["toolRef"] == "workbench.cosmos3.super_benchmark"
    assert raw["resources"]["b200-node"]["kubernetes"]["pod_config"]["spec"][
        "volumes"
    ][0]["emptyDir"]["sizeLimit"] == "32Gi"
    assert raw["resources"]["b200-node"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ] == [
        {
            "name": "ray-node",
            "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
        }
    ]
    assert MODEL_REVISION in SPEC_PATH.parent.parent.parent.parent.joinpath(
        "src/npa/workbench/cosmos/super_benchmark.py"
    ).read_text(encoding="utf-8")
    assert WORKLOAD["guardrails"] is False


def test_workflow_renders_exact_vendor_digest_and_real_command(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src")
    wrapper = "registry.example.invalid/operator/npa-cosmos3-super-benchmark@sha256:" + (
        "1" * 64
    )
    spec = load_spec_for_submit(
        SPEC_PATH, config_overrides={"runtime_image": wrapper}
    )
    plan = build_plan(spec, run_id="cosmos3-super-test")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="cosmos3-super-test",
        options=SkypilotRenderOptions(),
    )
    docs = [item for item in yaml.safe_load_all(rendered) if item]
    assert docs[1]["resources"]["image_id"] == f"docker:{wrapper}"
    assert docs[1]["resources"]["accelerators"] == "B200:8"
    assert docs[1]["config"]["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "name"
    ] == "ray-node"
    assert "npa workbench cosmos3 super-benchmark" in docs[1]["run"]
    assert "--attempts 24" in docs[1]["run"]
    assert "--suite primary" in docs[1]["run"]
    assert "--gpu-family B200" in docs[1]["run"]
    hints = secret_env_hints_for_plan(plan.steps)
    assert "HF_TOKEN" in hints
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE" in hints


def test_workflow_can_select_complete_b200_suite(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src")
    spec = load_spec_for_submit(
        SPEC_PATH,
        config_overrides={
            "runtime_image": (
                "registry.example.invalid/operator/benchmark@sha256:" + "3" * 64
            ),
            "suite": "b200-full",
        },
    )
    plan = build_plan(spec, run_id="cosmos3-super-full-test")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="cosmos3-super-full-test",
        options=SkypilotRenderOptions(),
    )
    docs = [item for item in yaml.safe_load_all(rendered) if item]
    assert "--suite b200-full" in docs[1]["run"]
