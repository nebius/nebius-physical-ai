"""Keep the public sample graph executable with durable data and native GPU stages."""

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides

_SPEC = (
    Path(__file__).resolve().parents[4] / "workflows/main/rgbd-scan-to-policy-demo.yaml"
)
_ISAAC = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64
_CPU = "registry.example.invalid/npa-sonic@sha256:" + "b" * 64


def _operator_spec():
    return merge_config_overrides(
        load_spec(_SPEC), {"isaac_image": _ISAAC, "assembly_image": _CPU}
    )


def test_public_sample_plan_needs_no_operator_capture_or_cases():
    spec = _operator_spec()
    plan = build_plan(spec, run_id="sample-contract")
    assert [step.state for step in plan.steps] == [
        "sample",
        "reconstruct",
        "prepare",
        "cases",
        "physics",
        "policy",
        "train",
        "evaluate",
    ]
    for step in plan.steps:
        assert all(argument != "" for argument in step.argv)
    assert "npa.workbench.nurec.navigation_sample" in plan.steps[0].argv
    policy = plan.steps[5].argv
    assert policy[policy.index("--num-envs") + 1] == "4000"
    assert policy[policy.index("--iterations") + 1] == "500"
    assert policy[policy.index("--episode-steps") + 1] == "300"
    assert any(
        output["uri"].endswith("/reports/index.html")
        for output in plan.steps[-1].outputs
    )


def test_sample_render_preserves_cpu_geometry_and_rtx_native_runtime(monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    spec = _operator_spec()
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="sample-render"),
        run_id="sample-render",
        options=SkypilotRenderOptions(
            registry="registry.example.invalid", materialize_registry_secrets=False
        ),
    )
    tasks = [item for item in yaml.safe_load_all(rendered) if item and "run" in item]
    assert len(tasks) == 8
    gpu_indices = [
        index for index, task in enumerate(tasks) if "accelerators" in task["resources"]
    ]
    assert gpu_indices == [4, 6, 7]
    for index in gpu_indices:
        assert tasks[index]["resources"]["accelerators"] == "RTXPRO6000:1"
        assert "@sha256:" in tasks[index]["resources"]["image_id"]
        pod = tasks[index]["config"]["kubernetes"]["pod_config"]["spec"]
        assert pod["runtimeClassName"] == "nvidia"
        assert {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"} in pod[
            "containers"
        ][0]["env"]
    assert "usd-core==26.8" in tasks[2]["run"]
    assert all(task["envs"]["NPA_SRC_OVERLAY"] == "1" for task in tasks)
