"""GR00T N1.7 reference workflow contract coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.cli.groot import (
    DEFAULT_MODEL,
    GROOT_FINETUNE_MANIFEST,
    GROOT_MODEL_VERSION,
    GROOT_REPO_REF,
    GROOT_RUNTIME_VERSION,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "groot-1-7-finetune.yaml"
)


def _option_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_groot_workflow_defaults_to_real_single_gpu_n1_7_training() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-single")
    step = plan.steps[0]

    assert spec.name == "groot-1-7-finetune"
    assert spec.config["base_model"] == DEFAULT_MODEL
    assert GROOT_MODEL_VERSION == "1.7"
    assert GROOT_RUNTIME_VERSION == "0.1.0"
    assert len(GROOT_REPO_REF) == 40
    assert step.tool_ref == "workbench.groot.finetune"
    assert step.resources_profile["accelerators"] == "H100:1"
    assert _option_value(step.argv, "--runtime") == "local"
    assert _option_value(step.argv, "--base-model") == DEFAULT_MODEL
    assert _option_value(step.argv, "--num-gpus") == "1"
    assert _option_value(step.argv, "--global-batch-size") == "1"
    assert _option_value(step.argv, "--run-id") == "groot-single"
    assert step.inputs == [
        {
            "uri": "s3://example-bucket/groot-1-7-finetune/groot-single/data/",
            "schema": "nvidia.groot.lerobot.v2",
        }
    ]
    assert step.outputs == [
        {
            "uri": (
                "s3://example-bucket/groot-1-7-finetune/groot-single/"
                f"checkpoints/{GROOT_FINETUNE_MANIFEST}"
            ),
            "schema": "npa.groot.finetune.v1",
        }
    ]


@pytest.mark.parametrize("gpu_count", [2, 3, 4])
def test_groot_workflow_gpu_count_reaches_plan_scheduler_and_render(
    gpu_count: int,
) -> None:
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id=f"groot-{gpu_count}gpu",
        config_overrides={"gpu_count": str(gpu_count)},
        render_options=SkypilotRenderOptions(
            registry="cr.example.invalid/workbench",
            materialize_registry_secrets=False,
        ),
    )
    try:
        step = prepared.plan.steps[0]
        expected_accelerators = f"H100:{gpu_count}"
        assert step.resources_profile["accelerators"] == expected_accelerators
        assert _option_value(step.argv, "--num-gpus") == str(gpu_count)
        assert _option_value(step.argv, "--global-batch-size") == str(gpu_count)

        scheduler = build_scheduler_plan(
            prepared.spec,
            prepared.plan.steps,
            run_id=f"groot-{gpu_count}gpu",
        )
        assert scheduler["tasks"][0]["resources"]["accelerators"] == (
            expected_accelerators
        )

        documents = [
            doc
            for doc in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            )
            if doc
        ]
        task = documents[1]
        assert task["resources"]["accelerators"] == expected_accelerators
        assert f"--num-gpus {gpu_count}" in task["run"]
        assert f"--global-batch-size {gpu_count}" in task["run"]
        assert f"--run-id groot-{gpu_count}gpu" in task["run"]
    finally:
        prepared.temp_dir.cleanup()


@pytest.mark.parametrize("gpu_count", ["0", "-1"])
def test_groot_workflow_rejects_non_positive_gpu_count(gpu_count: str) -> None:
    with pytest.raises(NpaWorkflowError, match="accelerator count must be >= 1"):
        prepare_npa_workflow_for_submit(
            SPEC_PATH,
            run_id="groot-invalid",
            config_overrides={"gpu_count": gpu_count},
            render_options=SkypilotRenderOptions(
                registry="cr.example.invalid/workbench",
                materialize_registry_secrets=False,
            ),
        )
