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


def test_groot_workflow_defaults_to_validated_real_eight_gpu_n1_7_training() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-default-8gpu")
    step = plan.steps[2]

    assert spec.name == "groot-1-7-finetune"
    assert [item.state for item in plan.steps] == [
        "prepare_split",
        "baseline_eval",
        "finetune",
        "posttrain_eval",
        "compare_learning",
        "emit_mcap",
        "emit_rrd",
        "publish",
    ]
    assert [item.tool_ref for item in plan.steps] == [
        "workflow.groot.prepare_split",
        "workbench.groot.baseline_eval",
        "workbench.groot.finetune",
        "workbench.groot.posttrain_eval",
        "workflow.groot.compare_learning",
        "workflow.groot.emit_learning_mcap",
        "workflow.groot.emit_learning_rrd",
        "workflow.groot.publish_learning",
    ]
    assert spec.config["base_model"] == DEFAULT_MODEL
    assert GROOT_MODEL_VERSION == "1.7"
    assert GROOT_RUNTIME_VERSION == "0.1.0"
    assert len(GROOT_REPO_REF) == 40
    assert step.tool_ref == "workbench.groot.finetune"
    assert step.resources_profile["accelerators"] == "H100:8"
    assert _option_value(step.argv, "--runtime") == "local"
    assert _option_value(step.argv, "--base-model") == DEFAULT_MODEL
    assert _option_value(step.argv, "--num-gpus") == "8"
    assert _option_value(step.argv, "--nccl-transport") == "auto"
    assert _option_value(step.argv, "--global-batch-size") == "8"
    assert _option_value(step.argv, "--run-id") == "groot-default-8gpu"
    assert step.inputs == [
        {
            "uri": "s3://example-bucket/groot-1-7-finetune/groot-default-8gpu/data/train/",
            "schema": "nvidia.groot.lerobot.v2.train-split",
        }
    ]
    assert step.outputs == [
        {
            "uri": (
                "s3://example-bucket/groot-1-7-finetune/groot-default-8gpu/"
                f"checkpoints/posttrain/{GROOT_FINETUNE_MANIFEST}"
            ),
            "schema": "npa.groot.finetune.v1",
        },
        {
            "uri": (
                "s3://example-bucket/groot-1-7-finetune/groot-default-8gpu/"
                "checkpoints/posttrain/"
            ),
            "schema": "nvidia.groot.n1.7.finetuned-checkpoint",
        },
    ]


@pytest.mark.parametrize("gpu_count", [1, 2, 3, 4, 8])
def test_groot_workflow_gpu_count_reaches_plan_scheduler_and_render(
    gpu_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id=f"groot-{gpu_count}gpu",
        config_overrides={
            "gpu_count": str(gpu_count),
            "nccl_transport": "socket",
        },
        render_options=SkypilotRenderOptions(
            registry="cr.example.invalid/workbench",
            materialize_registry_secrets=False,
        ),
    )
    try:
        step = prepared.plan.steps[2]
        expected_accelerators = f"H100:{gpu_count}"
        assert step.resources_profile["accelerators"] == expected_accelerators
        assert _option_value(step.argv, "--num-gpus") == str(gpu_count)
        assert _option_value(step.argv, "--nccl-transport") == "socket"
        assert _option_value(step.argv, "--global-batch-size") == "8"

        scheduler = build_scheduler_plan(
            prepared.spec,
            prepared.plan.steps,
            run_id=f"groot-{gpu_count}gpu",
        )
        assert scheduler["tasks"][2]["resources"]["accelerators"] == (
            expected_accelerators
        )
        assert [task["name"] for task in scheduler["tasks"]] == [
            "prepare_split",
            "baseline_eval",
            "finetune",
            "posttrain_eval",
            "compare_learning",
            "emit_mcap",
            "emit_rrd",
            "publish",
        ]

        documents = [
            doc
            for doc in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            )
            if doc
        ]
        task = documents[3]
        assert len(documents) == 9
        assert task["resources"]["accelerators"] == expected_accelerators
        assert task["config"]["kubernetes"]["pod_config"]["spec"][
            "securityContext"
        ] == {"runAsUser": 0, "runAsGroup": 0}
        assert documents[2]["config"]["kubernetes"]["pod_config"]["spec"][
            "securityContext"
        ] == {"runAsUser": 0, "runAsGroup": 0}
        assert f"--num-gpus {gpu_count}" in task["run"]
        assert "--nccl-transport socket" in task["run"]
        assert "--global-batch-size 8" in task["run"]
        assert f"--run-id groot-{gpu_count}gpu" in task["run"]
        assert "groot_learning prepare-split" in documents[1]["run"]
        assert "groot_learning baseline-eval" in documents[2]["run"]
        assert "groot_learning posttrain-eval" in documents[4]["run"]
        assert "transformers==4.57.3" in documents[2]["setup"]
        assert "transformers==4.57.3" in documents[3]["setup"]
        assert "transformers==4.57.3" in documents[4]["setup"]
        assert "groot_learning compare-learning" in documents[5]["run"]
        assert "groot_learning emit-mcap" in documents[6]["run"]
        assert "groot_learning emit-rrd" in documents[7]["run"]
        assert "groot_learning publish" in documents[8]["run"]
        assert "pyarrow>=15,<22" in documents[1]["setup"]
        assert "[viz]" in documents[7]["setup"]
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
