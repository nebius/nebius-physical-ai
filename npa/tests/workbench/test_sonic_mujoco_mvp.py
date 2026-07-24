from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SONIC_MVP_YAML = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "skypilot"
    / "sonic-locomotion-finetuning.yaml"
)


def _patch_registry_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.workbench.sonic import workflow as sonic_workflow

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="fresh-test-token\n", stderr="")

    monkeypatch.setattr(sonic_workflow.subprocess, "run", fake_run)


def _docs(plan) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc]


def test_h100_selects_combined_sonic_mujoco_image() -> None:
    from npa.deploy.images import container_image_for_tool, sonic_image_entry

    entry = sonic_image_entry(gpu_target="h100")
    assert entry["id"] == "sonic-mujoco-h100-mvp"
    assert entry["name"] == "npa-sonic-mujoco"
    assert entry["tag"] == "0.1.3-mvp"
    assert (
        container_image_for_tool("sonic", registry="registry.example/workbench", gpu_target="h100")
        == "registry.example/workbench/npa-sonic-mujoco:0.1.3-mvp"
    )


def test_sonic_mvp_materializer_sets_h100_spot_region_and_docker_payload(monkeypatch) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    _patch_registry_auth(monkeypatch)
    plan = materialize_sonic_workflow(
        SONIC_MVP_YAML,
        run_id="sonic-mvp-proof",
        registry="registry.example/workbench",
        registry_server="registry.example",
        registry_username="operator",
        registry_password="redacted-test-token",
        gpu_target="h100",
        use_spot=True,
        s3_endpoint="https://storage.eu-north1.nebius.cloud",
        s3_bucket="proof-bucket",
        s3_prefix="sonic-mvp-proof/sonic-mvp-proof",
        env_overrides={"SONIC_PAYLOAD_MODE": "docker", "SONIC_MAX_ITERATIONS": "1"},
    )

    docs = _docs(plan)
    assert docs[0] == {"name": "sonic-locomotion-finetuning", "execution": "serial"}
    assert [doc["name"] for doc in docs[1:]] == [
        "sonic-retarget-motion",
        "sonic-g1-finetune",
        "sonic-mujoco-eval",
    ]
    retarget = docs[1]
    assert retarget["resources"]["cloud"] == "kubernetes"
    assert retarget["resources"]["image_id"] == "docker:registry.example/workbench/npa-retargeting:0.1.1"
    assert retarget["envs"]["AWS_PROFILE"] == "nebius"
    assert retarget["envs"]["AWS_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    for task in docs[2:]:
        resources = task["resources"]
        envs = task["envs"]
        assert "image_id" not in resources
        assert resources["cloud"] == "nebius"
        assert resources["region"] == "eu-north1"
        assert resources["accelerators"] == "H100:1"
        assert resources["use_spot"] is True
        assert envs["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic-mujoco:0.1.3-mvp"
        assert envs["SONIC_PAYLOAD_MODE"] == "docker"
        assert envs["SONIC_GPU_TYPE"] == "h100"
        assert envs["SONIC_IMAGE_VARIANT"] == "sonic-mujoco-h100-mvp"
        assert envs["SKYPILOT_DOCKER_USERNAME"] == "operator"
        assert envs["SKYPILOT_DOCKER_PASSWORD"] == "redacted-test-token"
        assert envs["SKYPILOT_DOCKER_SERVER"] == "registry.example"

    train_env = docs[2]["envs"]
    eval_env = docs[3]["envs"]
    assert train_env["SONIC_RUN_REAL_TRAIN"] == "1"
    assert train_env["AWS_PROFILE"] == "nebius"
    assert eval_env["AWS_PROFILE"] == "nebius"
    assert train_env["SONIC_TRAIN_MODE"] == "finetune"
    assert train_env["SONIC_CHECKPOINT_PATH"] == "sonic_release/last.pt"
    assert train_env["SONIC_RUNTIME_INSTALL_TRAINING_DEPS"] == "1"
    assert train_env["WANDB_MODE"] == "disabled"
    assert train_env["WANDB_DISABLED"] == "true"
    assert train_env["SONIC_OUTPUT_PREFIX"] == "sonic-mvp-proof/sonic-mvp-proof/training/"
    assert eval_env["SONIC_OUTPUT_PREFIX"] == "sonic-mvp-proof/sonic-mvp-proof/mujoco-eval/"
    assert eval_env["SONIC_FINE_TUNED_CHECKPOINT_URI"] == (
        "s3://proof-bucket/sonic-mvp-proof/sonic-mvp-proof/training/checkpoints/last.pt"
    )
    assert '[ "${S3_BUCKET}" = "proof-bucket" ]' not in docs[2]["run"]
    assert '[ "${S3_BUCKET}" = "proof-bucket" ]' not in docs[3]["run"]
    assert "placeholder_bucket=" in docs[2]["run"]
    assert "placeholder_bucket=" in docs[3]["run"]
    assert "sudo -E docker" in docs[2]["setup"]
    assert "sudo -E docker" in docs[3]["setup"]
    assert "docker_cmd run --rm --gpus all" in docs[2]["run"]
    assert "docker_cmd run --rm --gpus all" in docs[3]["run"]
    assert "-e AWS_PROFILE" in docs[2]["run"]
    assert "-e AWS_PROFILE" in docs[3]["run"]
    assert "--entrypoint /bin/bash" in docs[2]["run"]
    assert "pip install --user --no-cache-dir" in docs[2]["run"]
    assert "\"open3d>=0.18,<0.20\"" in docs[2]["run"]
    assert "\"vector-quantize-pytorch>=1.14,<2\"" in docs[2]["run"]
    assert "-e WANDB_MODE" in docs[2]["run"]
    assert "-e WANDB_DISABLED" in docs[2]["run"]
    assert "NPA_SONIC_OUTPUT" in docs[3]["run"]
    assert "SONIC_MUJOCO_METRICS_PATH" in docs[3]["run"]
    assert "mujoco_eval_metrics.json" in docs[3]["run"]
    assert "mujoco-eval" in docs[3]["run"]
    assert plan.region == "eu-north1"


def test_sonic_mvp_materializer_rejects_me_west1() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    with pytest.raises(ValueError, match="exclude me-west1"):
        materialize_sonic_workflow(
            SONIC_MVP_YAML,
            run_id="sonic-mvp-proof",
            registry="registry.example/workbench",
            gpu_target="h100",
            s3_endpoint="https://storage.eu-north1.nebius.cloud",
            s3_bucket="proof-bucket",
            region="me-west1",
        )
