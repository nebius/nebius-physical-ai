from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SONIC_TRAIN_STANDALONE_YAML = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "skypilot"
    / "sonic-train-standalone.yaml"
)


def _task_envs(plan) -> tuple[dict, dict]:
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    task = docs[1]
    return task["resources"], task["envs"]


def _task_doc(plan) -> dict:
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    return docs[1]


def _patch_nebius_token(monkeypatch: pytest.MonkeyPatch, token: str = "fresh-token") -> list[list[str]]:
    from npa.workbench.sonic import workflow as sonic_workflow

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{token}\n", stderr="")

    monkeypatch.setattr(sonic_workflow.subprocess, "run", fake_run)
    return calls


def test_sonic_materializer_adds_nebius_registry_auth_for_vm_tasks(monkeypatch) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    calls = _patch_nebius_token(monkeypatch)

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="cr.eu-north1.nebius.cloud/registry-id",
        gpu_target="h100",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, envs = _task_envs(plan)
    assert calls == [["nebius", "iam", "get-access-token"]]
    assert resources["cloud"] == "nebius"
    assert resources["accelerators"] == "H100:1"
    assert resources["cpus"] == 16
    assert resources["memory"] == 200
    assert "image_id" not in resources
    assert envs["SKYPILOT_DOCKER_USERNAME"] == "iam"
    assert envs["SKYPILOT_DOCKER_PASSWORD"] == "fresh-token"
    assert envs["SKYPILOT_DOCKER_SERVER"] == "cr.eu-north1.nebius.cloud"
    assert plan.registry_auth_username == "iam"
    assert plan.registry_auth_server == "cr.eu-north1.nebius.cloud"
    assert plan.registry_auth_source == "nebius-iam-token"
    assert "fresh-token" not in repr(plan)


def test_sonic_materializer_honors_byo_registry_auth_for_vm_tasks() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        registry_username="customer",
        registry_password="customer-token",
        registry_server="https://registry.example/",
        gpu_target="l40s",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    _, envs = _task_envs(plan)
    assert envs["SKYPILOT_DOCKER_USERNAME"] == "customer"
    assert envs["SKYPILOT_DOCKER_PASSWORD"] == "customer-token"
    assert envs["SKYPILOT_DOCKER_SERVER"] == "registry.example"
    assert plan.registry_auth_source == "explicit"


def test_sonic_materializer_can_enable_spot_for_vm_tasks() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        gpu_target="h100",
        region="eu-north1",
        use_spot=True,
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, _ = _task_envs(plan)
    assert resources["use_spot"] is True
    assert resources["region"] == "eu-north1"


def test_sonic_materializer_skips_registry_auth_for_kubernetes_targets(monkeypatch) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    calls = _patch_nebius_token(monkeypatch)

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="cr.eu-north1.nebius.cloud/registry-id",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, envs = _task_envs(plan)
    assert resources["cloud"] == "kubernetes"
    assert resources["image_id"] == "docker:cr.eu-north1.nebius.cloud/registry-id/npa-sonic:0.1.2-k8s"
    assert "SKYPILOT_DOCKER_PASSWORD" not in envs
    assert calls == []


@pytest.mark.parametrize(
    ("gpu_target", "accelerators", "cpus", "memory"),
    [
        ("h100", "H100:1", 16, 200),
        ("H200", "H200:1", 16, 200),
        ("L40S", "L40S:1", 16, 64),
        ("b200", "B200:1", 20, 224),
    ],
)
def test_sonic_materializer_defaults_vm_accelerators_by_gpu_target(
    gpu_target: str,
    accelerators: str,
    cpus: int,
    memory: int,
) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        gpu_target=gpu_target,
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, _ = _task_envs(plan)
    assert resources["cloud"] == "nebius"
    assert resources["accelerators"] == accelerators
    assert resources["cpus"] == cpus
    assert resources["memory"] == memory


def test_sonic_materializer_uses_default_vm_runtime_and_docker_payload() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        registry_username="customer",
        registry_password="customer-token",
        registry_server="registry.example",
        gpu_target="h100",
        region="eu-north1",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        env_overrides={"SONIC_RUN_REAL_TRAIN": "1", "SONIC_MAX_ITERATIONS": "1"},
    )

    task = _task_doc(plan)
    assert "image_id" not in task["resources"]
    assert task["resources"]["region"] == "eu-north1"
    assert task["envs"]["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic:0.1.2"
    assert task["envs"]["SONIC_RUN_REAL_TRAIN"] == "1"
    assert task["envs"]["SONIC_MAX_ITERATIONS"] == "1"
    assert '"${docker_cmd[@]}" login' in task["setup"]
    assert '"${docker_cmd[@]}" pull "registry.example/workbench/npa-sonic:0.1.2"' in task["setup"]
    assert "--gpus all" in task["run"]
    assert '"${docker_cmd[@]}" run' in task["run"]
    assert "--entrypoint /bin/bash" in task["run"]
    assert "/entrypoint.sh train" in task["run"]
    assert "sonic_proof_status.json" in task["run"]
    assert "python3 -m pip install --quiet boto3" in task["run"]
    assert "s3.upload_file" in task["run"]
    assert 'exit "${docker_status}"' in task["run"]
    assert (
        "AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN HF_TOKEN WANDB_API_KEY "
        "WANDB_DISABLED WANDB_DIR"
    ) in task["run"]
