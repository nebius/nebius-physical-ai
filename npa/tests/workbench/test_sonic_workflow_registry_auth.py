from __future__ import annotations

from pathlib import Path
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
# Frozen raw-task fixture, not a shipped template: what these exercise is the submit
# WRAPPER's materializer, which still accepts a customer's own SkyPilot YAML.
# See npa/tests/fixtures/skypilot/README.md.
SONIC_TRAIN_STANDALONE_YAML = (
    ROOT / "npa/tests/fixtures/skypilot/sonic-train-standalone.yaml"
)


def _task_docs(plan) -> tuple[dict, dict]:
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    task = docs[1]
    return task["resources"], task["envs"]


def test_sonic_materializer_rejects_quarantined_vm_image() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    with pytest.raises(ValueError, match="quarantined"):
        materialize_sonic_workflow(
            SONIC_TRAIN_STANDALONE_YAML,
            run_id="sonic-proof",
            registry="ghcr.io/nebius/nebius-physical-ai",
            gpu_target="h100",
            s3_endpoint="https://storage.example",
            s3_bucket="proof-bucket",
        )


def test_sonic_materializer_skips_registry_auth_for_kubernetes_docker_payload() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        registry_username="operator",
        registry_password="redacted-test-token",
        registry_server="https://registry.example/",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        env_overrides={"SONIC_PAYLOAD_MODE": "docker"},
    )

    resources, envs = _task_docs(plan)
    assert "image_id" not in resources
    assert envs["SONIC_PAYLOAD_MODE"] == "docker"
    assert "SKYPILOT_DOCKER_USERNAME" not in envs
    assert "SKYPILOT_DOCKER_PASSWORD" not in envs
    assert plan.registry_auth_source == ""
    assert "docker_login_if_configured" in plan.yaml_text
    assert "docker login" in plan.yaml_text


def test_sonic_materializer_skips_registry_auth_for_kubernetes_targets(
    monkeypatch,
) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="ghcr.io/nebius/nebius-physical-ai",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, envs = _task_docs(plan)
    assert resources["cloud"] == "kubernetes"
    assert "SKYPILOT_DOCKER_PASSWORD" not in envs


@pytest.mark.parametrize("gpu_target", ["h100", "H200", "L40S"])
def test_sonic_materializer_rejects_quarantined_gpu_targets(gpu_target: str) -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    with pytest.raises(ValueError, match="quarantined"):
        materialize_sonic_workflow(
            SONIC_TRAIN_STANDALONE_YAML,
            run_id="sonic-proof",
            registry="registry.example/workbench",
            gpu_target=gpu_target,
            s3_endpoint="https://storage.example",
            s3_bucket="proof-bucket",
        )


def test_sonic_materializer_ignores_spot_for_active_kubernetes_image() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-proof",
        registry="registry.example/workbench",
        gpu_target="gpu-rtx6000",
        use_spot=True,
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
    )

    resources, _ = _task_docs(plan)
    assert resources["cloud"] == "kubernetes"
    assert "use_spot" not in resources
