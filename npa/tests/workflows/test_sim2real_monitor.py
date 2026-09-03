"""Tests for Sim2Real workflow status monitor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from npa.workflows.sim2real.monitor import (
    OperatorConfig,
    _stage_states,
    get_sim2real_workflow_status,
    load_operator_config,
    normalize_staged_run_id,
    orchestrator_job_name,
)


def _mock_s3_client(
    existing_keys: set[str], *, prefixes: set[str] | None = None
) -> MagicMock:
    prefixes = prefixes or set()

    client = MagicMock()

    def head_object(Bucket, Key):  # noqa: N803
        if Key in existing_keys:
            return {}
        import botocore.exceptions

        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "not found"}},
            "HeadObject",
        )

    def list_objects_v2(Bucket, Prefix, MaxKeys=1):  # noqa: N803
        del Bucket, MaxKeys
        if Prefix in prefixes or any(key.startswith(Prefix) for key in existing_keys):
            return {"KeyCount": 1}
        return {"KeyCount": 0}

    def get_object(Bucket, Key):  # noqa: N803
        del Bucket
        body = MagicMock()
        body.read.return_value = existing_keys[Key].encode("utf-8")
        return {"Body": body}

    client._s3.head_object.side_effect = head_object
    client._s3.list_objects_v2.side_effect = list_objects_v2
    client._s3.get_object.side_effect = get_object
    return client


def _write_operator_config(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "storage": {
            "bucket": "s3://operator-bucket/run-prefix",
            "endpoint_url": "https://storage.example",
            "k8s_context": "operator-context",
        }
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("registry_config", "expected"),
    [
        (
            {"container_registry": "registry.example/container"},
            "registry.example/container",
        ),
        (
            {"storage": {"registry": "registry.example/storage"}},
            "registry.example/storage",
        ),
        ({"registry": "registry.example/top-level"}, "registry.example/top-level"),
    ],
)
def test_explicit_operator_pack_preserves_legacy_registry_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry_config: dict[str, object],
    expected: str,
) -> None:
    config_path = tmp_path / "operator-pack.yaml"
    storage_registry = registry_config.get("storage", {})
    non_storage_config = {
        key: value for key, value in registry_config.items() if key != "storage"
    }
    storage = {
        "bucket": "s3://operator-bucket/run-prefix",
        "endpoint_url": "https://storage.example",
        "k8s_context": "operator-context",
        **storage_registry,
    }
    _write_operator_config(config_path, storage=storage, **non_storage_config)
    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", str(config_path))
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/ambient")
    monkeypatch.delenv("NPA_SIM2REAL_REGISTRY", raising=False)

    operator = load_operator_config()

    assert operator.registry == expected


def test_explicit_operator_registry_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "operator-pack.yaml"
    _write_operator_config(
        config_path,
        container_registry="registry.example/container/",
        registry="registry.example/top-level",
        storage={
            "bucket": "operator-bucket",
            "k8s_context": "operator-context",
            "registry": "registry.example/storage",
        },
    )
    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", str(config_path))
    monkeypatch.setenv("NPA_SIM2REAL_REGISTRY", "registry.example/environment/")

    assert load_operator_config().registry == "registry.example/environment"

    monkeypatch.delenv("NPA_SIM2REAL_REGISTRY")
    assert load_operator_config().registry == "registry.example/container"


def test_implicit_operator_config_cannot_redirect_public_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

    config_path = tmp_path / ".npa" / "config.yaml"
    _write_operator_config(
        config_path,
        container_registry="registry.invalid/container",
        registry="registry.invalid/top-level",
        storage={
            "bucket": "operator-bucket",
            "k8s_context": "operator-context",
            "registry": "registry.invalid/storage",
        },
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NPA_SIM2REAL_OPERATOR_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_REGISTRY", raising=False)
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/environment")

    assert load_operator_config().registry == DEFAULT_CONTAINER_REGISTRY


def test_missing_explicit_operator_config_does_not_fall_back_to_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_operator_config(tmp_path / ".npa" / "config.yaml")
    missing_path = tmp_path / "missing-operator-pack.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", str(missing_path))

    with pytest.raises(ValueError, match=r"missing operator config .*missing-operator-pack"):
        load_operator_config()


def test_unreadable_explicit_operator_config_does_not_fall_back_to_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_operator_config(tmp_path / ".npa" / "config.yaml")
    config_path = tmp_path / "operator-pack.yaml"
    _write_operator_config(config_path)
    config_path.chmod(0)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", str(config_path))

    try:
        with pytest.raises(ValueError, match="unable to read operator config"):
            load_operator_config()
    finally:
        config_path.chmod(0o600)


def test_stage_states_prefers_workflow_state_over_missing_s3_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_state = {
        "status": "preamble_completed",
        "updated_at": "2026-06-18T18:08:18Z",
        "train_envs_uri": "s3://demo-bucket/sim2real-b/run/envs/train/envs.jsonl",
        "env_count": 10000,
        "components": [
            {"name": "stage_02_assets", "tier": "WORKS"},
            {"name": "stage_03_augment", "tier": "WORKS"},
            {"name": "stage_04_06_env_gen_split_tokens", "tier": "WORKS"},
        ],
        "stage_records": [
            {
                "path": "/tmp/run/stage_01_trigger/trigger.json",
                "payload": {"stage": 1, "schema": "npa.sim2real.trigger.v1"},
            }
        ],
    }
    state_key = "sim2real-b/run-1/state/workflow_state.json"
    client = _mock_s3_client(
        {
            state_key: json.dumps(workflow_state),
            "sim2real-b/run-1/augment/cosmos2-transfer-result.json": "{}",
        }
    )

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    for stage_name in (
        "stage_01_trigger",
        "stage_02_assets",
        "stage_03_augment",
        "stage_04_envs_raw",
        "stage_05_envs_train",
        "stage_06_tokens",
    ):
        assert stages[stage_name]["state"] == "SUCCEEDED", stage_name
    assert stages["stage_02_assets"]["source"] == "workflow_state_status"


def test_stage_states_detects_consumed_asset_specs_on_s3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = {
        "sim2real-b/run-1/stage_02_assets/consumed_scene_spec.json",
        "sim2real-b/run-1/stage_02_assets/consumed_robot_spec.json",
    }
    client = _mock_s3_client(keys)

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    assert stages["stage_02_assets"]["state"] == "SUCCEEDED"
    assert stages["stage_02_assets"]["source"] == "s3_artifact"


def test_stage_states_augment_requires_cosmos2_transfer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = {"sim2real-b/run-1/augment/manifest.json"}
    client = _mock_s3_client(keys)

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    assert stages["stage_03_augment"]["state"] == "PENDING"

    keys.add("sim2real-b/run-1/augment/cosmos2-transfer-result.json")
    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    assert stages["stage_03_augment"]["state"] == "SUCCEEDED"


def test_stage_states_tokens_succeeds_when_folded_into_envgen_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = {
        "sim2real-b/run-1/envs/train/envs.jsonl",
        "sim2real-b/run-1/envs/heldout/envs.jsonl",
    }
    client = _mock_s3_client(keys)

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    assert stages["stage_06_tokens"]["state"] == "SUCCEEDED"
    assert stages["stage_05_envs_train"]["state"] == "SUCCEEDED"


def test_stage_states_infers_trigger_from_later_stage_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = {"sim2real-b/run-1/augment/cosmos2-transfer-result.json"}
    client = _mock_s3_client(keys)

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    stages = _stage_states(
        bucket="demo-bucket",
        run_id="run-1",
        s3_prefix="sim2real-b",
        endpoint="https://storage.example",
    )
    assert stages["stage_01_trigger"]["state"] == "SUCCEEDED"
    assert stages["stage_01_trigger"]["source"] == "inferred_from_later_stage"


def test_orchestrator_job_name() -> None:
    assert orchestrator_job_name("demo-run") == "sim2real-demo-run"
    assert (
        orchestrator_job_name("sim2real-pr258-final-gpu-20260805t180511z-2900ceae")
        == "sim2real-pr258-final-gpu-20260805t180511z-2900ceae"
    )


def test_normalize_staged_run_id_strips_polluted_submit_line() -> None:
    polluted = "sim2real-staged-20260615t120000z job=sim2real-staged-20260615t120000z"
    assert normalize_staged_run_id(polluted) == "sim2real-staged-20260615t120000z"


def test_sim2real_workflow_status_includes_eval_metrics_from_workflow_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = OperatorConfig(
        bucket="demo-bucket",
        endpoint_url="https://storage.example",
        registry="cr.example/registry",
        k8s_context="demo-context",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.load_operator_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.resolve_kubeconfig",
        lambda context: tmp_path / "kubeconfig",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_orchestrator_status",
        lambda **kwargs: {
            "job_name": orchestrator_job_name("run-1"),
            "found": False,
            "phase": "MISSING",
            "active": 0,
            "succeeded": 0,
            "failed": 0,
            "pod_phase": "",
            "pod_reason": "",
        },
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_sibling_summary",
        lambda **kwargs: [],
    )

    workflow_state = {
        "status": "completed",
        "updated_at": "2026-06-15T18:08:18Z",
        "final_eval": {"success_rate": 0.72, "threshold": 0.55},
        "final_decision": {
            "decision": "promote_checkpoint",
            "success_rate": 0.72,
            "threshold": 0.55,
        },
        "components": [{"name": "stage_10_eval_heldout", "tier": "WORKS"}],
    }
    state_key = "sim2real-b/run-1/state/workflow_state.json"
    report_key = "sim2real-b/run-1/reports/sim2real-report.json"
    client = _mock_s3_client(
        {
            state_key: json.dumps(workflow_state),
            report_key: json.dumps({"status": "completed"}),
        }
    )

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    result = get_sim2real_workflow_status("run-1")
    assert result["eval_metrics"]["success_rate"] == 0.72
    assert result["eval_metrics"]["threshold"] == 0.55
    assert result["eval_metrics"]["decision"] == "promote_checkpoint"


def test_sim2real_workflow_status_eval_metrics_fallback_to_heldout_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = OperatorConfig(
        bucket="demo-bucket",
        endpoint_url="https://storage.example",
        registry="cr.example/registry",
        k8s_context="demo-context",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.load_operator_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.resolve_kubeconfig",
        lambda context: tmp_path / "kubeconfig",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_orchestrator_status",
        lambda **kwargs: {
            "job_name": orchestrator_job_name("run-1"),
            "found": False,
            "phase": "MISSING",
            "active": 0,
            "succeeded": 0,
            "failed": 0,
            "pod_phase": "",
            "pod_reason": "",
        },
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_sibling_summary",
        lambda **kwargs: [],
    )

    heldout_key = "sim2real-b/run-1/eval/heldout/report.json"
    decision_key = "sim2real-b/run-1/outer_loop/decision.json"
    client = _mock_s3_client(
        {
            heldout_key: json.dumps({"success_rate": 0.41, "threshold": 0.55}),
            decision_key: json.dumps(
                {
                    "decision": "loop_back",
                    "success_rate": 0.41,
                    "threshold": 0.55,
                }
            ),
        }
    )

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    result = get_sim2real_workflow_status("run-1")
    assert result["eval_metrics"]["success_rate"] == 0.41
    assert result["eval_metrics"]["decision"] == "loop_back"


def test_sim2real_workflow_status_marks_failed_image_pull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = OperatorConfig(
        bucket="demo-bucket",
        endpoint_url="https://storage.example",
        registry="cr.example/registry",
        k8s_context="demo-context",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.load_operator_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.resolve_kubeconfig",
        lambda context: tmp_path / "kubeconfig",
    )

    from npa.workflows.sim2real.k8s_client import (
        ContainerSnapshot,
        JobSnapshot,
        PodSnapshot,
    )

    snapshot = JobSnapshot(
        name=orchestrator_job_name("run-1"),
        namespace="default",
        uid="job-uid",
        resource_version="1",
        state="running",
        active=1,
        succeeded=0,
        failed=0,
        deleting=False,
        condition_type="",
        condition_reason="",
        condition_message="",
        pods=(
            PodSnapshot(
                name="pod",
                uid="pod-uid",
                owner_uid="job-uid",
                phase="Pending",
                node_name="",
                deletion_timestamp="",
                scheduled_status="False",
                scheduled_reason="Unschedulable",
                resource_requests={},
                containers=(
                    ContainerSnapshot(
                        name="controller",
                        image="registry/controller@sha256:" + "a" * 64,
                        image_id="",
                        restart_count=0,
                        waiting_reason="ImagePullBackOff",
                    ),
                ),
            ),
        ),
    )

    class FakeK8sClient:
        def snapshot_if_exists(self, *args, **kwargs):
            return snapshot

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._structured_k8s_client",
        lambda **kwargs: FakeK8sClient(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_sibling_summary",
        lambda **kwargs: [],
    )

    fake_client = MagicMock()
    fake_client._s3.head_object.side_effect = Exception("missing")
    fake_client._s3.list_objects_v2.return_value = {"KeyCount": 0}
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: fake_client,
    )

    result = get_sim2real_workflow_status("run-1")
    assert result["status"] == "FAILED"
    assert result["pod_reason"] == "ImagePullBackOff"
    assert result["stages"]["stage_01_trigger"]["state"] == "PENDING"


def test_sim2real_module_status_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "run_id": "run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.watch_sim2real_status",
        lambda run_id, **kwargs: payload,
    )
    from npa.workflows.sim2real.cli import main

    assert main(["status", "run-1", "--json"]) == 0


def test_workflow_cli_status_sim2real_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    payload = {
        "run_id": "sim2real-staged-run-1",
        "status": "RUNNING",
        "current_stage": "stage_03_augment",
        "run_prefix_uri": "s3://demo-bucket/sim2real-b/sim2real-staged-run-1/",
        "stages": {"stage_01_trigger": {"state": "SUCCEEDED", "tier": ""}},
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda run_id, **kwargs: True,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.get_sim2real_workflow_status",
        lambda run_id, **kwargs: payload,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "workflow", "status", "sim2real-staged-run-1", "--json"],
    )
    assert result.exit_code == 0
    body = json.loads(result.stdout)
    assert body["run_id"] == "sim2real-staged-run-1"
    assert body["status"] == "RUNNING"


def test_retired_sim2real_runbook_is_not_shipped() -> None:

    root = Path(__file__).resolve().parents[2]
    assert not (root / "workflows" / "sim2real.yaml").exists()
    assert (
        root / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"
    ).is_file()
