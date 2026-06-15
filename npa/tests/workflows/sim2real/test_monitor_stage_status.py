"""Stage-status resolution for the Sim2Real workflow monitor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from npa.workflows.sim2real.monitor import (
    OperatorConfig,
    _apply_infer_from_later,
    _stage_states,
    _workflow_completion_index,
    _workflow_stage_succeeded,
    get_sim2real_workflow_status,
)


def _fake_s3(existing_keys: set[str], prefixes: set[str] | None = None) -> MagicMock:
    client = MagicMock()
    prefixes = prefixes or set()

    def head_object(Bucket: str, Key: str) -> dict[str, str]:
        del Bucket
        if Key in existing_keys:
            return {}
        import botocore.exceptions

        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "not found"}},
            "HeadObject",
        )

    def list_objects_v2(Bucket: str, Prefix: str, MaxKeys: int = 1) -> dict[str, int]:
        del Bucket, MaxKeys
        for prefix in prefixes:
            if Prefix.startswith(prefix) or prefix.startswith(Prefix):
                return {"KeyCount": 1}
        return {"KeyCount": 0}

    client._s3.head_object.side_effect = head_object
    client._s3.list_objects_v2.side_effect = list_objects_v2
    return client


def test_workflow_state_marks_preamble_stages_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_state = {
        "schema": "npa.sim2real.workflow_state.v1",
        "status": "preamble_completed",
        "updated_at": "2026-06-15T12:00:00Z",
        "train_envs_uri": "s3://demo-bucket/sim2real-b/run-1/envs/train/envs.jsonl",
        "components": [
            {"name": "stage_01_trigger", "tier": "WORKS"},
            {"name": "stage_02_assets", "tier": "WORKS"},
            {"name": "stage_03_augment", "tier": "WORKS"},
            {"name": "stage_04_06_env_gen_split_tokens", "tier": "WORKS"},
        ],
        "stage_records": [
            {
                "path": "/tmp/stage_01_trigger/trigger.json",
                "payload": {"stage": 1, "created_at": "2026-06-15T11:59:00Z"},
            }
        ],
    }
    client = _fake_s3(
        {
            "sim2real-b/run-1/state/workflow_state.json",
        }
    )
    client._s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(workflow_state).encode("utf-8"))
    }
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
    assert stages["stage_02_assets"]["state"] == "SUCCEEDED"
    assert stages["stage_06_tokens"]["state"] == "SUCCEEDED"
    assert stages["stage_07_actions_train"]["state"] == "PENDING"
    assert stages["stage_01_trigger"]["completed_at"] == "2026-06-15T12:00:00Z"


def test_stage_02_requires_both_consumed_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _fake_s3(
        {
            "sim2real-b/run-1/stage_02_assets/consumed_scene_spec.json",
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
    assert stages["stage_02_assets"]["state"] == "PENDING"

    client = _fake_s3(
        {
            "sim2real-b/run-1/stage_02_assets/consumed_scene_spec.json",
            "sim2real-b/run-1/stage_02_assets/consumed_robot_spec.json",
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
    assert stages["stage_02_assets"]["state"] == "SUCCEEDED"
    assert stages["stage_02_assets"]["source"] == "s3_artifact"


def test_stage_06_succeeds_on_train_envs_jsonl_not_phantom_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _fake_s3(
        {
            "sim2real-b/run-1/envs/train/envs.jsonl",
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
    assert stages["stage_06_tokens"]["state"] == "SUCCEEDED"
    assert stages["stage_05_envs_train"]["state"] == "SUCCEEDED"


def test_stage_01_inferred_when_later_stage_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _fake_s3(
        {
            "sim2real-b/run-1/augment/manifest.json",
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
    assert stages["stage_03_augment"]["state"] == "SUCCEEDED"
    assert stages["stage_01_trigger"]["state"] == "SUCCEEDED"
    assert stages["stage_01_trigger"]["source"] == "inferred_from_later_stage"


def test_workflow_completion_index_maps_envgen_component() -> None:
    workflow_state = {
        "components": [
            {"name": "stage_04_06_env_gen_split_tokens", "tier": "WORKS"},
        ],
        "stage_records": [],
    }
    index = _workflow_completion_index(workflow_state)
    assert "stage_05_envs_train" in index
    assert index["stage_05_envs_train"]["tier"] == "WORKS"


def test_apply_infer_from_later_only_for_flagged_stages() -> None:
    stages = {
        "stage_01_trigger": {"state": "PENDING", "name": "stage_01_trigger"},
        "stage_02_assets": {"state": "SUCCEEDED", "name": "stage_02_assets"},
    }
    _apply_infer_from_later(stages)
    assert stages["stage_01_trigger"]["state"] == "SUCCEEDED"
    assert stages["stage_02_assets"]["state"] == "SUCCEEDED"


def test_workflow_stage_succeeded_reads_component_tier() -> None:
    from npa.workflows.sim2real.monitor import _STAGE_SPECS

    workflow_state = {
        "status": "running",
        "updated_at": "2026-06-15T12:00:00Z",
        "components": [{"name": "stage_03_augment", "tier": "SEAM"}],
        "stage_records": [],
    }
    index = _workflow_completion_index(workflow_state)
    spec = next(item for item in _STAGE_SPECS if item.name == "stage_03_augment")
    resolved = _workflow_stage_succeeded(
        "stage_03_augment",
        workflow_state=workflow_state,
        completion_index=index,
        spec=spec,
    )
    assert resolved is not None
    assert resolved["tier"] == "SEAM"


def test_get_sim2real_workflow_status_no_early_pending_when_later_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        "npa.workflows.sim2real.monitor._kubectl_json",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor._k8s_sibling_summary",
        lambda **kwargs: [],
    )

    workflow_state = {
        "status": "preamble_completed",
        "updated_at": "2026-06-15T12:00:00Z",
        "train_envs_uri": "s3://demo-bucket/sim2real-b/run-1/envs/train/envs.jsonl",
        "components": [
            {"name": "stage_04_06_env_gen_split_tokens", "tier": "WORKS"},
        ],
        "stage_records": [],
    }
    client = _fake_s3(
        {
            "sim2real-b/run-1/state/workflow_state.json",
            "sim2real-b/run-1/envs/train/envs.jsonl",
        }
    )
    client._s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(workflow_state).encode("utf-8"))
    }
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.StorageClient.from_environment",
        lambda **kwargs: client,
    )

    result = get_sim2real_workflow_status("run-1")
    assert result["stages"]["stage_01_trigger"]["state"] == "SUCCEEDED"
    assert result["stages"]["stage_06_tokens"]["state"] == "SUCCEEDED"
    assert result["current_stage"] == "stage_07_actions_train"
