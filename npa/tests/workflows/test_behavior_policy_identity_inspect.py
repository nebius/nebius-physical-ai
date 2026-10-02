"""Exercise the read-only BEHAVIOR policy identity inspector."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.behavior_challenge import campaign
from npa.workflows.behavior_challenge import nonreporting_train as train
from npa.workflows.behavior_challenge import policy_identity_inspect as inspect
from npa.workflows.behavior_challenge import serving_identity


def _artifact(marker: str) -> dict:
    return {"sha256": marker * 64, "bytes": 10}


def _policy_args(
    checkpoint: Path, *, kind: str = "comet50", train_experience: bool = False
):
    return SimpleNamespace(
        policy_kind=kind,
        policy_execution_variant="native",
        policy_archive=checkpoint,
        policy_task_name=None,
        policy_native_binding=None,
        train_experience=train_experience,
        **{field: None for field in serving_identity._INPUT_FIELDS},
    )


def _campaign_panel(args, split: str = "development") -> dict:
    checkpoint = {
        "sha256": serving_identity.file_digest(args.policy_archive),
        "bytes": args.policy_archive.stat().st_size,
    }
    policy = campaign.freeze_policy_identity(
        "inspected-policy",
        {"checkpoint": checkpoint, "serving": serving_identity.serving_artifact(args)},
    )
    tasks = ["fixture_task", *(f"task-{index}" for index in range(99))]
    return campaign.declare_panel(policy, tasks, ["fixture_task"], split)


def _train_contracts() -> tuple[dict, dict, dict]:
    mapping = {
        "schema": "npa.behavior.nonreporting-train-task-mapping.v1",
        "split": "train",
        "task": "fixture_task",
        "data_namespace": "fixture_dataset",
        "data_task_id": 7,
        "source_manifest": _artifact("1"),
        "split_manifest": _artifact("2"),
        "mapping_artifact": _artifact("3"),
    }
    science = {
        "behavior_upstream_commit": campaign.UPSTREAM_COMMIT,
        "task_registry": _artifact("4"),
        "dataset": _artifact("5"),
        "dataset_view": _artifact("6"),
        "normalization": _artifact("7"),
        "tokenizer": _artifact("8"),
        "action_semantics": _artifact("9"),
    }
    evaluator = {
        "argv_contract": _artifact("a"),
        "evaluator_source": _artifact("b"),
        "controller_source": _artifact("c"),
        "robot_config": _artifact("d"),
        "rng_contract": _artifact("e"),
        "wrapper": "fixture.Wrapper",
        "mode": "train",
        "num_envs": 1,
        "num_rollouts": 1,
        "write_video": True,
        "max_steps_argument": None,
        "model_prediction_horizon": 32,
        "executed_prefix": 32,
        "fresh_policy_process_per_case": True,
        "qualification_process_discarded": True,
    }
    return mapping, science, evaluator


def _train_panel(args) -> dict:
    mapping, science, evaluator = _train_contracts()
    protocol = train.declare_train_protocol(
        "fixture_task",
        mapping,
        [{"instance_id": 0, "rollout_id": 0}],
        science,
        evaluator,
    )
    artifacts = {
        "checkpoint": {
            "sha256": serving_identity.file_digest(args.policy_archive),
            "bytes": args.policy_archive.stat().st_size,
        },
        "serving": serving_identity.serving_artifact(args),
    }
    return train.declare_train_panel(
        protocol, campaign.freeze_policy_identity("train-policy", artifacts)
    )


def _write_panel(path: Path, panel: dict) -> None:
    path.write_text(json.dumps(panel, sort_keys=True) + "\n")


@pytest.mark.parametrize("split", ["development", "report"])
def test_inspector_matches_production_digests_without_effects(
    tmp_path: Path, split: str
):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n")
    args = _policy_args(checkpoint)
    panel = _campaign_panel(args, split)
    panel_path = tmp_path / "panel.json"
    _write_panel(panel_path, panel)

    result = inspect.inspect_policy_identity(args, panel_path)

    assert result["policy"]["checkpoint"] == panel["policy"]["artifacts"]["checkpoint"]
    assert result["policy"]["serving"] == panel["policy"]["artifacts"]["serving"]
    assert not any(result["effects"].values())
    assert set(tmp_path.iterdir()) == {checkpoint, panel_path}


def test_inspector_reports_expected_and_actual_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n")
    args = _policy_args(checkpoint)
    panel = _campaign_panel(args)
    panel_path = tmp_path / "panel.json"
    _write_panel(panel_path, panel)
    checkpoint.write_text('{"changed":true}\n')

    with pytest.raises(ValueError, match=r"expected=.*actual="):
        inspect.inspect_policy_identity(args, panel_path)


def test_inspector_rejects_settings_and_source_mismatch(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n")
    args = _policy_args(checkpoint)
    panel = _campaign_panel(args)
    panel_path = tmp_path / "panel.json"
    _write_panel(panel_path, panel)
    args.policy_execution_variant = "transition-refresh"
    with pytest.raises(ValueError, match=r"serving.*expected=.*actual="):
        inspect.inspect_policy_identity(args, panel_path)
    args.policy_execution_variant = "native"
    monkeypatch.setattr(serving_identity, "_serving_source", lambda _: {"x": "0" * 64})
    with pytest.raises(ValueError, match=r"serving.*expected=.*actual="):
        inspect.inspect_policy_identity(args, panel_path)


def test_inspector_rejects_train_scope_without_train_flag(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n")
    train_args = _policy_args(checkpoint, kind="comet-trained", train_experience=True)
    panel_path = tmp_path / "train-panel.json"
    _write_panel(panel_path, _train_panel(train_args))

    with pytest.raises(ValueError, match="TRAIN panels require"):
        inspect.inspect_policy_identity(
            _policy_args(checkpoint, kind="comet-trained"), panel_path
        )

    development_path = tmp_path / "development-panel.json"
    _write_panel(development_path, _campaign_panel(_policy_args(checkpoint)))
    with pytest.raises(ValueError, match="DEV/REPORT panels forbid"):
        inspect.inspect_policy_identity(
            _policy_args(checkpoint, train_experience=True), development_path
        )


@pytest.mark.parametrize("train_experience", [False, True])
def test_policy_identity_inspect_cli_emits_one_json_document(
    tmp_path: Path, train_experience: bool
):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n")
    kind = "comet-trained" if train_experience else "comet50"
    args = _policy_args(checkpoint, kind=kind, train_experience=train_experience)
    panel_path = tmp_path / "panel.json"
    panel = _train_panel(args) if train_experience else _campaign_panel(args)
    _write_panel(panel_path, panel)
    command = [
        sys.executable,
        "-m",
        "npa.workflows.behavior_challenge",
        "policy-identity-inspect",
        "--panel-path",
        str(panel_path),
        "--policy-kind",
        kind,
        "--policy-archive",
        str(checkpoint),
    ]
    if train_experience:
        command.append("--train-experience")
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=os.environ.copy()
    )

    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == (
        "checkpoint_and_serving_identity_verified"
    )
