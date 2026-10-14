from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.lerobot.policy_container import INTERCHANGE_MANIFEST_FILENAME
from npa.workflows.sim2real.smolvla_reload_stage import (
    SmolVLAReloadError,
    build_parser,
    build_reload_receipt,
    materialize_checkpoint,
)


def _runtime(tmp_path: Path, **manifest_updates):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / INTERCHANGE_MANIFEST_FILENAME).write_text(
        json.dumps({"sealed": True}), encoding="utf-8"
    )
    manifest = {
        "backend": "lerobot_smolvla",
        "policy_identity": "policy-1",
        "robot_identity": "robot-6dof",
        "task_identity": "surface-place-1",
        "checkpoint_sha256": "a" * 64,
        "normalization_identity": "b" * 64,
        "normalization_files": [
            {"path": "policy_preprocessor.json", "sha256": "c" * 64}
        ],
        "action": {"names": ["j1", "gripper"]},
        "measured_state_names": ["j1", "finger"],
        "cameras": [{"key": "wrist"}],
    }
    manifest.update(manifest_updates)
    return SimpleNamespace(
        checkpoint_path=checkpoint,
        manifest=manifest,
        device="cuda",
        policy=SimpleNamespace(),
        preprocessor=SimpleNamespace(),
        postprocessor=SimpleNamespace(),
        validation=SimpleNamespace(to_dict=lambda: {"status": "loadable"}),
    )


def _preparation() -> dict:
    return {
        "policy_compatibility": {
            "backend": "lerobot_smolvla",
            "policy_identity": "policy-1",
            "robot_identity": "robot-6dof",
            "task_identity": "surface-place-1",
            "checkpoint_sha256": "a" * 64,
            "normalization_identity": "b" * 64,
        },
        "policy": {
            "action": {"names": ["j1", "gripper"]},
            "measured_state_names": ["j1", "finger"],
            "camera_keys": ["wrist"],
        },
    }


def test_reload_receipt_binds_native_objects_to_preparation(tmp_path: Path) -> None:
    receipt = build_reload_receipt(
        runtime=_runtime(tmp_path),
        preparation=_preparation(),
        checkpoint_source="s3://example/checkpoint/",
        hardware_topology="single-rtxpro6000",
    )
    assert receipt["status"] == "native_checkpoint_reloaded"
    assert receipt["interchange"]["checkpoint_sha256"] == "a" * 64
    assert receipt["inference_or_task_success_measured"] is False


def test_reload_rejects_checkpoint_from_another_robot(tmp_path: Path) -> None:
    with pytest.raises(SmolVLAReloadError, match="differs"):
        build_reload_receipt(
            runtime=_runtime(tmp_path, robot_identity="other-robot"),
            preparation=_preparation(),
            checkpoint_source="s3://example/checkpoint/",
            hardware_topology="single-rtxpro6000",
        )


def test_checkpoint_source_must_be_immutable_or_private(tmp_path: Path) -> None:
    with pytest.raises(SmolVLAReloadError, match="40-hex"):
        materialize_checkpoint("hf://lerobot/smolvla_base@main", tmp_path / "out")


def test_reload_cli_requires_explicit_binding_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint-uri",
            "s3://example/checkpoint/",
            "--preparation-uri",
            "s3://example/preparation.json",
            "--hardware-topology",
            "single-rtxpro6000",
            "--output-uri",
            "s3://example/reload.json",
            "--component-root-uri",
            "s3://example/run/",
        ]
    )
    assert args.hardware_topology == "single-rtxpro6000"
