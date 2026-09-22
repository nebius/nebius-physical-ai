"""Verify the released SHAwn specialist without weakening stock RLC contracts."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.behavior_challenge import __main__ as behavior_main
from npa.workflows.behavior_challenge import rlc_policy, rlc_specialist


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_rows() -> list[dict]:
    rows = [
        {
            "path": f"param/{index}",
            "shape": [1],
            "dtype": "bfloat16",
            "variable_type": "Param",
            "sha256": str(index).zfill(64),
        }
        for index in range(74)
    ]
    rows.append(
        {
            "path": "action_correlation_cholesky",
            "shape": [960, 960],
            "dtype": "float32",
            "variable_type": "Intermediate",
            "sha256": rlc_specialist.CORRELATION_SHA256,
        }
    )
    return rows


def _regular_files(root: Path, count: int) -> tuple[list[dict], dict[str, str]]:
    rows = []
    files = {}
    for index in range(count):
        path = root / f"regular-{index}"
        path.write_bytes(f"regular-{index}".encode())
        rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "git_blob_sha1": rlc_specialist._git_blob_sha1(path),
            }
        )
        files[path.name] = _sha256(path)
    return rows, files


def test_public_specialist_manifest_pins_release_and_seventeen_files():
    manifest = rlc_specialist.checkpoint_manifest()

    assert manifest["declared_task_ids"] == [1, 7, 18, 21]
    assert len(manifest["files"]) == 17
    assert sum("sha256" in row for row in manifest["files"]) == 4
    assert all("git_blob_sha1" in row for row in manifest["files"])


def test_specialist_file_identity_distinguishes_lfs_payloads(monkeypatch, tmp_path):
    regular = tmp_path / "regular"
    payload = tmp_path / "payload"
    regular.write_bytes(b"regular Git object")
    payload.write_bytes(b"downloaded LFS payload, not its pointer")
    rows, files = _regular_files(tmp_path, 15)
    rows.extend(
        [
            {
                "path": "regular",
                "bytes": regular.stat().st_size,
                "git_blob_sha1": rlc_specialist._git_blob_sha1(regular),
            },
            {
                "path": "payload",
                "bytes": payload.stat().st_size,
                "git_blob_sha1": "0" * 40,
                "sha256": _sha256(payload),
            },
        ]
    )
    files.update({"regular": _sha256(regular), "payload": _sha256(payload)})
    monkeypatch.setattr(rlc_specialist, "checkpoint_manifest", lambda: {"files": rows})
    monkeypatch.setattr(rlc_specialist, "_verify_raw_topology", lambda _root: None)

    rlc_specialist.verify_checkpoint_files(tmp_path, files)

    corrupted = bytearray(payload.read_bytes())
    corrupted[0] ^= 1
    payload.write_bytes(corrupted)
    files["payload"] = _sha256(payload)
    with pytest.raises(ValueError, match="bytes differ"):
        rlc_specialist.verify_checkpoint_files(tmp_path, files)


def test_specialist_regular_file_requires_git_blob_identity(monkeypatch, tmp_path):
    rows, files = _regular_files(tmp_path, 17)
    rows[0]["git_blob_sha1"] = "0" * 40
    monkeypatch.setattr(rlc_specialist, "checkpoint_manifest", lambda: {"files": rows})
    monkeypatch.setattr(rlc_specialist, "_verify_raw_topology", lambda _root: None)

    with pytest.raises(ValueError, match="bytes differ"):
        rlc_specialist.verify_checkpoint_files(tmp_path, files)


def test_specialist_raw_topology_rejects_wrong_leaf_set(tmp_path):
    metadata = {
        "tree_metadata": {
            "('params', 'weight', 'value')": {"value_metadata": {"write_shape": [1]}}
        }
    }
    path = tmp_path / "params/_METADATA"
    path.parent.mkdir()
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="raw 75-leaf topology differs"):
        rlc_specialist._verify_raw_topology(tmp_path)


@pytest.mark.parametrize("defect", ["correlation-hash", "parameter-dtype"])
def test_specialist_loaded_state_requires_exact_native_contract(monkeypatch, defect):
    model = SimpleNamespace(correlation_loaded=True)
    policy = SimpleNamespace(_model=model)
    monkeypatch.setattr(rlc_specialist, "_state_rows", lambda _model: _state_rows())

    summary = rlc_specialist.verify_loaded_state(policy)

    assert summary["typed_leaf_count"] == 75
    assert summary["param_bfloat16_count"] == 74

    wrong = _state_rows()
    if defect == "correlation-hash":
        wrong[-1] = {**wrong[-1], "sha256": "0" * 64}
    else:
        wrong[0] = {**wrong[0], "dtype": "float32"}
    monkeypatch.setattr(rlc_specialist, "_state_rows", lambda _model: wrong)
    with pytest.raises(ValueError, match="post-load native state differs"):
        rlc_specialist.verify_loaded_state(policy)


@pytest.mark.parametrize("task_id", [1, 7, 18, 21])
def test_specialist_accepts_only_declared_tasks(monkeypatch, task_id):
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_args: None)
    monkeypatch.setattr(
        rlc_policy, "_task_checkpoint", lambda *_args: (task_id, "checkpoint_2")
    )
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_root=Path("/source"),
        upstream_root=Path("/upstream"),
    )

    assert rlc_policy._verify_task(
        args, {"recipe": {"split": "development", "tasks": ["task"]}}
    ) == (task_id, "shawn-task-specialist")


def test_specialist_rejects_other_rlc_tasks(monkeypatch):
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_args: None)
    monkeypatch.setattr(
        rlc_policy, "_task_checkpoint", lambda *_args: (0, "checkpoint_2")
    )
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_root=Path("/source"),
        upstream_root=Path("/upstream"),
    )

    with pytest.raises(ValueError, match="does not support"):
        rlc_policy._verify_task(
            args, {"recipe": {"split": "development", "tasks": ["task"]}}
        )


def test_specialist_rejects_report_admission(monkeypatch):
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_args: None)
    monkeypatch.setattr(
        rlc_policy, "_task_checkpoint", lambda *_args: (1, "checkpoint_2")
    )
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_root=Path("/source"),
        upstream_root=Path("/upstream"),
    )

    with pytest.raises(ValueError, match="development-only"):
        rlc_policy._verify_task(
            args, {"recipe": {"split": "report", "tasks": ["task"]}}
        )


def test_specialist_command_enables_post_load_contract(tmp_path):
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_python=Path("/runtime/python"),
        policy_root=Path("/source"),
        policy_checkpoint=Path("/checkpoint"),
        policy_execution_variant="native",
        port=8000,
    )

    command = rlc_policy._command(args, 1, tmp_path)

    assert "--specialist-state-contract" in command
    assert command[command.index("--execution-variant") + 1] == "native"


def test_specialist_provenance_is_memory_unqualified_and_unranked(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    plan = {"recipe": {"policy_checkpoint_sha256": "a" * 64}}

    rlc_policy._record_specialist(output, ["python", "server"], {}, plan)

    record = json.loads((output / "policy-provenance.json").read_text())
    assert record["kind"] == "rlc-specialist"
    assert record["supported_task_ids"] == [1, 7, 18, 21]
    assert record["memory_compliance"] == "unverified"
    assert (
        record["status"]
        == "local_development_only_memory_unverified_not_rollout_ranked"
    )


def test_managed_cli_exposes_specialist_kind():
    parser = argparse.ArgumentParser()
    behavior_main._add_policy_arguments(parser)

    args = parser.parse_args(["--policy-kind", "rlc-specialist"])

    assert args.policy_kind == "rlc-specialist"
    assert args.policy_execution_variant == "native"
