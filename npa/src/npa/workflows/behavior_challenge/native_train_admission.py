"""Validate one exact TRAIN-only native-RLC semantic collector admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from npa.workflows.behavior_challenge.nonreporting_train import (
    validate_train_panel,
    validate_train_partition,
)

PANEL_ID = "4d871670192fed6941251e5e4d290c45df684c24dbedea21cf6f09dec69beef9"
CASE_ID = "030daf5bf852b976df5dd4e47a31ff0a20152b1bf514845fbb8cc703345e344b"
POLICY_ID = "ea0debacc6233a3081bf6cd827e330a3ef4ce87b7bb3a481c13e80e15352db5e"
CHECKPOINT = {
    "sha256": "9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea",
    "bytes": 12645726756,
    "member_root": "checkpoint_2",
}
TREE = {
    "sha256": "7be3886e73cd98245c38c7fd81aab99fb74c5476e6ba634c78e59ee2ebe9edb8",
    "bytes": 12645720778,
    "files": 27,
    "directories": 10,
}
EXACT_FILES = {
    "panel": {
        "path": "frozen/native-train-panel.json",
        "bytes": 4335,
        "sha256": "a63155faeeb6f69da0fa862076704dff50c236c73e0b72e196c80e333f89ad39",
    },
    "partition": {
        "path": "frozen/native-train-partition.json",
        "bytes": 416,
        "sha256": "cefb60fdf39f3175b35c1a0eeb33e66e44d554d6c363e6ecb2b01ef199290ecf",
    },
    "qualification": {
        "path": "evidence/ACTUAL-RENDER3-NATIVE-QUALIFICATION.json",
        "bytes": 2677,
        "sha256": "d9f6f6b3cb2c7874bd34870eb79b1e8b604b80150b637459c270f8491f2bdfaf",
    },
    "bootstrap_fullread": {
        "path": "evidence/ACTUAL-NATIVE-BOOTSTRAP-FULLREAD.json",
        "bytes": 748194,
        "sha256": "229f6bea06d111ec3294ae03dd40ca17dccf1f1f92381232f7540ed79f81001c",
    },
    "semantic_go": {
        "path": "evidence/SEMANTIC-TRACE-V4-GO.json",
        "bytes": 2886,
        "sha256": "048b2e55dca91a2b359e42d5b46ca1f1883f0249abbb62dece8ef5103243fc82",
    },
    "semantic_module": {
        "path": "native_semantic_trace.py",
        "bytes": 20361,
        "sha256": "fc418c220ab38c5aed9c9f7a9d6a493aeec6ba79cb98a1866505ba146afe4638",
    },
    "official_q": {
        "path": "train_official_q.py",
        "bytes": 9505,
        "sha256": "0ddfcfdc5360368f903f20c16853d67738f2df8bdf12d4fae0ddcd48c28c150b",
    },
    "lossless_arrays": {
        "path": "native_train_arrays.py",
        "bytes": 11553,
        "sha256": "66da1997fbacd1c120277c8d676bb5f660dc0f75c784b6f7464e23a5c6d054a0",
    },
}


def identity(path: Path) -> dict[str, int | str]:
    """Return one exact regular-file identity.

    Args:
        path: File to hash.
    Returns:
        Byte count and SHA-256 digest.
    Raises:
        ValueError: The path is not a regular file.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"native TRAIN authority is not a regular file: {path}")
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def exact_json(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Load one exact admitted JSON object.

    Args:
        root: Admission package root.
        row: Relative path and exact file identity.
    Returns:
        Parsed JSON object.
    Raises:
        ValueError: File bytes differ from the admission.
        TypeError: The JSON value is not an object.
    """
    path = root / row["path"]
    expected = {name: row[name] for name in ("bytes", "sha256")}
    if identity(path) != expected:
        raise ValueError(f"native TRAIN authority differs: {row['path']}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(row["path"])
    return value


def _validate_policy_authority(root: Path) -> None:
    qualification = exact_json(root, EXACT_FILES["qualification"])
    execution = qualification.get("execution_tree", {})
    observed_tree = {
        "sha256": execution.get("content_sha256"),
        "bytes": execution.get("total_uncompressed_bytes"),
        "files": execution.get("file_count"),
        "directories": execution.get("directory_count"),
    }
    if (
        qualification.get("schema")
        != "npa.private.render123-assigned-baseline-qualification.v2"
        or qualification.get("policy") != "native"
        or qualification.get("development_or_report_read") is not False
        or qualification.get("policy_loaded") is not False
        or qualification.get("simulator_started") is not False
        or qualification.get("archive")
        != {
            "path": "rlc-checkpoint_2.zip",
            "bytes": CHECKPOINT["bytes"],
            "sha256": CHECKPOINT["sha256"],
        }
        or execution.get("declared_member_root") != CHECKPOINT["member_root"]
        or observed_tree != TREE
        or execution.get("complete_regular_file_bijection") is not True
    ):
        raise ValueError("native checkpoint qualification differs")
    bootstrap = exact_json(root, EXACT_FILES["bootstrap_fullread"])
    if (
        bootstrap.get("status") != "all_eight_bootstrap_originals_fully_read_hashjoined"
        or bootstrap.get("setup", {}).get("development_or_report_read") is not False
        or bootstrap.get("setup", {}).get("policy_or_case_started") is not False
    ):
        raise ValueError("native source bootstrap authority differs")


def _validate_public_source(
    source_root: Path, expected_commit: str, expected_q: dict[str, Any]
) -> None:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("public native TRAIN source root differs")
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if (
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None
        or revision != expected_commit
        or dirty
    ):
        raise ValueError("public native TRAIN source commit differs")
    public = (
        source_root / "npa/src/npa/workflows/behavior_challenge/train_official_q.py"
    )
    if identity(public) != expected_q or expected_q != {
        key: EXACT_FILES["official_q"][key] for key in ("bytes", "sha256")
    }:
        raise ValueError("public official-Q source differs")


def _package_root(admission_path: Path) -> Path:
    path = admission_path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("native TRAIN admission is not a regular file")
    current = path.parent
    while current != Path(current.anchor):
        if current.is_symlink() or not current.is_dir():
            raise ValueError("native TRAIN admission package path differs")
        current = current.parent
    return path.parent.resolve()


def _validate_envelope(value: dict[str, Any]) -> None:
    required = {
        "claims",
        "development_or_report_allowed",
        "panel",
        "partition",
        "policy",
        "policy_authority",
        "public_source",
        "schema",
        "semantic_trace",
        "split",
        "status",
        "task",
    }
    if (
        set(value) != required
        or value.get("schema") != "npa.private.native-rlc-train-collector-admission.v1"
        or value.get("status")
        != "exact_train_panel_native_policy_and_audit_trace_admitted"
        or value.get("split") != "train"
        or value.get("task") != "picking_up_trash"
        or value.get("development_or_report_allowed") is not False
        or value.get("policy", {}).get("identity_sha256") != POLICY_ID
        or value.get("policy", {}).get("checkpoint_archive") != CHECKPOINT
        or value.get("policy", {}).get("checkpoint_tree") != TREE
        or value.get("semantic_trace", {}).get("scope") != "audit_only"
        or value.get("semantic_trace", {}).get("lossless_arrays")
        != EXACT_FILES["lossless_arrays"]
    ):
        raise ValueError("native TRAIN admission envelope differs")
    _validate_envelope_rows(value)


def _validate_envelope_rows(value: dict[str, Any]) -> None:
    panel = {**EXACT_FILES["panel"], "panel_id": PANEL_ID, "case_id": CASE_ID}
    partition = dict(EXACT_FILES["partition"])
    if (
        value["panel"] != panel
        or value["partition"] != partition
        or value["policy_authority"]
        != {
            "qualification": EXACT_FILES["qualification"],
            "bootstrap_fullread": EXACT_FILES["bootstrap_fullread"],
        }
    ):
        raise ValueError("native TRAIN authority rows differ")
    claims = value["claims"]
    claim_names = {
        "comet_checkpoint_or_experience",
        "development_or_report_used",
        "gpu_actions",
        "monitor_trained",
        "policy_changed",
        "provider_actions",
    }
    if (
        not isinstance(claims, dict)
        or set(claims) != claim_names
        or any(claims.values())
    ):
        raise ValueError("native TRAIN admission makes an unsupported claim")
    _validate_policy_row(value["policy"])


def _validate_policy_row(policy: dict[str, Any]) -> None:
    expected = {
        "checkpoint": "checkpoint_2",
        "checkpoint_archive": CHECKPOINT,
        "checkpoint_tree": TREE,
        "execution_variant": "native",
        "identity_sha256": POLICY_ID,
        "normalization": "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json",
        "policy_id": "rlc-native-v393",
        "source_commits": {
            ".": "ca556f74a455cef7987a2be4537b5ac85cc56dd7",
            "BEHAVIOR-1K": "684a83050ddd398de231e6aa7fc605bc34458d4b",
            "openpi": "01177e0242a1c7e8fad2547caa0e987def614cda",
        },
        "task_id": 1,
    }
    if policy != expected:
        raise ValueError("native TRAIN policy row differs")


def _validate_semantic_authority(value: dict[str, Any]) -> None:
    semantic = value["semantic_trace"]
    if (
        set(semantic)
        != {
            "archive",
            "independent_go",
            "lossless_arrays",
            "manifest",
            "module",
            "scope",
        }
        or semantic.get("scope") != "audit_only"
        or semantic.get("lossless_arrays") != EXACT_FILES["lossless_arrays"]
        or semantic.get("module")
        != {name: EXACT_FILES["semantic_module"][name] for name in ("bytes", "sha256")}
        or semantic.get("independent_go")
        != {name: EXACT_FILES["semantic_go"][name] for name in ("bytes", "sha256")}
        or semantic.get("archive")
        != {
            "bytes": 12190,
            "sha256": "a29f700446b57d9e36775ac0419fe25b8f83fc55dba598592febba9e13baa650",
        }
        or semantic.get("manifest")
        != {
            "bytes": 1539,
            "sha256": "bb2c1fa8ad87a952a4709fa22ec2f9bdf4de25cf07d8301c637729d5a74547f8",
        }
    ):
        raise ValueError("native TRAIN semantic authority differs")


def _validate_panel(root: Path) -> None:
    panel = validate_train_panel(exact_json(root, EXACT_FILES["panel"]))
    partition = validate_train_partition(
        exact_json(root, EXACT_FILES["partition"]), panel
    )
    if (
        panel["panel_id"] != PANEL_ID
        or panel["policy_binding_sha256"] != POLICY_ID
        or panel["case_count"] != 1
        or panel["cases"][0]["split"] != "train"
        or panel["cases"][0]["instance_id"] != 200
        or panel["cases"][0]["rollout_id"] != 0
        or partition["worker_count"] != 1
    ):
        raise ValueError("native TRAIN panel or partition differs")


def validate(admission_path: Path, *, source_root: Path) -> dict[str, Any]:
    """Validate the exact policy, TRAIN panel, trace, and public source gate.

    Args:
        admission_path: Operation-owned admission envelope.
        source_root: Clean public Workbench checkout used by the worker.
    Returns:
        Validated admission object.
    Raises:
        ValueError: An admission, policy, panel, trace, or source join differs.
    """
    root = _package_root(admission_path)
    value = json.loads(admission_path.read_bytes())
    _validate_envelope(value)
    _validate_semantic_authority(value)
    _validate_panel(root)
    for name in ("semantic_go", "semantic_module", "official_q", "lossless_arrays"):
        row = EXACT_FILES[name]
        if identity(root / row["path"]) != {
            key: row[key] for key in ("bytes", "sha256")
        }:
            raise ValueError(f"native TRAIN {name} differs")
    _validate_policy_authority(root)
    public = value.get("public_source", {})
    if set(public) != {"commit", "train_official_q"}:
        raise ValueError("public native TRAIN source row differs")
    _validate_public_source(
        source_root, public.get("commit", ""), public.get("train_official_q", {})
    )
    return value


__all__ = ["EXACT_FILES", "identity", "validate"]
