"""Bind recorded TRAIN samples to the prompt used by the Comet wrapper."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


SCHEMA = "npa.behavior.comet-effective-prompt.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SOURCE_FILES = {
    "task_mapping": "scripts/task_mapping.json",
    "wrapper": "src/openpi/shared/eval_b1k_wrapper.py",
    "tokenizer": "src/openpi/models/tokenizer.py",
    "transforms": "src/openpi/transforms.py",
    "training_config": "src/openpi/training/config.py",
}
_TOKENIZER_CONTRACT = {
    "implementation": "PaligemmaTokenizer",
    "model_uri": "gs://big_vision/paligemma_tokenizer.model",
    "model_bytes_status": "not_observed_bind_at_training_projection",
}
_DERIVATION_KEYS = {
    "schema",
    "status",
    "config",
    "qualification",
    "source_commit",
    "observed_prompt_artifact",
    "prompt_binding",
    "derivation_sha256",
}


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Comet prompt source member differs")
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _literal(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 512
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("Comet effective prompt differs")
    return value


def effective_prompt_binding(
    source_root: Path, task_name: str, task_id: int, override: str | None
) -> dict:
    """Resolve and fingerprint the exact effective Comet TRAIN prompt.

    Args:
        source_root: Verified OpenPI source root used by the wrapper.
        task_name: Metadata lookup key passed to ``B1KPolicyWrapper``.
        task_id: Expected task index in the pinned mapping.
        override: Optional literal prompt applied after wrapper construction.
    Returns:
        Canonical prompt and tokenizer-source binding.
    Raises:
        ValueError: Mapping, prompt, or source members differ.
        OSError: A source member cannot be read.
    """
    paths = {name: source_root / relative for name, relative in _SOURCE_FILES.items()}
    mapping = json.loads(paths["task_mapping"].read_text())
    row = mapping.get(task_name)
    if not isinstance(row, dict) or row.get("task_index") != task_id:
        raise ValueError("Comet prompt task mapping differs")
    mapped = _literal(row.get("task"))
    effective = mapped if override is None else _literal(override)
    payload = {
        "schema": SCHEMA,
        "task_name": task_name,
        "task_id": task_id,
        "effective_prompt": effective,
        "source_kind": "task_mapping_default"
        if override is None
        else "literal_override",
        "source_files": {name: _file_identity(path) for name, path in paths.items()},
        "tokenizer_contract": dict(_TOKENIZER_CONTRACT),
    }
    return {**payload, "binding_sha256": _digest(payload)}


def validate_prompt_binding(value: object) -> dict:
    """Validate a canonical effective-prompt binding.

    Args:
        value: Candidate prompt binding.
    Returns:
        Unchanged validated mapping.
    Raises:
        ValueError: Fields, prompt, source identities, or digest differ.
    """
    keys = {
        "schema",
        "task_name",
        "task_id",
        "effective_prompt",
        "source_kind",
        "source_files",
        "tokenizer_contract",
        "binding_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Comet prompt binding fields differ")
    if value["schema"] != SCHEMA or value["source_kind"] not in {
        "task_mapping_default",
        "literal_override",
    }:
        raise ValueError("Comet prompt binding scope differs")
    _literal(value["effective_prompt"])
    if not isinstance(value["task_name"], str) or type(value["task_id"]) is not int:
        raise ValueError("Comet prompt task identity differs")
    _validate_source_files(value["source_files"])
    if value["tokenizer_contract"] != _TOKENIZER_CONTRACT:
        raise ValueError("Comet prompt tokenizer contract differs")
    payload = {name: item for name, item in value.items() if name != "binding_sha256"}
    if value["binding_sha256"] != _digest(payload):
        raise ValueError("Comet prompt binding digest differs")
    return value


def derive_legacy_prompt_binding(
    config_path: Path,
    qualification_path: Path,
    source_root: Path,
    *,
    observed_prompt_artifact: Path | None = None,
) -> dict:
    """Derive a non-mutating prompt binding for one legacy v1 recording.

    Args:
        config_path: Original v1 TRAIN configuration file.
        qualification_path: Original discarded-load qualification receipt.
        source_root: Exact frozen source used for serving.
        observed_prompt_artifact: Preserved runtime log containing the effective
            prompt, required when the old config has no explicit override.
    Returns:
        Derivation receipt joined to the original config and qualification.
    Raises:
        ValueError: Original evidence or resolved prompt differs.
        OSError: Evidence or source files cannot be read.
    """
    config, qualification, binding, override = _legacy_binding_inputs(
        config_path, qualification_path, source_root
    )
    observed = _observed_prompt_identity(observed_prompt_artifact, binding, override)
    payload = {
        "schema": "npa.behavior.legacy-train-prompt-derivation.v1",
        "status": "legacy_recording_prompt_derived_without_mutation",
        "config": _file_identity(config_path),
        "qualification": _file_identity(qualification_path),
        "source_commit": config["source_commit"],
        "observed_prompt_artifact": observed,
        "prompt_binding": binding,
    }
    return {**payload, "derivation_sha256": _digest(payload)}


def _legacy_binding_inputs(config_path, qualification_path, source_root):
    config = _json_file(config_path, "Legacy TRAIN config")
    qualification = _json_file(qualification_path, "Legacy qualification")
    if (
        config.get("schema") != "npa.behavior.train-experience-config.v1"
        or config.get("status") != "train_only_recording_enabled"
        or config.get("split") != "train"
        or qualification.get("schema")
        != "npa.behavior.comet-native-serving-load-qualification.v1"
        or qualification.get("status") != "checkpoint_loaded_with_explicit_case_rng"
    ):
        raise ValueError("Legacy TRAIN prompt derivation requires config v1")
    case = config.get("case")
    if not isinstance(case, dict) or qualification.get("task") != case.get("task"):
        raise ValueError("Legacy TRAIN prompt task differs")
    _validate_legacy_lineage(config, qualification, case)
    override = config.get("policy_prompt_override")
    if override != qualification.get("task_prompt_override"):
        raise ValueError("Legacy TRAIN prompt override differs")
    _verify_source_revision(source_root, config.get("source_commit"))
    binding = effective_prompt_binding(
        source_root, case["task"], _task_id(source_root, case["task"]), override
    )
    return config, qualification, binding, override


def validate_legacy_prompt_derivation(value: object, config_path: Path) -> dict:
    """Validate a legacy derivation and join it to one recording config.

    Args:
        value: Candidate derivation receipt.
        config_path: Original config inside the recording being projected.
    Returns:
        Unchanged validated receipt.
    Raises:
        ValueError: Receipt fields, digest, config, or prompt task differ.
    """
    if not isinstance(value, dict) or set(value) != _DERIVATION_KEYS:
        raise ValueError("Legacy TRAIN prompt derivation fields differ")
    _validate_file_identity(value["qualification"])
    observed = value["observed_prompt_artifact"]
    if observed is not None:
        _validate_file_identity(observed)
    payload = {
        name: item for name, item in value.items() if name != "derivation_sha256"
    }
    config = _json_file(config_path, "Legacy TRAIN config")
    binding = validate_prompt_binding(value["prompt_binding"])
    if _legacy_derivation_differs(value, config, binding, payload, config_path):
        raise ValueError("Legacy TRAIN prompt derivation differs")
    return value


def legacy_prompt_derivation_sha256(value: object) -> str:
    """Return the canonical identity an admission must pin for a derivation.

    Args:
        value: Complete legacy derivation receipt.
    Returns:
        SHA-256 of its canonical JSON representation.
    """
    return _digest(value)


def _legacy_derivation_differs(value, config, binding, payload, config_path):
    return (
        value["schema"] != "npa.behavior.legacy-train-prompt-derivation.v1"
        or value["status"] != "legacy_recording_prompt_derived_without_mutation"
        or value["config"] != _file_identity(config_path)
        or value["source_commit"] != config.get("source_commit")
        or binding["task_name"] != config.get("case", {}).get("task")
        or (
            binding["source_kind"] == "task_mapping_default"
            and value["observed_prompt_artifact"] is None
        )
        or value["derivation_sha256"] != _digest(payload)
    )


def _json_file(path: Path, purpose: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{purpose} differs")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{purpose} differs")
    return value


def _verify_source_revision(source_root: Path, expected: object) -> None:
    revision_result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    changes_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    changes = changes_result.stdout.strip()
    if revision_result.returncode or changes_result.returncode:
        raise ValueError("Legacy TRAIN prompt source is not a Git checkout")
    if revision != expected or changes:
        raise ValueError("Legacy TRAIN prompt source revision differs")


def _observed_prompt_identity(path: Path | None, binding: dict, override: object):
    if path is None:
        if override is None:
            raise ValueError("Legacy default prompt requires a runtime log artifact")
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Legacy observed prompt artifact differs")
    observed = []
    for line in path.read_text().splitlines():
        marker = "self.task_prompt="
        if marker not in line:
            continue
        try:
            value = ast.literal_eval(line.split(marker, 1)[1])
        except (SyntaxError, ValueError):
            raise ValueError("Legacy observed prompt artifact differs") from None
        observed.append(value)
    if not observed or any(value != binding["effective_prompt"] for value in observed):
        raise ValueError("Legacy observed prompt artifact differs")
    return _file_identity(path)


def _task_id(source_root: Path, task_name: str) -> int:
    mapping = json.loads((source_root / _SOURCE_FILES["task_mapping"]).read_text())
    row = mapping.get(task_name)
    if not isinstance(row, dict) or type(row.get("task_index")) is not int:
        raise ValueError("Legacy TRAIN task mapping differs")
    return row["task_index"]


def _validate_legacy_lineage(config: dict, qualification: dict, case: dict) -> None:
    expected = {
        "case_id": case.get("case_id"),
        "instance_id": case.get("instance_id"),
        "rollout_id": case.get("rollout_id"),
        "checkpoint_content_sha256": config.get("checkpoint_sha256"),
        "rng_contract_sha256": config.get("rng_contract_sha256"),
        "source_commit": config.get("source_commit"),
    }
    if any(qualification.get(name) != value for name, value in expected.items()):
        raise ValueError("Legacy TRAIN prompt qualification lineage differs")


def _validate_source_files(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(_SOURCE_FILES):
        raise ValueError("Comet prompt source identities differ")
    for identity in value.values():
        _validate_file_identity(identity)


def _validate_file_identity(identity: object) -> None:
    if (
        not isinstance(identity, dict)
        or set(identity) != {"bytes", "sha256"}
        or type(identity["bytes"]) is not int
        or identity["bytes"] <= 0
        or not isinstance(identity["sha256"], str)
        or _DIGEST.fullmatch(identity["sha256"]) is None
    ):
        raise ValueError("Comet prompt source identity differs")


__all__ = [
    "derive_legacy_prompt_binding",
    "effective_prompt_binding",
    "legacy_prompt_derivation_sha256",
    "validate_legacy_prompt_derivation",
    "validate_prompt_binding",
]
