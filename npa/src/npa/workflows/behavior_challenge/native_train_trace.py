"""Finalize and validate audit-only Native RLC TRAIN traces."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .protocol import file_digest

CONFIG_SCHEMA = "npa.behavior.native-rlc-train-trace-config.v1"
MANIFEST_SCHEMA = "npa.behavior.native-rlc-train-trace-manifest.v2"
CORE_MEMBERS = ("config.json", "semantic.jsonl", "native-stage.jsonl")
MANIFEST_KEYS = {
    "schema",
    "status",
    "config",
    "action_count",
    "members",
    "development_or_report_used",
    "privileged_labels_entered_policy_inputs",
    "training_ready",
    "lossless_allowed_tensors",
}


def _identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native TRAIN trace member differs")
    return {"bytes": path.stat().st_size, "sha256": file_digest(path)}


def write_config(root: Path, value: dict[str, Any]) -> dict[str, int | str]:
    """Write the externally admitted trace identity before policy execution.

    Args:
        root: Fresh trace directory.
        value: Externally admitted trace configuration.
    Returns:
        Exact identity of the durable configuration.
    Raises:
        FileExistsError: The trace root already exists.
        ValueError: The configuration differs from the TRAIN-only contract.
    """
    validate_config(value)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(mode=0o700)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with (root / "config.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _identity(root / "config.json")


def validate_config(value: Any) -> dict[str, Any]:
    """Validate one Native-only, nonreporting TRAIN trace configuration.

    Args:
        value: Candidate configuration object.
    Returns:
        Validated configuration.
    Raises:
        ValueError: Fields or scope differ from the trace contract.
    """
    required = {
        "schema",
        "status",
        "case",
        "panel_id",
        "policy_identity_sha256",
        "checkpoint_sha256",
        "source_commits",
        "semantic_component",
        "admission_sha256",
        "development_or_report_used",
        "policy_changed",
        "training_ready",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema") != CONFIG_SCHEMA
        or value.get("status") != "native_train_audit_trace_enabled"
        or value.get("development_or_report_used") is not False
        or value.get("policy_changed") is not False
        or value.get("training_ready") is not False
        or value.get("case", {}).get("split") != "train"
    ):
        raise ValueError("Native TRAIN trace config differs")
    _validate_semantic_component(value["semantic_component"])
    return value


def _validate_semantic_component(value: Any) -> None:
    from .native_train_admission import EXACT_FILES

    identities = {
        "module": "semantic_module",
        "official_q": "official_q",
        "lossless_arrays": "lossless_arrays",
    }
    if not isinstance(value, dict) or set(value) != {*identities, "source"}:
        raise ValueError("Native TRAIN semantic component differs")
    for field, authority in identities.items():
        expected = {key: EXACT_FILES[authority][key] for key in ("bytes", "sha256")}
        if value[field] != expected:
            raise ValueError("Native TRAIN semantic component differs")
    if not isinstance(value["source"], dict) or not value["source"]:
        raise ValueError("Native TRAIN semantic source differs")


def _jsonl_fragment(
    root: Path, *, filename: str, schema: str, status: str
) -> list[dict[str, Any]]:
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native TRAIN trace fragment differs")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError("Native TRAIN trace fragment is empty")
    terminal = json.loads((root / "terminal.json").read_bytes())
    if (
        terminal.get("schema") != schema
        or terminal.get("status") != status
        or terminal.get("action_count") != len(rows)
        or terminal.get("rows") != _identity(path)
    ):
        raise ValueError("Native TRAIN trace fragment terminal differs")
    return rows


def _write_manifest(root: Path, terminal: dict[str, Any]) -> dict[str, Any]:
    _validate_tree(root, manifest_required=False)
    members = {
        path.relative_to(root).as_posix(): _identity(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete_native_rlc_train_trace",
        "config": json.loads((root / "config.json").read_bytes()),
        "action_count": terminal["action_count"],
        "members": members,
        "development_or_report_used": False,
        "privileged_labels_entered_policy_inputs": False,
        "training_ready": False,
        "lossless_allowed_tensors": True,
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    with (root / "trace-manifest.json").open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def finalize_trace(output: Path) -> dict[str, Any]:
    """Join evaluator and policy fragments and write the success manifest last.

    Args:
        output: Completed case output directory.
    Returns:
        Final trace manifest.
    Raises:
        ValueError: Fragment content, identities, or chronology differ.
        FileExistsError: Final members already exist.
    """
    from .native_semantic_trace import write_trace

    root = output / "native-train-trace"
    validate_config(json.loads((root / "config.json").read_bytes()))
    semantic = _jsonl_fragment(
        output / "native-trace-evaluator",
        filename="semantic.jsonl",
        schema="npa.behavior.native-train-semantic-fragment.v1",
        status="official_post_apply_rows_recorded",
    )
    stages = _jsonl_fragment(
        output / "native-trace-policy",
        filename="native-stage.jsonl",
        schema="npa.private.native-train-stage-fragment.v1",
        status="native_actions_and_stage_votes_recorded",
    )
    temporary = output / "native-train-trace-finalized"
    terminal = write_trace(temporary, semantic, stages)
    for name in ("semantic.jsonl", "native-stage.jsonl"):
        os.link(temporary / name, root / name)
    _link_tree(output / "native-trace-policy/arrays", root / "arrays/policy")
    _link_tree(output / "native-trace-evaluator/arrays", root / "arrays/evaluator")
    validate_lossless_arrays(root, semantic, stages)
    return _write_manifest(root, terminal)


def _link_tree(source: Path, target: Path) -> None:
    """Hard-link one finalized regular-file tree without following links."""
    if source.is_symlink() or not source.is_dir() or target.exists():
        raise ValueError("Native TRAIN array source differs")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Native TRAIN array source contains a link")
        relative = path.relative_to(source)
        if path.is_dir():
            (target / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target / relative)
        else:
            raise ValueError("Native TRAIN array member differs")


def _npz(path: Path) -> dict[str, Any]:
    """Load a lossless shard with pickle explicitly disabled."""
    import numpy as np

    if path.is_symlink() or not path.is_file():
        raise ValueError("Native TRAIN array shard differs")
    with np.load(path, allow_pickle=False) as value:
        return {name: np.array(value[name], copy=True) for name in value.files}


def _terminal(root: Path, schema: str, status: str) -> dict[str, Any]:
    value = json.loads((root / "terminal.json").read_bytes())
    if value.get("schema") != schema or value.get("status") != status:
        raise ValueError("Native TRAIN array terminal differs")
    return value


def validate_lossless_arrays(
    root: Path, semantic: list[dict[str, Any]], stages: list[dict[str, Any]]
) -> None:
    """Join every lossless allowed tensor to the reviewed semantic trace."""
    policy = root / "arrays/policy"
    evaluator = root / "arrays/evaluator"
    policy_terminal, evaluator_terminal = _array_terminals(policy, evaluator)
    count = len(stages)
    if (
        policy_terminal["action_count"] != count
        or evaluator_terminal["action_count"] != count
    ):
        raise ValueError("Native TRAIN array action count differs")
    returned = _shard_rows(
        policy, policy_terminal["returned_action_shards"], "returned"
    )
    applied = _shard_rows(
        evaluator, evaluator_terminal["applied_action_shards"], "applied"
    )
    decisions = _shard_rows(policy, policy_terminal["decision_shards"], "decision")
    _validate_action_rows(returned, applied, semantic, stages)
    _validate_decision_rows(decisions, returned, semantic, stages)
    _validate_final_observation(evaluator, semantic[-1], count, _observation_sha256())
    if (
        policy_terminal["inference_count"] != len(decisions)
        or policy_terminal["model_horizon"] != 30
        or policy_terminal["executed_prefix"] != 20
    ):
        raise ValueError("Native TRAIN decision cadence differs")


def _array_terminals(policy: Path, evaluator: Path):
    policy_terminal = _terminal(
        policy,
        "npa.behavior.native-train-policy-arrays.v1",
        "allowed_decisions_chunks_and_returned_actions_recorded",
    )
    evaluator_terminal = _terminal(
        evaluator,
        "npa.behavior.native-train-evaluator-arrays.v1",
        "applied_actions_and_final_observation_recorded",
    )
    policy_keys = {
        "schema",
        "status",
        "action_count",
        "inference_count",
        "model_horizon",
        "executed_prefix",
        "decision_shards",
        "returned_action_shards",
    }
    evaluator_keys = {
        "schema",
        "status",
        "action_count",
        "applied_action_shards",
        "final_observation",
    }
    if set(policy_terminal) != policy_keys or set(evaluator_terminal) != evaluator_keys:
        raise ValueError("Native TRAIN array terminal fields differ")
    return policy_terminal, evaluator_terminal


def _validate_action_rows(returned, applied, semantic, stages) -> None:
    import numpy as np

    from .native_train_arrays import array_identity

    if len(returned) != len(stages) or len(applied) != len(stages):
        raise ValueError("Native TRAIN array row count differs")
    for index, (semantic_row, stage) in enumerate(zip(semantic, stages, strict=True)):
        returned_row = dict(returned[index])
        applied_row = dict(applied[index])
        returned_action = returned_row.pop("returned_action")
        applied_action = applied_row.pop("applied_action")
        if (
            int(returned_row.pop("action_index")) != index
            or int(applied_row.pop("frame_index")) != index
            or returned_row
            or applied_row
            or returned_action.dtype != np.dtype("float32")
            or applied_action.dtype != np.dtype("float32")
            or not np.array_equal(returned_action, applied_action)
            or array_identity(returned_action)["sha256"] != stage["action_sha256"]
            or array_identity(applied_action)["sha256"]
            != semantic_row["applied_action_sha256"]
        ):
            raise ValueError("Native TRAIN returned/applied action join differs")


def _validate_decision_rows(decision_rows, returned_rows, semantic, stages) -> None:
    observation_sha256 = _observation_sha256()
    decisions = {int(row["action_index"]): row for row in decision_rows}
    if len(decisions) != len(decision_rows):
        raise ValueError("Native TRAIN decision index differs")
    _validate_wrapper_chunks(decision_rows, returned_rows)
    for index, stage in enumerate(stages):
        if stage["new_inference"]:
            if index not in decisions:
                raise ValueError("Native TRAIN decision row is absent")
            returned = dict(decisions.pop(index))
            _validate_decision(returned, stage, semantic, index, observation_sha256)
        elif index in decisions:
            raise ValueError("Native TRAIN unexpected decision row")
    if decisions:
        raise ValueError("Native TRAIN decision rows are unconsumed")


def _observation_sha256():
    from .native_train_arrays import observation_sha256

    return observation_sha256


def _shard_rows(root: Path, rows: Any, stem: str) -> list[dict[str, Any]]:
    """Load an exact contiguous terminal-declared NPZ shard sequence."""
    from .native_train_arrays import ACTION_ROWS_PER_SHARD, DECISION_ROWS_PER_SHARD

    if not isinstance(rows, list) or not rows:
        raise ValueError("Native TRAIN shard inventory differs")
    result = []
    limits = {
        "decision": DECISION_ROWS_PER_SHARD,
        "returned": ACTION_ROWS_PER_SHARD,
        "applied": ACTION_ROWS_PER_SHARD,
    }
    maximum = limits[stem]
    for ordinal, row in enumerate(rows):
        required = {"start", "end", "path", "bytes", "sha256"}
        if not isinstance(row, dict):
            raise ValueError("Native TRAIN shard row differs")
        path = root / row.get("path", "")
        length = row.get("end", 0) - row.get("start", 0)
        if (
            set(row) != required
            or row["start"] != len(result)
            or row["end"] <= row["start"]
            or length > maximum
            or (ordinal < len(rows) - 1 and length != maximum)
            or row["path"] != f"{stem}-{row['start']:06d}-{row['end']:06d}.npz"
            or _identity(path) != {key: row[key] for key in ("bytes", "sha256")}
        ):
            raise ValueError("Native TRAIN shard row differs")
        shard = _npz(path)
        lengths = {value.shape[0] for value in shard.values()}
        if lengths != {row["end"] - row["start"]}:
            raise ValueError("Native TRAIN shard length differs")
        for offset in range(row["end"] - row["start"]):
            result.append({name: value[offset] for name, value in shard.items()})
    return result


def _validate_decision(returned, stage, semantic, index, observation_sha256) -> None:
    import numpy as np

    from .native_train_arrays import ACTION_DIMENSION, MODEL_HORIZON
    from .native_train_arrays import allowed_observation

    action_index = int(returned.pop("action_index"))
    ordinal = int(returned.pop("inference_ordinal"))
    raw = returned.pop("raw_model_actions")
    wrapper_actions = returned.pop("raw_wrapper_actions")
    observation = returned
    if (
        action_index != index
        or ordinal != stage["inference_ordinal"]
        or raw.shape != (MODEL_HORIZON, ACTION_DIMENSION)
        or wrapper_actions.shape != (20, ACTION_DIMENSION)
        or returned.keys()
        != {
            "robot_r1::proprio",
            "robot_r1::robot_r1:zed_link:Camera:0::rgb",
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
        }
    ):
        raise ValueError("Native TRAIN decision shard differs")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
        or not np.issubdtype(wrapper_actions.dtype, np.number)
        or not np.isfinite(wrapper_actions).all()
    ):
        raise ValueError("Native TRAIN decision tensor differs")
    allowed_observation(observation)
    if (
        index > 0
        and observation_sha256(observation)
        != semantic[index - 1]["allowed_observation_sha256"]
    ):
        raise ValueError("Native TRAIN next-decision observation join differs")


def _validate_wrapper_chunks(decisions, returned_rows) -> None:
    """Join raw wrapper actions to their exact float32 returned prefix."""
    import numpy as np

    starts = [int(row["action_index"]) for row in decisions]
    if not starts or starts[0] != 0:
        raise ValueError("Native TRAIN first decision differs")
    for ordinal, row in enumerate(decisions):
        start = starts[ordinal]
        end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(returned_rows)
        if end != min(start + 20, len(returned_rows)):
            raise ValueError("Native TRAIN decision cadence differs")
        wrapper_actions = row["raw_wrapper_actions"]
        for offset, returned in enumerate(returned_rows[start:end]):
            expected = np.asarray(wrapper_actions[offset], dtype=np.float32)
            if not np.array_equal(expected, returned["returned_action"]):
                raise ValueError("Native TRAIN wrapper/returned action join differs")


def _validate_final_observation(evaluator, final_semantic, count, observation_sha256):
    terminal = json.loads((evaluator / "terminal.json").read_bytes())
    row = terminal.get("final_observation", {})
    path = evaluator / row.get("path", "")
    if (
        set(row) != {"path", "bytes", "sha256"}
        or row.get("path") != "final-observation.npz"
        or _identity(path) != {key: row.get(key) for key in ("bytes", "sha256")}
    ):
        raise ValueError("Native TRAIN final observation identity differs")
    final = _npz(path)
    if (
        int(final.pop("after_action_index")) != count - 1
        or int(final.pop("observation_index")) != count
    ):
        raise ValueError("Native TRAIN final observation index differs")
    if observation_sha256(final) != final_semantic["allowed_observation_sha256"]:
        raise ValueError("Native TRAIN final observation join differs")


def _validate_manifest(
    root: Path, value: dict[str, Any], semantic: list[dict[str, Any]]
) -> None:
    config = json.loads((root / "config.json").read_bytes())
    members = {
        path.relative_to(root).as_posix(): _identity(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != "trace-manifest.json"
    }
    if (
        set(value) != MANIFEST_KEYS
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("status") != "complete_native_rlc_train_trace"
        or value.get("action_count") != len(semantic)
        or value.get("development_or_report_used") is not False
        or value.get("privileged_labels_entered_policy_inputs") is not False
        or value.get("training_ready") is not False
        or value.get("lossless_allowed_tensors") is not True
        or validate_config(value.get("config")) != config
        or semantic[0]["source"] != value["config"]["semantic_component"]["source"]
        or value.get("members") != members
    ):
        raise ValueError("Native TRAIN trace manifest differs")


def _validate_tree(root: Path, *, manifest_required: bool) -> set[str]:
    """Reject links, special files, extra directories, and unknown members."""
    directories = set()
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("Native TRAIN trace contains a link")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise ValueError("Native TRAIN trace member type differs")
    expected_directories = {"arrays", "arrays/policy", "arrays/evaluator"}
    if directories != expected_directories or not _valid_tree_files(
        files, manifest_required
    ):
        raise ValueError("Native TRAIN trace tree differs")
    return files


def _valid_tree_files(files: set[str], manifest_required: bool) -> bool:
    required = {"config.json", "semantic.jsonl", "native-stage.jsonl"}
    if manifest_required:
        required.add("trace-manifest.json")
    patterns = {
        "arrays/policy/terminal.json",
        "arrays/evaluator/terminal.json",
        "arrays/evaluator/final-observation.npz",
    }
    dynamic = files - required - patterns
    return required | patterns <= files and all(
        name.endswith(".npz")
        and (
            name.startswith("arrays/policy/decision-")
            or name.startswith("arrays/policy/returned-")
            or name.startswith("arrays/evaluator/applied-")
        )
        for name in dynamic
    )


def _declared_tree_files(root: Path) -> set[str]:
    policy, evaluator = _array_terminals(
        root / "arrays/policy", root / "arrays/evaluator"
    )
    declared = []
    groups = (
        ("arrays/policy", policy["decision_shards"]),
        ("arrays/policy", policy["returned_action_shards"]),
        ("arrays/evaluator", evaluator["applied_action_shards"]),
    )
    for directory, rows in groups:
        declared.extend(f"{directory}/{row['path']}" for row in rows)
    if len(declared) != len(set(declared)):
        raise ValueError("Native TRAIN shard path is duplicated")
    return {
        "config.json",
        "semantic.jsonl",
        "native-stage.jsonl",
        "trace-manifest.json",
        "arrays/policy/terminal.json",
        "arrays/evaluator/terminal.json",
        "arrays/evaluator/final-observation.npz",
        *declared,
    }


def validate_finalized_trace(root: Path) -> dict[str, Any]:
    """Validate a complete immutable Native trace without simulator imports.

    Args:
        root: Final trace directory.
    Returns:
        Validated trace manifest.
    Raises:
        ValueError: Root, inventory, chronology, labels, or manifest differ.
    """
    from .native_semantic_trace import validate_join

    manifest_path = root / "trace-manifest.json"
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink():
        raise ValueError("Native TRAIN trace root differs")
    actual = _validate_tree(root, manifest_required=True)
    value = json.loads(manifest_path.read_bytes())
    semantic = [
        json.loads(row) for row in (root / "semantic.jsonl").read_text().splitlines()
    ]
    stages = [
        json.loads(row)
        for row in (root / "native-stage.jsonl").read_text().splitlines()
    ]
    validate_join(semantic, stages)
    validate_lossless_arrays(root, semantic, stages)
    _validate_manifest(root, value, semantic)
    if actual != _declared_tree_files(root) or actual != {
        *value["members"],
        "trace-manifest.json",
    }:
        raise ValueError("Native TRAIN trace inventory differs")
    return value
