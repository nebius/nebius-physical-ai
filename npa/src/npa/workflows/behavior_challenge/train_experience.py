"""Record and validate lossless TRAIN policy experience without privileged inputs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

try:
    from semantic_monitor.interface import (
        DEPTH_KEYS,
        PROPRIO_KEY,
        RGB_KEYS,
        validate_evaluation_observation,
    )
except ModuleNotFoundError:
    from .semantic_monitor.interface import (
        DEPTH_KEYS,
        PROPRIO_KEY,
        RGB_KEYS,
        validate_evaluation_observation,
    )

ACTION_DIMENSION = 23
ACTION_HORIZON = 32
DECISION_CADENCE = "model_decision_observation_with_all_applied_actions"
CONFIG_SCHEMA = "npa.behavior.train-experience-config.v1"
MANIFEST_SCHEMA = "npa.behavior.train-experience-manifest.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TERMINAL_CONTRACTS = {
    "all_official_actions_applied": (
        "npa.behavior.train-experience-evaluator.v1",
        {
            "schema",
            "status",
            "episode_count",
            "frame_count",
            "decision_shards",
            "transition_shards",
            "final_observation_shards",
            "events",
        },
    ),
    "raw_emissions_and_returned_actions_recorded": (
        "npa.behavior.train-experience-policy.v1",
        {
            "schema",
            "status",
            "emission_count",
            "action_count",
            "emission_shards",
            "action_shards",
            "events",
        },
    ),
}


def file_identity(path: Path) -> dict[str, object]:
    """Return a regular file's byte identity.

    Args:
        path: File to hash.
    Returns:
        Byte count and SHA-256.
    Raises:
        ValueError: The path is not a regular non-symlink file.
        OSError: The file cannot be read.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience member must be a regular file")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def array_identity(value: Any) -> dict[str, object]:
    """Return an exact C-order identity for one finite numeric array.

    Args:
        value: Tensor-like numeric value.
    Returns:
        Shape, dtype, bytes, and SHA-256 of canonical C bytes.
    Raises:
        ValueError: The value is not a finite numeric array.
    """
    array = _numeric(value)
    payload = np.ascontiguousarray(array).tobytes()
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _numeric(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("TRAIN experience arrays must be finite and numeric")
    return array


def allowed_observation(
    observation: Mapping[str, Any], *, include_depth: bool
) -> dict[str, np.ndarray]:
    """Copy only allowed onboard arrays from one evaluator observation.

    Args:
        observation: Raw official evaluator observation.
        include_depth: Whether all three allowed depth arrays are retained.
    Returns:
        Exact unbatched allowed arrays, excluding every privileged leaf.
    Raises:
        ValueError: Required arrays, shapes, dtypes, or depth cadence differ.
    """
    selected_keys = {PROPRIO_KEY, *RGB_KEYS}
    if include_depth:
        selected_keys.update(DEPTH_KEYS)
    if not selected_keys.issubset(observation):
        raise ValueError("TRAIN experience observation lacks enabled onboard arrays")
    selected = {name: _unbatched(observation[name], name) for name in selected_keys}
    validate_evaluation_observation(selected)
    return selected


def _unbatched(value: Any, name: str) -> np.ndarray:
    array = _numeric(value)
    expected_ranks = (
        {1} if name == PROPRIO_KEY else {2, 3} if name in DEPTH_KEYS else {3}
    )
    if array.ndim in expected_ranks:
        return np.array(array, copy=True)
    if array.shape[0] == 1 and array.ndim - 1 in expected_ranks:
        return np.array(array[0], copy=True)
    raise ValueError("TRAIN experience observation batch shape differs")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    if (
        path.exists()
        or path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class LosslessShardWriter:
    """Write bounded, lossless NPZ shards and exact member indexes."""

    def __init__(self, root: Path, *, records_per_shard: int) -> None:
        if records_per_shard <= 0:
            raise ValueError("TRAIN experience shard size must be positive")
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        self.root = root
        self.records_per_shard = records_per_shard
        self.pending: list[dict[str, np.ndarray]] = []
        self.rows: list[dict[str, object]] = []

    def append(self, arrays: Mapping[str, Any]) -> None:
        """Append one exact homogeneous record.

        Args:
            arrays: Named numeric arrays with stable shapes and dtypes.
        Returns:
            None.
        Raises:
            ValueError: Array inventory or layout differs within a shard.
        """
        record = {
            name: np.array(_numeric(value), copy=True) for name, value in arrays.items()
        }
        if not record:
            raise ValueError("TRAIN experience shard record is empty")
        if self.pending:
            _same_layout(self.pending[0], record)
        self.pending.append(record)
        if len(self.pending) == self.records_per_shard:
            self._flush()

    def close(self) -> list[dict[str, object]]:
        """Flush pending records and return immutable shard rows.

        Args:
            None.
        Returns:
            Exact shard rows in order.
        Raises:
            OSError: A shard cannot be written or synchronized.
        """
        self._flush()
        _atomic_json(self.root / "index.json", {"shards": self.rows})
        return list(self.rows)

    def _flush(self) -> None:
        if not self.pending:
            return
        name = f"part-{len(self.rows):06d}.npz"
        target = self.root / name
        temporary = target.with_suffix(".npz.tmp")
        stacked = {
            key: np.stack([record[key] for record in self.pending])
            for key in self.pending[0]
        }
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **stacked)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        self.rows.append(_shard_row(target, stacked, len(self.pending)))
        self.pending.clear()


def _same_layout(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    if set(left) != set(right):
        raise ValueError("TRAIN experience shard array inventory differs")
    for name in left:
        if (
            left[name].shape != right[name].shape
            or left[name].dtype != right[name].dtype
        ):
            raise ValueError("TRAIN experience shard array layout differs")


def _shard_row(
    path: Path, arrays: Mapping[str, np.ndarray], count: int
) -> dict[str, object]:
    return {
        "path": path.name,
        "count": count,
        **file_identity(path),
        "arrays": {
            name: array_identity(value) for name, value in sorted(arrays.items())
        },
    }


def write_experience_config(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Create one exact TRAIN-only experience configuration.

    Args:
        root: Fresh run-owned experience directory.
        value: Candidate configuration with case and policy identities.
    Returns:
        Validated copied configuration.
    Raises:
        ValueError: Scope, cadence, or identities differ.
        FileExistsError: The root already exists.
    """
    config = validate_experience_config(dict(value))
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    _atomic_json(root / "config.json", config)
    return config


def validate_experience_config(value: object) -> dict[str, Any]:
    """Validate an exact model-decision TRAIN experience configuration.

    Args:
        value: Candidate configuration.
    Returns:
        Unchanged validated configuration.
    Raises:
        ValueError: Fields, split, cadence, case, or identities differ.
    """
    keys = {
        "schema",
        "status",
        "split",
        "cadence",
        "action_horizon",
        "include_depth",
        "case",
        "panel_sha256",
        "policy_identity_sha256",
        "checkpoint_sha256",
        "rng_contract_sha256",
        "source_commit",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("TRAIN experience configuration fields differ")
    case = value["case"]
    if not isinstance(case, dict) or set(case) != {
        "case_id",
        "task",
        "instance_id",
        "rollout_id",
        "split",
    }:
        raise ValueError("TRAIN experience case fields differ")
    _validate_config_values(value, case)
    return value


def _validate_config_values(value: dict, case: dict) -> None:
    identities = (
        "panel_sha256",
        "policy_identity_sha256",
        "checkpoint_sha256",
        "rng_contract_sha256",
    )
    if any(
        not isinstance(value[name], str) or _DIGEST.fullmatch(value[name]) is None
        for name in identities
    ):
        raise ValueError("TRAIN experience identity differs")
    if (
        value["schema"] != CONFIG_SCHEMA
        or value["status"] != "train_only_recording_enabled"
        or value["split"] != "train"
        or value["cadence"] != DECISION_CADENCE
        or value["action_horizon"] != ACTION_HORIZON
        or type(value["include_depth"]) is not bool
        or case["split"] != "train"
        or case["rollout_id"] != 0
        or type(case["instance_id"]) is not int
        or not isinstance(case["case_id"], str)
        or not isinstance(case["task"], str)
        or re.fullmatch(r"[0-9a-f]{40}", str(value["source_commit"])) is None
    ):
        raise ValueError("TRAIN experience configuration scope differs")


class EvaluatorExperienceRecorder:
    """Record model-decision observations and all official applied actions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = validate_experience_config(
            json.loads((root / "config.json").read_text())
        )
        self.directory = root / "evaluator"
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError(self.directory)
        self.directory.mkdir()
        self.events = self.directory / "events.jsonl"
        self.events.open("xb").close()
        self.decisions = LosslessShardWriter(
            self.directory / "decisions", records_per_shard=8
        )
        self.transitions = LosslessShardWriter(
            self.directory / "transitions", records_per_shard=128
        )
        self.final_observation = LosslessShardWriter(
            self.directory / "final-observation", records_per_shard=1
        )
        self.frame = 0
        self.episodes = 0
        self.pending: tuple[dict[str, np.ndarray], np.ndarray] | None = None
        self.last_observation: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        """Record a fresh official evaluator episode boundary.

        Args:
            None.
        Returns:
            None.
        Raises:
            ValueError: An action remains unapplied at reset.
        """
        if self.pending is not None or self.episodes != 0:
            raise ValueError("TRAIN experience reset has an unapplied action")
        _append_json(
            self.events,
            {
                "schema": "npa.behavior.train-experience-episode-start.v1",
                "episode_ordinal": self.episodes,
                "frame_index": self.frame,
                "monotonic_ns": time.monotonic_ns(),
            },
        )
        self.episodes += 1

    def record_policy(
        self, observation: Mapping[str, Any], action: Any, started: int, completed: int
    ) -> None:
        """Record one returned action and its exact allowed input identity.

        Args:
            observation: Raw evaluator observation passed to the policy client.
            action: Returned singleton action.
            started: Monotonic time before the policy call.
            completed: Monotonic time after the policy call.
        Returns:
            None.
        Raises:
            ValueError: Chronology, observation, or action differs.
        """
        if self.pending is not None or not 0 <= started <= completed:
            raise ValueError("TRAIN experience policy chronology differs")
        allowed = allowed_observation(
            observation, include_depth=self.config["include_depth"]
        )
        action_array = _action(action)
        if self.frame % ACTION_HORIZON == 0:
            self.decisions.append(
                {"frame_index": np.asarray(self.frame, dtype=np.int64), **allowed}
            )
        _append_json(
            self.events,
            _policy_event(self.frame, allowed, action_array, started, completed),
        )
        self.pending = allowed, action_array

    def record_applied(
        self, observation: Mapping[str, Any], action: Any, started: int, completed: int
    ) -> None:
        """Record one action after the official evaluator accepts it.

        Args:
            observation: Post-apply raw evaluator observation.
            action: Action passed to the official evaluator.
            started: Monotonic time before the original apply call.
            completed: Monotonic time after it returned.
        Returns:
            None.
        Raises:
            ValueError: The action, observation, or chronology differs.
        """
        if self.pending is None or not 0 <= started <= completed:
            raise ValueError("TRAIN experience apply chronology differs")
        before, returned = self.pending
        applied = _action(action)
        if array_identity(returned) != array_identity(applied):
            raise ValueError("Returned and officially applied actions differ")
        after = allowed_observation(
            observation, include_depth=self.config["include_depth"]
        )
        self.transitions.append(
            {
                "frame_index": np.asarray(self.frame, dtype=np.int64),
                "pre_proprio": before[PROPRIO_KEY],
                "applied_action": applied,
                "post_proprio": after[PROPRIO_KEY],
            }
        )
        _append_json(
            self.events, _applied_event(self.frame, applied, after, started, completed)
        )
        self.pending = None
        self.frame += 1
        self.last_observation = after

    def close(self) -> dict[str, Any]:
        """Flush all shards and write the evaluator terminal.

        Args:
            None.
        Returns:
            Exact evaluator terminal.
        Raises:
            ValueError: An action remains unapplied or no episode was observed.
        """
        if (
            self.pending is not None
            or self.episodes != 1
            or self.frame < 1
            or self.last_observation is None
        ):
            raise ValueError("TRAIN experience evaluator ended incomplete")
        self.final_observation.append(
            {
                "frame_index": np.asarray(self.frame, dtype=np.int64),
                **self.last_observation,
            }
        )
        value = {
            "schema": "npa.behavior.train-experience-evaluator.v1",
            "status": "all_official_actions_applied",
            "episode_count": self.episodes,
            "frame_count": self.frame,
            "decision_shards": self.decisions.close(),
            "transition_shards": self.transitions.close(),
            "final_observation_shards": self.final_observation.close(),
            "events": file_identity(self.events),
        }
        _atomic_json(self.directory / "terminal.json", value)
        return value


def _action(value: Any) -> np.ndarray:
    action = _numeric(value)
    if action.shape == (1, ACTION_DIMENSION):
        action = action[0]
    if action.shape != (ACTION_DIMENSION,):
        raise ValueError("TRAIN experience action shape differs")
    return np.array(action, copy=True)


def _official_transport_matches(raw: np.ndarray, returned: dict) -> bool:
    if (
        not isinstance(returned, dict)
        or returned.get("dtype") != np.dtype(np.float32).str
    ):
        return False
    return array_identity(raw.astype(np.float32)) == returned


def _observation_identities(
    value: Mapping[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    return {name: array_identity(array) for name, array in sorted(value.items())}


def _policy_event(frame, observation, action, started, completed) -> dict[str, object]:
    return {
        "schema": "npa.behavior.train-experience-policy-return.v1",
        "frame_index": frame,
        "decision_observation_stored": frame % ACTION_HORIZON == 0,
        "observation": _observation_identities(observation),
        "returned_action": array_identity(action),
        "policy_call_started_monotonic_ns": started,
        "policy_call_completed_monotonic_ns": completed,
        "privileged_state_recorded": False,
    }


def _applied_event(frame, action, observation, started, completed) -> dict[str, object]:
    return {
        "schema": "npa.behavior.train-experience-official-apply.v1",
        "frame_index": frame,
        "official_evaluator_applied_action": array_identity(action),
        "post_apply_observation": _observation_identities(observation),
        "apply_started_monotonic_ns": started,
        "apply_completed_monotonic_ns": completed,
    }


class PolicyExperienceRecorder:
    """Record raw model emissions and their returned-action mapping."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = validate_experience_config(
            json.loads((root / "config.json").read_text())
        )
        self.directory = root / "policy"
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError(self.directory)
        self.directory.mkdir()
        self.emissions = LosslessShardWriter(
            self.directory / "emissions", records_per_shard=16
        )
        self.actions = LosslessShardWriter(
            self.directory / "actions", records_per_shard=128
        )
        self.events = self.directory / "events.jsonl"
        self.events.open("xb").close()
        self.chunk: np.ndarray | None = None
        self.chunk_ordinal = -1
        self.offset = 0
        self.action_count = 0

    def reset(self) -> None:
        """Reset chunk mapping at an official evaluator episode boundary.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.chunk = None
        self.offset = 0

    def record_emission(
        self, actions: Any, before: str, after: str, started: int, completed: int
    ) -> None:
        """Record one raw 32-action model emission.

        Args:
            actions: Exact policy `infer` action array.
            before: RNG identity before inference.
            after: RNG identity after inference.
            started: Monotonic time before delegated inference.
            completed: Monotonic time after delegated inference.
        Returns:
            None.
        Raises:
            ValueError: Shape, RNG identity, or chronology differs.
        """
        chunk = _validated_emission(actions, before, after, started, completed)
        self.chunk_ordinal += 1
        self.offset = 0
        self.chunk = np.array(chunk, copy=True)
        self.emissions.append(
            {
                "chunk_ordinal": np.asarray(self.chunk_ordinal, dtype=np.int64),
                "actions": self.chunk,
            }
        )
        _append_json(
            self.events,
            _emission_event(
                self.chunk_ordinal, self.chunk, before, after, started, completed
            ),
        )

    def record_action(self, action: Any, inference_ordinal: int) -> None:
        """Bind one returned server action to its raw source chunk.

        Args:
            action: Exact action returned by the wrapper.
            inference_ordinal: Existing native server inference ordinal.
        Returns:
            None.
        Raises:
            ValueError: No emission exists or bytes and ordinals differ.
        """
        returned = _action(action)
        if self.chunk is None or inference_ordinal != self.chunk_ordinal:
            raise ValueError("TRAIN experience action lacks its source emission")
        if self.offset >= len(self.chunk) or array_identity(returned) != array_identity(
            self.chunk[self.offset]
        ):
            raise ValueError("TRAIN experience action differs from its source chunk")
        self.actions.append(
            {
                "action_index": np.asarray(self.action_count, dtype=np.int64),
                "chunk_ordinal": np.asarray(self.chunk_ordinal, dtype=np.int64),
                "chunk_offset": np.asarray(self.offset, dtype=np.int64),
                "returned_action": returned,
            }
        )
        self.action_count += 1
        self.offset += 1

    def close(self) -> dict[str, Any]:
        """Flush all policy shards and write the policy terminal.

        Args:
            None.
        Returns:
            Exact policy terminal.
        Raises:
            OSError: Shards or terminal cannot be synchronized.
        """
        value = {
            "schema": "npa.behavior.train-experience-policy.v1",
            "status": "raw_emissions_and_returned_actions_recorded",
            "emission_count": self.chunk_ordinal + 1,
            "action_count": self.action_count,
            "emission_shards": self.emissions.close(),
            "action_shards": self.actions.close(),
            "events": file_identity(self.events),
        }
        _atomic_json(self.directory / "terminal.json", value)
        return value


class RecordingPolicy:
    """Delegate exactly one inference while recording its raw action chunk."""

    def __init__(self, delegate: Any, recorder: PolicyExperienceRecorder) -> None:
        self.delegate = delegate
        self.recorder = recorder

    def infer(self, inputs: Mapping[str, Any]) -> Any:
        """Return the delegate result unchanged after lossless recording.

        Args:
            inputs: Existing Comet inference input mapping.
        Returns:
            The exact object returned by the delegate.
        Raises:
            ValueError: Result or RNG chronology differs from the contract.
        """
        before_key = self.delegate._rng
        before = _jax_key_identity(before_key)
        started = time.monotonic_ns()
        result = self.delegate.infer(inputs)
        completed = time.monotonic_ns()
        after = _jax_key_identity(self.delegate._rng)
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("TRAIN experience inference result differs")
        self.recorder.record_emission(
            result["actions"], before, after, started, completed
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _jax_key_identity(key: Any) -> str:
    import jax

    payload = np.asarray(jax.random.key_data(key)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _validated_emission(actions, before, after, started, completed) -> np.ndarray:
    chunk = _numeric(actions)
    if (
        chunk.shape != (ACTION_HORIZON, ACTION_DIMENSION)
        or not 0 <= started <= completed
    ):
        raise ValueError("TRAIN experience model emission differs")
    if (
        _DIGEST.fullmatch(before) is None
        or _DIGEST.fullmatch(after) is None
        or before == after
    ):
        raise ValueError("TRAIN experience emission RNG differs")
    return chunk


def _emission_event(ordinal, chunk, before, after, started, completed) -> dict:
    return {
        "schema": "npa.behavior.train-experience-emission.v1",
        "chunk_ordinal": ordinal,
        "actions": array_identity(chunk),
        "rng_before_sha256": before,
        "rng_after_sha256": after,
        "inference_started_monotonic_ns": started,
        "inference_completed_monotonic_ns": completed,
    }


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience journal differs")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("TRAIN experience journal row differs")
    return rows


def finalize_experience(root: Path, output: Path) -> dict[str, Any]:
    """Validate both recorders and bind them to official TRAIN outcome bytes.

    Args:
        root: Run-owned experience root.
        output: Official evaluator case output directory.
    Returns:
        Immutable complete experience manifest.
    Raises:
        ValueError: Chronology, actions, scope, outcome, or inventory differs.
    """
    _validate_experience_root(root)
    config = validate_experience_config(json.loads((root / "config.json").read_text()))
    evaluator = _terminal(
        root / "evaluator/terminal.json", "all_official_actions_applied"
    )
    policy = _terminal(
        root / "policy/terminal.json", "raw_emissions_and_returned_actions_recorded"
    )
    policy_events = _json_lines(root / "policy/events.jsonl")
    evaluator_events = _json_lines(root / "evaluator/events.jsonl")
    _validate_terminal_artifacts(root, evaluator, policy)
    _validate_chronology(config, evaluator, policy, evaluator_events, policy_events)
    _validate_recorded_arrays(root, evaluator_events, policy_events)
    metrics_path, metrics = _official_metrics(output, config["case"])
    if metrics["steps"] != evaluator["frame_count"]:
        raise ValueError("TRAIN experience frame count differs from official steps")
    metrics_copy = root / "official-metrics.json"
    _copy_exact_file(metrics_path, metrics_copy)
    members = _member_inventory(root)
    value = _experience_manifest(
        config, evaluator, policy, metrics_copy, metrics, members
    )
    _atomic_json(root / "experience-manifest.json", value)
    return value


def validate_finalized_experience(
    root: Path, *, official_output: Path | None = None
) -> dict[str, Any]:
    """Validate one immutable finalized TRAIN experience.

    Args:
        root: Finalized experience root.
        official_output: Optional official evaluator output used before publication.
    Returns:
        Exact validated experience manifest.
    Raises:
        ValueError: The root, manifest, chronology, outcome, or members differ.
    """
    _validate_experience_root(root)
    manifest = _terminal_manifest(root / "experience-manifest.json")
    config = validate_experience_config(json.loads((root / "config.json").read_text()))
    evaluator = _terminal(
        root / "evaluator/terminal.json", "all_official_actions_applied"
    )
    policy = _terminal(
        root / "policy/terminal.json", "raw_emissions_and_returned_actions_recorded"
    )
    evaluator_events = _json_lines(root / "evaluator/events.jsonl")
    policy_events = _json_lines(root / "policy/events.jsonl")
    _validate_terminal_artifacts(root, evaluator, policy)
    _validate_chronology(config, evaluator, policy, evaluator_events, policy_events)
    _validate_recorded_arrays(root, evaluator_events, policy_events)
    metrics = _metrics_member(root, config["case"])
    if official_output is not None:
        _validate_official_metrics_source(root, official_output, config["case"])
    _validate_manifest_summary(root, manifest, config, evaluator, policy, metrics)
    if manifest.get("members") != _member_inventory(root):
        raise ValueError("TRAIN experience manifest member inventory differs")
    return manifest


def _validate_experience_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("TRAIN experience root must be a real directory")


def _copy_exact_file(source: Path, destination: Path) -> None:
    if (
        source.is_symlink()
        or not source.is_file()
        or destination.exists()
        or destination.is_symlink()
    ):
        raise ValueError("TRAIN experience official metrics path differs")
    payload = source.read_bytes()
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _terminal_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience success manifest differs")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("TRAIN experience success manifest differs")
    return value


def _experience_manifest(config, evaluator, policy, metrics_path, metrics, members):
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete_exact_train_experience",
        "config": config,
        "cadence": DECISION_CADENCE,
        "episode_count": evaluator["episode_count"],
        "frame_count": evaluator["frame_count"],
        "emission_count": policy["emission_count"],
        "official_metrics": file_identity(metrics_path),
        "q_score": metrics["q_score"]["final"],
        "success": metrics["success"],
        "steps": metrics["steps"],
        "members": members,
        "privileged_state_entered_policy_inputs": False,
        "development_or_report_used": False,
    }


def _terminal(path: Path, status: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience recorder terminal differs")
    value = json.loads(path.read_text())
    schema, keys = _TERMINAL_CONTRACTS[status]
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("schema") != schema
        or value.get("status") != status
    ):
        raise ValueError("TRAIN experience recorder terminal differs")
    return value


def _shard_index(directory: Path) -> list[dict[str, Any]]:
    path = directory / "index.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience shard index differs")
    value = json.loads(path.read_text())
    rows = value.get("shards") if isinstance(value, dict) else None
    names = (
        [row.get("path") for row in rows if isinstance(row, dict)]
        if isinstance(rows, list)
        else []
    )
    if (
        not isinstance(rows, list)
        or not rows
        or len(names) != len(rows)
        or len(set(names)) != len(names)
    ):
        raise ValueError("TRAIN experience shard index differs")
    return rows


def _validate_terminal_artifacts(root: Path, evaluator: dict, policy: dict) -> None:
    expected = (
        (evaluator, "decision_shards", root / "evaluator/decisions"),
        (evaluator, "transition_shards", root / "evaluator/transitions"),
        (
            evaluator,
            "final_observation_shards",
            root / "evaluator/final-observation",
        ),
        (policy, "emission_shards", root / "policy/emissions"),
        (policy, "action_shards", root / "policy/actions"),
    )
    for terminal, name, directory in expected:
        if terminal.get(name) != _shard_index(directory):
            raise ValueError("TRAIN experience terminal shard index differs")
    if evaluator.get("events") != file_identity(root / "evaluator/events.jsonl"):
        raise ValueError("TRAIN experience evaluator events identity differs")
    if policy.get("events") != file_identity(root / "policy/events.jsonl"):
        raise ValueError("TRAIN experience policy events identity differs")


def _validate_chronology(
    config, evaluator, policy, evaluator_events, policy_events
) -> None:
    frames = evaluator.get("frame_count")
    expected_emissions = (frames + ACTION_HORIZON - 1) // ACTION_HORIZON
    if (
        type(frames) is not int
        or frames <= 0
        or evaluator.get("episode_count") != 1
        or policy.get("action_count") != frames
        or policy.get("emission_count") != expected_emissions
    ):
        raise ValueError("TRAIN experience action/emission chronology differs")
    starts = _events(evaluator_events, "episode-start.v1")
    returns = _events(evaluator_events, "policy-return.v1")
    applied = _events(evaluator_events, "official-apply.v1")
    emissions = _events(policy_events, "emission.v1")
    expected_schemas = ["npa.behavior.train-experience-episode-start.v1"]
    expected_schemas.extend(
        schema
        for _ in range(frames)
        for schema in (
            "npa.behavior.train-experience-policy-return.v1",
            "npa.behavior.train-experience-official-apply.v1",
        )
    )
    if (
        len(starts) != evaluator.get("episode_count")
        or len(returns) != frames
        or len(applied) != frames
        or [row.get("schema") for row in evaluator_events] != expected_schemas
    ):
        raise ValueError("TRAIN experience evaluator journal chronology differs")
    if len(emissions) != expected_emissions or len(policy_events) != expected_emissions:
        raise ValueError("TRAIN experience emission journal chronology differs")
    _join_observations(returns, applied)
    _join_actions(returns, applied)
    if config["cadence"] != DECISION_CADENCE:
        raise ValueError("TRAIN experience cadence differs")


def _events(rows: list[dict], suffix: str) -> list[dict]:
    return [row for row in rows if row.get("schema", "").endswith(suffix)]


def _join_observations(returns: list[dict], applied: list[dict]) -> None:
    for index in range(len(returns) - 1):
        if (
            applied[index]["post_apply_observation"]
            != returns[index + 1]["observation"]
        ):
            raise ValueError("Post-apply observation differs from next policy input")
    for index, row in enumerate(returns):
        if row.get("frame_index") != index or row.get(
            "decision_observation_stored"
        ) != (index % ACTION_HORIZON == 0):
            raise ValueError("TRAIN experience decision cadence differs")
        started = row.get("policy_call_started_monotonic_ns")
        completed = row.get("policy_call_completed_monotonic_ns")
        apply_started = applied[index].get("apply_started_monotonic_ns")
        apply_completed = applied[index].get("apply_completed_monotonic_ns")
        times = (started, completed, apply_started, apply_completed)
        if not all(type(value) is int for value in times):
            raise ValueError("TRAIN experience monotonic chronology differs")
        if not started <= completed <= apply_started <= apply_completed:
            raise ValueError("TRAIN experience monotonic chronology differs")
        if index + 1 < len(returns) and apply_completed > returns[index + 1].get(
            "policy_call_started_monotonic_ns", -1
        ):
            raise ValueError("TRAIN experience monotonic chronology differs")


def _join_actions(returns: list[dict], applied: list[dict]) -> None:
    for index, (returned, executed) in enumerate(zip(returns, applied, strict=True)):
        if returned.get("frame_index") != index or executed.get("frame_index") != index:
            raise ValueError("TRAIN experience action index differs")
        if not isinstance(returned.get("returned_action"), dict) or not isinstance(
            executed.get("official_evaluator_applied_action"), dict
        ):
            raise ValueError("TRAIN experience action identity differs")


def _load_shards(directory: Path) -> dict[str, np.ndarray]:
    rows = _shard_index(directory)
    fields: dict[str, list[np.ndarray]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("count")) is not int:
            raise ValueError("TRAIN experience shard row differs")
        relative = Path(str(row.get("path", "")))
        path = directory / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise ValueError("TRAIN experience shard path differs")
        if file_identity(path) != {name: row[name] for name in ("bytes", "sha256")}:
            raise ValueError("TRAIN experience shard identity differs")
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != set(row["arrays"]):
                raise ValueError("TRAIN experience shard inventory differs")
            for name in shard.files:
                array = np.array(shard[name], copy=True)
                if array_identity(array) != row["arrays"][name]:
                    raise ValueError("TRAIN experience shard array differs")
                if array.ndim < 1 or array.shape[0] != row["count"]:
                    raise ValueError("TRAIN experience shard record count differs")
                fields.setdefault(name, []).append(array)
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def _validate_recorded_arrays(root: Path, evaluator_events, policy_events) -> None:
    decisions = _load_shards(root / "evaluator/decisions")
    transitions = _load_shards(root / "evaluator/transitions")
    final_observation = _load_shards(root / "evaluator/final-observation")
    actions = _load_shards(root / "policy/actions")
    emissions = _load_shards(root / "policy/emissions")
    returns = [
        row
        for row in evaluator_events
        if row.get("schema", "").endswith("policy-return.v1")
    ]
    _validate_array_fields(
        root, decisions, transitions, final_observation, actions, emissions
    )
    expected = np.arange(len(returns), dtype=np.int64)
    if not np.array_equal(actions["action_index"], expected):
        raise ValueError("TRAIN experience policy action indices differ")
    ordinals = np.arange(len(emissions["actions"]), dtype=np.int64)
    if not np.array_equal(emissions["chunk_ordinal"], ordinals):
        raise ValueError("TRAIN experience emission indices differ")
    _validate_emission_events(emissions, policy_events)
    if not np.array_equal(actions["chunk_ordinal"], expected // ACTION_HORIZON):
        raise ValueError("TRAIN experience action chunk ordinals differ")
    if not np.array_equal(actions["chunk_offset"], expected % ACTION_HORIZON):
        raise ValueError("TRAIN experience action chunk offsets differ")
    for index, row in enumerate(returns):
        chunk = emissions["actions"][int(actions["chunk_ordinal"][index])]
        offset = int(actions["chunk_offset"][index])
        returned = actions["returned_action"][index]
        if array_identity(returned) != array_identity(chunk[offset]):
            raise ValueError("TRAIN experience source chunk mapping differs")
        if not _official_transport_matches(returned, row["returned_action"]):
            raise ValueError("Official Float32 transport action differs")
    _validate_transition_arrays(transitions, returns, evaluator_events)
    _validate_decision_arrays(decisions, returns)
    _validate_final_observation(final_observation, evaluator_events)


def _validate_emission_events(emissions, events) -> None:
    previous = None
    for ordinal, (chunk, event) in enumerate(
        zip(emissions["actions"], events, strict=True)
    ):
        if event.get("chunk_ordinal") != ordinal or event.get(
            "actions"
        ) != array_identity(chunk):
            raise ValueError("TRAIN experience emission journal differs")
        before = event.get("rng_before_sha256")
        after = event.get("rng_after_sha256")
        started = event.get("inference_started_monotonic_ns")
        completed = event.get("inference_completed_monotonic_ns")
        if (
            not isinstance(before, str)
            or not isinstance(after, str)
            or _DIGEST.fullmatch(before) is None
            or _DIGEST.fullmatch(after) is None
            or before == after
        ):
            raise ValueError("TRAIN experience emission RNG differs")
        if previous is not None and (before != previous[0] or started < previous[1]):
            raise ValueError("TRAIN experience emission sequence differs")
        if (
            not all(type(value) is int for value in (started, completed))
            or started > completed
        ):
            raise ValueError("TRAIN experience emission timing differs")
        previous = after, completed


def _validate_array_fields(
    root, decisions, transitions, final_observation, actions, emissions
) -> None:
    decision_fields = {"frame_index", PROPRIO_KEY, *RGB_KEYS}
    if json.loads((root / "config.json").read_text())["include_depth"]:
        decision_fields.update(DEPTH_KEYS)
    _require_fields(decisions, decision_fields, "decision")
    _require_fields(final_observation, decision_fields, "final observation")
    _require_fields(
        transitions,
        {"frame_index", "pre_proprio", "applied_action", "post_proprio"},
        "transition",
    )
    _require_fields(
        actions,
        {"action_index", "chunk_ordinal", "chunk_offset", "returned_action"},
        "policy action",
    )
    _require_fields(emissions, {"chunk_ordinal", "actions"}, "emission")


def _require_fields(value: dict, expected: set[str], purpose: str) -> None:
    if set(value) != expected:
        raise ValueError(f"TRAIN experience {purpose} array fields differ")


def _validate_transition_arrays(transitions, returns, events) -> None:
    applied = _events(events, "official-apply.v1")
    frames = np.arange(len(returns), dtype=np.int64)
    if not np.array_equal(transitions["frame_index"], frames):
        raise ValueError("TRAIN experience transition indices differ")
    for index, (returned, executed) in enumerate(zip(returns, applied, strict=True)):
        action = transitions["applied_action"][index]
        if array_identity(action) != executed["official_evaluator_applied_action"]:
            raise ValueError("TRAIN experience applied action shard differs")
        if array_identity(action) != returned["returned_action"]:
            raise ValueError("Evaluator returned and applied action bytes differ")
        if (
            array_identity(transitions["pre_proprio"][index])
            != returned["observation"][PROPRIO_KEY]
        ):
            raise ValueError("TRAIN experience transition pre-state differs")
        if (
            array_identity(transitions["post_proprio"][index])
            != executed["post_apply_observation"][PROPRIO_KEY]
        ):
            raise ValueError("TRAIN experience transition post-state differs")


def _validate_decision_arrays(decisions, returns) -> None:
    frames = np.arange(0, len(returns), ACTION_HORIZON, dtype=np.int64)
    if not np.array_equal(decisions["frame_index"], frames):
        raise ValueError("TRAIN experience decision indices differ")
    for offset, frame in enumerate(frames):
        expected = returns[int(frame)]["observation"]
        actual = {
            name: array_identity(value[offset])
            for name, value in decisions.items()
            if name != "frame_index"
        }
        if actual != expected:
            raise ValueError("TRAIN experience decision observation differs")


def _validate_final_observation(value, events) -> None:
    applied = _events(events, "official-apply.v1")
    frame = len(applied)
    if not np.array_equal(value["frame_index"], np.asarray([frame])):
        raise ValueError("TRAIN experience final observation index differs")
    actual = {
        name: array_identity(array[0])
        for name, array in value.items()
        if name != "frame_index"
    }
    if actual != applied[-1]["post_apply_observation"]:
        raise ValueError("TRAIN experience final observation differs")


def _official_metrics(
    output: Path, case: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    path = output / "json" / f"{stem}.json"
    value = json.loads(path.read_text())
    _validate_metrics(value, case)
    return path, value


def _validate_metrics(value: object, case: Mapping[str, Any]) -> None:
    if (
        not isinstance(value, dict)
        or value.get("task") != case["task"]
        or value.get("instance_id") != case["instance_id"]
        or value.get("rollout_id") != case["rollout_id"]
        or type(value.get("steps")) is not int
        or type(value.get("success")) is not bool
        or not isinstance(value.get("q_score"), dict)
    ):
        raise ValueError("TRAIN experience official outcome differs")
    score = value["q_score"].get("final")
    if type(score) not in {int, float} or not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("TRAIN experience official score differs")


def _metrics_member(root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "official-metrics.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN experience official metrics member differs")
    value = json.loads(path.read_text())
    _validate_metrics(value, case)
    return value


def _validate_official_metrics_source(
    root: Path, output: Path, case: Mapping[str, Any]
) -> None:
    source_path, source = _official_metrics(output, case)
    member_path = root / "official-metrics.json"
    member = _metrics_member(root, case)
    if source != member or file_identity(source_path) != file_identity(member_path):
        raise ValueError("TRAIN experience official metrics source differs")


def _validate_manifest_summary(
    root, manifest, config, evaluator, policy, metrics
) -> None:
    expected = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete_exact_train_experience",
        "config": config,
        "cadence": DECISION_CADENCE,
        "episode_count": evaluator["episode_count"],
        "frame_count": evaluator["frame_count"],
        "emission_count": policy["emission_count"],
        "official_metrics": file_identity(root / "official-metrics.json"),
        "q_score": metrics["q_score"]["final"],
        "success": metrics["success"],
        "steps": metrics["steps"],
        "privileged_state_entered_policy_inputs": False,
        "development_or_report_used": False,
    }
    if set(manifest) != set(expected) | {"members"} or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("TRAIN experience success manifest summary differs")


def _member_inventory(root: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path == root / "experience-manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("TRAIN experience contains a symlink")
        if path.is_file():
            result.append(
                {"path": path.relative_to(root).as_posix(), **file_identity(path)}
            )
        elif not path.is_dir():
            raise ValueError("TRAIN experience contains a special member")
    return result


__all__ = [
    "ACTION_HORIZON",
    "EvaluatorExperienceRecorder",
    "PolicyExperienceRecorder",
    "RecordingPolicy",
    "allowed_observation",
    "array_identity",
    "file_identity",
    "finalize_experience",
    "validate_finalized_experience",
    "validate_experience_config",
    "write_experience_config",
]
