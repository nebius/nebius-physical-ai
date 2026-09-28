"""Record lossless allowed Native RLC TRAIN tensors without policy changes."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

RGB_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
PROPRIO_KEY = "robot_r1::proprio"
ACTION_DIMENSION = 23
MODEL_HORIZON = 30
EXECUTED_PREFIX = 20
ACTION_ROWS_PER_SHARD = 256
DECISION_ROWS_PER_SHARD = 8


def array_identity(value: Any) -> dict[str, Any]:
    """Return the exact dtype, shape, bytes, and digest of a numeric array."""
    array = _numeric(value)
    payload = np.ascontiguousarray(array).tobytes()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": len(payload),
        "sha256": hashlib.sha256(header + b"\0" + payload).hexdigest(),
    }


def allowed_observation(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy only the three official RGB views and 61-value proprioception."""
    state = _unbatch(value[PROPRIO_KEY], 1)
    if state.dtype != np.float32 or state.shape != (61,):
        raise ValueError("Native TRAIN proprioception differs")
    result = {PROPRIO_KEY: state}
    for key in RGB_KEYS:
        image = _unbatch(value[key], 3)
        size = 720 if "zed_link" in key else 480
        if image.dtype != np.uint8 or image.shape not in {
            (size, size, 3),
            (size, size, 4),
        }:
            raise ValueError("Native TRAIN RGB differs")
        result[key] = np.array(image[..., :3], copy=True)
    return result


def observation_sha256(value: Mapping[str, Any]) -> str:
    """Hash an allowed observation using the reviewed trace convention."""
    selected = {name: array_identity(array)["sha256"] for name, array in value.items()}
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _numeric(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Native TRAIN array differs")
    return np.array(array, copy=True)


def _unbatch(value: Any, rank: int) -> np.ndarray:
    array = _numeric(value)
    if array.ndim == rank + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != rank:
        raise ValueError("Native TRAIN observation batch differs")
    return array


def _action(value: Any) -> np.ndarray:
    action = _numeric(value)
    if action.shape == (1, ACTION_DIMENSION):
        action = action[0]
    if action.shape != (ACTION_DIMENSION,):
        raise ValueError("Native TRAIN action differs")
    return action


def _file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class _ShardWriter:
    """Write fixed-shape numeric rows to exclusive compressed NPZ shards."""

    def __init__(self, root: Path, stem: str, rows_per_shard: int) -> None:
        self.root = root
        self.stem = stem
        self.rows_per_shard = rows_per_shard
        self.pending: list[dict[str, np.ndarray]] = []
        self.rows: list[dict[str, Any]] = []
        self.count = 0

    def append(self, value: Mapping[str, Any]) -> None:
        row = {name: _numeric(array) for name, array in value.items()}
        if self.pending:
            expected = {
                name: (array.dtype.str, array.shape)
                for name, array in self.pending[0].items()
            }
            observed = {
                name: (array.dtype.str, array.shape) for name, array in row.items()
            }
            if observed != expected:
                raise ValueError("Native TRAIN shard fields differ")
        self.pending.append(row)
        self.count += 1
        if len(self.pending) == self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.pending:
            return
        start = self.count - len(self.pending)
        end = self.count
        path = self.root / f"{self.stem}-{start:06d}-{end:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        stacked = {
            name: np.stack([row[name] for row in self.pending])
            for name in self.pending[0]
        }
        with path.open("xb") as stream:
            np.savez_compressed(stream, **stacked)
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(
            {"start": start, "end": end, "path": path.name, **_file_identity(path)}
        )
        self.pending.clear()

    def close(self) -> list[dict[str, Any]]:
        self._flush()
        if not self.rows:
            raise ValueError("Native TRAIN shard set is empty")
        return self.rows


class NativePolicyArrays:
    """Capture decision inputs, raw emissions, executed chunks, and returned actions."""

    def __init__(self, root: Path, wrapper: Any) -> None:
        _create_root(root)
        self.root = root
        self.wrapper = wrapper
        self.inference = 0
        self.action = 0
        self.pending_raw: np.ndarray | None = None
        self.decisions = _ShardWriter(root, "decision", DECISION_ROWS_PER_SHARD)
        self.actions = _ShardWriter(root, "returned", ACTION_ROWS_PER_SHARD)
        self.original_infer = wrapper.policy.infer
        wrapper.policy.infer = MethodType(self._infer, wrapper.policy)

    def _infer(self, delegate: Any, *args: Any, **kwargs: Any) -> Any:
        del delegate
        if self.pending_raw is not None:
            raise ValueError("Native TRAIN emission remains unconsumed")
        result = self.original_infer(*args, **kwargs)
        if not isinstance(result, Mapping):
            raise ValueError("Native TRAIN inference result differs")
        raw = _numeric(result.get("actions"))
        if raw.shape == (1, MODEL_HORIZON, ACTION_DIMENSION):
            raw = raw[0]
        if raw.shape != (MODEL_HORIZON, ACTION_DIMENSION):
            raise ValueError("Native TRAIN raw model horizon differs")
        self.pending_raw = raw
        return result

    def record(
        self, observation: Mapping[str, Any], action: Any, stage: Mapping[str, Any]
    ) -> None:
        """Record one unchanged action and any decision that produced it."""
        index = self.action
        if stage.get("action_index") != index:
            raise ValueError("Native TRAIN stage/action chronology differs")
        self.actions.append(
            {"action_index": np.asarray(index), "returned_action": _action(action)}
        )
        if stage.get("new_inference"):
            self._record_decision(observation, stage, index)
        elif self.pending_raw is not None:
            raise ValueError("Native TRAIN unexpected raw emission")
        self.action += 1

    def _record_decision(self, observation, stage, index) -> None:
        if self.pending_raw is None or stage.get("inference_ordinal") != self.inference:
            raise ValueError("Native TRAIN decision chronology differs")
        executed = _numeric(self.wrapper.last_actions)
        if executed.shape == (1, EXECUTED_PREFIX, ACTION_DIMENSION):
            executed = executed[0]
        if executed.shape != (EXECUTED_PREFIX, ACTION_DIMENSION):
            raise ValueError("Native TRAIN executed chunk differs")
        self.decisions.append(
            {
                "action_index": np.asarray(index),
                "inference_ordinal": np.asarray(self.inference),
                "raw_model_actions": self.pending_raw,
                "raw_wrapper_actions": executed,
                **allowed_observation(observation),
            }
        )
        self.pending_raw = None
        self.inference += 1

    def close(self) -> dict[str, Any]:
        """Write a terminal only after every emission was consumed."""
        if self.pending_raw is not None or self.action < 1 or self.inference < 1:
            raise ValueError("Native TRAIN policy arrays ended incomplete")
        return _write_terminal(
            self.root,
            {
                "schema": "npa.behavior.native-train-policy-arrays.v1",
                "status": "allowed_decisions_chunks_and_returned_actions_recorded",
                "action_count": self.action,
                "inference_count": self.inference,
                "model_horizon": MODEL_HORIZON,
                "executed_prefix": EXECUTED_PREFIX,
                "decision_shards": self.decisions.close(),
                "returned_action_shards": self.actions.close(),
            },
        )


class NativeEvaluatorArrays:
    """Capture exact applied actions and the final allowed observation."""

    def __init__(self, root: Path) -> None:
        _create_root(root)
        self.root = root
        self.actions = _ShardWriter(root, "applied", ACTION_ROWS_PER_SHARD)
        self.final: dict[str, np.ndarray] | None = None

    def record(self, observation: Mapping[str, Any], action: Any, frame: int) -> None:
        """Record one official applied action and retain its post-apply endpoint."""
        if frame != self.actions.count:
            raise ValueError("Native TRAIN applied-action chronology differs")
        self.actions.append(
            {"frame_index": np.asarray(frame), "applied_action": _action(action)}
        )
        self.final = allowed_observation(observation)

    def close(self) -> dict[str, Any]:
        """Write the terminal with one materialized final observation."""
        if self.actions.count < 1 or self.final is None:
            raise ValueError("Native TRAIN evaluator arrays ended incomplete")
        path = self.root / "final-observation.npz"
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                after_action_index=np.asarray(self.actions.count - 1),
                observation_index=np.asarray(self.actions.count),
                **self.final,
            )
            stream.flush()
            os.fsync(stream.fileno())
        return _write_terminal(
            self.root,
            {
                "schema": "npa.behavior.native-train-evaluator-arrays.v1",
                "status": "applied_actions_and_final_observation_recorded",
                "action_count": self.actions.count,
                "applied_action_shards": self.actions.close(),
                "final_observation": {"path": path.name, **_file_identity(path)},
            },
        )


def _write_terminal(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    path = root / "terminal.json"
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return value


def _create_root(root: Path) -> None:
    """Create one fresh recorder root before fragment inventory checks."""
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(mode=0o700, parents=True)
