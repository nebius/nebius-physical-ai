"""Collect raw task-1 TRAIN semantics without deriving stage labels."""

from __future__ import annotations
from copy import deepcopy
from enum import IntEnum
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence
import numpy as np
from npa.workflows.behavior_challenge import train_official_q

SCHEMA = "npa.private.native-train-semantic-trace-row.v1"
STAGE_SCHEMA = "npa.private.native-train-stage-trace-row.v1"
CANS = tuple(f"can__of__soda.n.01_{index}" for index in range(1, 4))
ASHCAN = "ashcan.n.01_1"
RGB_KEYS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
PROPRIO_KEY = "robot_r1::proprio"
UNDEFINED_FIELDS = (
    "target_can_slot",
    "near",
    "grasp_attempted",
    "release_attempted",
    "phase",
    "promotion_label",
)
SOURCE_MEMBERS = {
    "bddl3/bddl/predicates.py": {
        "bytes": 6730,
        "sha256": "414cd2292a256c2304b1308499d34b65edec4a6af3f065f724242c186ab840f6",
    },
    "OmniGibson/omnigibson/eval/evaluator.py": {
        "bytes": 34580,
        "sha256": "7c66958717945b7c2f618cb6d779828ebcd945c4c4968193360d787f576cb4e5",
    },
    "OmniGibson/omnigibson/controllers/controller_base.py": {
        "bytes": 35161,
        "sha256": "48dbf51cfcbdd891a3fa0c0c3134fe457c5802e317af94d2b4833e3d1b588e46",
    },
    "OmniGibson/omnigibson/tasks/behavior_task.py": {
        "bytes": 43560,
        "sha256": "0daeda0263295b84ac703fdd7561571c0fd65661c090c9ebae9f9f802aca58c7",
    },
    "OmniGibson/omnigibson/metrics/task_metric.py": {
        "bytes": 3414,
        "sha256": "9f9c20fa5c3b8e57c46d81e04e71eb2b63c007809a43631a6b4fe3c791ceef8e",
    },
    "OmniGibson/omnigibson/robots/robot.py": {
        "bytes": 199887,
        "sha256": "d30f98b0c4688899d1a6e767f6c4894576860aa66075c2a17c77af5a7b906297",
    },
    "OmniGibson/omnigibson/utils/bddl_utils.py": {
        "bytes": 74395,
        "sha256": "8c9e62414299848f51995384704dbd03bf83986e5cfbb5788671d8f8aa3baa40",
    },
}
RLC_MEMBER = {
    "path": "src/b1k/shared/eval_b1k_wrapper.py",
    "bytes": 14066,
    "sha256": "59711eefc2829cfee9db30d0985794dd7a12024843e3c7271fcc8364f5b5c97e",
}
RUNTIME_WRAPPERS = {
    (14066, "59711eefc2829cfee9db30d0985794dd7a12024843e3c7271fcc8364f5b5c97e"),
    (14046, "1b75af0bb28b815929ab941ffca8f00cc7adf291d8d5e4ab098c6524a4dd5453"),
}
OFFICIAL_Q_VALIDATOR = {
    "bytes": 9505,
    "sha256": "0ddfcfdc5360368f903f20c16853d67738f2df8bdf12d4fae0ddcd48c28c150b",
}
ROW_KEYS = {
    "schema",
    "frame_index",
    "applied_action_sha256",
    "allowed_observation_sha256",
    "can_predicates",
    "goal_option_masks",
    "official_q",
    "undefined_fields",
    "source",
}
CAN_KEYS = {"slot", "object_scope_name", "inside_target", "grasped_by_arm"}
STAGE_KEYS = {
    "schema",
    "action_index",
    "inference_ordinal",
    "new_inference",
    "stage_before",
    "raw_predicted_stage",
    "stage_after",
    "action_sha256",
    "source",
}


def file_identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source member differs")
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def source_authority(
    behavior_root: Path,
    rlc_root: Path,
    *,
    runtime_wrapper_path: Path | None = None,
) -> dict[str, Any]:
    """Validate exact official evaluator and native wrapper bytes."""
    official = {name: file_identity(behavior_root / name) for name in SOURCE_MEMBERS}
    native = file_identity(rlc_root / RLC_MEMBER["path"])
    if official != SOURCE_MEMBERS or native != {
        key: RLC_MEMBER[key] for key in ("bytes", "sha256")
    }:
        raise ValueError("semantic trace source authority differs")
    runtime = file_identity(runtime_wrapper_path or rlc_root / RLC_MEMBER["path"])
    if (runtime["bytes"], runtime["sha256"]) not in RUNTIME_WRAPPERS:
        raise ValueError("semantic trace runtime wrapper differs")
    return {
        "behavior_commit": "6cbf70b075816096e9be53958780769f3264d25d",
        "rlc_commit": "ca556f74a455cef7987a2be4537b5ac85cc56dd7",
        "official": deepcopy(official),
        "native_wrapper": deepcopy(RLC_MEMBER),
        "runtime_wrapper": runtime,
    }


def array_sha256(value: Any) -> str:
    """Hash dtype, shape, and contiguous bytes for one numeric array."""
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ValueError("trace array differs")
    if not np.isfinite(array).all():
        raise ValueError("trace array is not finite")
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(
        header + b"\0" + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def allowed_observation_sha256(observation: Mapping[str, Any]) -> str:
    """Hash the exact three onboard RGB views and 61-value proprioception."""
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    selected = {}
    state = _unbatched(observation[PROPRIO_KEY], {1}, "proprioception")
    if state.shape != (61,) or not np.issubdtype(state.dtype, np.number):
        raise ValueError("semantic trace proprioception differs")
    selected[PROPRIO_KEY] = array_sha256(state)
    for key in RGB_KEYS:
        image = _unbatched(observation[key], {3}, "RGB")
        size = 720 if "zed_link" in key else 480
        if image.shape not in {(size, size, 3), (size, size, 4)}:
            raise ValueError("semantic trace RGB shape differs")
        if image.dtype != np.uint8:
            raise ValueError("semantic trace RGB dtype differs")
        selected[key] = array_sha256(image[..., :3])
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _unbatched(value: Any, ranks: set[int], name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim in ranks:
        return array
    if array.ndim - 1 in ranks and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"semantic trace {name} batch differs")


def _scalar_bool(value: Any, name: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise ValueError(f"{name} differs")
    return bool(array)


def _grasp_bool(value: Any) -> bool:
    """Map the exact upstream three-state grasp enum without bool coercion."""
    enum = type(value)
    if (
        not isinstance(value, IntEnum)
        or enum.__name__ != "IsGraspingState"
        or enum.__module__ != "omnigibson.controllers.controller_base"
    ):
        raise ValueError("raw grasp predicate differs")
    if value.name == "TRUE" and int(value) == 1:
        return True
    if value.name == "FALSE" and int(value) == -1:
        return False
    if value.name == "UNKNOWN" and int(value) == 0:
        raise ValueError("raw grasp predicate is unknown")
    raise ValueError("raw grasp predicate differs")


def _masks(value: Any) -> list[list[bool]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("official goal masks differ")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("official goal option differs")
        if any(type(item) is not bool for item in row):
            raise ValueError("official goal predicate differs")
        rows.append(list(row))
    return rows


class EvaluatorSemanticObserver:
    """Read raw post-apply predicates from one exact evaluator environment."""

    def __init__(
        self,
        evaluator: Any,
        official_q_observer: Any,
        source: Mapping[str, Any],
        *,
        inside_predicate: Any | None = None,
    ):
        if inside_predicate is None:
            from bddl import predicates

            inside_predicate = predicates.Inside

        states = getattr(evaluator, "instance_eval_states", None)
        if not isinstance(states, list) or len(states) != 1:
            raise ValueError("semantic trace requires one evaluator environment")
        self.accessor = states[0].env_accessor
        self.official_q = official_q_observer
        self.inside_predicate = inside_predicate
        self.source = deepcopy(dict(source))
        scope = self.accessor.object_scope
        if any(name not in scope for name in (*CANS, ASHCAN)):
            raise ValueError("task-1 object scope differs")
        arms = tuple(self.accessor.robot.arm_names)
        if set(arms) != {"left", "right"} or len(arms) != 2:
            raise ValueError("task-1 robot arms differ")
        self.arms = tuple(sorted(arms))

    def observe(
        self, frame_index: int, applied_action: Any, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Read raw predicates after the unchanged official apply call."""
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("semantic trace frame differs")
        task = self.accessor.shared_env.task
        scope = self.accessor.object_scope
        cans = []
        for slot, name in enumerate(CANS):
            obj = scope[name]
            grasp = {
                arm: _grasp_bool(
                    self.accessor.robot.is_grasping(arm=arm, candidate_obj=obj),
                )
                for arm in self.arms
            }
            inside = _scalar_bool(
                task._evaluate_predicate(
                    self.accessor.env_idx, self.inside_predicate, name, ASHCAN
                ),
                "raw inside predicate",
            )
            cans.append(
                {
                    "slot": slot,
                    "object_scope_name": name,
                    "inside_target": inside,
                    "grasped_by_arm": grasp,
                }
            )
        return {
            "schema": SCHEMA,
            "frame_index": frame_index,
            "applied_action_sha256": array_sha256(applied_action),
            "allowed_observation_sha256": allowed_observation_sha256(observation),
            "can_predicates": cans,
            "goal_option_masks": _masks(self.accessor.get_goal_option_satisfaction()),
            "official_q": self.official_q.observe(frame_index),
            "undefined_fields": list(UNDEFINED_FIELDS),
            "source": deepcopy(self.source),
        }


class StageTrace:
    """Record raw native stage state and model proposal for returned actions."""

    def __init__(self, source: Mapping[str, Any]) -> None:
        self.source = deepcopy(dict(source))
        self.pending: dict[str, int] | None = None
        self.rows: list[dict[str, Any]] = []

    def record_inference(
        self, inputs: Mapping[str, Any], result: Mapping[str, Any]
    ) -> None:
        """Capture exact current stage and raw argmax emitted by PiBehavior."""
        if self.pending is not None:
            raise ValueError("unconsumed native stage inference")
        before = np.asarray(inputs.get("subtask_state"))
        logits = np.asarray(result.get("subtask_logits"))
        if before.shape != () or not np.issubdtype(before.dtype, np.integer):
            raise ValueError("native stage input differs")
        valid_logits = (
            logits.ndim == 1
            and logits.size >= 1
            and np.issubdtype(logits.dtype, np.floating)
            and not np.isnan(logits).any()
            and not np.isposinf(logits).any()
            and np.isfinite(logits).any()
        )
        if not valid_logits:
            raise ValueError("native raw stage proposal differs")
        self.pending = {
            "stage_before": int(before),
            "raw_predicted_stage": int(np.argmax(logits)),
        }

    def record_action(
        self,
        *,
        action_index: int,
        inference_ordinal: int,
        new_inference: bool,
        stage_before: int,
        stage_after: int,
        action: Any,
    ) -> dict[str, Any]:
        """Bind one returned action to raw proposal and accepted native stage."""
        integers = (action_index, inference_ordinal, stage_before, stage_after)
        if any(type(item) is not int or item < 0 for item in integers):
            raise ValueError("native stage chronology differs")
        if action_index != len(self.rows) or type(new_inference) is not bool:
            raise ValueError("native action index differs")
        proposal = None
        if new_inference:
            if self.pending is None or self.pending["stage_before"] != stage_before:
                raise ValueError("native stage inference join differs")
            proposal = self.pending["raw_predicted_stage"]
            self.pending = None
        elif self.pending is not None:
            raise ValueError("native stage proposal cadence differs")
        row = {
            "schema": STAGE_SCHEMA,
            "action_index": action_index,
            "inference_ordinal": inference_ordinal,
            "new_inference": new_inference,
            "stage_before": stage_before,
            "raw_predicted_stage": proposal,
            "stage_after": stage_after,
            "action_sha256": array_sha256(action),
            "source": deepcopy(self.source),
        }
        self.rows.append(row)
        return deepcopy(row)


class NativeStageObserver:
    """Observe one exact native RLC wrapper without changing returned actions."""

    def __init__(
        self, wrapper: Any, trace: StageTrace, *, wrapper_path: Path | None = None
    ) -> None:
        path = wrapper_path or Path(inspect.getsourcefile(type(wrapper)) or "")
        if file_identity(path) != trace.source.get("runtime_wrapper"):
            raise ValueError("live native wrapper source differs")
        self.wrapper = wrapper
        self.trace = trace
        self.native_update = wrapper.update_current_stage
        wrapper.update_current_stage = MethodType(self._update, wrapper)

    def _update(self, native_wrapper: Any, logits: Any) -> Any:
        before = int(native_wrapper.current_stage)
        self.trace.record_inference(
            {"subtask_state": np.asarray(before, dtype=np.int64)},
            {"subtask_logits": logits},
        )
        return self.native_update(logits)

    def act(self, observation: Mapping[str, Any]) -> Any:
        """Return the exact native action after recording its stage chronology."""
        stage_before = int(self.wrapper.current_stage)
        predictions_before = int(self.wrapper.prediction_count)
        action = self.wrapper.act(observation)
        predictions_after = int(self.wrapper.prediction_count)
        new_inference = predictions_after == predictions_before + 1
        if predictions_after not in {predictions_before, predictions_before + 1}:
            raise ValueError("native inference cadence differs")
        if new_inference:
            if self.trace.pending is None:
                raise ValueError("native stage proposal is missing")
            stage_before = self.trace.pending["stage_before"]
        elif predictions_after == 0:
            raise ValueError("native action precedes its first inference")
        self.trace.record_action(
            action_index=len(self.trace.rows),
            inference_ordinal=predictions_after - 1,
            new_inference=new_inference,
            stage_before=stage_before,
            stage_after=int(self.wrapper.current_stage),
            action=action,
        )
        return action


def _validate_can_rows(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("raw can predicates differ")
    for slot, row in enumerate(value):
        if (
            not isinstance(row, Mapping)
            or set(row) != CAN_KEYS
            or row["slot"] != slot
            or row["object_scope_name"] != CANS[slot]
            or type(row["inside_target"]) is not bool
            or set(row["grasped_by_arm"]) != {"left", "right"}
            or any(type(item) is not bool for item in row["grasped_by_arm"].values())
        ):
            raise ValueError("raw can predicate row differs")


def validate_join(
    semantic_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require one raw semantic and stage row for every applied action."""
    if not semantic_rows or len(semantic_rows) != len(stage_rows):
        raise ValueError("semantic trace action alignment differs")
    if file_identity(Path(train_official_q.__file__)) != OFFICIAL_Q_VALIDATOR:
        raise ValueError("official Q validator source differs")
    official_rows = []
    for semantic, stage in zip(semantic_rows, stage_rows, strict=True):
        if not isinstance(semantic, Mapping) or not isinstance(stage, Mapping):
            raise ValueError("semantic trace row fields differ")
        if set(semantic) != ROW_KEYS or set(stage) != STAGE_KEYS:
            raise ValueError("semantic trace row fields differ")
        official_rows.append(semantic["official_q"])
    validated_q = train_official_q.validate_official_q_rows(
        official_rows,
        train_official_q.recompute_official_q,
        train_official_q.OFFICIAL_Q_SOURCE,
    )
    for frame, (semantic, stage) in enumerate(
        zip(semantic_rows, stage_rows, strict=True)
    ):
        _validate_can_rows(semantic["can_predicates"])
        goal_option_masks = _masks(semantic["goal_option_masks"])
        if (
            semantic["schema"] != SCHEMA
            or stage["schema"] != STAGE_SCHEMA
            or semantic["frame_index"] != frame
            or stage["action_index"] != frame
            or semantic["applied_action_sha256"] != stage["action_sha256"]
            or semantic["source"] != stage["source"]
            or semantic["undefined_fields"] != list(UNDEFINED_FIELDS)
        ):
            raise ValueError("semantic trace action/frame join differs")
        official_q = validated_q[frame]
        if (
            official_q["frame_index"] != frame
            or goal_option_masks != official_q["current_satisfied_options"]
        ):
            raise ValueError("semantic trace official Q join differs")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _write_exclusive(path: Path, payload: bytes) -> dict[str, int | str]:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return file_identity(path)


def write_trace(
    root: Path,
    semantic_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write one locally durable audit-only trace after exact cross-process joins."""
    validate_join(semantic_rows, stage_rows)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(mode=0o700)
    members = {
        "semantic.jsonl": _write_exclusive(
            root / "semantic.jsonl", _jsonl_bytes(semantic_rows)
        ),
        "native-stage.jsonl": _write_exclusive(
            root / "native-stage.jsonl", _jsonl_bytes(stage_rows)
        ),
    }
    terminal = {
        "schema": "npa.private.native-train-semantic-trace.v1",
        "status": "raw_train_trace_complete_audit_only",
        "action_count": len(semantic_rows),
        "members": members,
        "source": deepcopy(dict(semantic_rows[0]["source"])),
        "undefined_fields": list(UNDEFINED_FIELDS),
        "training_ready": False,
        "policy_changed": False,
    }
    payload = json.dumps(terminal, indent=2, sort_keys=True).encode() + b"\n"
    _write_exclusive(root / "terminal.json", payload)
    return terminal


__all__ = [
    "EvaluatorSemanticObserver",
    "NativeStageObserver",
    "SCHEMA",
    "STAGE_SCHEMA",
    "StageTrace",
    "UNDEFINED_FIELDS",
    "allowed_observation_sha256",
    "array_sha256",
    "source_authority",
    "validate_join",
    "write_trace",
]
