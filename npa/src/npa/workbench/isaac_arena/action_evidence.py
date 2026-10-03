"""Measure policy actions at the real policy-to-environment boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .errors import IsaacArenaError
from .hashing import file_sha256

ACTION_EVIDENCE_SCHEMA = "npa.isaac-arena.executed-actions.v1"
ACTION_SEQUENCE_SCHEMA = "npa.isaac-arena.action-sequence.v1"
ACTION_EVIDENCE_FILENAME = "simulator-action-evidence.json"
_ACTION_STATE_ATTRIBUTE = "_npa_executed_action_evidence"
ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE = 1e-6


def _base_environment(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _canonical_action(action: Any) -> tuple[Any, bytes, str]:
    import numpy as np

    value = action
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    if (
        array.ndim < 1
        or not array.size
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise IsaacArenaError("executed policy action must be a finite numeric tensor")
    canonical = np.ascontiguousarray(array, dtype="<f8")
    header = json.dumps(
        {
            "schema": ACTION_SEQUENCE_SCHEMA,
            "shape": list(canonical.shape),
            "dtype": "float64-little-endian",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded = len(header).to_bytes(8, "big") + header + canonical.tobytes()
    return canonical, encoded, hashlib.sha256(encoded).hexdigest()


def _sequence_sha256(step_hashes: Iterable[str]) -> str:
    payload = json.dumps(
        {
            "schema": ACTION_SEQUENCE_SCHEMA,
            "action_step_sha256": list(step_hashes),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _trailing_held_steps(interstep_deltas: list[float], action_steps: int) -> int:
    """Count the numerically held final-action run, including its first action."""

    if not action_steps:
        return 0
    count = 1
    for delta in reversed(interstep_deltas):
        if delta > ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE:
            break
        count += 1
    return count


def _hashes_and_deltas_consistent(
    step_hashes: list[str], interstep_deltas: list[float]
) -> bool:
    """Reject impossible summaries without retaining private action values."""

    return len(interstep_deltas) == max(0, len(step_hashes) - 1) and all(
        delta == 0.0
        for previous, current, delta in zip(
            step_hashes, step_hashes[1:], interstep_deltas
        )
        if previous == current
    )


def action_sequence_evidence(actions: Any) -> dict[str, Any]:
    """Hash a prepared step-major action tensor without retaining action values."""

    import numpy as np

    values = np.asarray(actions)
    if values.ndim < 2 or values.shape[0] < 1:
        raise IsaacArenaError("prepared replay actions must be a step-major tensor")
    records = [_canonical_action(values[index]) for index in range(values.shape[0])]
    arrays = [record[0] for record in records]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise IsaacArenaError(
            "prepared replay action shapes changed within the episode"
        )
    flattened = np.concatenate([array.reshape(-1) for array in arrays])
    deltas = [
        float(np.max(np.abs(current - previous)))
        for previous, current in zip(arrays, arrays[1:])
    ]
    hashes = [record[2] for record in records]
    trailing_held_steps = _trailing_held_steps(deltas, len(records))
    nonzero_fraction = float(np.mean(np.abs(flattened) > 1e-6))
    action_abs_max = float(np.max(np.abs(flattened)))
    return {
        "schema": ACTION_SEQUENCE_SCHEMA,
        "steps": len(records),
        "action_shape": list(arrays[0].shape),
        "action_step_sha256": hashes,
        "sequence_sha256": _sequence_sha256(hashes),
        "finite_actions": True,
        "action_abs_max": action_abs_max,
        "action_nonzero_fraction": nonzero_fraction,
        "nonzero_actions": action_abs_max >= 1e-4 and nonzero_fraction >= 0.001,
        "distinct_action_steps": len(set(hashes)),
        "interstep_delta_abs_max": deltas,
        "action_step_delta_abs_max": max(deltas, default=0.0),
        "varied_actions": len(set(hashes)) > 1 and max(deltas, default=0.0) >= 1e-6,
        "held_action_delta_abs_max_tolerance": ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE,
        "trailing_held_action_steps": trailing_held_steps,
        "trailing_held_action_fraction": trailing_held_steps / len(records),
    }


def configure_action_evidence(
    env: Any, output_dir: str | Path, policy_type: str, rank: int = 0
) -> None:
    """Start one sanitized action journal before the upstream rollout."""

    if rank != 0:
        raise IsaacArenaError("visual action evidence supports only local rank zero")
    if not isinstance(policy_type, str) or not policy_type:
        raise IsaacArenaError("action evidence requires an explicit policy type")
    base = _base_environment(env)
    if hasattr(base, _ACTION_STATE_ATTRIBUTE):
        raise IsaacArenaError("action evidence is already configured")
    setattr(
        base,
        _ACTION_STATE_ATTRIBUTE,
        {
            "path": Path(output_dir) / ACTION_EVIDENCE_FILENAME,
            "policy_type": policy_type,
            "arrays": [],
            "step_hashes": [],
            "finalized": False,
        },
    )


def record_executed_action(env: Any, action: Any, action_step: int) -> None:
    """Record an action only after the corresponding native env.step returns."""

    base = _base_environment(env)
    state = getattr(base, _ACTION_STATE_ATTRIBUTE, None)
    if not isinstance(state, dict) or state.get("finalized"):
        raise IsaacArenaError("action evidence is not active")
    expected = len(state["arrays"]) + 1
    if type(action_step) is not int or action_step != expected:
        raise IsaacArenaError("executed policy action steps are not contiguous")
    array, _encoded, digest = _canonical_action(action)
    if state["arrays"] and array.shape != state["arrays"][0].shape:
        raise IsaacArenaError("executed policy action shape changed within the episode")
    state["arrays"].append(array)
    state["step_hashes"].append(digest)


def _journal_payload(state: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    arrays = state["arrays"]
    hashes = state["step_hashes"]
    if arrays:
        flattened = np.concatenate([array.reshape(-1) for array in arrays])
        deltas = [
            float(np.max(np.abs(current - previous)))
            for previous, current in zip(arrays, arrays[1:])
        ]
        action_abs_max = float(np.max(np.abs(flattened)))
        nonzero_fraction = float(np.mean(np.abs(flattened) > 1e-6))
        action_shape = list(arrays[0].shape)
    else:
        deltas = []
        action_abs_max = 0.0
        nonzero_fraction = 0.0
        action_shape = []
    distinct = len(set(hashes))
    maximum_delta = max(deltas, default=0.0)
    trailing_held_steps = _trailing_held_steps(deltas, len(arrays))
    return {
        "schema": ACTION_EVIDENCE_SCHEMA,
        "source": "upstream policy.get_action to env.step boundary",
        "recording_phase": "after_native_env_step_return",
        "policy_type": state["policy_type"],
        "executed_steps": len(arrays),
        "action_steps": list(range(1, len(arrays) + 1)),
        "action_shape": action_shape,
        "action_step_sha256": hashes,
        "sequence_sha256": _sequence_sha256(hashes),
        "finite_actions": True,
        "action_abs_max": action_abs_max,
        "action_nonzero_fraction": nonzero_fraction,
        "nonzero_actions": action_abs_max >= 1e-4 and nonzero_fraction >= 0.001,
        "distinct_action_steps": distinct,
        "interstep_delta_abs_max": deltas,
        "action_step_delta_abs_max": maximum_delta,
        "varied_actions": distinct > 1 and maximum_delta >= 1e-6,
        "held_action_delta_abs_max_tolerance": ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE,
        "trailing_held_action_steps": trailing_held_steps,
        "trailing_held_action_fraction": (
            trailing_held_steps / len(arrays) if arrays else 0.0
        ),
        "synthetic_padding_steps": 0,
        "raw_actions_retained": False,
    }


def finalize_action_evidence(env: Any) -> None:
    """Atomically retain the sanitized action journal on every rollout exit."""

    base = _base_environment(env)
    state = getattr(base, _ACTION_STATE_ATTRIBUTE, None)
    if not isinstance(state, dict) or state.get("finalized"):
        return
    payload = _journal_payload(state)
    path = state["path"]
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    state["finalized"] = True
    state["arrays"].clear()


def validate_action_evidence(
    record: Any,
    *,
    policy_type: str,
    expected_steps: int,
    require_nonzero_varied: bool,
    maximum_trailing_held_fraction: float | None = None,
) -> dict[str, Any]:
    """Validate an action journal supplied to the shared acceptance contract."""

    if not isinstance(record, dict):
        raise IsaacArenaError("visual acceptance requires measured executed actions")
    hashes = record.get("action_step_sha256")
    trailing_steps = record.get("trailing_held_action_steps")
    trailing_fraction = record.get("trailing_held_action_fraction")
    interstep_deltas = record.get("interstep_delta_abs_max")
    numeric = (
        record.get("action_abs_max"),
        record.get("action_nonzero_fraction"),
        record.get("action_step_delta_abs_max"),
    )
    valid = (
        record.get("schema") == ACTION_EVIDENCE_SCHEMA
        and record.get("source") == "upstream policy.get_action to env.step boundary"
        and record.get("recording_phase") == "after_native_env_step_return"
        and record.get("policy_type") == policy_type
        and type(record.get("executed_steps")) is int
        and record["executed_steps"] == expected_steps
        and record.get("action_steps") == list(range(1, expected_steps + 1))
        and isinstance(record.get("action_shape"), list)
        and bool(record["action_shape"])
        and all(type(value) is int and value > 0 for value in record["action_shape"])
        and isinstance(hashes, list)
        and len(hashes) == expected_steps
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in hashes
        )
        and record.get("sequence_sha256") == _sequence_sha256(hashes)
        and record.get("finite_actions") is True
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in numeric
        )
        and isinstance(interstep_deltas, list)
        and len(interstep_deltas) == max(0, expected_steps - 1)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in interstep_deltas
        )
        and _hashes_and_deltas_consistent(hashes, interstep_deltas)
        and math.isclose(
            float(record["action_step_delta_abs_max"]),
            max(interstep_deltas, default=0.0),
            abs_tol=1e-12,
        )
        and type(record.get("distinct_action_steps")) is int
        and record["distinct_action_steps"] == len(set(hashes))
        and expected_steps > 0
        and record.get("held_action_delta_abs_max_tolerance")
        == ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE
        and type(trailing_steps) is int
        and trailing_steps == _trailing_held_steps(interstep_deltas, expected_steps)
        and isinstance(trailing_fraction, (int, float))
        and not isinstance(trailing_fraction, bool)
        and math.isfinite(float(trailing_fraction))
        and math.isclose(
            float(trailing_fraction), trailing_steps / expected_steps, abs_tol=1e-12
        )
        and record.get("synthetic_padding_steps") == 0
        and record.get("raw_actions_retained") is False
    )
    if require_nonzero_varied:
        valid = (
            valid
            and record.get("nonzero_actions") is True
            and float(record["action_abs_max"]) >= 1e-4
            and float(record["action_nonzero_fraction"]) >= 0.001
            and record.get("varied_actions") is True
            and record["distinct_action_steps"] > 1
            and float(record["action_step_delta_abs_max"]) >= 1e-6
        )
    if maximum_trailing_held_fraction is not None:
        valid = (
            valid
            and 0 <= maximum_trailing_held_fraction < 1
            and float(trailing_fraction) <= maximum_trailing_held_fraction
        )
    if not valid:
        raise IsaacArenaError(
            "visual acceptance requires measured nonzero varied actions without padding"
        )
    return record


def validate_prepared_action_sequence(
    record: Any,
    *,
    expected_steps: int,
    maximum_trailing_held_fraction: float,
) -> dict[str, Any]:
    """Validate the private replay tensor's sanitized sequence commitment."""

    if not isinstance(record, dict):
        raise IsaacArenaError("replay has no prepared action-sequence commitment")
    hashes = record.get("action_step_sha256")
    trailing_steps = record.get("trailing_held_action_steps")
    trailing_fraction = record.get("trailing_held_action_fraction")
    interstep_deltas = record.get("interstep_delta_abs_max")
    numeric = (
        record.get("action_abs_max"),
        record.get("action_nonzero_fraction"),
        record.get("action_step_delta_abs_max"),
    )
    valid = (
        record.get("schema") == ACTION_SEQUENCE_SCHEMA
        and record.get("steps") == expected_steps
        and isinstance(record.get("action_shape"), list)
        and bool(record["action_shape"])
        and all(type(value) is int and value > 0 for value in record["action_shape"])
        and isinstance(hashes, list)
        and len(hashes) == expected_steps
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in hashes
        )
        and record.get("sequence_sha256") == _sequence_sha256(hashes)
        and record.get("finite_actions") is True
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in numeric
        )
        and isinstance(interstep_deltas, list)
        and len(interstep_deltas) == max(0, expected_steps - 1)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in interstep_deltas
        )
        and _hashes_and_deltas_consistent(hashes, interstep_deltas)
        and math.isclose(
            float(record["action_step_delta_abs_max"]),
            max(interstep_deltas, default=0.0),
            abs_tol=1e-12,
        )
        and record.get("nonzero_actions") is True
        and float(record["action_abs_max"]) >= 1e-4
        and float(record["action_nonzero_fraction"]) >= 0.001
        and record.get("distinct_action_steps") == len(set(hashes))
        and record.get("varied_actions") is True
        and record["distinct_action_steps"] > 1
        and float(record["action_step_delta_abs_max"]) >= 1e-6
        and expected_steps > 0
        and record.get("held_action_delta_abs_max_tolerance")
        == ACTION_HOLD_DELTA_ABS_MAX_TOLERANCE
        and type(trailing_steps) is int
        and trailing_steps == _trailing_held_steps(interstep_deltas, expected_steps)
        and isinstance(trailing_fraction, (int, float))
        and not isinstance(trailing_fraction, bool)
        and math.isfinite(float(trailing_fraction))
        and math.isclose(
            float(trailing_fraction), trailing_steps / expected_steps, abs_tol=1e-12
        )
        and 0 <= maximum_trailing_held_fraction < 1
        and float(trailing_fraction) <= maximum_trailing_held_fraction
    )
    if not valid:
        raise IsaacArenaError(
            "replay prepared actions are not finite, nonzero, varied, and horizon-complete"
        )
    return record


def read_action_evidence(
    run_dir: Path, *, policy_type: str, expected_steps: int
) -> dict[str, Any]:
    """Load, hash, and validate the current run's sanitized action journal."""

    path = run_dir / ACTION_EVIDENCE_FILENAME
    try:
        if path.is_symlink() or not path.is_file():
            raise IsaacArenaError("current run has no executed-action evidence")
        content = path.read_bytes()
        record = json.loads(content)
    except (OSError, ValueError) as exc:
        raise IsaacArenaError(
            "current run has no valid executed-action evidence"
        ) from exc
    validate_action_evidence(
        record,
        policy_type=policy_type,
        expected_steps=expected_steps,
        require_nonzero_varied=False,
    )
    return {
        **record,
        "file": {
            "path": path.name,
            "bytes": len(content),
            "sha256": file_sha256(path),
        },
    }
