"""Serve one qualified native Comet checkpoint with explicit per-case RNG."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import http
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from pathlib import PurePosixPath

import numpy as np

try:
    from comet_policy import (
        CONFIG_NAME,
        SOURCE_COMMIT,
        build_source_overlay,
        policy_observation,
        validate_action,
        verify_source,
    )
    from evaluator_versions import UPSTREAM_COMMITS
    from evaluator_wire import EvaluatorWire
except ModuleNotFoundError:
    from .comet_policy import (
        CONFIG_NAME,
        SOURCE_COMMIT,
        build_source_overlay,
        policy_observation,
        validate_action,
        verify_source,
    )
    from .evaluator_versions import UPSTREAM_COMMITS
    from .evaluator_wire import EvaluatorWire


def _key_identity(key) -> str:
    import jax

    return hashlib.sha256(np.asarray(jax.random.key_data(key)).tobytes()).hexdigest()


def _keys_equal(left, right) -> bool:
    import jax

    return np.array_equal(
        np.asarray(jax.random.key_data(left)),
        np.asarray(jax.random.key_data(right)),
    )


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _process_identity(args, initial_rng: str) -> str:
    return _canonical_digest(_process_identity_payload(args, initial_rng))


def _process_identity_payload(args, initial_rng: str) -> dict:
    return {
        "schema": "npa.behavior.comet-native-serving-process-identity.v1",
        "case_id": args.case_id,
        "task": args.task_name,
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "case_seed": args.case_seed,
        "checkpoint_content_sha256": args.checkpoint_sha256,
        "rng_contract_sha256": args.rng_contract_sha256,
        "trace_configuration_sha256": args.trace_configuration_sha256,
        "initial_rng_sha256": initial_rng,
    }


@contextmanager
def _source_working_directory(root: Path):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _canonical_asset_id(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "%" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is None
            for part in path.parts
        )
    ):
        raise ValueError("Native Comet asset ID differs")
    return value


def _load_policy(args, overlay: Path):
    sys.path[:0] = [str(overlay), str(args.source_root / "packages/openpi-client/src")]
    import jax
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training.config import get_config

    mapping = json.loads((args.source_root / "scripts/task_mapping.json").read_text())
    row = mapping.get(args.task_name)
    if (
        not isinstance(row, dict)
        or row.get("task_index") != args.task_id
        or not isinstance(row.get("task"), str)
        or not row["task"].strip()
    ):
        raise ValueError("Native Comet source task mapping differs")
    checkpoint = args.checkpoint / str(args.manager_step)
    asset_id = _canonical_asset_id(args.asset_id)
    norm_stats = checkpoint.joinpath(
        "assets", *PurePosixPath(asset_id).parts, "norm_stats.json"
    )
    if norm_stats.is_symlink() or not norm_stats.is_file():
        raise ValueError("Native Comet serving normalization asset differs")
    policy = create_trained_policy(get_config(CONFIG_NAME), checkpoint)
    explicit = jax.random.key(args.case_seed)
    policy._rng = explicit
    if not _keys_equal(policy._rng, explicit):
        raise ValueError("Native policy explicit initial RNG differs")
    with _source_working_directory(args.source_root):
        wrapper = B1KPolicyWrapper(
            policy,
            task_name=args.task_name,
            control_mode="receeding_horizon",
            max_len=32,
            fine_grained_level=0,
        )
    return wrapper, policy


def _atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(path)
        return
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class ActionTrace:
    """Append exact sent actions and observed gripper proprioception."""

    def __init__(self, path: Path | None, args):
        self.path = path
        self.args = args
        self.rows = 0
        if path is not None:
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("xb").close()

    def append(
        self,
        action: np.ndarray,
        observation: dict,
        before: str,
        after: str,
        inference_ordinal: int,
    ):
        action = np.asarray(action)
        state = np.asarray(observation["robot_r1::proprio"])
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("Native trace action differs")
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError("Native trace proprioception differs")
        action_index = self.rows
        self.rows += 1
        if self.path is None:
            return
        row = {
            "schema": "npa.behavior.comet-native-action-trace-row.v1",
            "case_id": self.args.case_id,
            "task": self.args.task_name,
            "instance_id": self.args.instance_id,
            "rollout_id": self.args.rollout_id,
            "action_index": action_index,
            "inference_ordinal": inference_ordinal,
            "sent_action": action.tolist(),
            "left_command": float(action[14]),
            "right_command": float(action[22]),
            "left_gripper_proprio": state[24:26].tolist(),
            "right_gripper_proprio": state[49:51].tolist(),
            "checkpoint_content_sha256": self.args.checkpoint_sha256,
            "rng_contract_sha256": self.args.rng_contract_sha256,
            "trace_configuration_sha256": self.args.trace_configuration_sha256,
            "process_identity_sha256": self.args.process_identity_sha256,
            "initial_rng_sha256": self.args.initial_rng_sha256,
            "rng_before_sha256": before,
            "rng_after_sha256": after,
        }
        with self.path.open("ab") as stream:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())


class ProcessProgress:
    """Append one durable row after each action is successfully sent."""

    def __init__(self, path: Path, args):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.args = args
        self.rows = 0
        self._stream = path.open("xb")

    def append(self, current_rng: str, inference_count: int) -> dict:
        self.rows += 1
        value = _process_receipt(
            self.args,
            self.args.initial_rng_sha256,
            current_rng,
            inference_count,
            self.rows,
        )
        self._stream.write(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return value

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def _process_receipt(
    args,
    initial_rng: str,
    current_rng: str,
    inference_count: int,
    action_count: int,
) -> dict:
    return {
        "schema": "npa.behavior.comet-native-serving-process.v1",
        "status": "ready" if action_count == 0 else "sent_action_recorded",
        "process_identity_sha256": args.process_identity_sha256,
        "case_id": args.case_id,
        "task": args.task_name,
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "case_seed": args.case_seed,
        "checkpoint_content_sha256": args.checkpoint_sha256,
        "rng_contract_sha256": args.rng_contract_sha256,
        "trace_configuration_sha256": args.trace_configuration_sha256,
        "initial_rng_sha256": initial_rng,
        "current_rng_sha256": current_rng,
        "inference_count": inference_count,
        "action_count": action_count,
        "trace_enabled": args.action_trace is not None,
    }


def validate_process_receipt(value: object, expected: dict | None = None) -> dict:
    """Validate one exact native serving process receipt.

    Args:
        value: Candidate ready or progress receipt.
        expected: Optional ready receipt that fixes stable process fields.
    Returns:
        The unchanged validated receipt.
    Raises:
        ValueError: Fields, identities, or chronology differ.
    """
    required = {
        "schema",
        "status",
        "process_identity_sha256",
        "case_id",
        "task",
        "instance_id",
        "rollout_id",
        "case_seed",
        "checkpoint_content_sha256",
        "rng_contract_sha256",
        "trace_configuration_sha256",
        "initial_rng_sha256",
        "current_rng_sha256",
        "inference_count",
        "action_count",
        "trace_enabled",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Native Comet process receipt fields differ")
    for name in (
        "process_identity_sha256",
        "checkpoint_content_sha256",
        "rng_contract_sha256",
        "trace_configuration_sha256",
        "initial_rng_sha256",
        "current_rng_sha256",
    ):
        if (
            not isinstance(value[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None
        ):
            raise ValueError("Native Comet process receipt identity differs")
    actions = value["action_count"]
    inferences = value["inference_count"]
    if (
        value["schema"] != "npa.behavior.comet-native-serving-process.v1"
        or isinstance(actions, bool)
        or not isinstance(actions, int)
        or actions < 0
        or isinstance(inferences, bool)
        or not isinstance(inferences, int)
        or inferences < 0
        or inferences > actions
        or value["status"] != ("ready" if actions == 0 else "sent_action_recorded")
        or (actions == 0 and inferences != 0)
        or not isinstance(value["trace_enabled"], bool)
        or not isinstance(value["case_id"], str)
        or not value["case_id"]
        or not isinstance(value["task"], str)
        or not value["task"]
        or isinstance(value["instance_id"], bool)
        or not isinstance(value["instance_id"], int)
        or value["instance_id"] < 0
        or value["rollout_id"] != 0
        or isinstance(value["case_seed"], bool)
        or not isinstance(value["case_seed"], int)
        or not 0 <= value["case_seed"] < 2**32
    ):
        raise ValueError("Native Comet process receipt chronology differs")
    identity_payload = {
        "schema": "npa.behavior.comet-native-serving-process-identity.v1",
        "case_id": value["case_id"],
        "task": value["task"],
        "instance_id": value["instance_id"],
        "rollout_id": value["rollout_id"],
        "case_seed": value["case_seed"],
        "checkpoint_content_sha256": value["checkpoint_content_sha256"],
        "rng_contract_sha256": value["rng_contract_sha256"],
        "trace_configuration_sha256": value["trace_configuration_sha256"],
        "initial_rng_sha256": value["initial_rng_sha256"],
    }
    if value["process_identity_sha256"] != _canonical_digest(identity_payload):
        raise ValueError("Native Comet process identity differs")
    if expected is not None:
        stable = required - {
            "status",
            "current_rng_sha256",
            "inference_count",
            "action_count",
        }
        if any(value[name] != expected[name] for name in stable):
            raise ValueError("Native Comet process receipt lineage differs")
    return value


def _load_qualification(args) -> dict:
    return {
        "schema": "npa.behavior.comet-native-serving-load-qualification.v1",
        "status": "checkpoint_loaded_with_explicit_case_rng",
        "case_id": args.case_id,
        "task": args.task_name,
        "instance_id": args.instance_id,
        "rollout_id": args.rollout_id,
        "case_seed": args.case_seed,
        "checkpoint_content_sha256": args.checkpoint_sha256,
        "manager_step": args.manager_step,
        "asset_id": args.asset_id,
        "source_commit": SOURCE_COMMIT,
        "config_name": CONFIG_NAME,
        "rng_contract_sha256": args.rng_contract_sha256,
        "trace_configuration_sha256": args.trace_configuration_sha256,
        "initial_rng_sha256": args.initial_rng_sha256,
        "process_identity_sha256": args.process_identity_sha256,
        "inference_count": 0,
    }


def validate_load_qualification(value: object, expected: dict) -> dict:
    """Validate the discarded loader receipt and explicit RNG identity.

    Args:
        value: Candidate load-qualification receipt.
        expected: Exact stable fields derived from the admitted case.
    Returns:
        The unchanged validated receipt.
    Raises:
        ValueError: Fields, process identity, or admitted values differ.
    """
    keys = {
        "schema",
        "status",
        "case_id",
        "task",
        "instance_id",
        "rollout_id",
        "case_seed",
        "checkpoint_content_sha256",
        "manager_step",
        "asset_id",
        "source_commit",
        "config_name",
        "rng_contract_sha256",
        "trace_configuration_sha256",
        "initial_rng_sha256",
        "process_identity_sha256",
        "inference_count",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Native Comet serving-load qualification fields differ")
    stable = keys - {"initial_rng_sha256", "process_identity_sha256"}
    if set(expected) != stable or any(value[name] != expected[name] for name in stable):
        raise ValueError("Native Comet serving-load qualification differs")
    for name in ("initial_rng_sha256", "process_identity_sha256"):
        if (
            not isinstance(value[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None
        ):
            raise ValueError("Native Comet serving-load qualification identity differs")
    identity_payload = {
        "schema": "npa.behavior.comet-native-serving-process-identity.v1",
        "case_id": value["case_id"],
        "task": value["task"],
        "instance_id": value["instance_id"],
        "rollout_id": value["rollout_id"],
        "case_seed": value["case_seed"],
        "checkpoint_content_sha256": value["checkpoint_content_sha256"],
        "rng_contract_sha256": value["rng_contract_sha256"],
        "trace_configuration_sha256": value["trace_configuration_sha256"],
        "initial_rng_sha256": value["initial_rng_sha256"],
    }
    if value["process_identity_sha256"] != _canonical_digest(identity_payload):
        raise ValueError("Native Comet serving-load process identity differs")
    return value


def validate_trace_row(value: object, ready: dict, action_index: int) -> dict:
    """Validate one action diagnostic against its serving process.

    Args:
        value: Candidate trace row.
        ready: Validated immutable process ready receipt.
        action_index: Required contiguous action index.
    Returns:
        The unchanged validated row.
    Raises:
        ValueError: Lineage, action, proprioception, or RNG fields differ.
    """
    required = {
        "schema",
        "case_id",
        "task",
        "instance_id",
        "rollout_id",
        "action_index",
        "inference_ordinal",
        "sent_action",
        "left_command",
        "right_command",
        "left_gripper_proprio",
        "right_gripper_proprio",
        "checkpoint_content_sha256",
        "rng_contract_sha256",
        "trace_configuration_sha256",
        "process_identity_sha256",
        "initial_rng_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Native Comet action trace fields differ")
    stable = {
        "case_id",
        "task",
        "instance_id",
        "rollout_id",
        "checkpoint_content_sha256",
        "rng_contract_sha256",
        "trace_configuration_sha256",
        "process_identity_sha256",
        "initial_rng_sha256",
    }
    if (
        value["schema"] != "npa.behavior.comet-native-action-trace-row.v1"
        or value["action_index"] != action_index
        or any(value[name] != ready[name] for name in stable)
        or isinstance(value["inference_ordinal"], bool)
        or not isinstance(value["inference_ordinal"], int)
        or value["inference_ordinal"] < 0
    ):
        raise ValueError("Native Comet action trace lineage differs")
    action = value["sent_action"]
    left = value["left_gripper_proprio"]
    right = value["right_gripper_proprio"]
    if (
        not all(isinstance(item, list) for item in (action, left, right))
        or len(action) != 23
        or len(left) != 2
        or len(right) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in action + left + right
        )
        or not np.isfinite(np.asarray(action + left + right, dtype=np.float64)).all()
        or value["left_command"] != action[14]
        or value["right_command"] != action[22]
    ):
        raise ValueError("Native Comet action trace values differ")
    for name in ("rng_before_sha256", "rng_after_sha256"):
        if (
            not isinstance(value[name], str)
            or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None
        ):
            raise ValueError("Native Comet action trace RNG differs")
    return value


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _connection(
    websocket,
    wrapper,
    policy,
    args,
    trace: ActionTrace,
    progress: ProcessProgress,
):
    import jax
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    wire = EvaluatorWire(args.upstream_commit)
    wrapper.reset()
    inference_count = 0
    await websocket.send(packer.pack({"policy": "comet-native-train"}))
    async for payload in websocket:
        observation = msgpack_numpy.unpackb(payload)
        if wire.is_reset(observation):
            wrapper.reset()
            continue
        selected = wire.observation_for_policy(observation)
        inputs = policy_observation(selected)
        before_key = policy._rng
        expected_after, _sample_key = jax.random.split(before_key)
        action = validate_action(wrapper.act(inputs))
        if _keys_equal(policy._rng, expected_after):
            inference_ordinal = inference_count
            inference_count += 1
        elif _keys_equal(policy._rng, before_key) and inference_count > 0:
            inference_ordinal = inference_count - 1
        else:
            raise ValueError(
                "Native policy RNG differs from its chunk/inference schedule"
            )
        before = _key_identity(before_key)
        after = _key_identity(policy._rng)
        await websocket.send(packer.pack({"action": wire.action_for_evaluator(action)}))
        progress.append(after, inference_count)
        trace.append(
            action,
            selected,
            before,
            after,
            inference_ordinal,
        )


async def _serve(
    wrapper, policy, args, trace: ActionTrace, progress: ProcessProgress
) -> None:
    from websockets.asyncio.server import serve

    lock = asyncio.Lock()
    connected = False

    async def handle(websocket):
        nonlocal connected
        if connected or lock.locked():
            await websocket.close(code=1013, reason="Evaluator already connected")
            return
        connected = True
        async with lock:
            await _connection(websocket, wrapper, policy, args, trace, progress)

    async with serve(
        handle,
        "127.0.0.1",
        args.port,
        compression=None,
        max_size=16 * 1024 * 1024,
        process_request=_health,
    ) as server:
        await server.serve_forever()


def parser() -> argparse.ArgumentParser:
    """Build the private native serving-process argument parser."""
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--manager-step", type=int, required=True)
    value.add_argument("--asset-id", required=True)
    value.add_argument("--task-id", type=int, required=True)
    value.add_argument("--task-name", required=True)
    value.add_argument("--port", type=int, required=True)
    value.add_argument(
        "--upstream-commit", choices=tuple(UPSTREAM_COMMITS.values()), required=True
    )
    value.add_argument("--case-id", required=True)
    value.add_argument("--instance-id", type=int, required=True)
    value.add_argument("--rollout-id", type=int, required=True)
    value.add_argument("--case-seed", type=int, required=True)
    value.add_argument("--checkpoint-sha256", required=True)
    value.add_argument("--rng-contract-sha256", required=True)
    value.add_argument("--trace-configuration-sha256", required=True)
    value.add_argument("--process-receipt", type=Path, required=True)
    value.add_argument("--action-trace", type=Path)
    value.add_argument("--qualify-only", action="store_true")
    value.add_argument("--qualification-output", type=Path)
    return value


def _main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    verify_source(args.source_root)
    with tempfile.TemporaryDirectory(prefix="npa-comet-native-source-") as temporary:
        overlay = build_source_overlay(args.source_root, Path(temporary))
        wrapper, policy = _load_policy(args, overlay)
        args.initial_rng_sha256 = _key_identity(policy._rng)
        args.process_identity_sha256 = _process_identity(args, args.initial_rng_sha256)
        if args.qualify_only:
            if args.qualification_output is None:
                raise ValueError("Qualification output is required")
            _atomic_json(
                args.qualification_output,
                _load_qualification(args),
            )
            return
        trace = ActionTrace(args.action_trace, args)
        progress = ProcessProgress(
            args.process_receipt.with_name("native-process-progress.jsonl"), args
        )
        _atomic_json(
            args.process_receipt,
            _process_receipt(
                args,
                args.initial_rng_sha256,
                args.initial_rng_sha256,
                0,
                0,
            ),
        )
        try:
            asyncio.run(_serve(wrapper, policy, args, trace, progress))
        finally:
            progress.close()


if __name__ == "__main__":
    _main()
