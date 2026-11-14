"""Pinned OpenPI WebSocket protocol and strict pi0.5 response validation."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any


class Pi05PolicyClientError(RuntimeError):
    """Raised when server transport or policy output violates the contract."""


def _pack_array(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise Pi05PolicyClientError(f"unsupported array dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: dict[bytes, Any]) -> Any:
    import numpy as np

    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class Pi05PolicyClient:
    """Small exact implementation of pinned ``openpi-client`` wire semantics."""

    def __init__(
        self, host: str, port: int = 8000, *, auth_token: str | None = None
    ) -> None:
        import msgpack
        from websockets.sync.client import connect

        self._msgpack = msgpack
        token = auth_token or os.environ.get("NPA_OPENPI_SERVICE_AUTH_TOKEN", "")
        if not token:
            raise Pi05PolicyClientError("pi0.5 policy service auth token is required")
        self._connection = connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            additional_headers={"Authorization": f"Bearer {token}"},
        )
        metadata = self._connection.recv()
        if isinstance(metadata, str):
            raise Pi05PolicyClientError("OpenPI metadata must be binary msgpack")
        self.metadata = msgpack.unpackb(metadata, object_hook=_unpack_array)

    def infer(
        self,
        *,
        exterior: Any,
        wrist: Any,
        joint_position: Any,
        gripper_position: Any,
        prompt: str,
        capture_age_s: float | None = None,
        source_wrist_sha256: str = "",
        sampling_noise: Any | None = None,
        sampling_sequence: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        observation = _observation(
            exterior=exterior,
            wrist=wrist,
            joint_position=joint_position,
            gripper_position=gripper_position,
            prompt=prompt,
        )
        if capture_age_s is not None and not 0.0 <= capture_age_s <= 1.0 / 15.0:
            raise Pi05PolicyClientError("wrist capture is stale at policy query")
        wrist_sha256 = _validated_wrist_hash(observation, source_wrist_sha256)
        sampling = _sampling_request(self.metadata, sampling_noise, sampling_sequence)
        if sampling:
            observation.update(sampling["payload"])
        return self._send_inference(observation, wrist_sha256, capture_age_s, sampling)

    def infer_typed(
        self,
        *,
        observation: Any,
        robot: Any,
        policy: Any,
        sampling_noise: Any | None = None,
        sampling_sequence: str = "",
    ) -> tuple[Any, dict[str, Any]]:
        """Translate the neutral contract to pinned DROID/OpenPI wire aliases.

        This adapter is deliberately exact: pi0.5 is qualified only for the
        seven-axis Franka action/state shape and never pads a different arm.
        """

        import numpy as np

        if policy.backend != "openpi_pi05":
            raise Pi05PolicyClientError("typed OpenPI adapter requires openpi_pi05")
        if (
            observation.policy_identity != policy.identity
            or observation.robot_identity != robot.identity
            or observation.task_identity != policy.task_identity
        ):
            raise Pi05PolicyClientError(
                "typed observation identities do not match OpenPI profile"
            )
        if len(robot.arm_joint_names) != 7:
            raise Pi05PolicyClientError(
                "pi0.5 DROID checkpoint requires exactly seven arm joints; padding is forbidden"
            )
        if tuple(observation.measured_state_names) != tuple(robot.measured_state_names):
            raise Pi05PolicyClientError(
                "typed measured state names/order do not match the robot"
            )
        views = {view.role: view for view in observation.views}
        if set(views) != {"exterior", "wrist"}:
            raise Pi05PolicyClientError(
                "pi0.5 requires exactly one exterior and one wrist view"
            )
        if any(
            view.layout != "HWC" or view.image_size != (224, 224)
            for view in views.values()
        ):
            raise Pi05PolicyClientError(
                "pi0.5 requires calibrated 224x224 HWC RGB views"
            )
        exterior = np.frombuffer(views["exterior"].rgb8, dtype=np.uint8).reshape(
            224, 224, 3
        )
        wrist = np.frombuffer(views["wrist"].rgb8, dtype=np.uint8).reshape(224, 224, 3)
        state = np.asarray(observation.measured_state, dtype=np.float32)
        arm = state[:7]
        gripper = np.asarray([state[7]], dtype=np.float32)
        wrist_age = (
            views["wrist"].state_timestamp_s - views["wrist"].capture_timestamp_s
        )
        actions, evidence = self.infer(
            exterior=exterior,
            wrist=wrist,
            joint_position=arm,
            gripper_position=gripper,
            prompt=observation.prompt,
            capture_age_s=wrist_age,
            source_wrist_sha256=views["wrist"].rgb_sha256,
            sampling_noise=sampling_noise,
            sampling_sequence=sampling_sequence,
        )
        from npa.workflows.sim2real.interchange_contract import PolicyOutput

        output = PolicyOutput(
            observation_input_identity=observation.input_identity,
            policy_identity=policy.identity,
            checkpoint_sha256=policy.checkpoint_sha256,
            action=policy.action,
            action_chunk=tuple(tuple(float(value) for value in row) for row in actions),
            generated_timestamp_s=time.time(),
        )
        return output, evidence

    def _send_inference(
        self,
        observation: dict[str, Any],
        wrist_sha256: str,
        capture_age_s: float | None,
        sampling: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        started = time.perf_counter()
        payload = self._msgpack.packb(observation, default=_pack_array)
        wire, wire_wrist_sha256 = _verify_wire_roundtrip(
            self._msgpack, payload, wrist_sha256
        )
        self._connection.send(payload)
        response = self._connection.recv()
        if isinstance(response, str):
            raise Pi05PolicyClientError(
                f"OpenPI server returned text error: {response}"
            )
        decoded = self._msgpack.unpackb(response, object_hook=_unpack_array)
        actions, evidence = _validated_response(
            decoded, started, sampling, expected_wrist_sha256=wrist_sha256
        )
        evidence["wire_input"] = _wire_receipt(
            payload, wire, wire_wrist_sha256, capture_age_s
        )
        return actions, evidence


def _validated_wrist_hash(observation: dict[str, Any], source_hash: str) -> str:
    wrist_hash = _array_sha256(observation["observation/wrist_image_left"])
    if source_hash and source_hash != wrist_hash:
        raise Pi05PolicyClientError("wrist bytes changed before wire serialization")
    return wrist_hash


def _wire_receipt(
    payload: bytes, wire: dict[str, Any], wrist_hash: str, capture_age_s: float | None
) -> dict[str, Any]:
    return {
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "exterior_rgb_sha256": _array_sha256(wire["observation/exterior_image_1_left"]),
        "wrist_rgb_sha256": wrist_hash,
        "wrist_source_hash_matched": True,
        "capture_age_s": capture_age_s,
    }


def _verify_wire_roundtrip(
    msgpack: Any, payload: bytes, expected: str
) -> tuple[dict, str]:
    wire = msgpack.unpackb(payload, object_hook=_unpack_array)
    wrist_hash = _array_sha256(wire["observation/wrist_image_left"])
    if wrist_hash != expected:
        raise Pi05PolicyClientError("wrist bytes changed in msgpack roundtrip")
    return wire, wrist_hash


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sampling_request(
    metadata: dict[str, Any], noise: Any | None, sequence: str
) -> dict[str, Any] | None:
    import numpy as np

    if noise is None and not sequence:
        return None
    protocol = metadata.get("npa_explicit_noise_protocol") or {}
    expected_shape = tuple(protocol.get("shape") or ())
    value = np.asarray(noise)
    if protocol.get("schema") != "npa.openpi.pi05-explicit-noise.v1":
        raise Pi05PolicyClientError("server lacks the pi0.5 explicit-noise protocol")
    if value.dtype != np.float32 or value.shape != expected_shape:
        raise Pi05PolicyClientError("sampling noise violates server model shape/dtype")
    if not np.isfinite(value).all() or not re.fullmatch(
        r"[a-zA-Z0-9_.:-]{1,128}", sequence
    ):
        raise Pi05PolicyClientError("sampling noise or sequence identity is invalid")
    return {
        "payload": {
            "npa/sampling_noise": value,
            "npa/sampling_sequence": sequence,
        },
        "sha256": _array_sha256(value),
        "shape": list(value.shape),
        "sequence": sequence,
    }


def _validated_response(
    decoded: dict[str, Any],
    started: float,
    sampling: dict[str, Any] | None,
    *,
    expected_wrist_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    actions = np.asarray(decoded.get("actions"))
    if actions.dtype != np.float64 or actions.ndim != 2 or actions.shape[1] != 8:
        raise Pi05PolicyClientError(
            f"OpenPI actions must be float64[T,8], got {actions.dtype} {actions.shape}"
        )
    if actions.shape[0] < 15 or not np.isfinite(actions).all():
        raise Pi05PolicyClientError("OpenPI actions lack a finite 15-step horizon")
    identity = _sampler_noise_identity(decoded)
    if sampling and (
        identity != sampling["sha256"]
        or decoded.get("npa_sampler_noise_shape") != sampling["shape"]
        or decoded.get("npa_sampling_sequence") != sampling["sequence"]
    ):
        raise Pi05PolicyClientError("server did not bind actions to requested noise")
    transform = _validated_transform_receipt(
        decoded, expected_wrist_sha256, required=bool(sampling)
    )
    evidence = {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "finite": True,
        "round_trip_ms": (time.perf_counter() - started) * 1000.0,
        "server_timing": decoded.get("server_timing", {}),
        "sampler_noise_sha256": identity,
        "sampler_noise_shape": decoded.get("npa_sampler_noise_shape"),
        "sampling_sequence": decoded.get("npa_sampling_sequence"),
        "model_wrist_transform": transform,
    }
    return actions, evidence


def _validated_transform_receipt(
    decoded: dict[str, Any], expected_wrist_sha256: str, *, required: bool
) -> dict[str, Any] | None:
    transform = decoded.get("npa_wrist_transform_receipt")
    if required and (
        not isinstance(transform, dict)
        or transform.get("schema") != "npa.openpi.pi05-live-wrist-transform.v1"
        or transform.get("source_wrist_sha256") != expected_wrist_sha256
        or transform.get("left_wrist_mask") is not True
        or transform.get("right_wrist_mask") is not False
    ):
        raise Pi05PolicyClientError(
            "server did not bind the real wrist bytes to the model input"
        )
    return transform if isinstance(transform, dict) else None


def _sampler_noise_identity(decoded: dict[str, Any]) -> str | None:
    identity = decoded.get("npa_sampler_noise_sha256")
    if identity is None:
        return None
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise Pi05PolicyClientError("server sampler-noise identity is malformed")
    return identity


def _observation(
    *,
    exterior: Any,
    wrist: Any,
    joint_position: Any,
    gripper_position: Any,
    prompt: str,
) -> dict[str, Any]:
    import numpy as np

    checks = {
        "exterior": (np.asarray(exterior), (224, 224, 3), np.dtype("uint8")),
        "wrist": (np.asarray(wrist), (224, 224, 3), np.dtype("uint8")),
        "joint": (np.asarray(joint_position), (7,), np.dtype("float32")),
        "gripper": (np.asarray(gripper_position), (1,), np.dtype("float32")),
    }
    for name, (value, shape, dtype) in checks.items():
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise Pi05PolicyClientError(
                f"{name} violates OpenPI input contract: {value.shape} {value.dtype}"
            )
    if not str(prompt).strip():
        raise Pi05PolicyClientError("OpenPI prompt must be non-empty")
    return {
        "observation/exterior_image_1_left": checks["exterior"][0],
        "observation/wrist_image_left": checks["wrist"][0],
        "observation/joint_position": checks["joint"][0],
        "observation/gripper_position": checks["gripper"][0],
        "prompt": prompt,
    }
