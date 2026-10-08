"""Pinned OpenPI WebSocket protocol and strict pi0.5 response validation."""

from __future__ import annotations

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

    def __init__(self, host: str, port: int = 8000) -> None:
        import msgpack
        from websockets.sync.client import connect

        self._msgpack = msgpack
        self._connection = connect(
            f"ws://{host}:{port}", compression=None, max_size=None
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
    ) -> tuple[Any, dict[str, Any]]:
        observation = _observation(
            exterior=exterior,
            wrist=wrist,
            joint_position=joint_position,
            gripper_position=gripper_position,
            prompt=prompt,
        )
        started = time.perf_counter()
        payload = self._msgpack.packb(observation, default=_pack_array)
        self._connection.send(payload)
        response = self._connection.recv()
        if isinstance(response, str):
            raise Pi05PolicyClientError(
                f"OpenPI server returned text error: {response}"
            )
        decoded = self._msgpack.unpackb(response, object_hook=_unpack_array)
        return _validated_response(decoded, started)


def _validated_response(
    decoded: dict[str, Any], started: float
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    actions = np.asarray(decoded.get("actions"))
    if actions.dtype != np.float64 or actions.ndim != 2 or actions.shape[1] != 8:
        raise Pi05PolicyClientError(
            f"OpenPI actions must be float64[T,8], got {actions.dtype} {actions.shape}"
        )
    if actions.shape[0] < 15 or not np.isfinite(actions).all():
        raise Pi05PolicyClientError("OpenPI actions lack a finite 15-step horizon")
    evidence = {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "finite": True,
        "round_trip_ms": (time.perf_counter() - started) * 1000.0,
        "server_timing": decoded.get("server_timing", {}),
    }
    return actions, evidence


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
