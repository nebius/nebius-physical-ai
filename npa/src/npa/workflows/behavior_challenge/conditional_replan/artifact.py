"""Deterministically export and run conditional-replanning gate artifacts."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import validate_semantic_config
from .extraction import FEATURE_DIMENSION

ARTIFACT_SCHEMA = "npa.behavior.parent-diagnostic-conditional-gate-artifact.v1"


def expected_arrays() -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Return the exact public artifact array inventory.

    Args:
        None.
    Returns:
        Mapping from array name to shape and dtype.
    Raises:
        None.
    """
    f32 = np.dtype(np.float32)
    return {
        "input_mean": ((FEATURE_DIMENSION,), f32),
        "input_std": ((FEATURE_DIMENSION,), f32),
        "target_mean": ((1,), f32),
        "target_std": ((1,), f32),
        "linear_control.weight": ((1, FEATURE_DIMENSION), f32),
        "linear_control.bias": ((1,), f32),
        "mlp.0.weight": ((256, FEATURE_DIMENSION), f32),
        "mlp.0.bias": ((256,), f32),
        "mlp.2.weight": ((64, 256), f32),
        "mlp.2.bias": ((64,), f32),
        "mlp.4.weight": ((1, 64), f32),
        "mlp.4.bias": ((1,), f32),
    }


@dataclass(frozen=True)
class GateArtifact:
    """Hold validated NumPy arrays and public artifact metadata.

    Args:
        arrays: Exact model parameters and fit-only scalers.
        metadata: Versioned artifact metadata.
    Returns:
        None.
    Raises:
        None.
    """

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def validated(self) -> GateArtifact:
        """Validate the complete artifact contract.

        Args:
            None.
        Returns:
            This artifact after validation.
        Raises:
            ValueError: If inventory, arrays, or metadata differ.
        """
        if set(self.arrays) != set(expected_arrays()):
            raise ValueError("conditional gate array inventory differs")
        for name, (shape, dtype) in expected_arrays().items():
            value = self.arrays[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"conditional gate array differs: {name}")
        if np.any(self.arrays["input_std"] <= 0) or np.any(
            self.arrays["target_std"] <= 0
        ):
            raise ValueError("conditional gate scaler is not positive")
        if self.metadata.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("conditional gate metadata schema differs")
        _validate_selection(self.metadata.get("selection"))
        if self.metadata.get("development_or_report_used") is not False:
            raise ValueError("conditional gate data scope differs")
        validate_semantic_config(
            self.metadata.get("feature_config"),
            self.metadata.get("target_config"),
            self.metadata.get("model_config"),
            self.metadata.get("gate_config"),
        )
        return self


def _validate_selection(selection: Any) -> None:
    if not isinstance(selection, dict) or set(selection) != {"linear_control", "mlp"}:
        raise ValueError("conditional gate selection metadata differs")
    if any(
        type(epoch) is not int or not 1 <= epoch <= 200 for epoch in selection.values()
    ):
        raise ValueError("conditional gate selected epoch differs")


def write_artifact(path: Path, artifact: GateArtifact) -> None:
    """Write one deterministic, pickle-free NPZ artifact.

    Args:
        path: New output path.
        artifact: Validated artifact to serialize.
    Returns:
        None.
    Raises:
        FileExistsError: If the destination exists.
        ValueError: If the artifact differs from the contract.
    """
    artifact.validated()
    arrays = dict(artifact.arrays)
    metadata = json.dumps(artifact.metadata, sort_keys=True, separators=(",", ":"))
    arrays["metadata_json_utf8"] = np.frombuffer(metadata.encode(), np.uint8).copy()
    with path.open("xb") as stream:
        stream.write(deterministic_npz(arrays))


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Encode arrays as a deterministic uncompressed NumPy archive.

    Args:
        arrays: Named arrays to serialize.
    Returns:
        Complete NPZ bytes with canonical members and timestamps.
    Raises:
        ValueError: If a member name is unsafe.
    """
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            if not name or "/" in name or "\\" in name:
                raise ValueError("unsafe NPZ member name")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def load_artifact(path: Path) -> GateArtifact:
    """Load and validate an immutable artifact without Torch.

    Args:
        path: NPZ artifact path.
    Returns:
        Validated NumPy-only artifact.
    Raises:
        ValueError: If the artifact representation differs.
    """
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: np.ascontiguousarray(values[name]) for name in values.files}
    encoded = arrays.pop("metadata_json_utf8", None)
    if encoded is None or encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise ValueError("conditional gate metadata encoding differs")
    metadata = json.loads(encoded.tobytes().decode())
    return GateArtifact(arrays, metadata).validated()


def predict_scaled(
    artifact: GateArtifact, features: np.ndarray, model: str
) -> np.ndarray:
    """Run canonical NumPy inference in scaled target space.

    Args:
        artifact: Validated exported artifact.
        features: Float32 matrix shaped ``(rows, 1233)``.
        model: ``linear_control`` or ``mlp``.
    Returns:
        Float32 scaled predictions shaped ``(rows, 1)``.
    Raises:
        ValueError: If the model or feature representation differs.
    """
    arrays = artifact.validated().arrays
    value = np.asarray(features)
    if (
        value.ndim != 2
        or value.shape[1] != FEATURE_DIMENSION
        or value.dtype != np.float32
    ):
        raise ValueError("conditional gate feature matrix differs")
    value = ((value - arrays["input_mean"]) / arrays["input_std"]).astype(np.float32)
    if model == "linear_control":
        return _linear(value, arrays, "linear_control")
    if model != "mlp":
        raise ValueError("unknown conditional gate model")
    value = gelu_exact(_linear(value, arrays, "mlp.0"))
    value = gelu_exact(_linear(value, arrays, "mlp.2"))
    return _linear(value, arrays, "mlp.4")


def predict_decoded(
    artifact: GateArtifact, features: np.ndarray, model: str
) -> np.ndarray:
    """Decode canonical NumPy predictions into error differences.

    Args:
        artifact: Validated exported artifact.
        features: Float32 feature matrix.
        model: Exported model name.
    Returns:
        Float32 decoded predictions.
    Raises:
        ValueError: If inference produces non-finite output.
    """
    scaled = predict_scaled(artifact, features, model).reshape(-1)
    arrays = artifact.arrays
    result = scaled * arrays["target_std"][0] + arrays["target_mean"][0]
    if not np.isfinite(result).all():
        raise ValueError("decoded prediction is non-finite")
    return result.astype(np.float32)


def gelu_exact(value: np.ndarray) -> np.ndarray:
    """Evaluate the qualified exact-erf GELU in float32.

    Args:
        value: Input array.
    Returns:
        Float32 GELU values.
    Raises:
        None.
    """
    flat = value.astype(np.float64, copy=False).reshape(-1)
    erf = np.fromiter((math.erf(item / math.sqrt(2.0)) for item in flat), np.float64)
    return (0.5 * flat * (1.0 + erf)).reshape(value.shape).astype(np.float32)


def _linear(
    value: np.ndarray, arrays: dict[str, np.ndarray], prefix: str
) -> np.ndarray:
    return (value @ arrays[f"{prefix}.weight"].T + arrays[f"{prefix}.bias"]).astype(
        np.float32
    )
