"""Verify the frozen released SHAwn RLC specialist and its loaded native state."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_REPOSITORY = "Shawn3636/pi05-rft-behavior1k"
MODEL_REVISION = "5f0354b905ef5a5673c38eab364e439926f08e70"
SOURCE_REPOSITORY = "Sunliu36/Behavior1kChallenge_Solution_by_SHAWN"
SOURCE_COMMIT = "a608b89c419694b97f944108f67dc0232f4085b8"
SUPPORTED_TASK_IDS = frozenset({1, 7, 18, 21})
NORMALIZATION = "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json"
CORRELATION_SHA256 = "f4592b717febbd3046b8bbdc7b9f2a0c4b1bc2836463663c5095da6a40687c8c"
TOPOLOGY_SHA256 = "8b3910680962f4f5150b54883428215dcab3648581b1c11945244bc336e97ced"


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest() -> dict:
    """Load the immutable public specialist checkpoint inventory.

    Returns:
        The validated specialist checkpoint manifest.
    Raises:
        ValueError: The pinned release identity differs.
    """
    path = Path(__file__).with_name("rlc-specialist-checkpoint.json")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("repository") != MODEL_REPOSITORY
        or manifest.get("revision") != MODEL_REVISION
        or manifest.get("source_repository") != SOURCE_REPOSITORY
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest.get("declared_task_ids") != sorted(SUPPORTED_TASK_IDS)
    ):
        raise ValueError("Specialist checkpoint manifest identity differs")
    return manifest


def _verify_raw_topology(root: Path) -> None:
    metadata = json.loads((root / "params/_METADATA").read_text())["tree_metadata"]
    rows = []
    for encoded, value in metadata.items():
        parts = ast.literal_eval(encoded)
        if parts[0] != "params" or parts[-1] != "value":
            raise ValueError("Specialist raw topology path differs")
        rows.append(
            {
                "path": "/".join(parts[1:-1]),
                "shape": value["value_metadata"]["write_shape"],
            }
        )
    encoded = json.dumps(
        sorted(rows, key=lambda row: row["path"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(rows) != 75 or hashlib.sha256(encoded).hexdigest() != TOPOLOGY_SHA256:
        raise ValueError("Specialist raw 75-leaf topology differs")


def verify_checkpoint_files(root: Path, files: dict[str, str]) -> None:
    """Verify every extracted release payload and the raw model topology.

    Args:
        root: Extracted specialist checkpoint root.
        files: SHA-256 inventory independently derived from the supplied archive.
    Returns:
        None.
    Raises:
        ValueError: File identities or the raw 75-leaf topology differ.
    """
    rows = checkpoint_manifest().get("files")
    expected = {row.get("path"): row for row in rows or []}
    if len(expected) != 17 or set(files) != set(expected):
        raise ValueError("Specialist checkpoint file set differs")
    for relative, identity in expected.items():
        path = root / relative
        if path.stat().st_size != identity.get("bytes"):
            raise ValueError("Specialist checkpoint file size differs")
        if "sha256" in identity:
            matches = files[relative] == identity["sha256"]
        else:
            matches = _git_blob_sha1(path) == identity.get("git_blob_sha1")
        if not matches:
            raise ValueError("Specialist checkpoint bytes differ")
    _verify_raw_topology(root)


def _state_rows(model) -> list[dict]:
    import jax
    from flax import nnx

    rows = []
    for path, variable in nnx.state(model).flat_state().items():
        array = np.asarray(jax.device_get(variable.value))
        rows.append(
            {
                "path": "/".join(map(str, path)),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "variable_type": getattr(variable.type, "__name__", str(variable.type)),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def verify_loaded_state(policy) -> dict:
    """Require the exact qualified native state before server readiness.

    Args:
        policy: Loaded native policy whose model state is ready for inference.
    Returns:
        Verified typed-state summary for provenance and tests.
    Raises:
        ValueError: Typed leaves or native correlation readiness differ.
    """
    rows = _state_rows(policy._model)
    expected_correlation = {
        "path": "action_correlation_cholesky",
        "shape": [960, 960],
        "dtype": "float32",
        "variable_type": "Intermediate",
        "sha256": CORRELATION_SHA256,
    }
    correlations = [row for row in rows if row["path"] == expected_correlation["path"]]
    params = sum(
        row["variable_type"] == "Param" and row["dtype"] == "bfloat16" for row in rows
    )
    if (
        len(rows) != 75
        or params != 74
        or correlations != [expected_correlation]
        or getattr(policy._model, "correlation_loaded", None) is not True
    ):
        raise ValueError("Specialist post-load native state differs")
    return {
        "typed_leaf_count": 75,
        "param_bfloat16_count": 74,
        "correlation": expected_correlation,
    }
