"""Define the generated regression recipe and portable artifact integrity checks."""

import hashlib
import json
import math
import random
import re

IMAGE = "docker.io/pytorch/pytorch@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407"
TORCH_VERSION = "2.13.0+cu130"
RECIPE = {
    "seed": 7,
    "steps": 32,
    "samples": 1024,
    "features": 8,
    "heldout_samples": 1024,
    "learning_rate": 0.1,
    "momentum": 0.8,
}
INITIAL = [0.02 * (index - 3.5) for index in range(8)] + [-0.125]


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _dataset(seed, samples):
    generator = random.Random(seed)
    rows = [[generator.gauss(0, 1) for _ in range(8)] for _ in range(samples)]
    targets = [
        sum(value * (index + 1) / 8 for index, value in enumerate(row)) + 0.25
        for row in rows
    ]
    return rows, targets


def _check_identity(source_sha, payload_sha, run_id):
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or source_sha == "0" * 40:
        raise ValueError("candidate source SHA must contain exactly 40 hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", payload_sha) or payload_sha == "0" * 64:
        raise ValueError("payload SHA must contain exactly 64 hex characters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run ID must be a simple identifier")


def _close(actual, expected, label, tolerance=1e-8):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise TypeError(f"{label}: expected a real number")
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f"{label}: numerical mismatch")


def _validate_integrity(manifest, artifacts):
    if manifest.get("schema") != "cuda-regression-manifest/v1":
        raise ValueError("artifact manifest schema differs")
    if set(manifest.get("artifacts", {})) != {"checkpoint.pt", "metrics.json"}:
        raise ValueError("artifact manifest inventory differs")
    for name, record in manifest["artifacts"].items():
        data = artifacts[name]
        if not data or record != {"sha256": _sha256(data), "size": len(data)}:
            raise ValueError(f"artifact integrity mismatch: {name}")
