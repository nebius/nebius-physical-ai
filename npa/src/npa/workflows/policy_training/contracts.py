"""Validate immutable episode, checkpoint, and evaluation handoffs."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def digest(payload: Any) -> str:
    """Hash a JSON artifact independently of formatting.

    Args:
        payload: JSON-serializable content.
    Returns:
        Canonical SHA-256 digest.
    Raises:
        ValueError: Content contains non-finite numbers.
    """
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def probability(value: Any) -> float:
    """Validate a probability without accepting booleans or non-finite values.

    Args:
        value: Numeric probability.
    Returns:
        Validated floating-point value.
    Raises:
        ValueError: Value is outside the probability domain.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and between zero and one")
    return float(value)


def checkpoint(payload: dict[str, Any]) -> dict[str, str]:
    """Require an explicit checkpoint location and content digest.

    Args:
        payload: Completed training result.
    Returns:
        Checkpoint identity.
    Raises:
        ValueError: The checkpoint identity is incomplete.
    """
    value = payload.get("checkpoint", {})
    sha = value.get("sha256", "")
    if not value.get("uri") or len(sha) != 64:
        raise ValueError("checkpoint requires a URI and SHA-256 digest")
    if any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("checkpoint SHA-256 must be lowercase hexadecimal")
    return {"uri": value["uri"], "sha256": sha}


def approved_checkpoint(uri: str, *, _reference=False) -> dict[str, str]:
    """Load only a checkpoint that passed its measured gate.

    Args:
        uri: Gate artifact location.
        _reference: Internal teaching-demo validation only.
    Returns:
        Approved checkpoint identity.
    Raises:
        ValueError: The gate did not promote.
    """
    from npa.workbench.dataset.storage import read_json_uri

    gate = read_json_uri(uri)
    if gate.get("decision") != "promote_checkpoint":
        raise ValueError("checkpoint has not passed its evaluation gate")
    _validate_engine(gate, reference=_reference)
    return checkpoint(gate)


def _validate_engine(payload, *, reference=False):
    engine = payload.get("engine")
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("result requires the actual execution engine")
    if not reference and engine.strip() == "numpy-planar-reference":
        raise ValueError("reference engine cannot qualify a production checkpoint")
