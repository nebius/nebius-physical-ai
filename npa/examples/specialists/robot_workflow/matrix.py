"""Validate prepared Fetch scenes without invoking a model or simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def load_matrix(path: Path) -> tuple[dict, str]:
    """Read fixed scene inputs and explicit physical negative controls.

    Args: path: UTF-8 JSON scene matrix.
    Returns: Validated matrix and SHA-256 of its original bytes.
    Raises: ValueError: Input schema, identities, scenes or seeds are invalid.
    """
    from npa.workbench.token_factory.robot_scene import RobotScene

    raw = path.read_bytes()
    matrix = json.loads(raw)
    if (
        set(matrix) != {"schema", "cases"}
        or matrix["schema"] != "npa.robot-workflow.v1"
    ):
        raise ValueError("unsupported prepared scene matrix")
    if not isinstance(matrix["cases"], list) or not matrix["cases"]:
        raise ValueError("scene matrix must contain cases")
    seen = set()
    for case in matrix["cases"]:
        if set(case) != {"id", "scene", "seed", "controller", "expected_accepted"}:
            raise ValueError("unexpected scene case fields")
        identifier = case["id"]
        if not isinstance(identifier, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]*", identifier
        ):
            raise ValueError("case id must be a simple name")
        if identifier in seen or type(case["seed"]) is not int or case["seed"] < 0:
            raise ValueError("duplicate case id or invalid seed")
        seen.add(identifier)
        if case["controller"] not in {"pick_place", "open_gripper"}:
            raise ValueError("unsupported controller")
        if type(case["expected_accepted"]) is not bool:
            raise ValueError("expected acceptance must be boolean")
        RobotScene.model_validate(case["scene"])
    return matrix, hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a local artifact without loading its full contents into memory.

    Args: path: Existing regular file.
    Returns: SHA-256 hexadecimal digest.
    Raises: OSError: File cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
