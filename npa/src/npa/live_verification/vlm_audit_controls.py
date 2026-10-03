"""Generate frozen synthetic visual controls for real hosted paired and blinded-preference checks."""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from npa.clients.token_factory import DEFAULT_BASE_URL

PAIRED_MODELS = ("MiniMaxAI/MiniMax-M3", "google/gemma-3-27b-it")
CONTROL_LABELS = {"inside": True, "outside": False, "blank": False}
AUDIT_CASES_BY_KIND = {"paired": "paired-judges", "preference": "blinded-preference"}
GENERATED_CONTROL_NAMES_BY_KIND = {"paired": tuple(CONTROL_LABELS)}
PAIRED_CONTROL_TASK = (
    "Describe the visible shapes and their positions. Judge whether a red square "
    "is fully inside the green rectangular outline."
)
PAIRED_CONTROL_RUBRIC = (
    "Score 1 only if a red square is fully inside the green rectangular outline. "
    "Score 0 if the red square is outside, or either shape is absent. "
    "Judge only the submitted image. Return bare JSON without Markdown fences."
)


PREFERENCE_MODELS = ("MiniMaxAI/MiniMax-M3",)
CONTROL_PAIRS = {
    "inside-first": ("inside", "outside", "baseline"),
    "inside-second": ("outside", "inside", "candidate"),
    "identical": ("inside", "inside", "tie"),
}
PREFERENCE_CONTROL_TASK = (
    "Compare how well the visible shapes satisfy the visual requirement."
)
PREFERENCE_CONTROL_RUBRIC = (
    "Prefer the image with a red square fully inside the green rectangular outline. "
    "A square outside the outline does not satisfy the requirement. "
    "If the images are visually equivalent under this rule, return tie. "
    "Judge only visible shapes, not hidden physical state."
)

GENERATED_CONTROL_NAMES_BY_KIND["preference"] = tuple(CONTROL_PAIRS)


def audit_controls(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve named controls while retaining the single-request operator format.

    Args:
        case: One configured audit case.
    Returns:
        Nonempty mapping of control labels to request/expectation objects.
    Raises:
        ValueError: The configured controls are empty or malformed.
    """
    controls = case.get("controls", {"operator": case})
    if not isinstance(controls, dict) or not controls:
        raise ValueError("audit controls must be a nonempty object")
    if any(not isinstance(value, dict) for value in controls.values()):
        raise ValueError("audit control must be an object")
    return controls


def configured_audit_cases(
    config: dict[str, Any],
    *,
    available_cases: tuple[str, ...],
    required_kind: str | None = None,
) -> tuple[str, ...]:
    """Select one validated audit family without collecting unrelated operations.

    Args:
        config: Private generated or operator audit configuration.
        available_cases: Operations implemented by the caller.
        required_kind: Explicit runner selection, if supplied.
    Returns:
        The single configured case; generated controls retain their frozen order.
    Raises:
        ValueError: Selection is unknown, mixed, unavailable, or inconsistent.
    """
    cases = config.get("cases") if isinstance(config, dict) else None
    if not isinstance(cases, dict) or len(cases) != 1:
        raise ValueError("invalid_audit_case_selection")
    case = next(iter(cases))
    kind = next(
        (key for key, value in AUDIT_CASES_BY_KIND.items() if value == case), None
    )
    if kind is None or case not in available_cases or not isinstance(cases[case], dict):
        raise ValueError("invalid_audit_case_selection")
    if (required_kind is not None and required_kind != kind) or (
        "audit_kind" in config and config["audit_kind"] != kind
    ):
        raise ValueError("invalid_audit_case_selection")
    if "control_schema" in config:
        _validate_generated_controls(config, case, kind)
    return (case,)


def _validate_generated_controls(config: dict[str, Any], case: str, kind: str) -> None:
    controls = audit_controls(config["cases"][case])
    if (
        config.get("audit_kind") != kind
        or config["control_schema"] != f"npa.{kind}_visual_controls.v1"
        or tuple(controls) != GENERATED_CONTROL_NAMES_BY_KIND.get(kind)
    ):
        raise ValueError("invalid_generated_audit_controls")


def _control_png(name: str) -> bytes:
    image = Image.new("RGB", (640, 480), "white")
    if name != "blank":
        draw = ImageDraw.Draw(image)
        draw.rectangle((360, 130, 610, 430), outline="green", width=12)
        draw.ellipse((40, 30, 130, 120), fill="blue")
        left = 420 if name == "inside" else 170
        draw.rectangle((left, 230, left + 90, 320), fill="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generated_paired_config(directory: Path) -> dict[str, Any]:
    """Create private deterministic images and frozen paired visual expectations.

    Args:
        directory: New fixture directory inside the private run evidence root.
    Returns:
        Audit configuration; it contains no credential values.
    Raises:
        OSError: Fixtures cannot be created exclusively.
    """
    directory.mkdir(mode=0o700)
    controls = {}
    for name, expected in CONTROL_LABELS.items():
        path = directory / f"{name}.png"
        pixels = _control_png(name)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(pixels)
        controls[name] = {
            "request": _control_request(path),
            "input_sha256": hashlib.sha256(pixels).hexdigest(),
            "expectations": {
                "primary.result.passed": expected,
                "secondary.result.passed": expected,
                "escalation_required": False,
            },
        }
    return {
        "audit_kind": "paired",
        "control_schema": "npa.paired_visual_controls.v1",
        "cases": {"paired-judges": {"controls": controls}},
    }


def _control_request(path: Path) -> dict[str, Any]:
    return {
        "input_path": str(path),
        "primary_model": PAIRED_MODELS[0],
        "secondary_model": PAIRED_MODELS[1],
        "endpoint_url": DEFAULT_BASE_URL,
        "api_key_env": "NEBIUS_TOKEN_FACTORY_KEY",
        "task": PAIRED_CONTROL_TASK,
        "rubric": PAIRED_CONTROL_RUBRIC,
        "frame_selection": "final",
        "max_frames": 1,
        "success_threshold": 0.8,
    }


def generated_preference_config(directory: Path) -> dict[str, Any]:
    """Create private images and frozen preferences before real hosted inference.

    Args:
        directory: New fixture directory inside the private evidence root.
    Returns:
        Credential-free configuration containing three counterbalanced controls.
    Raises:
        OSError: The private images cannot be created exclusively.
    """
    directory.mkdir(mode=0o700)
    controls = {}
    for name, (first, second, expected) in CONTROL_PAIRS.items():
        paths, hashes = [], []
        for index, image in enumerate((first, second)):
            path = directory / f"{name}-{index}.png"
            pixels = _control_png(image)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(pixels)
            paths.append(path)
            hashes.append(hashlib.sha256(pixels).hexdigest())
        controls[name] = {
            "request": {
                "baseline_path": str(paths[0]),
                "candidate_path": str(paths[1]),
                "model": PREFERENCE_MODELS[0],
                "endpoint_url": DEFAULT_BASE_URL,
                "api_key_env": "NEBIUS_TOKEN_FACTORY_KEY",
                "task": PREFERENCE_CONTROL_TASK,
                "rubric": PREFERENCE_CONTROL_RUBRIC,
            },
            "baseline_sha256": hashes[0],
            "candidate_sha256": hashes[1],
            "expectations": {
                "mapped_preferences": [expected, expected],
                "escalation_required": False,
                "status": (
                    "consistent_tie"
                    if expected == "tie"
                    else f"consistent_{expected}_preference"
                ),
            },
        }
    return {
        "control_schema": "npa.preference_visual_controls.v1",
        "audit_kind": "preference",
        "cases": {"blinded-preference": {"controls": controls}},
    }
