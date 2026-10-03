"""Generate frozen synthetic visual controls for real hosted paired-judge checks."""

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
CONTROL_TASK = (
    "Describe the visible shapes and their positions. Judge whether a red square "
    "is fully inside the green rectangular outline."
)
CONTROL_RUBRIC = (
    "Score 1 only if a red square is fully inside the green rectangular outline. "
    "Score 0 if the red square is outside, or either shape is absent. "
    "Judge only the submitted image. Return bare JSON without Markdown fences."
)


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
        "task": CONTROL_TASK,
        "rubric": CONTROL_RUBRIC,
        "frame_selection": "final",
        "max_frames": 1,
        "success_threshold": 0.8,
    }
