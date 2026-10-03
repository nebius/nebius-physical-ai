"""Generate frozen synthetic visual controls for real hosted blinded-preference checks."""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from npa.clients.token_factory import DEFAULT_BASE_URL

PREFERENCE_MODELS = ("MiniMaxAI/MiniMax-M3",)
CONTROL_PAIRS = {
    "inside-first": ("inside", "outside", "baseline"),
    "inside-second": ("outside", "inside", "candidate"),
    "identical": ("inside", "inside", "tie"),
}
CONTROL_TASK = "Compare how well the visible shapes satisfy the visual requirement."
CONTROL_RUBRIC = (
    "Prefer the image with a red square fully inside the green rectangular outline. "
    "A square outside the outline does not satisfy the requirement. "
    "If the images are visually equivalent under this rule, return tie. "
    "Judge only visible shapes, not hidden physical state."
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
                "task": CONTROL_TASK,
                "rubric": CONTROL_RUBRIC,
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
        "cases": {"blinded-preference": {"controls": controls}},
    }
