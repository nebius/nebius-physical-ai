"""Validate the writable NVIDIA Kit state boundary used by Isaac runtimes."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
from typing import Mapping


def validated_kit_args(environ: Mapping[str, str] | None = None) -> str:
    """Return launcher arguments bound to one strict child of writable state."""

    environment = os.environ if environ is None else environ
    state_text = environment.get("NPA_ISAAC_STATE_DIR", "").strip()
    state = Path(state_text)
    if not state_text or not state.is_absolute() or not state.is_dir():
        raise RuntimeError("NPA_ISAAC_STATE_DIR must be an existing absolute directory")
    if not os.access(state, os.W_OK):
        raise RuntimeError("NPA_ISAAC_STATE_DIR is not writable")

    raw = environment.get("NPA_ISAAC_KIT_ARGS", "").strip()
    try:
        arguments = shlex.split(raw)
    except ValueError as error:
        raise RuntimeError("NPA_ISAAC_KIT_ARGS is malformed") from error
    roots = _portable_roots(arguments)
    if len(roots) != 1:
        raise RuntimeError("NPA_ISAAC_KIT_ARGS must declare one --portable-root")

    portable = Path(roots[0])
    if not portable.is_absolute():
        raise RuntimeError("Kit portable root must be absolute")
    resolved_state = state.resolve(strict=True)
    resolved_portable = portable.resolve(strict=False)
    if resolved_portable == resolved_state or not resolved_portable.is_relative_to(
        resolved_state
    ):
        raise RuntimeError("Kit portable root must be inside NPA_ISAAC_STATE_DIR")
    return raw


def _portable_roots(arguments: list[str]) -> list[str]:
    """Extract portable roots while rejecting incomplete flag spellings."""

    roots: list[str] = []
    for index, value in enumerate(arguments):
        if value == "--portable-root":
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise RuntimeError("--portable-root requires a path")
            roots.append(arguments[index + 1])
        elif value.startswith("--portable-root="):
            root = value.partition("=")[2]
            if not root:
                raise RuntimeError("--portable-root requires a path")
            roots.append(root)
    return roots
