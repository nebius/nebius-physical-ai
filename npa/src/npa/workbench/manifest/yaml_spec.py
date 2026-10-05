"""YAML surface: a thin waiter over Runtime.invoke.

A YAML invocation spec names the tool/command/inputs; this translates it
to Runtime.invoke. Syntax translation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .runtime import Runtime, InvocationResult


def invoke_spec(
    runtime: Runtime, spec_path: str | Path, backend: str = "local"
) -> InvocationResult:
    spec: dict[str, Any] = yaml.safe_load(Path(spec_path).read_text())
    return runtime.invoke(
        tool=spec["tool"],
        version=str(spec["version"]),
        command=spec["command"],
        inputs=spec.get("inputs") or {},
        surface="yaml",
        backend=backend,
    )
