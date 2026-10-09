"""SDK surface: a thin waiter over Runtime.invoke.

Syntax translation only — no invocation semantics live here.
"""

from __future__ import annotations

from typing import Any

from .runtime import Runtime, InvocationResult


def invoke(
    runtime: Runtime,
    tool: str,
    version: str,
    command: str,
    inputs: dict[str, Any] | None = None,
    backend: str = "local",
) -> InvocationResult:
    return runtime.invoke(
        tool=tool,
        version=version,
        command=command,
        inputs=inputs,
        surface="sdk",
        backend=backend,
    )
