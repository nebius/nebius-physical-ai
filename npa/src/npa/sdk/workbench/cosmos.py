"""Compatibility imports for Cosmos workbench SDK functions."""

from __future__ import annotations

from npa.workbench.cosmos import (
    Cosmos3AccessConfig,
    Cosmos3AccessError,
    Cosmos3CheckResult,
    Cosmos3FetchResult,
    Cosmos3ServeConfig,
    check,
    check_cosmos3_access,
    fetch,
    fetch_cosmos3_artifacts,
)

__all__ = [
    "Cosmos3AccessConfig",
    "Cosmos3AccessError",
    "Cosmos3CheckResult",
    "Cosmos3FetchResult",
    "Cosmos3ServeConfig",
    "check",
    "check_cosmos3_access",
    "fetch",
    "fetch_cosmos3_artifacts",
]
