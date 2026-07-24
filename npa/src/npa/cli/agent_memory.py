"""Compatibility shim: run/experiment memory now lives in the shipped package.

The real implementation moved to ``npa/src/npa/agent_backend/memory.py`` (Phase G:
shipped importable package instead of embed). This shim preserves the historical
``npa.cli.agent_memory`` import path used by callers and tests.
"""

from __future__ import annotations

from npa.agent_backend.memory import (  # noqa: F401
    INDEX_KEY,
    MAX_INDEX_ENTRIES,
    MEMORY_KEY_PREFIX,
    InMemoryStore,
    JsonFileStore,
    RunMemory,
)

__all__ = [
    "INDEX_KEY",
    "MAX_INDEX_ENTRIES",
    "MEMORY_KEY_PREFIX",
    "InMemoryStore",
    "JsonFileStore",
    "RunMemory",
]
