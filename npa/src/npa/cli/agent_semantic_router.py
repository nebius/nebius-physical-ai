"""Compatibility shim for the shipped semantic-router implementation."""

from __future__ import annotations

from npa.agent_backend import semantic_router as _impl
from npa.agent_backend.semantic_router import (
    MODE_ACTION,
    MODE_INTENT,
    MODE_NONE,
    SOURCE_CACHE,
    SOURCE_KEYWORD,
    SOURCE_MODEL,
    SOURCE_NONE,
    classify_intent_semantic,
)

__all__ = [
    "MODE_ACTION",
    "MODE_INTENT",
    "MODE_NONE",
    "SOURCE_CACHE",
    "SOURCE_KEYWORD",
    "SOURCE_MODEL",
    "SOURCE_NONE",
    "classify_intent_semantic",
]


def __getattr__(name: str):
    """Preserve access to historical private helpers without duplicating logic."""
    return getattr(_impl, name)
