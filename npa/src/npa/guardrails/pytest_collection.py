"""Pytest collection guardrails."""

from __future__ import annotations

import pytest


def assert_nonzero_collection(count: int) -> None:
    """Reject test invocations that collected no tests."""

    if count <= 0:
        raise pytest.UsageError(
            "pytest collected 0 tests; refusing a false green guardrail run"
        )
