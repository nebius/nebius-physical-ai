from __future__ import annotations

import pytest

from npa.guardrails.pytest_collection import assert_nonzero_collection


def test_collection_guard_accepts_nonzero_collection() -> None:
    assert_nonzero_collection(1)


def test_collection_guard_rejects_zero_collection() -> None:
    with pytest.raises(pytest.UsageError, match="collected 0 tests"):
        assert_nonzero_collection(0)
