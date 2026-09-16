"""Validate deterministic CI partitioning for the full pytest suite."""

from types import SimpleNamespace

import pytest

import conftest as suite_conftest


def test_ci_shard_coordinates_are_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave ordinary local and unsharded test runs unchanged.

    Args:
        monkeypatch: Isolated environment mutation fixture.
    Returns:
        None.
    Raises:
        None.
    """

    monkeypatch.delenv("NPA_CI_SHARD_INDEX", raising=False)
    monkeypatch.delenv("NPA_CI_TOTAL_SHARDS", raising=False)
    assert suite_conftest._ci_shard_coordinates() is None


def test_ci_shards_cover_every_item_once() -> None:
    """Assign the complete collection without overlap between shards.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Shards overlap, omit an item, or become imbalanced.
    """

    items = [SimpleNamespace(nodeid=f"test_{index:02d}") for index in range(17)]
    shards = [
        suite_conftest._items_for_ci_shard(items, shard_index, 4)
        for shard_index in range(4)
    ]
    assigned = [item.nodeid for shard in shards for item in shard]
    assert sorted(assigned) == sorted(item.nodeid for item in items)
    assert len(assigned) == len(set(assigned))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize(
    ("index", "total"),
    [(None, "4"), ("1", None), ("zero", "4"), ("0", "4"), ("5", "4")],
)
def test_ci_shard_coordinates_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, index: str | None, total: str | None
) -> None:
    """Reject partial, non-numeric, and out-of-range shard coordinates.

    Args:
        monkeypatch: Isolated environment mutation fixture.
        index: Candidate one-based shard index.
        total: Candidate shard count.
    Returns:
        None.
    Raises:
        AssertionError: Invalid coordinates do not raise ``pytest.UsageError``.
    """

    monkeypatch.delenv("NPA_CI_SHARD_INDEX", raising=False)
    monkeypatch.delenv("NPA_CI_TOTAL_SHARDS", raising=False)
    if index is not None:
        monkeypatch.setenv("NPA_CI_SHARD_INDEX", index)
    if total is not None:
        monkeypatch.setenv("NPA_CI_TOTAL_SHARDS", total)
    with pytest.raises(pytest.UsageError):
        suite_conftest._ci_shard_coordinates()
