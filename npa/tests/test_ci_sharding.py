"""Validate deterministic CI partitioning for the full pytest suite."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest as suite_conftest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import merge_ci_test_timings as timing_merge  # noqa: E402


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


def test_ci_shards_balance_recorded_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spread slow modules using measured work rather than test count.

    Args:
        monkeypatch: Replaces the committed timing manifest.
        tmp_path: Holds a public synthetic timing manifest.
    Returns:
        None.
    Raises:
        AssertionError: Greedy partitioning concentrates measured slow work.
    """

    manifest = tmp_path / "durations.json"
    manifest.write_text(json.dumps({"tests/slow.py": 40.0, "tests/fast.py": 1.0}))
    monkeypatch.setattr(suite_conftest, "_CI_TIMING_MANIFEST", manifest)
    items = [
        *(SimpleNamespace(nodeid=f"tests/slow.py::test_{index}") for index in range(8)),
        *(SimpleNamespace(nodeid=f"tests/fast.py::test_{index}") for index in range(8)),
    ]
    shards = [
        suite_conftest._items_for_ci_shard(items, shard_index, 4)
        for shard_index in range(4)
    ]
    slow_counts = [
        sum(item.nodeid.startswith("tests/slow.py") for item in shard)
        for shard in shards
    ]
    assert slow_counts == [2, 2, 2, 2]


def test_ci_timing_artifacts_merge_by_module(tmp_path: Path) -> None:
    """Combine non-overlapping shard measurements into the next profile.

    Args:
        tmp_path: Holds synthetic trusted-main artifacts.
    Returns:
        None.
    Raises:
        AssertionError: Duplicate module measurements are not summed.
    """

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"tests/a.py": 1.25, "tests/b.py": 2.0}))
    second.write_text(json.dumps({"tests/a.py": 0.75}))
    assert timing_merge.merge_timings([first, second]) == {
        "tests/a.py": 2.0,
        "tests/b.py": 2.0,
    }


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
