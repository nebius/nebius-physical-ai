"""Represent and order verified conditional-replanning fit rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extraction import FEATURE_DIMENSION


@dataclass(frozen=True)
class FitData:
    """Hold verified TRAIN-only rows in canonical order.

    Args:
        features: Float32 feature matrix.
        continue_error: Float64 continued-proposal error per row.
        fresh_error: Float64 fresh-proposal error per row.
        distance: Float64 proposal distance per row.
        episodes: Integer episode identifier per row.
        rows: Integer within-episode row index.
        partitions: Unicode ``fit`` or ``validation`` label per row.
    Returns:
        None.
    Raises:
        None.
    """

    features: np.ndarray
    continue_error: np.ndarray
    fresh_error: np.ndarray
    distance: np.ndarray
    episodes: np.ndarray
    rows: np.ndarray
    partitions: np.ndarray

    def validated(self) -> FitData:
        """Validate shapes, dtypes, values, and canonical row order.

        Args:
            None.
        Returns:
            This fit-data object after validation.
        Raises:
            ValueError: If any row violates the public data contract.
        """
        count = len(self.features)
        expected = {
            "features": ((count, FEATURE_DIMENSION), np.dtype(np.float32)),
            "continue_error": ((count,), np.dtype(np.float64)),
            "fresh_error": ((count,), np.dtype(np.float64)),
            "distance": ((count,), np.dtype(np.float64)),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"fit data differs: {name}")
        _validate_labels(self, count)
        if canonical_order(self).tolist() != list(range(count)):
            raise ValueError("fit rows are not in canonical episode/row order")
        return self


def canonical_order(data: FitData) -> np.ndarray:
    """Return indices sorted by episode and within-episode row.

    Args:
        data: Candidate fit data.
    Returns:
        Integer canonical ordering indices.
    Raises:
        None.
    """
    return np.lexsort((data.rows, data.episodes))


def concatenate(shards: list[FitData]) -> FitData:
    """Combine verified episode shards into canonical order.

    Args:
        shards: One or more independently validated shards.
    Returns:
        Combined, canonical fit data.
    Raises:
        ValueError: If no shards are provided or episode rows overlap.
    """
    if not shards:
        raise ValueError("at least one fit shard is required")
    for shard in shards:
        shard.validated()
    names = FitData.__dataclass_fields__
    values = {
        name: np.concatenate([getattr(row, name) for row in shards]) for name in names
    }
    provisional = FitData(**values)
    order = canonical_order(provisional)
    result = FitData(**{name: value[order] for name, value in values.items()})
    return result.validated()


def _validate_labels(data: FitData, count: int) -> None:
    if data.episodes.shape != (count,) or data.rows.shape != (count,):
        raise ValueError("fit data row identities differ")
    if data.episodes.dtype.kind not in "iu" or data.rows.dtype.kind not in "iu":
        raise ValueError("fit data row identities are not integers")
    if data.partitions.shape != (count,) or data.partitions.dtype.kind != "U":
        raise ValueError("fit data partition representation differs")
    if not set(data.partitions.tolist()) <= {"fit", "validation"}:
        raise ValueError("fit data partition differs")
    pairs = list(zip(data.episodes.tolist(), data.rows.tolist(), strict=True))
    if len(pairs) != len(set(pairs)):
        raise ValueError("fit data contains duplicate episode rows")
