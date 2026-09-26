"""Tests for canonical fit row ordering and fail-closed validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.fit_data import concatenate


def test_concatenate_restores_canonical_episode_row_order(fit_data) -> None:
    slices = (slice(0, 8), slice(8, None))
    halves = [
        replace(
            fit_data,
            **{
                name: getattr(fit_data, name)[part]
                for name in fit_data.__dataclass_fields__
            },
        )
        for part in slices
    ]
    result = concatenate(list(reversed(halves)))
    assert list(zip(result.episodes, result.rows, strict=True)) == sorted(
        zip(result.episodes, result.rows, strict=True)
    )


def test_duplicate_rows_fail_closed(fit_data) -> None:
    duplicate = replace(fit_data, rows=np.zeros_like(fit_data.rows))
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.validated()
