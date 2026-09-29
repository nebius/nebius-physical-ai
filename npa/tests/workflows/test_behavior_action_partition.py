"""Check the action-only parameter filter in the native Flax runtime."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-matched-training"
)
sys.path.insert(0, str(IMPLEMENTATION))

from action_partition import freeze_filter  # noqa: E402


def test_native_freeze_filter_preserves_nonparameter_statistics() -> None:
    nnx = pytest.importorskip("flax.nnx")
    model = nnx.Dict(
        action_in_proj=nnx.Dict(bias=nnx.Param(np.ones(2, dtype=np.float32))),
        stage_predictor=nnx.Param(np.ones(2, dtype=np.float32)),
        future_parameter=nnx.Param(np.ones(2, dtype=np.float32)),
        action_correlation_cholesky=nnx.Intermediate(np.eye(2, dtype=np.float32)),
    )
    state = nnx.state(model)
    frozen = freeze_filter()
    assert set(state.filter(frozen).flat_state()) == {
        ("stage_predictor",),
        ("future_parameter",),
    }
    trainable = nnx.All(nnx.Param, nnx.Not(frozen))
    assert set(state.filter(trainable).flat_state()) == {
        ("action_in_proj", "bias"),
    }
