"""Synthetic fixtures for the portable conditional replanning package."""

from __future__ import annotations

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.artifact import (
    ARTIFACT_SCHEMA,
    GateArtifact,
    expected_arrays,
)
from npa.workflows.behavior_challenge.conditional_replan.contract import (
    fixed_semantic_config,
)
from npa.workflows.behavior_challenge.conditional_replan.fit_data import FitData


@pytest.fixture
def artifact() -> GateArtifact:
    """Return a finite artifact matching the qualified schema."""
    arrays = {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in expected_arrays().items()
    }
    arrays["input_std"].fill(1)
    arrays["target_std"].fill(1)
    semantics = fixed_semantic_config()
    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "selection": {"linear_control": 1, "mlp": 1},
        "development_or_report_used": False,
        "feature_config": semantics["features"],
        "target_config": semantics["target"],
        "model_config": semantics["models"],
        "gate_config": semantics["gate"],
    }
    return GateArtifact(arrays, metadata).validated()


@pytest.fixture
def fit_data() -> FitData:
    """Return small canonical fit and validation rows."""
    generator = np.random.default_rng(7)
    count = 16
    features = generator.normal(size=(count, 1233)).astype(np.float32)
    continued = generator.random(count).astype(np.float64)
    fresh = generator.random(count).astype(np.float64)
    return FitData(
        features,
        continued,
        fresh,
        np.square(continued - fresh),
        np.repeat(np.arange(4), 4),
        np.tile(np.arange(4), 4),
        np.asarray(["fit"] * 12 + ["validation"] * 4),
    ).validated()
