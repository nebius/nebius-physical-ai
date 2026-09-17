"""Verify visibility gating, unbiased camera sampling and physical-action losses."""

import numpy as np
import pytest

from npa.workflows.behavior_challenge.camera_evidence import (
    EvidenceThresholds,
    camera_weights,
    consistency_loss,
    sample_camera,
)


def test_search_frames_do_not_regularize_from_visible_tables():
    invariance, sufficiency = camera_weights([[0, 0, 0]], [[1, 0.7, 0.5]])
    np.testing.assert_array_equal(invariance, [[0, 0, 0]])
    np.testing.assert_array_equal(sufficiency, [[0, 0, 0]])


def test_wrist_only_visibility_preserves_the_informative_camera():
    invariance, sufficiency = camera_weights([[0, 0, 0.1]], [[1, 1, 0]])
    np.testing.assert_array_equal(invariance, [[1, 1, 0]])
    np.testing.assert_array_equal(sufficiency, [[0, 0, 1]])


def test_ambiguous_camera_is_neither_irrelevant_nor_sufficient():
    invariance, sufficiency = camera_weights([[0.2, 0.1, 0]], [[0, 0, 0]])
    np.testing.assert_array_equal(invariance, [[0, 0, 1]])
    np.testing.assert_array_equal(sufficiency, [[1, 0, 0]])


def test_interaction_geometry_requires_a_visible_focal_object():
    invariance, sufficiency = camera_weights([[0.01, 0.01, 0]], [[0.5, 0, 1]])
    assert sufficiency[0, 0] == 1
    assert sufficiency[0, 2] == 0
    assert invariance[0, 2] == 1


def test_importance_sampling_estimates_the_full_camera_sum():
    weights = np.tile([0.1, 0.3, 0.6], (100_000, 1))
    camera, total = sample_camera(weights, np.random.default_rng(17))
    costs = np.array([1, 4, 9])
    expected = (weights[0] * costs).sum()
    assert abs((total * costs[camera]).mean() - expected) < 0.03


def test_zero_weight_rows_have_finite_zero_importance():
    camera, total = sample_camera([[0, 0, 0], [0, 0, 1]], np.random.default_rng(0))
    np.testing.assert_array_equal(camera, [0, 2])
    np.testing.assert_array_equal(total, [0, 1])


def test_sampling_is_reproducible_and_skips_zero_weight_cameras():
    weights = np.tile([0, 0.4, 0.6], (100, 1))
    first, _ = sample_camera(weights, np.random.default_rng(3))
    second, _ = sample_camera(weights, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)


def test_consistency_excludes_action_padding_and_applies_importance():
    clean = np.zeros((2, 30, 32))
    corrupted = np.full_like(clean, 2)
    corrupted[..., 23:] = 1000
    loss = consistency_loss(clean, corrupted, np.array([2, 0]), action_dimensions=23)
    np.testing.assert_array_equal(loss, [8, 0])


@pytest.mark.parametrize('areas', [[[np.nan, 0]], [[-0.1, 0]], [[1.1, 0]], [], [0, 0]])
def test_invalid_visibility_is_rejected(areas):
    with pytest.raises(ValueError):
        camera_weights(areas, areas)


@pytest.mark.parametrize('arguments', [{'low': 0}, {'high': 1}, {'low': 0.9},
                                    {'global_visibility': 0}, {'focal': -1},
                                    {'interaction_weight': float('nan')}])
def test_invalid_thresholds_are_rejected(arguments):
    with pytest.raises(ValueError):
        EvidenceThresholds(**arguments)


def test_mismatched_camera_matrices_are_rejected():
    with pytest.raises(ValueError, match='matching shapes'):
        camera_weights([[0, 0]], [[0, 0, 0]])


@pytest.mark.parametrize('weights', [[[float('inf')]], [[-1]], [], [1]])
def test_invalid_sampling_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        sample_camera(weights, np.random.default_rng(0))


def test_invalid_physical_action_shape_is_rejected():
    with pytest.raises(ValueError, match='Real action dimensions'):
        consistency_loss(np.zeros((1, 30, 23)), np.zeros((1, 30, 23)), np.ones(1), action_dimensions=32)
