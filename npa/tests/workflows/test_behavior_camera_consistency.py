"""Exercise the real optional JAX objective and its training gradients."""

import numpy as np
import pytest

from npa.workflows.behavior_challenge.camera_consistency import regularization_losses

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


CAMERAS = ("head", "left_wrist", "right_wrist")


def images():
    # Reverse insertion order catches accidental reliance on dictionary ordering.
    return {name: jnp.ones((2, 16, 16, 3)) for name in reversed(CAMERAS)}


def objective(predict, weights):
    return regularization_losses(
        predict,
        images(),
        weights,
        jax.random.key(3),
        camera_names=CAMERAS,
        action_dimensions=2,
    )


def test_informative_wrist_is_preserved_under_both_objectives():
    weights = jnp.array([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]] * 2)

    def predict(views, key):
        # Shared random noise must cancel; only the informative wrist drives actions.
        value = views["left_wrist"].mean(axis=(1, 2, 3)) + jax.random.normal(key, (2,))
        return jnp.broadcast_to(value[:, None, None], (2, 4, 3))

    losses = jax.jit(lambda: objective(predict, weights))()
    np.testing.assert_array_equal(np.asarray(losses), 0.0)


def test_jitted_gradient_reduces_dependence_on_irrelevant_cameras():
    weights = jnp.array(
        [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )

    def loss(scale):
        def predict(views, key):
            value = sum(views[name].mean(axis=(1, 2, 3)) for name in CAMERAS)
            return jnp.broadcast_to(value[:, None, None] * scale, (2, 4, 3))

        invariance, sufficiency = objective(predict, weights)
        return (invariance + sufficiency).sum(), (invariance, sufficiency)

    (value, per_example), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        jnp.ones(3)
    )
    assert np.isfinite(value) and value > 0
    assert np.all(np.asarray(gradient[:2]) > 0)
    assert gradient[2] == 0  # Padding cannot influence the objective.
    assert per_example[0][0] > 0 and per_example[1][0] > 0
    assert per_example[0][1] == 0 and per_example[1][1] == 0


def test_no_visible_object_has_finite_zero_loss_and_gradient():
    def loss(scale):
        def predict(views, key):
            value = views["head"].mean(axis=(1, 2, 3)) * scale
            return jnp.broadcast_to(value[:, None, None], (2, 4, 3))

        return sum(value.sum() for value in objective(predict, jnp.zeros((2, 2, 3))))

    value, gradient = jax.jit(jax.value_and_grad(loss))(1.0)
    assert value == 0 and gradient == 0


def test_three_predictions_share_the_same_flow_key():
    keys = []

    def predict(views, key):
        keys.append(np.asarray(jax.random.key_data(key)))
        return jnp.ones((2, 4, 3))

    objective(predict, jnp.ones((2, 2, 3)))
    assert len(keys) == 3
    np.testing.assert_array_equal(keys[0], keys[1])
    np.testing.assert_array_equal(keys[0], keys[2])


@pytest.mark.parametrize("names", [("head",), ("head", "head", "right_wrist")])
def test_ambiguous_camera_mapping_is_rejected(names):
    with pytest.raises(ValueError, match="exactly once"):
        regularization_losses(
            None,
            images(),
            jnp.ones((2, 2, 3)),
            jax.random.key(0),
            camera_names=names,
            action_dimensions=2,
        )


def test_mismatched_evidence_is_rejected_before_prediction():
    with pytest.raises(ValueError, match="Evidence must match"):
        objective(None, jnp.ones((2, 3, 2)))
