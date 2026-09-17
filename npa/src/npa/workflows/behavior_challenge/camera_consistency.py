"""Run training-only EGR corruptions in the policy's optional JAX environment."""

from .camera_evidence import consistency_loss


def regularization_losses(predict_velocity, images, weights, key, *, camera_names, action_dimensions):
    """Compare clean and corrupted predictions with a shared flow realization.

    The caller adds these per-example losses to its imitation objective. Evidence
    stays outside the prediction callback. This function requires JAX in the
    policy training environment, without adding it to Workbench's dependencies.

    Args:
        predict_velocity: Differentiable callback accepting images and the shared
            prediction key, returning [batch, horizon, padded action dimensions].
        images: Mapping of camera names to normalized [batch, H, W, C] arrays.
        weights: Nonnegative [batch, 2, cameras] evidence weights; axis one orders
            invariance then sufficiency. Camera order must match ``camera_names``.
        key: JAX random key, distinct from the data pipeline's sampling generator.
        camera_names: Explicit camera ordering, preserved when JAX sorts dictionary keys.
        action_dimensions: Number of physical action coordinates, excluding padding.
    Returns:
        Invariance and sufficiency losses, each shaped [batch].
    Raises:
        ValueError: Camera images or evidence have incompatible shapes.
        ImportError: JAX is unavailable in the calling training environment.
    """
    import jax

    if len(set(camera_names)) != len(camera_names) or set(camera_names) != set(images):
        raise ValueError("Camera names must identify each image exactly once")
    images = {name: images[name] for name in camera_names}
    if not images or any(len(value.shape) != 4 for value in images.values()):
        raise ValueError("Camera images require nonempty [batch, height, width, channels] arrays")
    batch = next(iter(images.values())).shape[0]
    if any(value.shape[0] != batch or not all(value.shape) for value in images.values()):
        raise ValueError("Camera images must have matching nonempty batches")
    if weights.shape != (batch, 2, len(images)):
        raise ValueError("Evidence must match the image batch and camera order")
    prediction_key, selection_key, corruption_key = jax.random.split(key, 3)
    clean = predict_velocity(images, prediction_key)
    selection_keys = jax.random.split(selection_key, 2)
    corruption_keys = jax.random.split(corruption_key, 2)
    losses = []
    for index in range(2):
        selected, multiplier = _sample(weights[:, index], selection_keys[index])
        corrupted = _erase(images, selected, corruption_keys[index], preserve_selected=bool(index))
        velocity = predict_velocity(corrupted, prediction_key)
        losses.append(consistency_loss(clean, velocity, multiplier, action_dimensions=action_dimensions))
    return tuple(losses)


def _sample(weights, key):
    import jax
    import jax.numpy as jnp

    total = weights.sum(axis=-1)
    safe = jnp.where(total[:, None] > 0, weights, jnp.ones_like(weights))
    selected = jax.random.categorical(key, jnp.where(safe > 0, jnp.log(safe), -jnp.inf))
    return selected, total


def _erase(images, selected, key, *, preserve_selected):
    import jax
    import jax.numpy as jnp

    output = {}
    for camera, ((name, value), camera_key) in enumerate(zip(images.items(), jax.random.split(key, len(images)))):
        batch, height, width, _ = value.shape
        box = jax.random.uniform(camera_key, (batch, 4))
        rectangle_height = height * (0.2 + 0.4 * box[:, 0])
        rectangle_width = width * (0.2 + 0.4 * box[:, 1])
        top = (height - rectangle_height) * box[:, 2]
        left = (width - rectangle_width) * box[:, 3]
        rows = jnp.arange(height)[None, :, None]
        columns = jnp.arange(width)[None, None, :]
        mask = (rows >= top[:, None, None]) & (rows < (top + rectangle_height)[:, None, None])
        mask &= (columns >= left[:, None, None]) & (columns < (left + rectangle_width)[:, None, None])
        affected = selected != camera if preserve_selected else selected == camera
        output[name] = jnp.where((mask & affected[:, None, None])[..., None], 0.0, value)
    return output
