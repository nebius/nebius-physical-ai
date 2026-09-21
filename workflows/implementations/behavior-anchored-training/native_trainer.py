"""Wire the public stock-anchor objective into a native Flax NNX update."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

Path = tuple[str, ...]


def _path(value: Iterable[Any]) -> Path:
    return tuple(str(part) for part in value)


def _variable_type(variable: Any) -> str:
    semantic = getattr(variable, "type", type(variable))
    return getattr(semantic, "__name__", str(semantic))


def state_inventory(state: Any) -> list[dict[str, Any]]:
    """Return exact host-byte identities for every NNX state leaf.

    Args:
        state: NNX state with ``flat_state`` support.

    Returns:
        Stable leaf metadata and SHA-256 identities in path order.

    Raises:
        ValueError: A leaf has an object dtype.
    """
    import jax
    import numpy as np

    rows = []
    for path, variable in sorted(state.flat_state().items()):
        array = np.asarray(jax.device_get(variable.value))
        if array.dtype.hasobject:
            raise ValueError(f"state leaf has object dtype: {_path(path)}")
        data = array.tobytes(order="C")
        rows.append(
            {
                "path": "/".join(_path(path)),
                "variable_type": _variable_type(variable),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def validate_state_update(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    allowed_paths: Iterable[Path],
    *,
    require_change: bool,
) -> tuple[Path, ...]:
    """Require all changed bytes to remain inside an exact path allowlist.

    Args:
        before: Inventory captured before an update.
        after: Inventory captured after the update.
        allowed_paths: Complete state paths allowed to change.
        require_change: Whether at least one allowed leaf must change.

    Returns:
        Changed paths in stable order.

    Raises:
        ValueError: Metadata changed, a frozen leaf changed, or no change occurred.
    """
    previous = {row["path"]: row for row in before}
    current = {row["path"]: row for row in after}
    if set(previous) != set(current):
        raise ValueError("state paths changed during native update")
    allowed = {"/".join(path) for path in allowed_paths}
    changed = []
    for name in sorted(previous):
        left = previous[name]
        right = current[name]
        for field in ("variable_type", "shape", "dtype", "bytes"):
            if left[field] != right[field]:
                raise ValueError(f"state metadata changed at {name}")
        if left["sha256"] != right["sha256"]:
            if name not in allowed:
                raise ValueError(f"frozen state changed at {name}")
            changed.append(tuple(name.split("/")))
    if require_change and not changed:
        raise ValueError("native update did not change an allowlisted leaf")
    return tuple(changed)


def preserve_frozen_state(
    previous: Any, current: Any, allowed_paths: Iterable[Path]
) -> Any:
    """Restore all non-allowlisted values in an updated NNX state.

    Args:
        previous: State before the native update.
        current: State produced by the native update.
        allowed_paths: Complete paths permitted to retain new values.

    Returns:
        A state with current allowed leaves and previous frozen leaves.

    Raises:
        ValueError: State paths or variable metadata differ.
    """
    before = dict(previous.flat_state().items())
    after = dict(current.flat_state().items())
    if tuple(before) != tuple(after):
        raise ValueError("state paths changed during frozen-state restoration")
    allowed = frozenset(allowed_paths)
    for path, left in before.items():
        right = after[path]
        if (
            _variable_type(left) != _variable_type(right)
            or left.value.shape != right.value.shape
            or left.value.dtype != right.value.dtype
        ):
            raise ValueError(f"state metadata changed at {_path(path)}")
    return current.map(
        lambda path, variable: (
            variable
            if _path(path) in allowed
            else variable.replace(value=before[path].value)
        )
    )


def _trainer_contract(train_state, trainable_filter, allowed_paths, samples, weight):
    allowed = frozenset(allowed_paths)
    if not allowed:
        raise ValueError("native trainer requires a nonempty path allowlist")
    if samples <= 0:
        raise ValueError("num_flow_samples must be positive")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("anchor_weight must be finite and nonnegative")
    selected = train_state.params.filter(trainable_filter)
    paths = {_path(path) for path, _ in selected.flat_state().items()}
    if paths != allowed:
        raise ValueError("trainable filter and path allowlist differ")
    return allowed


def _trainer_values(
    train_state, trainable_filter, detailed_loss, samples, allowed_paths, weight
):
    from native_nnx_anchor import capture_parent_action_state

    allowed = _trainer_contract(
        train_state, trainable_filter, allowed_paths, samples, weight
    )
    parent = capture_parent_action_state(train_state, trainable_filter)
    return parent, trainable_filter, detailed_loss, samples, weight, allowed


def _require_finite_loss_gradients(loss, gradients):
    import jax
    import numpy as np

    leaves = jax.tree.leaves(gradients.to_pure_dict())
    if not np.isfinite(float(jax.device_get(loss))) or not all(
        np.isfinite(np.asarray(jax.device_get(leaf))).all() for leaf in leaves
    ):
        raise FloatingPointError("anchored loss or gradients are non-finite")


def _require_finite_params(updated_state):
    import jax
    import numpy as np
    from flax import nnx

    for path, variable in updated_state.filter(nnx.Param).flat_state().items():
        if not np.isfinite(np.asarray(jax.device_get(variable.value))).all():
            raise FloatingPointError(
                f"updated parameter is non-finite at {'/'.join(_path(path))}"
            )


@dataclass(frozen=True)
class NativeTrainer:
    """Hold the frozen parent and executable native anchored-loss contract."""

    parent_action_state: Any
    trainable_filter: Any
    detailed_loss: Callable[..., dict[str, Any]]
    num_flow_samples: int
    anchor_weight: float
    allowed_paths: frozenset[Path]

    @classmethod
    def initialize(
        cls,
        train_state: Any,
        trainable_filter: Any,
        detailed_loss: Callable[..., dict[str, Any]],
        *,
        num_flow_samples: int,
        allowed_paths: Iterable[Path],
        anchor_weight: float = 1.0,
    ) -> NativeTrainer:
        """Capture a frozen teacher after native state initialization.

        Args:
            train_state: Native state whose params contain canonical correlation.
            trainable_filter: Exact NNX action-parameter filter.
            detailed_loss: Native loss seam returning ``_flow_velocity``.
            num_flow_samples: Positive native flow-sample count.
            allowed_paths: Exact parameter paths selected by the filter.
            anchor_weight: Finite nonnegative anchor multiplier.
        Returns:
            Executable trainer wiring with independently stored parent state.
        Raises:
            ValueError: The filter, sample count, weight, or state contract differs.
        """
        values = _trainer_values(
            train_state,
            trainable_filter,
            detailed_loss,
            num_flow_samples,
            allowed_paths,
            anchor_weight,
        )
        return cls(*values)

    def loss_with_parent(
        self,
        model: Any,
        parent_action_state: Any,
        rng: Any,
        observation: Any,
        actions: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Evaluate the executable paired native objective."""
        import jax
        import jax.numpy as jnp
        from native_nnx_anchor import native_anchored_loss, paired_detailed_losses

        paired = paired_detailed_losses(
            model,
            parent_action_state,
            self.trainable_filter,
            rng,
            observation,
            actions,
            loss_method=self.detailed_loss,
            num_flow_samples=self.num_flow_samples,
        )
        objective, metrics = native_anchored_loss(
            paired,
            array_module=jnp,
            stop_gradient=jax.lax.stop_gradient,
            anchor_weight=self.anchor_weight,
        )
        return objective.total_loss, metrics

    def value_and_grad(
        self, model: Any, rng: Any, observation: Any, actions: Any
    ) -> tuple[tuple[Any, dict[str, Any]], Any]:
        """Differentiate only the exact selected action parameters."""
        from flax import nnx

        diff_state = nnx.DiffState(0, self.trainable_filter)

        def loss(selected, key, obs, target):
            return self.loss_with_parent(
                selected, self.parent_action_state, key, obs, target
            )

        compiled = nnx.jit(nnx.value_and_grad(loss, argnums=diff_state, has_aux=True))
        return compiled(model, rng, observation, actions)

    def parent_input_sharding(self, train_state_sharding: Any) -> Any:
        """Select and validate the frozen parent's exact input sharding.

        Args:
            train_state_sharding: Native train-state sharding with a ``params`` state.

        Returns:
            Sharding state for the independently passed parent action parameters.

        Raises:
            ValueError: The selected sharding paths differ from the action allowlist.
        """
        selected = train_state_sharding.params.filter(self.trainable_filter)
        paths = {_path(path) for path, _ in selected.flat_state().items()}
        if paths != self.allowed_paths:
            raise ValueError("parent input sharding and path allowlist differ")
        return selected

    def update(
        self, optimizer: Any, rng: Any, observation: Any, actions: Any
    ) -> tuple[Any, dict[str, Any], tuple[Path, ...]]:
        """Run one native optimizer update and validate its changed leaves.

        Args:
            optimizer: ``nnx.Optimizer`` configured with the same filter.
            rng, observation, actions: Native detailed-loss inputs.

        Returns:
            Total loss, metrics, and exact changed parameter paths.

        Raises:
            ValueError: Optimizer filtering differs or frozen state changes.
            FloatingPointError: Loss, gradients, or updated params are non-finite.
        """
        from flax import nnx

        if optimizer.wrt != self.trainable_filter:
            raise ValueError("optimizer and anchored trainer filters differ")
        before = state_inventory(nnx.state(optimizer.model))
        (loss, metrics), gradients = self.value_and_grad(
            optimizer.model, rng, observation, actions
        )
        _require_finite_loss_gradients(loss, gradients)
        optimizer.update(gradients)
        updated_state = nnx.state(optimizer.model)
        _require_finite_params(updated_state)
        after = state_inventory(updated_state)
        changed = validate_state_update(
            before, after, self.allowed_paths, require_change=True
        )
        return loss, metrics, changed
