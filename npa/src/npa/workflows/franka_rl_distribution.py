"""Bound learned Gaussian exploration while retaining native RSL-RL probability semantics."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from torch import nn
from torch.distributions import Normal


class BoundedGaussianDistribution(GaussianDistribution):
    """Learn state-independent scales through a smooth bounded parameterization.

    Sampling, log probability, entropy, and KL use the same native Normal.
    Only the scale is bounded; the actor still chooses every action mean.

    Args:
        output_dim: Number of policy action channels.
        init_std: Initial actual action standard deviation, strictly inside the bounds.
        min_std: Positive lower bound on action standard deviation.
        max_std: Finite upper bound on action standard deviation.
    Returns:
        None.
    Raises:
        ValueError: Dimensions or scale bounds are invalid.
    """

    def __init__(self, output_dim: int, init_std: float = 1.0,
                 min_std: float = 0.05, max_std: float = 1.5) -> None:
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("Gaussian output dimension must be a positive integer")
        if not all(math.isfinite(value) for value in (init_std, min_std, max_std)):
            raise ValueError("Gaussian scale bounds and initial scale must be finite")
        if not 0 < min_std < init_std < max_std:
            raise ValueError("Gaussian scales must satisfy 0 < min_std < init_std < max_std")
        # The native heteroscedastic subclass also initializes the base directly
        # to avoid registering an unused Gaussian scale parameter.
        Distribution.__init__(self, output_dim)
        self.register_buffer("std_bounds", torch.tensor([min_std, max_std]))
        raw = math.log((init_std - min_std) / (max_std - init_std))
        self.raw_std = nn.Parameter(torch.full((output_dim,), raw))
        self._distribution: Normal | None = None

    @property
    def learned_std(self) -> torch.Tensor:
        """Return the current actual scale without requiring a policy forward pass.

        Args:
            None.
        Returns:
            One differentiable standard deviation per action, within stored bounds.
        Raises:
            None.
        """
        return torch.lerp(self.std_bounds[0], self.std_bounds[1], self.raw_std.sigmoid())

    def update(self, mlp_output: torch.Tensor) -> None:
        """Construct the Normal used by every native stochastic policy operation.

        Args:
            mlp_output: Unrestricted actor means with actions on the final axis.
        Returns:
            None.
        Raises:
            RuntimeError: Actor output shape is incompatible with the action scales.
        """
        self._distribution = Normal(mlp_output, self.learned_std.expand_as(mlp_output), validate_args=False)
