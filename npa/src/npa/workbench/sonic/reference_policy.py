"""Reference locomotion actor used by the in-job SONIC train runtime.

This module imports ``torch`` at import time because the policy has to be a real
``torch.nn.Module`` whose class is importable when ``sonic export`` unpickles the
checkpoint. Import it lazily (inside a function) from anywhere that must stay
torch-free.

The layout mirrors the Isaac Lab locomotion observation SONIC trains against for
a 23-DoF Unitree G1: base angular velocity, projected gravity, the velocity
command, a gait phase, and the proprioceptive joint state plus the previous
action. Actions are joint position targets expressed as offsets from the
default pose.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

#: Per-group widths of the observation vector, in order.
OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("gait_phase", 2),
    ("joint_pos", 23),
    ("joint_vel", 23),
    ("last_action", 23),
)
DEFAULT_ACTION_DIM = 23
DEFAULT_OBS_DIM = sum(width for _, width in OBS_LAYOUT)


def obs_field_spec() -> list[dict[str, Any]]:
    """Observation spec fields in the layout ``sonic export`` records."""

    return [{"name": name, "dim": width} for name, width in OBS_LAYOUT]


class ReferenceLocomotionPolicy(nn.Module):
    """Deterministic MLP actor over normalized observations.

    ``forward`` consumes *normalized* observations. The normalization statistics
    live on ``.normalization`` so ``sonic export`` can bake them into the ONNX
    graph (``--normalize baked``) and the exported policy can be fed raw
    observations.
    """

    def __init__(
        self,
        observation_dim: int = DEFAULT_OBS_DIM,
        action_dim: int = DEFAULT_ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        layers: list[nn.Module] = []
        width = self.observation_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(width, int(size)))
            layers.append(nn.ELU())
            width = int(size)
        layers.append(nn.Linear(width, self.action_dim))
        self.net = nn.Sequential(*layers)
        # Read by `sonic export` to build the metadata sidecar `sonic eval`
        # consumes: normalization is baked into the ONNX graph, and the specs
        # describe the layout above.
        self.normalization: dict[str, Any] = {}
        self.obs_spec: dict[str, Any] = {"name": "obs", "fields": obs_field_spec()}
        self.action_spec: dict[str, Any] = {"name": "action", "dim": self.action_dim}
        self.control_dt = 0.02

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def gait_constants(
    action_dim: int = DEFAULT_ACTION_DIM,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-joint gait amplitude and phase offsets.

    Fixed (not random) so the teacher signal is reproducible across runs and the
    training loss curve is comparable between checkpoints.
    """

    index = torch.arange(action_dim, dtype=torch.float32, device=device)
    amplitude = 0.15 + 0.1 * torch.cos(index * 0.37)
    phase_offset = (index * (2.0 * math.pi / max(action_dim, 1))) % (2.0 * math.pi)
    return amplitude, phase_offset


def reference_action(obs: torch.Tensor, *, action_dim: int = DEFAULT_ACTION_DIM) -> torch.Tensor:
    """Teacher joint targets for a batch of raw observations.

    A command-conditioned central pattern generator with per-joint damping: the
    gait amplitude scales with the commanded speed, the phase comes from the
    observation's ``gait_phase`` sin/cos pair, and joint velocity is damped.
    This is the signal the actor is fitted to.
    """

    offsets = {}
    cursor = 0
    for name, width in OBS_LAYOUT:
        offsets[name] = (cursor, cursor + width)
        cursor += width

    def _slice(name: str) -> torch.Tensor:
        start, end = offsets[name]
        return obs[..., start:end]

    commands = _slice("velocity_commands")
    phase = _slice("gait_phase")
    joint_pos = _slice("joint_pos")[..., :action_dim]
    joint_vel = _slice("joint_vel")[..., :action_dim]

    amplitude, phase_offset = gait_constants(action_dim, device=obs.device)
    speed = torch.linalg.norm(commands, dim=-1, keepdim=True)
    angle = torch.atan2(phase[..., 0:1], phase[..., 1:2])
    swing = torch.sin(angle + phase_offset) * amplitude * torch.clamp(speed, 0.0, 2.0)
    return swing - 0.05 * joint_vel - 0.1 * joint_pos


def sample_observations(
    batch: int,
    *,
    observation_dim: int = DEFAULT_OBS_DIM,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample plausible raw locomotion observations.

    Each observation group is drawn from its own physically-shaped range (a
    gravity vector near -1 on z, bounded velocity commands, a phase on the unit
    circle) instead of one undifferentiated Gaussian, so the normalization stats
    recorded on the checkpoint describe a realistic input distribution.
    """

    def _randn(width: int) -> torch.Tensor:
        return torch.randn(batch, width, generator=generator, device=device)

    def _rand(width: int) -> torch.Tensor:
        return torch.rand(batch, width, generator=generator, device=device)

    gravity = torch.zeros(batch, 3, device=device)
    gravity[:, 2] = -1.0
    angle = _rand(1) * (2.0 * math.pi)
    groups = {
        "base_ang_vel": 0.4 * _randn(3),
        "projected_gravity": gravity + 0.05 * _randn(3),
        "velocity_commands": torch.cat(
            [1.5 * (_rand(1) - 0.2), 0.4 * (_rand(1) - 0.5), 0.6 * (_rand(1) - 0.5)], dim=-1
        ),
        "gait_phase": torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1),
        "joint_pos": 0.3 * _randn(23),
        "joint_vel": 0.8 * _randn(23),
        "last_action": 0.2 * _randn(23),
    }
    obs = torch.cat([groups[name] for name, _ in OBS_LAYOUT], dim=-1)
    if obs.shape[-1] == observation_dim:
        return obs
    if obs.shape[-1] > observation_dim:
        return obs[..., :observation_dim]
    pad = torch.zeros(batch, observation_dim - obs.shape[-1], device=device)
    return torch.cat([obs, pad], dim=-1)
