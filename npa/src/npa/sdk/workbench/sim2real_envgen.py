"""SDK helpers for Sim2Real env generation and action conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    SceneSpec,
    build_policy_image_contract,
    build_scene_spec,
    write_action_conditioned_envs,
    write_raw_shard,
    write_split_manifest,
)


def scene_spec(**overrides: Any) -> SceneSpec:
    return build_scene_spec(**overrides)


def raw_shard(config: EnvGenConfig, output_dir: str | Path) -> dict[str, Any]:
    return write_raw_shard(config, Path(output_dir))


def split(config: EnvGenConfig, output_dir: str | Path) -> dict[str, Any]:
    return write_split_manifest(config, Path(output_dir))


def actions(
    config: EnvGenConfig,
    output_dir: str | Path,
    *,
    policy_image: str,
    limit: int = 256,
    train_envs_uri: str = "",
    actions_uri: str = "",
) -> dict[str, Any]:
    return write_action_conditioned_envs(
        config,
        Path(output_dir),
        policy_image=policy_image,
        limit=limit,
        train_envs_uri=train_envs_uri,
        actions_uri=actions_uri,
    )


def policy_image_contract(*, train_envs_uri: str, actions_uri: str, default_policy_image: str) -> dict[str, Any]:
    return build_policy_image_contract(
        train_envs_uri=train_envs_uri,
        output_uri=actions_uri,
        default_policy_image=default_policy_image,
    )


__all__ = [
    "EnvGenConfig",
    "SceneSpec",
    "actions",
    "policy_image_contract",
    "raw_shard",
    "scene_spec",
    "split",
]
