"""Sim2Real env generation and action-conditioning commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    build_policy_image_contract,
    build_scene_spec,
    write_action_conditioned_envs,
    write_raw_shard,
    write_split_manifest,
)

app = typer.Typer(
    name="sim2real-envgen",
    help="Generate Sim2Real raw envs, split manifests, and action-conditioned train envs.",
    no_args_is_help=True,
)


@app.command("raw-shard")
def raw_shard_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    output_uri: str = typer.Option(..., "--output-uri"),
    env_count: int = typer.Option(10_000, "--env-count"),
    shard_index: int = typer.Option(0, "--shard-index"),
    shard_count: int = typer.Option(1, "--shard-count"),
    seed: int = typer.Option(42, "--seed"),
    byo_mesh_uri: str = typer.Option("", "--byo-mesh-uri"),
    augmented_frames_uri: str = typer.Option("", "--augmented-frames-uri"),
    output_dir: Path = typer.Option(Path("/tmp/npa-sim2real-envgen"), "--output-dir"),
) -> None:
    """Generate and upload one raw env shard."""

    config = _config(
        run_id=run_id,
        output_uri=output_uri,
        env_count=env_count,
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
        byo_mesh_uri=byo_mesh_uri,
        augmented_frames_uri=augmented_frames_uri,
    )
    typer.echo(json.dumps(write_raw_shard(config, output_dir), indent=2, sort_keys=True))


@app.command("split")
def split_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    output_uri: str = typer.Option(..., "--output-uri"),
    env_count: int = typer.Option(10_000, "--env-count"),
    train_fraction: float = typer.Option(0.8, "--train-fraction"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: Path = typer.Option(Path("/tmp/npa-sim2real-envgen"), "--output-dir"),
) -> None:
    """Generate and upload deterministic disjoint train/heldout split manifests."""

    config = _config(
        run_id=run_id,
        output_uri=output_uri,
        env_count=env_count,
        train_fraction=train_fraction,
        seed=seed,
    )
    typer.echo(json.dumps(write_split_manifest(config, output_dir), indent=2, sort_keys=True))


@app.command("actions")
def actions_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    output_uri: str = typer.Option(..., "--output-uri"),
    policy_image: str = typer.Option(..., "--policy-image"),
    train_envs_uri: str = typer.Option("", "--train-envs-uri"),
    actions_uri: str = typer.Option("", "--actions-uri"),
    env_count: int = typer.Option(10_000, "--env-count"),
    limit: int = typer.Option(256, "--limit"),
    seed: int = typer.Option(42, "--seed"),
    output_dir: Path = typer.Option(Path("/tmp/npa-sim2real-envgen-actions"), "--output-dir"),
) -> None:
    """Generate and upload action-conditioned train envs for a representative slice."""

    config = _config(run_id=run_id, output_uri=output_uri, env_count=env_count, seed=seed)
    typer.echo(
        json.dumps(
            write_action_conditioned_envs(
                config,
                output_dir,
                policy_image=policy_image,
                limit=limit,
                train_envs_uri=train_envs_uri,
                actions_uri=actions_uri,
            ),
            indent=2,
            sort_keys=True,
        )
    )


@app.command("policy-contract")
def policy_contract_cmd(
    train_envs_uri: str = typer.Option(..., "--train-envs-uri"),
    actions_uri: str = typer.Option(..., "--actions-uri"),
    policy_image: str = typer.Option(..., "--policy-image"),
) -> None:
    """Print the BYO policy image contract."""

    typer.echo(
        json.dumps(
            build_policy_image_contract(
                train_envs_uri=train_envs_uri,
                output_uri=actions_uri,
                default_policy_image=policy_image,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def _config(
    *,
    run_id: str,
    output_uri: str,
    env_count: int,
    train_fraction: float = 0.8,
    seed: int,
    shard_index: int = 0,
    shard_count: int = 1,
    byo_mesh_uri: str = "",
    augmented_frames_uri: str = "",
) -> EnvGenConfig:
    return EnvGenConfig(
        run_id=run_id,
        output_uri=output_uri,
        env_count=env_count,
        train_fraction=train_fraction,
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
        scene_spec=build_scene_spec(
            byo_mesh_uri=byo_mesh_uri,
            augmented_frames_uri=augmented_frames_uri,
        ),
    )
