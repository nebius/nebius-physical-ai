"""Execute native LIBERO-10 action-policy SFT and publish a verified DCP bundle."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from npa.workbench.cosmos.policy_artifacts import (
    file_digest,
    policy_workspace,
    publish_bundle,
    write_local_json,
)
from npa.workbench.cosmos.policy_contract import (
    ACTION_CONTRACT,
    DATASET_REVISION,
    EXPERIMENT,
    FRAMEWORK_REVISION,
    MODEL_REVISION,
    RECIPE,
    STATS,
    VAE_REVISION,
    TrainSettings,
)
from npa.workbench.cosmos.policy_runtime import prepare_training_runtime, run_native
from npa.workbench.dataset.storage import read_json_uri

TRAIN_SCHEMA = "npa.cosmos3.policy-train.v1"


def training_argv(repo: Path, settings: TrainSettings) -> list[str]:
    """Build native torchrun arguments without changing the action/data contract.

    Args:
        repo: Pinned framework checkout.
        settings: Explicit training hyperparameters and single-node topology.
    Returns:
        Executable argv for native FSDP training.
    Raises:
        ValueError: Requested checkpoint cadence cannot capture the final step.
    """
    cadence = min(settings.save_every, settings.iterations)
    if settings.iterations % cadence:
        raise ValueError("iterations must be divisible by save_every")
    return [
        str(repo / ".venv/bin/python"),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={settings.processes}",
        "-m",
        "cosmos_framework.scripts.train",
        "--sft-toml",
        str(repo / RECIPE),
        "--",
        "job.wandb_mode=disabled",
        "job.name=npa_libero_policy",
        f"trainer.max_iter={settings.iterations}",
        f"trainer.seed={settings.seed}",
        f"checkpoint.save_iter={cadence}",
        f"trainer.grad_accum_iter={settings.gradient_accumulation}",
        f"dataloader_train.max_samples_per_batch={settings.samples_per_rank}",
        f"model.config.parallelism.data_parallel_shard_degree={settings.processes}",
        "model.config.parallelism.data_parallel_replicate_degree=1",
    ]


def _fetch_inputs(
    root: Path, repo: Path, env: dict[str, str]
) -> tuple[Path, Path, Path]:
    from npa.workbench.cosmos.generate import resolve_hf_token

    token = resolve_hf_token()
    fetch_env = {**env, **({"HF_TOKEN": token} if token else {})}
    paths_file = root / "input-paths.json"
    run_native(
        [
            str(repo / ".venv/bin/python"),
            str(Path(__file__).with_name("policy_inputs.py")),
            "--dataset-revision",
            DATASET_REVISION,
            "--model-revision",
            MODEL_REVISION,
            "--vae-revision",
            VAE_REVISION,
            "--output-path",
            str(paths_file),
        ],
        cwd=repo,
        env=fetch_env,
        log=root / "artifacts/input-fetch.log",
    )
    paths = json.loads(paths_file.read_text())
    dataset, model, vae = (Path(paths[key]) for key in ("dataset", "model", "vae"))
    info = json.loads((dataset / "meta/info.json").read_text())
    if info.get("fps") != 20 or not list(dataset.glob("data/chunk-*/file-*.parquet")):
        raise ValueError("expected the native 20 Hz LIBERO-10 action dataset")
    write_local_json(
        root / "artifacts/dataset.json",
        {
            "repository": "nvidia/LIBERO_LeRobot_v3",
            "revision": DATASET_REVISION,
            "info_sha256": file_digest(dataset / "meta/info.json"),
        },
    )
    return dataset, model, vae


def _collect_checkpoint(output: Path, artifacts: Path, settings: TrainSettings) -> str:
    matches = list(output.rglob(f"checkpoints/iter_{settings.iterations:09d}"))
    if len(matches) != 1:
        raise ValueError(
            "native training did not publish the requested final checkpoint"
        )
    checkpoint = matches[0]
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        if not (directory / ".metadata").is_file() or not list(
            directory.glob("*.distcp")
        ):
            raise ValueError(f"incomplete native checkpoint component: {component}")
    # Keep the native .../checkpoints/iter_<step> layout required by its loader.
    relative = f"job/checkpoints/{checkpoint.name}"
    config = checkpoint.parent.parent / "config.yaml"
    if not config.is_file():
        raise ValueError("native training omitted the resolved configuration")
    # Both directories belong to this temporary workspace. Rename the completed
    # checkpoint instead of duplicating hundreds of GB on the worker's disk.
    target = artifacts / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.rename(target)
    shutil.copyfile(config, artifacts / "job/config.yaml")
    return relative


def _verify_processor_revision(model: Path) -> None:
    # The pinned native converter resolves its processor through a registry
    # entry using main. Refuse silent metadata drift from the pinned weights.
    reference = model.parent.parent / "refs/main"
    if not reference.is_file() or reference.read_text().strip() != MODEL_REVISION:
        raise ValueError(
            "native processor registry did not resolve the pinned model revision"
        )


def train_policy(*, input_path: str, output_path: str) -> dict[str, Any]:
    """Train actual Cosmos action heads on pinned LIBERO data and retain state.

    Args:
        input_path: Optional local/S3 TrainSettings JSON; empty uses native-scale settings.
        output_path: Private local/S3 destination for weights, logs, and completion manifest.
    Returns:
        Hash-bound checkpoint manifest after successful training and publication.
    Raises:
        ValueError: Invalid settings/data or incomplete checkpoint evidence.
        subprocess.CalledProcessError: Native setup, conversion, or training failed.
    """
    settings = TrainSettings.model_validate(
        read_json_uri(input_path) if input_path else {}
    )
    with policy_workspace(output_path, "train") as root:
        artifacts = root / "artifacts"
        artifacts.mkdir()
        repo, env = prepare_training_runtime(root)
        for log in root.glob("*.log"):
            shutil.copyfile(log, artifacts / log.name)
        dataset, model, vae = _fetch_inputs(root, repo, env)
        env.update(
            LIBERO_ROOT=str(dataset),
            WAN_VAE_PATH=str(vae),
            BASE_CHECKPOINT_PATH=str(root / "base-dcp"),
            IMAGINAIRE_OUTPUT_ROOT=str(root / "output"),
        )
        run_native(
            [
                str(repo / ".venv/bin/python"),
                "-m",
                "cosmos_framework.scripts.convert_model_to_dcp",
                "--checkpoint-path",
                str(model),
                "-o",
                env["BASE_CHECKPOINT_PATH"],
            ],
            cwd=repo,
            env=env,
            log=artifacts / "conversion.log",
        )
        _verify_processor_revision(model)
        elapsed = _run_training(repo, settings, env, artifacts)
        checkpoint = _collect_checkpoint(root / "output", artifacts, settings)
        shutil.copyfile(vae, artifacts / "Wan2.2_VAE.pth")
        shutil.copyfile(repo / STATS, artifacts / "action_stats.json")
        write_local_json(artifacts / "settings.json", settings.model_dump())
        report = _training_report(settings, checkpoint, elapsed, artifacts)
        return publish_bundle(artifacts, output_path, report, "training.json")


def _run_training(
    repo: Path, settings: TrainSettings, env: dict[str, str], artifacts: Path
) -> float:
    argv = training_argv(repo, settings)
    native_options = argv[argv.index("--sft-toml") :]
    separator = native_options.index("--")
    dryrun = [
        str(repo / ".venv/bin/python"),
        "-m",
        "cosmos_framework.scripts.train",
        *native_options[:separator],
        "--dryrun",
        *native_options[separator:],
    ]
    run_native(dryrun, cwd=repo, env=env, log=artifacts / "configuration.log")
    started = time.monotonic()
    run_native(argv, cwd=repo, env=env, log=artifacts / "training.log")
    return time.monotonic() - started


def _training_report(
    settings: TrainSettings, checkpoint: str, elapsed: float, artifacts: Path
) -> dict[str, Any]:
    return {
        "schema": TRAIN_SCHEMA,
        "status": "succeeded",
        "checkpoint": checkpoint,
        "framework_revision": FRAMEWORK_REVISION,
        "experiment": EXPERIMENT,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "processor_revision": MODEL_REVISION,
        "vae_revision": VAE_REVISION,
        "action_contract": ACTION_CONTRACT,
        "settings": settings.model_dump(),
        "training_seconds": elapsed,
        "training_gpu_seconds": settings.processes * elapsed,
        "stats_sha256": file_digest(artifacts / "action_stats.json"),
        "model_quality": "not_evaluated",
        "optimizer_state_retained": True,
    }
