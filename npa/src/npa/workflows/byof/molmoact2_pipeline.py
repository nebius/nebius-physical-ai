"""Execute the official MolmoAct2 LIBERO fine-tune and evaluation path.

This module deliberately does not extend the older ``workbench.molmoact``
planning stubs.  It is the run-time adapter for the separately pinned
MolmoAct2 BYOF image.  Every command moves a real input/output artifact and
then invokes either the upstream trainer, LeRobot evaluator, or upstream
open-loop evaluation script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


UPSTREAM_REPOSITORY = "https://github.com/allenai/molmoact2"
UPSTREAM_REVISION = "6070080a20321b4f498ab30f28e1d09ac465edb7"
LEROBOT_SOURCE_PATH = "experiments/lerobot"
LIBERO_DATASET = "allenai/MolmoAct2-LIBERO-Dataset"
LIBERO_DATASET_REVISION = "fe3ead447f44c0ea950396360b304cc2fb6be8f8"
BASE_CHECKPOINT = "allenai/MolmoAct2"
BASE_CHECKPOINT_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"


def _default_work_root() -> str:
    """Use a process-scoped directory on the selected temporary filesystem."""

    return str(Path(tempfile.gettempdir()) / f"npa-molmoact2-{os.getpid()}")


class MolmoAct2PipelineError(RuntimeError):
    """The exact upstream contract or a required artifact was unavailable."""


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _require_uri(value: str, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise MolmoAct2PipelineError(f"{name} is required")
    return value


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _download(uri: str, destination: Path) -> Path:
    if _is_s3(uri):
        _storage().download_path(uri, str(destination))
        return destination
    source = Path(uri)
    if not source.exists():
        raise MolmoAct2PipelineError(f"input does not exist: {uri}")
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination


def _upload(source: Path, uri: str) -> str:
    if _is_s3(uri):
        return _storage().upload_path(str(source), uri)
    target = Path(uri)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return str(target)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MolmoAct2PipelineError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise MolmoAct2PipelineError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upstream_root() -> Path:
    root = Path(os.environ.get("MOLMOACT2_UPSTREAM_ROOT", "/opt/molmoact2/experiments"))
    if not (root / "launch_scripts" / "train_lerobot.py").is_file():
        raise MolmoAct2PipelineError(
            f"pinned upstream trainer is missing under {root}; use the MolmoAct2 BYOF image"
        )
    return root


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, check=False)
    if result.returncode:
        raise MolmoAct2PipelineError(
            f"upstream command failed with exit {result.returncode}: {command[0]}"
        )


def _dataset_contract(root: Path) -> tuple[int, dict[str, Any]]:
    info = _read_json(root / "meta" / "info.json")
    total = info.get("total_episodes")
    if not isinstance(total, int) or total < 2:
        raise MolmoAct2PipelineError(
            "LIBERO dataset must contain at least two episodes"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise MolmoAct2PipelineError("LIBERO meta/info.json has no feature contract")
    required = {
        "action",
        "observation.state",
        "observation.images.image",
        "observation.images.wrist_image",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise MolmoAct2PipelineError(
            f"LIBERO dataset does not satisfy the official camera/state/action contract: {missing}"
        )
    return total, info


def _validate_heldout_fraction(heldout_fraction: float) -> None:
    if not 0.0 < heldout_fraction < 1.0:
        raise MolmoAct2PipelineError(
            "heldout_fraction must be strictly between 0 and 1"
        )


def _deterministic_split(
    total: int, revision: str, heldout_fraction: float
) -> tuple[list[int], list[int]]:
    _validate_heldout_fraction(heldout_fraction)
    ordered = sorted(
        range(total),
        key=lambda episode: hashlib.sha256(
            f"{LIBERO_DATASET}@{revision}:{episode}".encode()
        ).hexdigest(),
    )
    count = max(1, min(total - 1, round(total * heldout_fraction)))
    return sorted(ordered[count:]), sorted(ordered[:count])


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch the immutable official dataset and make a real episode split."""
    _validate_heldout_fraction(args.heldout_fraction)
    from huggingface_hub import snapshot_download

    work = Path(args.work_root).resolve() / "prepare"
    source = work / "source" / LIBERO_DATASET
    source.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=LIBERO_DATASET,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=str(source),
    )
    total, info = _dataset_contract(source)
    train, heldout = _deterministic_split(
        total, args.dataset_revision, args.heldout_fraction
    )
    split = {
        "schema": "npa.molmoact2.libero-episode-split.v1",
        "dataset": {"repo_id": LIBERO_DATASET, "revision": args.dataset_revision},
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "contracts": {
            "state_key": "observation.state",
            "action_key": "action",
            "camera_keys": [
                "observation.images.image",
                "observation.images.wrist_image",
            ],
            "normalize_gripper": False,
            "control_mode": "delta end-effector pose",
            "action_horizon": 10,
            "n_action_steps": 10,
        },
        "total_episodes": total,
        "train_episode_indices": train,
        "heldout_episode_indices": heldout,
        "metadata_sha256": _sha256(source / "meta" / "info.json"),
    }
    # Use upstream LeRobot's own reindexing/copy machinery.  The published
    # artifact contains train and heldout datasets only, rather than retaining
    # a second complete source copy beside two derived copies.
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    prepared = work / "prepared"
    split_dataset(
        LeRobotDataset(LIBERO_DATASET, root=source),
        {"train": train, "heldout": heldout},
        output_dir=prepared,
    )
    _write_json(prepared / "npa_episode_split.json", split)
    _upload(prepared, args.prepared_dataset_uri)
    manifest = {
        "schema": "npa.molmoact2.libero-prepare.v1",
        "prepared_dataset_uri": args.prepared_dataset_uri.rstrip("/") + "/",
        "split": split,
        "source_info": {
            "fps": info.get("fps"),
            "codebase_version": info.get("codebase_version"),
        },
    }
    report = work / "prepare.json"
    _write_json(report, manifest)
    _upload(report, args.output_uri)
    return manifest


def _download_prepared(
    prepared_uri: str, work: Path
) -> tuple[Path, Path, dict[str, Any]]:
    root = work / "prepared"
    _download(prepared_uri, root)
    train_root = root / "train"
    heldout_root = root / "heldout"
    _dataset_contract(train_root)
    _dataset_contract(heldout_root)
    split = _read_json(root / "npa_episode_split.json")
    if split.get("dataset", {}).get("revision") != LIBERO_DATASET_REVISION:
        raise MolmoAct2PipelineError(
            "prepared dataset revision is not the pinned official LIBERO revision"
        )
    return train_root, heldout_root, split


def _latest_inference_checkpoint(root: Path) -> Path:
    # The pinned upstream LeRobot evaluator documents ``step*-merged`` as the
    # inference artifact. Prefer it even when the trainer also leaves an
    # unsharded training export beside it.
    candidates = sorted(root.glob("step*-merged"), key=lambda path: path.name)
    if not candidates:
        candidates = sorted(root.glob("step*-unsharded"), key=lambda path: path.name)
    if not candidates:
        raise MolmoAct2PipelineError(
            "upstream trainer did not emit an inference checkpoint"
        )
    return candidates[-1]


def _download_base_checkpoint(work: Path) -> Path:
    """Materialize the exact foundation revision before native training.

    The upstream trainer accepts a local checkpoint directory but has no
    revision argument for a Hugging Face model id.  Supplying the bare model
    id would therefore make the manifest's immutable revision claim false.
    Deliberately leave ``token`` unspecified: the public LIBERO path works
    anonymously, while an operator-provided native Hub credential can still
    be forwarded by the standard runtime plumbing when it is available.
    """
    from huggingface_hub import snapshot_download

    checkpoint = work / "foundation-checkpoint"
    snapshot_download(
        repo_id=BASE_CHECKPOINT,
        repo_type="model",
        revision=BASE_CHECKPOINT_REVISION,
        local_dir=str(checkpoint),
    )
    if not (checkpoint / "config.json").is_file():
        raise MolmoAct2PipelineError(
            "pinned foundation checkpoint did not materialize config.json"
        )
    return checkpoint


def finetune(args: argparse.Namespace) -> dict[str, Any]:
    """Invoke the pinned upstream trainer, retaining its actual checkpoint bytes."""
    work = Path(args.work_root).resolve() / "finetune"
    data_root, _heldout_root, split = _download_prepared(
        args.prepared_dataset_uri, work
    )
    upstream = _upstream_root()
    output = work / "checkpoint"
    base_checkpoint = _download_base_checkpoint(work)
    local_lerobot_root = work / "lerobot-data-root" / "allenai"
    local_lerobot_root.mkdir(parents=True, exist_ok=True)
    local_dataset = local_lerobot_root / "MolmoAct2-LIBERO-Dataset"
    if local_dataset.is_symlink():
        # A runtime retry may preserve this work root. Reuse only the exact
        # prepared input; replacing an ordinary path could discard evidence.
        if local_dataset.resolve() != data_root.resolve():
            local_dataset.unlink()
    elif local_dataset.exists():
        raise MolmoAct2PipelineError(
            f"LeRobot dataset mount is not a symlink: {local_dataset}"
        )
    if not local_dataset.is_symlink():
        os.symlink(data_root, local_dataset)
    environment = dict(os.environ)
    environment.update(
        {
            "LEROBOT_DATA_ROOT": str(local_lerobot_root.parent),
            "HF_LEROBOT_HOME": str(local_lerobot_root.parent),
            "WANDB_MODE": "disabled",
            "WANDB_PROJECT": "npa-molmoact2",
        }
    )
    command = [
        "torchrun",
        "--standalone",
        "--nproc-per-node=1",
        "launch_scripts/train_lerobot.py",
        str(base_checkpoint),
        "libero",
        f"--save_folder={output}",
        # This is the upstream LoRA recipe: it causes the trainer's native
        # merged-checkpoint path to produce a complete inference artifact for
        # the downstream closed-loop evaluator.
        "--packing=false",
        "--dynamic_seq_len=true",
        "--ft_vlm=true",
        "--ft_action_expert=true",
        "--ft_embedding=lm_head",
        "--lora_enable=true",
        "--lora_rank=64",
    ]
    _run(command, cwd=upstream, env=environment)
    merged = _latest_inference_checkpoint(output)
    checkpoint_manifest = {
        "schema": "npa.molmoact2.libero-checkpoint.v1",
        "base_checkpoint": {
            "repo_id": BASE_CHECKPOINT,
            "revision": BASE_CHECKPOINT_REVISION,
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "lerobot_source_path": LEROBOT_SOURCE_PATH,
        },
        "prepared_dataset_uri": args.prepared_dataset_uri,
        "train_episode_indices": split["train_episode_indices"],
        "heldout_episode_indices": split["heldout_episode_indices"],
        "training_command": command,
        "inference_checkpoint_relative_path": str(merged.relative_to(output)),
        "telemetry": "disabled",
    }
    _write_json(output / "npa_checkpoint_manifest.json", checkpoint_manifest)
    _upload(output, args.checkpoint_uri)
    return checkpoint_manifest


def _download_checkpoint(
    checkpoint_uri: str, work: Path
) -> tuple[Path, dict[str, Any]]:
    checkpoint = work / "checkpoint"
    _download(checkpoint_uri, checkpoint)
    manifest = _read_json(checkpoint / "npa_checkpoint_manifest.json")
    merged = checkpoint / str(manifest.get("inference_checkpoint_relative_path", ""))
    if not merged.is_dir():
        raise MolmoAct2PipelineError(
            "checkpoint artifact has no merged MolmoAct2 inference checkpoint"
        )
    return merged, manifest


def rollout(args: argparse.Namespace) -> dict[str, Any]:
    """Run genuine closed-loop LIBERO episodes through upstream LeRobot."""
    work = Path(args.work_root).resolve() / "rollout"
    checkpoint, checkpoint_manifest = _download_checkpoint(args.checkpoint_uri, work)
    upstream = _upstream_root()
    output = work / "rollouts"
    environment = dict(os.environ)
    environment["WANDB_MODE"] = "disabled"
    command = [
        "lerobot-eval",
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={checkpoint}",
        "--policy.device=cuda",
        "--policy.seq_len=2100",
        "--policy.norm_tag=libero",
        "--policy.inference_action_mode=continuous",
        "--policy.enable_inference_cuda_graph=true",
        "--env.type=libero",
        f"--eval.n_episodes={args.n_episodes}",
        "--eval.batch_size=1",
        f"--output_dir={output}",
    ]
    _run(command, cwd=upstream, env=environment)
    json_files = sorted(output.rglob("*.json"))
    videos = sorted(output.rglob("*.mp4"))
    if not json_files or not videos:
        raise MolmoAct2PipelineError(
            "closed-loop evaluator did not produce JSON metrics and MP4 evidence"
        )
    report = {
        "schema": "npa.molmoact2.libero-rollout.v1",
        "checkpoint_uri": args.checkpoint_uri,
        "checkpoint_manifest_sha256": _sha256(
            work / "checkpoint" / "npa_checkpoint_manifest.json"
        ),
        "evaluation_command": command,
        "json_reports": [str(path.relative_to(output)) for path in json_files],
        "mp4_reports": [str(path.relative_to(output)) for path in videos],
    }
    _write_json(output / "npa_rollout_manifest.json", report)
    _upload(output, args.rollout_uri)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Compute held-out action error through the upstream MolmoAct2 inference path."""
    work = Path(args.work_root).resolve() / "evaluate"
    _train_root, heldout_root, split = _download_prepared(
        args.prepared_dataset_uri, work
    )
    checkpoint, checkpoint_manifest = _download_checkpoint(args.checkpoint_uri, work)
    rollout_root = work / "rollout"
    _download(args.rollout_uri, rollout_root)
    rollout = _read_json(rollout_root / "npa_rollout_manifest.json")
    heldout = split.get("heldout_episode_indices")
    if not isinstance(heldout, list) or not heldout:
        raise MolmoAct2PipelineError(
            "prepared split has no held-out episode for action evaluation"
        )
    upstream = _upstream_root()
    output = work / "heldout-action"
    command = [
        "python",
        "scripts/run_open_loop_inference_lerobot.py",
        "--dataset",
        LIBERO_DATASET,
        f"--dataset_root={heldout_root}",
        "--episode_idx=0",
        f"--checkpoint={checkpoint}",
        f"--output_dir={output}",
        "--device=cuda",
        "--norm_tag=libero",
        "--inference_action_mode=continuous",
        "--num_steps=10",
    ]
    _run(command, cwd=upstream, env=dict(os.environ))
    action = _read_json(output / "summary.json")
    if action.get("average_mse_raw") is None:
        raise MolmoAct2PipelineError(
            "upstream held-out evaluator emitted no raw action MSE"
        )
    result = {
        "schema": "npa.molmoact2.libero-evaluation.v1",
        "checkpoint_uri": args.checkpoint_uri,
        "prepared_dataset_uri": args.prepared_dataset_uri,
        "rollout_uri": args.rollout_uri,
        "heldout_source_episode_index": heldout[0],
        "heldout_reindexed_episode_index": 0,
        "action_metrics": {
            "average_mse_raw": action["average_mse_raw"],
            "average_mse_raw_nongripper": action.get("average_mse_raw_nongripper"),
            "average_mse_raw_gripper": action.get("average_mse_raw_gripper"),
            "normalization_mode": action.get("normalization_mode"),
            "normalization_mask": action.get("normalization_mask"),
        },
        "closed_loop_artifacts": rollout,
        "checkpoint_manifest": checkpoint_manifest,
        "action_evaluation_command": command,
    }
    _write_json(output / "evaluation.json", result)
    _upload(output, args.evaluation_uri)
    return result


def visualize(args: argparse.Namespace) -> dict[str, Any]:
    """Emit a factual Rerun recording and preserve evaluator-produced video bytes."""
    work = Path(args.work_root).resolve() / "visualize"
    evaluation_root = work / "evaluation"
    rollout_root = work / "rollout"
    _download(args.evaluation_uri, evaluation_root)
    _download(args.rollout_uri, rollout_root)
    evaluation = _read_json(evaluation_root / "evaluation.json")
    videos = sorted(rollout_root.rglob("*.mp4"))
    if not videos:
        raise MolmoAct2PipelineError(
            "rollout artifact has no evaluator-produced MP4 to preserve"
        )
    try:
        import rerun as rr
    except ImportError as exc:
        raise MolmoAct2PipelineError(
            "rerun is required to emit the factual recording"
        ) from exc
    artifact = work / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    rrd = artifact / "libero-evaluation.rrd"
    rr.init(
        "npa-molmoact2-libero",
        recording_id="npa-molmoact2-libero",
        default_enabled=True,
    )
    rr.save(str(rrd))
    rr.log(
        "provenance/upstream_revision",
        rr.TextDocument(UPSTREAM_REVISION, media_type="text/plain"),
    )
    rr.log(
        "metrics/heldout_action",
        rr.Scalars(
            {
                "mse_raw": float(evaluation["action_metrics"]["average_mse_raw"]),
            }
        ),
    )
    rr.log("artifacts/rollout_video", rr.AssetVideo(path=str(videos[0])))
    rr.disconnect()
    video = artifact / videos[0].name
    shutil.copy2(videos[0], video)
    result = {
        "schema": "npa.molmoact2.libero-visualization.v1",
        "rrd": rrd.name,
        "mp4": video.name,
        "evaluation_sha256": _sha256(evaluation_root / "evaluation.json"),
        "provenance": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
        },
    }
    _write_json(artifact / "visualization.json", result)
    _upload(artifact, args.artifact_uri)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default=_default_work_root())
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--prepared-dataset-uri", required=True)
    prepare_parser.add_argument("--output-uri", required=True)
    prepare_parser.add_argument("--dataset-revision", default=LIBERO_DATASET_REVISION)
    prepare_parser.add_argument("--heldout-fraction", type=float, default=0.1)
    finetune_parser = sub.add_parser("finetune")
    finetune_parser.add_argument("--prepared-dataset-uri", required=True)
    finetune_parser.add_argument("--checkpoint-uri", required=True)
    rollout_parser = sub.add_parser("rollout")
    rollout_parser.add_argument("--checkpoint-uri", required=True)
    rollout_parser.add_argument("--rollout-uri", required=True)
    rollout_parser.add_argument("--n-episodes", type=int, default=10)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--prepared-dataset-uri", required=True)
    evaluate_parser.add_argument("--checkpoint-uri", required=True)
    evaluate_parser.add_argument("--rollout-uri", required=True)
    evaluate_parser.add_argument("--evaluation-uri", required=True)
    visualize_parser = sub.add_parser("visualize")
    visualize_parser.add_argument("--rollout-uri", required=True)
    visualize_parser.add_argument("--evaluation-uri", required=True)
    visualize_parser.add_argument("--artifact-uri", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "prepare": prepare,
        "finetune": finetune,
        "rollout": rollout,
        "evaluate": evaluate,
        "visualize": visualize,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
