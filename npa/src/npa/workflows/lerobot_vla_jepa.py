"""Run pinned native LeRobot VLA-JEPA stages with sealed artifact handoffs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from npa.workbench.lerobot.policy_container import validate_lerobot_checkpoint
from npa.workflows.lerobot_transfer_data import (
    file_sha256,
    materialize,
    publish,
    tree_hashes,
    write_json,
)

LEROBOT_VERSION = "0.6.1"
LEROBOT_SOURCE_REVISION = "7e241bd630a3719a56157a497ce5d08f244784f1"
PRETRAIN_REPO = "lerobot/VLA-JEPA-Pretrain"
PRETRAIN_REVISION = "e946c3e5b538d760f4b4ff239d1b1c12090c041d"
LIBERO_REPO = "HuggingFaceVLA/libero"
LIBERO_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"
QWEN_REPO = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
VJEPA_REPO = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


def _ids(value: str) -> list[int]:
    """Parse sorted, unique non-negative task identifiers from JSON."""
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("task ids must be a non-empty JSON list")
    if any(type(item) is not int or item < 0 for item in parsed):
        raise ValueError("task ids must contain non-negative integer values")
    if len(set(parsed)) != len(parsed):
        raise ValueError("task ids must be unique")
    return sorted(parsed)


def _require_fraction(value: float) -> float:
    """Require a meaningful strict training fraction."""
    if not 0 < value < 1:
        raise ValueError("train fraction must be strictly between zero and one")
    return value


def _snapshot(repo: str, revision: str, repo_type: str, cache_root: Path) -> Path:
    """Fetch one immutable Hub snapshot and atomically mark it ready."""
    from huggingface_hub import snapshot_download

    cache_root.mkdir(parents=True, exist_ok=True)
    path = Path(
        snapshot_download(
            repo_id=repo,
            repo_type=repo_type,
            revision=revision,
            cache_dir=str(cache_root),
        )
    )
    marker = (
        cache_root
        / "npa-vla-jepa-ready"
        / f"{repo.replace('/', '--')}--{revision}.json"
    )
    temporary = marker.with_suffix(".tmp")
    write_json(
        temporary,
        {"repo": repo, "repo_type": repo_type, "revision": revision},
    )
    temporary.replace(marker)
    return path


def _runtime_cache() -> Path:
    """Choose the operator cache without embedding fetched bytes in an image."""
    return Path(os.environ.get("HF_HOME") or "/tmp/npa-vla-jepa-hf-cache")


def _dataset_rows(dataset: Path, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Read and validate the numeric LIBERO frame contract."""
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 10:
        raise ValueError("VLA-JEPA requires LeRobot v3 LIBERO data at 10 Hz")
    features = info.get("features") or {}
    required = {
        "action": [7],
        "observation.state": [8],
        "observation.image": [3, 256, 256],
        "observation.image2": [3, 256, 256],
    }
    for name, shape in required.items():
        if (features.get(name) or {}).get("shape") != shape:
            raise ValueError(f"LIBERO VLA-JEPA feature mismatch for {name}")
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "action",
        "observation.state",
    ]
    rows = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    if len(rows) != info.get("total_frames"):
        raise ValueError("LIBERO frame count differs from pinned metadata")
    return rows


def _episodes_by_task(rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Return complete episode ids keyed by their single task id."""
    tasks: dict[int, set[int]] = {}
    for row in rows:
        episode = int(row["episode_index"])
        tasks.setdefault(episode, set()).add(int(row["task_index"]))
    if any(len(values) != 1 for values in tasks.values()):
        raise ValueError("each LIBERO episode must have exactly one task id")
    result: dict[int, list[int]] = {}
    for episode, values in tasks.items():
        result.setdefault(next(iter(values)), []).append(episode)
    return {task: sorted(episodes) for task, episodes in result.items()}


def _selected_episodes(
    by_task: dict[int, list[int]], heldout_tasks: list[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split trainable task episodes deterministically while sealing held-out tasks."""
    unknown = set(heldout_tasks) - set(by_task)
    if unknown:
        raise ValueError(
            f"held-out task ids are absent from dataset: {sorted(unknown)}"
        )
    generator = np.random.default_rng(seed)
    train, reserved = [], []
    for task, episodes in sorted(by_task.items()):
        if task in heldout_tasks:
            reserved.extend(episodes)
            continue
        ordered = list(generator.permutation(episodes))
        keep = max(1, int(np.floor(len(ordered) * fraction)))
        train.extend(ordered[:keep])
        reserved.extend(ordered[keep:])
    if not train or not reserved:
        raise ValueError(
            "task/episode split must leave both training and reserved episodes"
        )
    return sorted(train), sorted(reserved)


def _training_stats(
    rows: list[dict[str, Any]], selected: set[int], original: dict[str, Any]
) -> dict[str, Any]:
    """Replace only VLA-JEPA's numeric normalizers with training-only statistics."""
    stats = dict(original)
    selected_rows = [row for row in rows if int(row["episode_index"]) in selected]
    if not selected_rows:
        raise ValueError("no frames remain for selected training episodes")
    for name in ("action", "observation.state"):
        values = np.asarray([row[name] for row in selected_rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite values in {name}")
        stats[name] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "count": [len(values)],
        }
    return stats


def prepare_dataset(output: Path, args: argparse.Namespace) -> None:
    """Fetch LIBERO data, seal task-disjoint splits, and rewrite numeric stats."""
    cache = _runtime_cache()
    source = _snapshot(args.dataset_repo, args.dataset_revision, "dataset", cache)
    dataset = output / "dataset"
    shutil.copytree(
        source,
        dataset,
        ignore=shutil.ignore_patterns(".cache", ".npa-vla-jepa-ready.json"),
    )
    info = json.loads((dataset / "meta" / "info.json").read_text())
    rows = _dataset_rows(dataset, info)
    source_hashes = tree_hashes(dataset)
    heldout_tasks = _ids(args.heldout_task_ids)
    train, reserved = _selected_episodes(
        _episodes_by_task(rows),
        heldout_tasks,
        _require_fraction(args.train_fraction),
        args.seed,
    )
    source_stats = json.loads((dataset / "meta" / "stats.json").read_text())
    write_json(
        dataset / "meta" / "stats.json", _training_stats(rows, set(train), source_stats)
    )
    write_json(
        output / "prepare.json",
        {
            "schema": "npa.lerobot-vla-jepa.prepare.v1",
            "dataset": {"repo": args.dataset_repo, "revision": args.dataset_revision},
            "train_episodes": train,
            "reserved_episodes": reserved,
            "heldout_task_ids": heldout_tasks,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
            "source_hashes_before_training_statistics": source_hashes,
            "normalization": "training-only action min/max and state mean/std; visual inputs remain identity-normalized by VLA-JEPA",
            "physical_robot_data": False,
            "dataset_kind": "LIBERO simulation demonstrations",
        },
    )


def _runtime_versions() -> dict[str, Any]:
    """Verify the exact LeRobot package and a usable CUDA runtime."""
    import torch

    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION or not torch.cuda.is_available():
        raise RuntimeError(
            f"requires LeRobot {LEROBOT_VERSION} and an available CUDA device"
        )
    return {
        "lerobot": version,
        "lerobot_source_revision": LEROBOT_SOURCE_REVISION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _support_paths(cache: Path) -> dict[str, Path]:
    """Materialize every model dependency at its immutable source revision."""
    models = {
        "pretrain": (PRETRAIN_REPO, PRETRAIN_REVISION),
        "qwen": (QWEN_REPO, QWEN_REVISION),
        "vjepa": (VJEPA_REPO, VJEPA_REVISION),
    }
    return {
        name: _snapshot(repo, revision, "model", cache)
        for name, (repo, revision) in models.items()
    }


def _train_command(
    prepared: Path, output: Path, args: argparse.Namespace, models: dict[str, Path]
) -> list[str]:
    """Build the upstream CLI invocation without changing its trainer semantics."""
    recipe = json.loads((prepared / "prepare.json").read_text())
    return [
        "lerobot-train",
        f"--policy.path={models['pretrain']}",
        f"--policy.qwen_model_name={models['qwen']}",
        f"--policy.jepa_encoder_name={models['vjepa']}",
        f"--dataset.repo_id={recipe['dataset']['repo']}",
        f"--dataset.root={prepared / 'dataset'}",
        f"--dataset.revision={recipe['dataset']['revision']}",
        f"--dataset.episodes={json.dumps(recipe['train_episodes'])}",
        f"--output_dir={output}",
        f"--steps={args.train_steps}",
        f"--batch_size={args.train_batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.train_steps}",
        "--eval_freq=0",
        "--wandb.enable=false",
        "--wandb.mode=disabled",
        "--policy.device=cuda",
    ]


def _run(command: list[str], log_path: Path) -> None:
    """Stream one native command and preserve its unmodified output."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def train(prepared: Path, output: Path, args: argparse.Namespace) -> None:
    """Fine-tune the upstream model and publish an independently loadable checkpoint."""
    if args.train_steps <= 0 or args.train_batch_size <= 0 or args.num_workers < 0:
        raise ValueError(
            "training steps and batch size must be positive; workers non-negative"
        )
    runtime = _runtime_versions()
    models = _support_paths(_runtime_cache())
    started = time.monotonic()
    training = output / "training"
    _run(_train_command(prepared, training, args, models), output / "train.log")
    checkpoint = training / "checkpoints" / "last" / "pretrained_model"
    validation = validate_lerobot_checkpoint(checkpoint).to_dict()
    shutil.copytree(checkpoint, output / "checkpoint")
    write_json(
        output / "training.json",
        {
            "schema": "npa.lerobot-vla-jepa.training.v1",
            "prepare_sha256": file_sha256(prepared / "prepare.json"),
            "checkpoint_hashes": tree_hashes(output / "checkpoint"),
            "checkpoint_validation": validation,
            "duration_seconds": time.monotonic() - started,
            "runtime": runtime,
            "pretrained": {"repo": PRETRAIN_REPO, "revision": PRETRAIN_REVISION},
            "supporting_models": {
                "qwen": {"repo": QWEN_REPO, "revision": QWEN_REVISION},
                "vjepa2": {"repo": VJEPA_REPO, "revision": VJEPA_REVISION},
            },
            "telemetry": "disabled",
            "claim": "native fine-tuning run; not a convergence or benchmark claim",
        },
    )


def _verify_training(training: Path, prepared: Path) -> dict[str, Any]:
    """Reject a checkpoint not produced from this exact prepared artifact."""
    record = json.loads((training / "training.json").read_text())
    if record.get("prepare_sha256") != file_sha256(prepared / "prepare.json"):
        raise ValueError("training does not derive from this prepared artifact")
    hashes = tree_hashes(training / "checkpoint")
    if not hashes or hashes != record.get("checkpoint_hashes"):
        raise ValueError("checkpoint bytes differ from training provenance")
    validate_lerobot_checkpoint(training / "checkpoint")
    return record


def _eval_command(
    checkpoint: Path, output: Path, args: argparse.Namespace, models: dict[str, Path]
) -> list[str]:
    """Build the native simulator evaluation command for explicit held-out tasks."""
    return [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        f"--policy.qwen_model_name={models['qwen']}",
        f"--policy.jepa_encoder_name={models['vjepa']}",
        "--policy.device=cuda",
        "--env.type=libero",
        f"--env.task={args.libero_suite}",
        f"--env.task_ids={args.heldout_task_ids}",
        f"--eval.n_episodes={args.eval_episodes}",
        f"--eval.batch_size={args.eval_batch_size}",
        f"--output_dir={output}",
        f"--seed={args.seed}",
    ]


def _inspect_videos(root: Path) -> dict[str, dict[str, Any]]:
    """Decode actual evaluation videos instead of trusting a file extension."""
    import av

    inspected = {}
    for path in sorted(root.rglob("*.mp4")):
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        if frames <= 0:
            raise ValueError(f"undecodable or empty rollout video: {path}")
        inspected[path.relative_to(root).as_posix()] = {
            "frames": frames,
            "sha256": file_sha256(path),
        }
    if not inspected:
        raise ValueError("native LeRobot evaluation did not produce an MP4 rollout")
    return inspected


def rollout(
    training: Path, prepared: Path, output: Path, args: argparse.Namespace
) -> None:
    """Run native VLA-JEPA simulator rollouts and preserve real evaluation output."""
    _verify_training(training, prepared)
    if args.eval_episodes <= 0 or args.eval_batch_size <= 0:
        raise ValueError("evaluation episodes and batch size must be positive")
    models = _support_paths(_runtime_cache())
    run_output = output / "native-eval"
    command = _eval_command(training / "checkpoint", run_output, args, models)
    _run(command, output / "rollout.log")
    metrics_path = run_output / "eval_info.json"
    if not metrics_path.exists():
        raise ValueError("native evaluation completed without eval_info.json")
    metrics = json.loads(metrics_path.read_text())
    write_json(
        output / "rollout.json",
        {
            "schema": "npa.lerobot-vla-jepa.rollout.v1",
            "training_sha256": file_sha256(training / "training.json"),
            "heldout_task_ids": _ids(args.heldout_task_ids),
            "suite": args.libero_suite,
            "native_metrics": metrics,
            "videos": _inspect_videos(run_output),
            "physical_robot_tested": False,
        },
    )


def evaluate(rollouts: Path, training: Path, prepared: Path, output: Path) -> None:
    """Numerically validate actual held-out native rollouts and checkpoint lineage."""
    training_record = _verify_training(training, prepared)
    rollout_record = json.loads((rollouts / "rollout.json").read_text())
    if rollout_record.get("training_sha256") != file_sha256(training / "training.json"):
        raise ValueError("rollout does not derive from this exact checkpoint")
    overall = (rollout_record.get("native_metrics") or {}).get("overall") or {}
    success = overall.get("pc_success")
    if not isinstance(success, (float, int)) or not 0 <= float(success) <= 1:
        raise ValueError("native held-out eval lacks a bounded pc_success metric")
    write_json(
        output / "heldout-evaluation.json",
        {
            "schema": "npa.lerobot-vla-jepa.heldout-evaluation.v1",
            "training_sha256": file_sha256(training / "training.json"),
            "rollout_sha256": file_sha256(rollouts / "rollout.json"),
            "checkpoint_hashes": training_record["checkpoint_hashes"],
            "heldout_task_ids": rollout_record["heldout_task_ids"],
            "suite": rollout_record["suite"],
            "pc_success": float(success),
            "native_metrics": rollout_record["native_metrics"],
            "rollout_videos": rollout_record["videos"],
            "physical_robot_tested": False,
            "claim": "held-out LIBERO simulator behavior only; not physical-robot success",
        },
    )


def _record_rrd(evaluation: dict[str, Any], output: Path, run_id: str) -> None:
    """Write a factual Rerun recording from the native held-out metric."""
    import rerun as rr

    output.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream("npa_lerobot_vla_jepa", recording_id=run_id)
    recording.save(str(output))
    recording.log(
        "provenance",
        rr.TextDocument(json.dumps(evaluation, sort_keys=True)),
        static=True,
    )
    recording.set_time("evaluation", sequence=0)
    recording.log("metrics/heldout_pc_success", rr.Scalars(evaluation["pc_success"]))
    recording.flush()
    recording.disconnect()


def _inspect_rrd(path: Path, run_id: str) -> str:
    """Decode the saved recording and require provenance plus real numeric content."""
    binary = str(Path(sys.executable).parent / "rerun")
    verified = subprocess.run(
        [binary, "rrd", "verify", str(path)], check=True, capture_output=True, text=True
    )
    decoded = subprocess.run(
        [binary, "rrd", "print", "-vv", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for expected in (
        "npa_lerobot_vla_jepa",
        run_id,
        "metrics/heldout_pc_success",
        "provenance",
    ):
        if expected not in decoded.stdout:
            raise ValueError(f"required Rerun content is missing: {expected}")
    return verified.stdout + decoded.stdout


def report(evaluation: Path, training: Path, output: Path, run_id: str) -> None:
    """Emit checkpoint provenance and independently inspect a factual RRD."""
    result = json.loads((evaluation / "heldout-evaluation.json").read_text())
    if result.get("training_sha256") != file_sha256(training / "training.json"):
        raise ValueError("evaluation does not match training provenance")
    _record_rrd(result, output / "vla-jepa.rrd", run_id)
    inspection = _inspect_rrd(output / "vla-jepa.rrd", run_id)
    (output / "vla-jepa.inspection.txt").write_text(inspection)
    write_json(
        output / "report.json",
        {
            "schema": "npa.lerobot-vla-jepa.report.v1",
            "evaluation_sha256": file_sha256(evaluation / "heldout-evaluation.json"),
            "checkpoint_hashes": result["checkpoint_hashes"],
            "rrd_sha256": file_sha256(output / "vla-jepa.rrd"),
            "heldout_pc_success": result["pc_success"],
            "physical_robot_tested": False,
            "ready_for_robot_deployment": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Define the five executable VLA-JEPA stage contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset-repo", default=LIBERO_REPO)
    prepare.add_argument("--dataset-revision", default=LIBERO_REVISION)
    prepare.add_argument("--heldout-task-ids", required=True)
    prepare.add_argument("--train-fraction", type=float, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    train = commands.add_parser("train")
    train.add_argument("--train-steps", type=int, required=True)
    train.add_argument("--train-batch-size", type=int, required=True)
    train.add_argument("--num-workers", type=int, required=True)
    rollout_stage = commands.add_parser("rollout")
    rollout_stage.add_argument("--heldout-task-ids", required=True)
    rollout_stage.add_argument("--libero-suite", required=True)
    rollout_stage.add_argument("--eval-episodes", type=int, required=True)
    rollout_stage.add_argument("--eval-batch-size", type=int, required=True)
    rollout_stage.add_argument("--seed", type=int, required=True)
    commands.add_parser("evaluate")
    report_stage = commands.add_parser("report")
    report_stage.add_argument("--run-id", required=True)
    for command in (
        prepare,
        train,
        rollout_stage,
        commands.choices["evaluate"],
        report_stage,
    ):
        command.add_argument("--output-path", required=True)
    for command in (train, rollout_stage, commands.choices["evaluate"], report_stage):
        command.add_argument("--input-path", required=True)
    rollout_stage.add_argument("--training-path", required=True)
    evaluate_stage = commands.choices["evaluate"]
    evaluate_stage.add_argument("--prepared-path", required=True)
    evaluate_stage.add_argument("--training-path", required=True)
    report_stage.add_argument("--training-path", required=True)
    return parser


def _execute(args: argparse.Namespace, workspace: Path, output: Path) -> None:
    """Materialize verified predecessors and execute exactly one real stage."""
    if args.stage == "prepare":
        prepare_dataset(output, args)
        return
    primary = materialize(args.input_path, workspace / "input")
    if args.stage == "train":
        train(primary, output, args)
        return
    training = materialize(args.training_path, workspace / "training")
    if args.stage == "rollout":
        rollout(training, primary, output, args)
        return
    if args.stage == "evaluate":
        prepared = materialize(args.prepared_path, workspace / "prepared")
        evaluate(primary, training, prepared, output)
        return
    report(primary, training, output, args.run_id)


def main(argv: list[str] | None = None) -> int:
    """Run a stage in a clean workspace and publish a checksum-sealed output."""
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="lerobot-vla-jepa-") as temporary:
        workspace = Path(temporary)
        output = workspace / "output"
        _execute(args, workspace, output)
        publish(output, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
