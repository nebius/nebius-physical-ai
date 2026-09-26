"""Evaluate a trained WAM checkpoint on every LIBERO-10 task using native servers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _environment(root, worker):
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=str(worker),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(31000 + worker),
        COSMOS_TRAINING="1",
        LD_LIBRARY_PATH="",
        WANDB_MODE="disabled",
        LIBERO_ROOT=str(root / "data/libero_10"),
        LIBERO_CONFIG_PATH=str(root / "simulation/libero-config"),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        PYTHONPATH=os.pathsep.join(
            str(root / path) for path in ("framework", "simulation/LIBERO")
        ),
        PATH=str(root / "framework/.venv/bin") + os.pathsep + env["PATH"],
        OMP_NUM_THREADS="4",
    )
    return env


def _server_command(args, checkpoint, worker, output):
    root = args.shared_root
    stats = root / "framework/cosmos_framework/data/generator/action/normalizer_stats"
    return [
        str(root / "framework/.venv/bin/python"),
        "-m",
        "cosmos_framework.scripts.action_policy_server_libero",
        "--experiment",
        "action_policy_libero_nano",
        "--checkpoint-path",
        str(checkpoint / "model"),
        "--experiment-overrides",
        "model.config.tokenizer.vae_path="
        + json.dumps(str(root / "vae/Wan2.2_VAE.pth")),
        "model.config.vlm_config.tokenizer.pretrained_model_name="
        + json.dumps(str(root / "tokenizer")),
        "--action-normalization",
        "quantile_rot",
        "--action-stats-path",
        str(stats / "libero_native_frame_wise_relative_rot6d.json"),
        "--raw-action-dim",
        "10",
        "--fps",
        "20",
        "--seed",
        str(args.seed),
        "--num-steps",
        "30",
        "--guidance",
        "1.0",
        "--host",
        "127.0.0.1",
        "--port",
        str(8000 + worker),
        "--http-400-on-error",
        "--output-dir",
        str(output / "server-output"),
    ]


def _client_command(args, worker, tasks, output):
    root = args.shared_root
    command = [
        str(root / "simulation/libenv/bin/python"),
        str(root / "framework/cosmos_framework/simulation/libero/closed_loop_eval.py"),
        "--server_url",
        f"http://127.0.0.1:{8000 + worker}",
        "--task_suite",
        "libero_10",
        "--task_ids",
        ",".join(map(str, tasks)),
        "--num_trials_per_task",
        str(args.trials),
        "--num_envs",
        str(args.envs),
        "--camera",
        "agentview,wrist",
        "--image_size",
        "256",
        "--action_space",
        "frame_wise_relative",
        "--rotation_space",
        "6d",
        "--action_dim",
        "10",
        "--gripper_mode",
        "zero_one",
        "--mujoco_gl",
        "osmesa",
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output / "rollouts"),
    ]
    if args.record_rollouts:
        command.extend(["--save_gifs", "--save_comparison"])
    return command


def _wait_ready(server, worker, checkpoint):
    while server.poll() is None:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{8000 + worker}/info", timeout=5
            ) as reply:
                info = json.load(reply)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
            continue
        if Path(info.get("checkpoint", "")).resolve() != checkpoint / "model":
            raise ValueError("policy server loaded an unexpected checkpoint")
        return info
    raise RuntimeError(f"policy server exited with status {server.returncode}")


def _stop(server):
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def _evaluate_worker(args, checkpoint, worker, tasks):
    output = args.output_dir / f"worker-{worker}"
    output.mkdir(mode=0o700)
    env = _environment(args.shared_root, worker)
    server_command = _server_command(args, checkpoint, worker, output)
    client_command = _client_command(args, worker, tasks, output)
    _write(
        output / "commands.json", {"server": server_command, "client": client_command}
    )
    started = time.monotonic()
    with (output / "server.log").open("x") as log:
        server = subprocess.Popen(
            server_command,
            cwd=args.shared_root / "framework",
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            _write(output / "server-info.json", _wait_ready(server, worker, checkpoint))
            with (output / "evaluation.log").open("x") as evaluation_log:
                subprocess.run(
                    client_command,
                    cwd=args.shared_root / "framework",
                    env=env,
                    stdout=evaluation_log,
                    stderr=evaluation_log,
                    check=True,
                )
        finally:
            _stop(server)
    _write(output / "completion.json", {"elapsed_seconds": time.monotonic() - started})
    return json.loads((output / "rollouts/summary.json").read_text())


def _task_results(summaries, trials):
    tasks = [task for summary in summaries for task in summary["task_results"]]
    if sorted(task["task_id"] for task in tasks) != list(range(10)):
        raise ValueError("evaluation must cover all ten tasks exactly once")
    for task in tasks:
        episodes = task["episode_results"]
        if sorted(item["episode"] for item in episodes) != list(range(trials)):
            raise ValueError("evaluation has missing or duplicate trials")
        for episode in episodes:
            if type(episode["success"]) is not bool or episode.get("error"):
                raise ValueError(
                    "evaluation contains infrastructure or inference errors"
                )
        successes = sum(item["success"] for item in episodes)
        if task["episodes"] != trials or task["successes"] != successes:
            raise ValueError("task aggregates do not match individual trials")
    return sorted(tasks, key=lambda task: task["task_id"])


def _wilson(successes, total):
    probability, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def _validate(args):
    if args.workers not in range(1, 9) or args.trials < 1 or args.envs < 1:
        raise ValueError("workers must be 1..8; trials and envs must be positive")
    if args.record_rollouts and args.envs != 1:
        raise ValueError(
            "native vectorized evaluation does not save rollout videos; use --envs 1"
        )
    settings = json.loads((args.run_dir / "run.json").read_text())
    job = args.run_dir / "output/cosmos3_wam/libero_10" / settings["name"]
    checkpoint = (job / "checkpoints" / f"iter_{args.step:09d}").resolve()
    if not (checkpoint / "model/.metadata").is_file():
        raise ValueError("trained model checkpoint is incomplete")
    if args.seed != settings["seed"]:
        raise ValueError("evaluation seed must match the recorded benchmark seed")
    prepared = json.loads((args.shared_root / "prepared.json").read_text())
    if prepared["sources"] != settings["sources"]:
        raise ValueError("evaluation inputs differ from training inputs")
    for directory, name in (
        ("framework", "framework"),
        ("simulation/LIBERO", "libero"),
    ):
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.shared_root / directory, text=True
        ).strip()
        if revision != settings["sources"][name]:
            raise ValueError(f"evaluation {name} revision differs from training")
    return checkpoint


def _evaluate_tasks(args, checkpoint):
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _evaluate_worker,
                args,
                checkpoint,
                worker,
                list(range(worker, 10, args.workers)),
            )
            for worker in range(args.workers)
        ]
        return _task_results([future.result() for future in futures], args.trials)


def _checkpoint_identity(checkpoint):
    manifest = {}
    for path in sorted((checkpoint / "model").iterdir()):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    if ".metadata" not in manifest or not any(
        name.endswith(".distcp") for name in manifest
    ):
        raise ValueError("model checkpoint has no complete distributed shards")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return digest, manifest


def _main(args):
    started = time.monotonic()
    checkpoint = _validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _write(
        args.output_dir / "settings.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    digest, manifest = _checkpoint_identity(checkpoint)
    _write(args.output_dir / "model-hashes.json", manifest)
    tasks = _evaluate_tasks(args, checkpoint)
    total, successes = 10 * args.trials, sum(task["successes"] for task in tasks)
    result = {
        "schema": "npa.cosmos3.wam-quality.v1",
        "model_manifest_sha256": digest,
        "step": args.step,
        "trials": total,
        "successes": successes,
        "success_rate": successes / total,
        "wilson_95_interval": _wilson(successes, total),
        "full_500_trial_evaluation": args.trials == 50,
        "threshold": 0.9,
        "threshold_met": args.trials == 50 and successes / total >= 0.9,
        "elapsed_seconds": time.monotonic() - started,
        "task_results": tasks,
    }
    _write(args.output_dir / "quality.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "task_results"}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-rollouts", action="store_true")
    os.umask(0o077)
    _main(parser.parse_args())
