"""Evaluate a trained WAM checkpoint on every LIBERO-10 task using native servers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
import os
import re
from pathlib import Path
import socket
import subprocess
import time


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


def _recording_options(enabled):
    if enabled:
        return ["--save_gifs", "--save_comparison"]
    return []


def _client_command(args, worker, tasks, output):
    root = args.shared_root
    return [
        str(root / "simulation/libenv/bin/python"),
        str(Path(__file__).with_name("simulation_client.py")),
        "--shared-root",
        str(root),
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
        "--timeout",
        "300",
        "--output_dir",
        str(output / "rollouts"),
        *_recording_options(args.record_rollouts),
    ]


def _server_info(worker):
    connection = http.client.HTTPConnection("127.0.0.1", 8000 + worker, timeout=5)
    try:
        connection.request("GET", "/info")
        response = connection.getresponse()
        if response.status != 200:
            raise ConnectionError("local policy server is not ready")
        return json.load(response)
    finally:
        connection.close()


def _wait_ready(server, worker, checkpoint, timeout=None):
    started = time.monotonic()
    while server.poll() is None:
        if timeout is not None and time.monotonic() - started >= timeout:
            raise TimeoutError("native policy server readiness timed out")
        try:
            info = _server_info(worker)
        except (OSError, http.client.HTTPException):
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


def _client_environment(env):
    # Each simulator process owns a software renderer; avoid nested CPU pools.
    return dict(
        env,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LP_NUM_THREADS="1",
    )


def _evaluate_worker(args, checkpoint, worker, tasks):
    output = args.output_dir / f"worker-{worker}"
    output.mkdir(mode=0o700)
    env = _environment(args.shared_root, worker)
    compile_disable = getattr(args, "torch_compile_disable", None)
    if compile_disable is not None:
        env["TORCH_COMPILE_DISABLE"] = "1" if compile_disable else "0"
    server_command = _server_command(args, checkpoint, worker, output)
    reload_only = getattr(args, "reload_only", False)
    client_command = (
        None if reload_only else _client_command(args, worker, tasks, output)
    )
    _write(
        output / "commands.json", {"server": server_command, "client": client_command}
    )
    started = time.monotonic()
    with (output / "server.log").open("x") as log:
        bounded = (
            reload_only or getattr(args, "source_training_report", None) is not None
        )
        if bounded:
            for port in (8000 + worker, 31000 + worker):
                with socket.socket() as listener:
                    # A previous completed run can leave loopback in TIME_WAIT.
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("127.0.0.1", port))
                    listener.listen(1)
        server = subprocess.Popen(
            server_command,
            cwd=args.shared_root / "framework",
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            info = (
                _wait_ready(server, worker, checkpoint, timeout=600)
                if bounded
                else _wait_ready(server, worker, checkpoint)
            )
            _write(output / "server-info.json", info)
            ready_seconds = time.monotonic() - started
            if reload_only:
                return {"server_ready_seconds": ready_seconds}
            rollout_started = time.monotonic()
            with (output / "evaluation.log").open("x") as evaluation_log:
                subprocess.run(
                    client_command,
                    cwd=args.shared_root / "framework",
                    env=_client_environment(env),
                    stdout=evaluation_log,
                    stderr=evaluation_log,
                    check=True,
                )
            rollout_seconds = time.monotonic() - rollout_started
        finally:
            _stop(server)
    _write(
        output / "completion.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "server_ready_seconds": ready_seconds,
            "rollout_seconds": rollout_seconds,
        },
    )
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


def _check_revision(directory, expected):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True
    ).strip()
    if revision != expected:
        raise ValueError("evaluation source revision differs from training")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=directory, check=True)


def _validate(args):
    settings = json.loads((args.run_dir / "run.json").read_text())
    if settings.get("launcher") == "torchrun" and "torch_compile_disable" in settings:
        args.torch_compile_disable = settings["torch_compile_disable"]
    if getattr(args, "require_transfer_receipt", False) and (
        getattr(args, "reload_only", False) or not args.source_training_report
    ):
        raise ValueError(
            "a transfer receipt requires scoring with a source training report"
        )
    rtx = (
        settings.get("launcher") == "torchrun"
        or settings.get("hardware", {}).get("label")
        == "RTX PRO 6000 Blackwell Server Edition"
    )
    for key, default in (("workers", 8), ("trials", 50), ("envs", 8)):
        if getattr(args, key) is None:
            setattr(args, key, 1 if rtx else default)
    if (
        rtx
        and not getattr(args, "reload_only", False)
        and not getattr(args, "source_training_report", None)
    ):
        raise ValueError("RTX scoring requires --source-training-report")
    if args.workers not in range(1, 9) or args.trials < 1 or args.envs < 1:
        raise ValueError("workers must be 1..8; trials and envs must be positive")
    if args.record_rollouts and args.envs != 1:
        raise ValueError(
            "native vectorized evaluation does not save rollout videos; use --envs 1"
        )
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", settings["name"])
        or not 1 <= args.step <= settings["steps"]
    ):
        raise ValueError("invalid run name or checkpoint step")
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
        _check_revision(args.shared_root / directory, settings["sources"][name])
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(checkpoint, strict=True):
    directory = checkpoint / "model"
    if strict and directory.is_symlink():
        raise ValueError("staged model directory is a symlink")
    manifest = {}
    for path in sorted(directory.iterdir()):
        if strict and (
            path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        ):
            raise ValueError(f"invalid staged model entry: {path.name}")
        if path.is_file():
            manifest[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    if ".metadata" not in manifest or not any(
        name.endswith(".distcp") for name in manifest
    ):
        raise ValueError("model checkpoint has no complete distributed shards")
    return _manifest_digest(manifest), manifest


def _manifest_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _source_model(report, run_bytes, step):
    if (
        report.get("schema") != "npa.cosmos3.wam-rtx-smoke.v1"
        or report["run_sha256"] != hashlib.sha256(run_bytes).hexdigest()
    ):
        raise ValueError("source training evidence differs from the scoring run")
    if (
        report.get("terminal_step") != step
        or report["completed_optimizer_updates"] != step
    ):
        raise ValueError(
            "source training evidence differs from the requested checkpoint step"
        )
    files = report["checkpoint_files"]
    if _manifest_digest(files) != report["checkpoint_sha256"]:
        raise ValueError("source checkpoint manifest digest mismatch")
    model = {
        name.removeprefix("model/"): entry
        for name, entry in files.items()
        if name.startswith("model/")
    }
    if ".metadata" not in model or not any(name.endswith(".distcp") for name in model):
        raise ValueError("source checkpoint has no complete model")
    for name, entry in model.items():
        if (
            Path(name).name != name
            or type(entry["bytes"]) is not int
            or entry["bytes"] <= 0
            or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError(f"invalid source model entry: {name!r}")
    return model


def _validate_transfer_measurements(receipt, expected):
    sizes = {
        "model_bytes": sum(v["bytes"] for v in expected.values()),
        "model_files": len(expected),
    }
    for field, value in sizes.items():
        if type(receipt.get(field)) is not int or receipt[field] != value:
            raise ValueError(f"transfer {field} differs from the source manifest")
    for field in (
        "model_transfer_seconds",
        "destination_hash_seconds",
        "total_receive_seconds",
    ):
        seconds = receipt.get(field)
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"invalid transfer timing: {field}")
    if (
        receipt["total_receive_seconds"]
        < receipt["model_transfer_seconds"] + receipt["destination_hash_seconds"]
    ):
        raise ValueError("total receive time is shorter than copy and verification")


def _scored_identity(args, checkpoint):
    started = time.monotonic()
    source = getattr(args, "source_training_report", None)
    if source is None:
        digest, observed = _checkpoint_identity(checkpoint, strict=False)
        return digest, observed, {}
    raw = source.read_bytes()
    report = json.loads(raw)
    run_bytes = (args.run_dir / "run.json").read_bytes()
    expected = _source_model(report, run_bytes, args.step)
    transferred = args.run_dir / "transfer.json"
    required = getattr(args, "require_transfer_receipt", False)
    if required and not transferred.is_file():
        raise ValueError("the verified transfer receipt is missing")
    if transferred.exists():
        receipt = json.loads(transferred.read_text())
        if (
            not isinstance(receipt, dict)
            or receipt.get("source_training_report_sha256")
            != hashlib.sha256(raw).hexdigest()
            or receipt.get("run_sha256") != report["run_sha256"]
            or receipt.get("model_manifest_sha256") != _manifest_digest(expected)
        ):
            raise ValueError("source report differs from the verified transfer")
        if required:
            _validate_transfer_measurements(receipt, expected)
    digest, observed = _checkpoint_identity(checkpoint)
    if expected != observed:
        raise ValueError(
            "destination model files differ from the source training checkpoint"
        )
    identity = {
        "run_sha256": report["run_sha256"],
        "run_id": json.loads(run_bytes).get("run_id"),
        "source_checkpoint_sha256": report["checkpoint_sha256"],
        "source_training_report_sha256": hashlib.sha256(raw).hexdigest(),
        "model_bytes": sum(entry["bytes"] for entry in expected.values()),
        "model_verification_seconds": time.monotonic() - started,
    }
    return digest, observed, identity


def _main(args):
    started = time.monotonic()
    checkpoint = _validate(args)
    reload_only = getattr(args, "reload_only", False)
    if reload_only:
        settings = json.loads((args.run_dir / "run.json").read_text())
        if args.workers != 1 or args.step != settings["steps"] or args.record_rollouts:
            raise ValueError("reload-only requires one worker and the terminal step")
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _write(
        args.output_dir / "settings.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    if reload_only:
        from report import _checkpoint

        observation = _evaluate_worker(args, checkpoint, 0, ())
        hash_started = time.monotonic()
        digest, manifest = _checkpoint(args.run_dir, settings)
        hash_seconds = time.monotonic() - hash_started
        _write(args.output_dir / "checkpoint-hashes.json", manifest)
        result = {
            "schema": "npa.cosmos3.wam-policy-reload.v1",
            "step": args.step,
            "loaded_sha256": digest,
            "checkpoint_hash_seconds": hash_seconds,
            "server_ready_seconds": observation["server_ready_seconds"],
            "elapsed_seconds": time.monotonic() - started,
        }
        _write(args.output_dir / "reload.json", result)
        print(json.dumps(result))
        return
    digest, manifest, identity = _scored_identity(args, checkpoint)
    _write(args.output_dir / "model-hashes.json", manifest)
    tasks = _evaluate_tasks(args, checkpoint)
    total, successes = 10 * args.trials, sum(task["successes"] for task in tasks)
    result = {
        "schema": "npa.cosmos3.wam-quality.v1",
        **identity,
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
    if identity:
        result["worker_timings"] = [
            json.loads(
                (args.output_dir / f"worker-{worker}/completion.json").read_text()
            )
            for worker in range(args.workers)
        ]
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
    parser.add_argument(
        "--source-training-report",
        type=Path,
        help="training-side checkpoint receipt for RTX scoring",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--require-transfer-receipt",
        action="store_true",
        help="Verify the receiver's copy and hash timings before scoring a transferred model",
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-rollouts", action="store_true")
    parser.add_argument(
        "--reload-only",
        action="store_true",
        help="Start one native policy server and verify terminal checkpoint identity",
    )
    os.umask(0o077)
    _main(parser.parse_args())
