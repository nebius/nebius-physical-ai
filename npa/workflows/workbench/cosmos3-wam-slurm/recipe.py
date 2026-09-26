"""Prepare and execute fixed-batch Cosmos3-Nano LIBERO WAM jobs under Slurm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def _settings(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError(
            "name must contain only letters, digits, underscores or dashes"
        )
    for value in (args.nodes, args.steps, args.samples_per_rank, args.global_batch):
        if value <= 0:
            raise ValueError("nodes, steps and batch sizes must be positive")
    microbatch = 8 * args.nodes * args.samples_per_rank
    if args.global_batch % microbatch:
        raise ValueError(
            "global batch must be divisible by 8 * nodes * samples per rank"
        )
    return {
        "schema": "npa.cosmos3.wam-slurm.v1",
        "name": args.name,
        "nodes": args.nodes,
        "gpus": 8 * args.nodes,
        "steps": args.steps,
        "samples_per_rank": args.samples_per_rank,
        "global_batch": args.global_batch,
        "gradient_accumulation": args.global_batch // microbatch,
        "profile": args.profile,
        "seed": args.seed,
        "shared_root": str(args.shared_root.resolve()),
        "sources": json.loads(Path(__file__).with_name("sources.json").read_text()),
    }


def _toml(settings):
    config = Path(__file__).with_name("train.toml.in").read_text()
    for key, value in settings.items():
        config = config.replace("@" + key.upper() + "@", str(value))
    return config


def _batch_script(settings, run):
    root = Path(settings["shared_root"])
    command = shlex.join(
        [
            str(root / "framework/.venv/bin/python"),
            str(run / "recipe.py"),
            "run-node",
            "--run-dir",
            str(run),
        ]
    )
    return f"""#!/usr/bin/env bash
# Submit from the shared jail filesystem, visible at the same path on every node.
#SBATCH --job-name={settings["name"]}
#SBATCH --nodes={settings["nodes"]}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --output={shlex.quote(str(run / "slurm-%j.out"))}
#SBATCH --error={shlex.quote(str(run / "slurm-%j.err"))}
set -euo pipefail
umask 077
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${{MASTER_PORT:-29500}}"
# Slurm 23.11 exports CPU-only TRES that conflicts with --gpus-per-task.
unset SLURM_TRES_PER_TASK
exec srun --ntasks={settings["nodes"]} --ntasks-per-node=1 \\
    --cpus-per-task=128 --gpus-per-task=8 --gpu-bind=none --kill-on-bad-exit=1 {command}
"""


def _plan(args):
    settings = _settings(args)
    run = args.run_dir.resolve()
    if any(char in str(run) for char in "\n\r"):
        raise ValueError("run directory cannot contain newlines")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copyfile(__file__, run / "recipe.py")
    shutil.copyfile(
        Path(__file__).with_name("distributed_preflight.py"),
        run / "distributed_preflight.py",
    )
    config = _toml(settings)
    settings["toml_sha256"] = hashlib.sha256(config.encode()).hexdigest()
    (run / "train.toml").write_text(config)
    (run / "train.sbatch").write_text(_batch_script(settings, run))
    _write_json(run / "run.json", settings)
    print(
        json.dumps(
            {
                "gpus": settings["gpus"],
                "global_batch": settings["global_batch"],
                "gradient_accumulation": settings["gradient_accumulation"],
                "status": "planned",
                "profile": settings["profile"],
            }
        )
    )


def _environment(settings, run):
    root = Path(settings["shared_root"])
    return dict(
        os.environ,
        COSMOS_TRAINING="1",
        LD_LIBRARY_PATH="",
        PYTHONPATH=str(root / "framework"),
        PATH=str(root / "framework/.venv/bin")
        + os.pathsep
        + os.environ.get("PATH", ""),
        WANDB_MODE="disabled",
        NPA_WAM_RUN_DIR=str(run),
        LIBERO_ROOT=str(root / "data/libero_10"),
        BASE_CHECKPOINT_PATH=str(root / "base-dcp"),
        WAN_VAE_PATH=str(root / "vae/Wan2.2_VAE.pth"),
        IMAGINAIRE_OUTPUT_ROOT=str(run / "output"),
        OMP_NUM_THREADS="4",
    )


def _native_options(settings, run):
    options = [
        "--sft-toml",
        str(run / "train.toml"),
        "--",
        f"trainer.seed={settings['seed']}",
        "model.config.vlm_config.tokenizer.pretrained_model_name="
        + json.dumps(str(Path(settings["shared_root"]) / "tokenizer")),
    ]
    if settings["profile"]:
        ranks = ",".join(str(rank * 8) for rank in range(settings["nodes"]))
        options += [
            "trainer.profiling.enable_profiling=true",
            "trainer.profiling.profile_freq=100",
            "trainer.profiling.profile_warmup=3",
            "trainer.profiling.profile_active=2",
            f"trainer.profiling.target_ranks=[{ranks}]",
        ]
    return options


def _training_argv(settings, run, rank):
    python = str(Path(settings["shared_root"]) / "framework/.venv/bin/python")
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=8",
        f"--nnodes={settings['nodes']}",
        f"--node_rank={rank}",
        f"--master_addr={os.environ['MASTER_ADDR']}",
        f"--master_port={os.environ.get('MASTER_PORT', '29500')}",
        "--max_restarts=0",
        "-m",
        "cosmos_framework.scripts.train",
        *_native_options(settings, run),
    ]


def _verify_inputs(settings, run):
    root = Path(settings["shared_root"])
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root / "framework", text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root / "framework",
        text=True,
    ).strip()
    if revision != settings["sources"]["framework"] or dirty:
        raise ValueError("framework must be the clean pinned revision")
    digest = hashlib.sha256((run / "train.toml").read_bytes()).hexdigest()
    if digest != settings["toml_sha256"]:
        raise ValueError("training TOML changed after planning")
    prepared = json.loads((root / "prepared.json").read_text())
    if prepared["sources"] != settings["sources"]:
        raise ValueError("prepared input revisions differ from planned revisions")
    for item in (
        "base-dcp/model/.metadata",
        "vae/Wan2.2_VAE.pth",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "data/libero_10/meta/info.json",
    ):
        if not (root / item).is_file():
            raise ValueError(f"missing prepared input: {item}")


def _verify_gpus():
    import torch

    names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if len(names) != 8 or any("B200" not in name for name in names):
        raise ValueError("each Slurm task must see exactly eight NVIDIA B200 GPUs")
    return {"gpu_names": names, "torch": torch.__version__, "cuda": torch.version.cuda}


def _preflight(settings, run, rank, command, env, log):
    dryrun = [
        command[0],
        "-m",
        "cosmos_framework.scripts.train",
        "--dryrun",
        *_native_options(settings, run),
    ]
    # Only rank zero writes resolved config during preflight; training resolves it again.
    if rank == 0:
        subprocess.run(
            dryrun,
            cwd=Path(settings["shared_root"]) / "framework",
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
    launcher = command[: command.index("--max_restarts=0") + 1]
    subprocess.run(
        [*launcher, str(run / "distributed_preflight.py")],
        cwd=run,
        env=env,
        stdout=log,
        stderr=log,
        check=True,
    )


def _run_node(args):
    run = args.run_dir.resolve()
    settings = json.loads((run / "run.json").read_text())
    rank = int(os.environ["SLURM_NODEID"])
    if (
        int(os.environ["SLURM_NNODES"]) != settings["nodes"]
        or not 0 <= rank < settings["nodes"]
    ):
        raise ValueError("Slurm allocation differs from the planned topology")
    _verify_inputs(settings, run)
    hardware = _verify_gpus()
    env = _environment(settings, run)
    command = _training_argv(settings, run, rank)
    record = {
        "rank": rank,
        "hardware": hardware,
        "started_unix": time.time(),
        "run_sha256": hashlib.sha256((run / "run.json").read_bytes()).hexdigest(),
    }
    _write_json(run / f"node-{rank}.started.json", record)
    with (run / f"node-{rank}.log").open("x") as log:
        _preflight(settings, run, rank, command, env, log)
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=Path(settings["shared_root"]) / "framework",
            env=env,
            stdout=log,
            stderr=log,
        )
    record.update(
        returncode=result.returncode,
        ended_unix=time.time(),
        train_process_seconds=time.monotonic() - started,
    )
    _write_json(run / f"node-{rank}.finished.json", record)
    return result.returncode


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Write a fresh private Slurm run directory")
    plan.add_argument("--shared-root", type=Path, required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--name", required=True)
    plan.add_argument("--nodes", type=int, required=True)
    plan.add_argument("--steps", type=int, required=True)
    plan.add_argument("--samples-per-rank", type=int, default=64)
    plan.add_argument("--global-batch", type=int, default=2048)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--profile", action="store_true")
    node = commands.add_parser(
        "run-node", help="Run one torchrun agent per allocated node"
    )
    node.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    os.umask(0o077)
    if arguments.command == "plan":
        _plan(arguments)
    else:
        sys.exit(_run_node(arguments))
