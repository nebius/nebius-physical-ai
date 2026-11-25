"""Prepare and execute fixed-batch Cosmos3-Nano LIBERO WAM jobs under Slurm or local torchrun."""

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
import uuid


# Planned receipts written before the RTX extension carry no ``hardware`` key.
# Treating those receipts as the B200 topology keeps earlier runs runnable.
SCHEMA_V1 = "npa.cosmos3.wam-slurm.v1"
SCHEMA_V2 = "npa.cosmos3.wam-slurm.v2"

B200 = {
    "label": "NVIDIA B200",
    "nodes": None,
    "gpus_per_node": (4, 8),
}
# Match the full device name from the reserved PCIe server inventory.
RTX_PRO_6000 = {
    "label": "RTX PRO 6000 Blackwell Server Edition",
    "names": ("NVIDIA RTX PRO 6000 Blackwell Server Edition",),
    "nodes": 1,
    "gpus_per_node": (8,),
}


def _hardware(name):
    if name == "b200":
        return {"default": True, "label": B200["label"]}
    if name == "rtx-pro-6000":
        return {"default": False, "label": RTX_PRO_6000["label"]}
    raise ValueError(f"unknown hardware setting: {name}")


def _topology(settings):
    """Return the GPU validation contract for a planned receipt.

    Receipts that predate the RTX extension have no ``hardware`` key; they were
    planned as B200 jobs, so the B200 contract applies and nothing is relaxed.
    """
    hardware = settings.get("hardware")
    if hardware is None:
        hardware = _hardware("b200")
    label = hardware.get("label") if isinstance(hardware, dict) else None
    if label == B200["label"]:
        return B200
    if label == RTX_PRO_6000["label"]:
        return RTX_PRO_6000
    raise ValueError(f"each Slurm task must see a supported GPU type, got {label!r}")


def _check_topology(hardware, nodes, gpus_per_node):
    label = hardware.get("label") if isinstance(hardware, dict) else None
    if label == B200["label"]:
        return
    if label != RTX_PRO_6000["label"]:
        raise ValueError(f"unknown hardware label: {label!r}")
    if nodes != RTX_PRO_6000["nodes"]:
        raise ValueError(
            "RTX PRO 6000 Blackwell Server Edition requires exactly one node"
        )
    if gpus_per_node not in RTX_PRO_6000["gpus_per_node"]:
        raise ValueError(
            "RTX PRO 6000 Blackwell Server Edition requires exactly eight GPUs per node"
        )


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
    hardware = _hardware(args.hardware)
    local = getattr(args, "launcher", "slurm") == "torchrun"
    if local and args.hardware != "rtx-pro-6000":
        raise ValueError("the local launcher requires rtx-pro-6000")
    _check_topology(hardware, args.nodes, args.gpus_per_node)
    if args.gpus_per_node not in (4, 8):
        raise ValueError("gpus per node must be four or eight")
    microbatch = args.gpus_per_node * args.nodes * args.samples_per_rank
    if args.global_batch % microbatch:
        raise ValueError(
            "global batch must be divisible by GPUs per node * nodes * samples per rank"
        )
    return {
        "schema": SCHEMA_V1 if hardware["default"] else SCHEMA_V2,
        **({} if hardware["default"] else {"hardware": hardware}),
        **(
            {
                "launcher": "torchrun",
                "run_id": str(uuid.uuid4()),
                "torch_compile_disable": os.environ.get("TORCH_COMPILE_DISABLE", "0")
                == "1",
            }
            if local
            else {}
        ),
        "name": args.name,
        "nodes": args.nodes,
        "gpus": args.gpus_per_node * args.nodes,
        "gpus_per_node": args.gpus_per_node,
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
#SBATCH --gpus-per-node={settings["gpus_per_node"]}
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
    --cpus-per-task=128 --gpus-per-task={settings["gpus_per_node"]} --gpu-bind=none --kill-on-bad-exit=1 {command}
"""


def _plan(args):
    settings = _settings(args)
    run = args.run_dir.resolve()
    if any(char in str(run) for char in "\n\r"):
        raise ValueError("run directory cannot contain newlines")
    local = settings.get("launcher") == "torchrun"
    trainer = Path(__file__).with_name("trainer_rank.py")
    if local and not trainer.is_file():
        raise ValueError("missing local trainer target: trainer_rank.py")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copyfile(__file__, run / "recipe.py")
    shutil.copyfile(
        Path(__file__).with_name("distributed_preflight.py"),
        run / "distributed_preflight.py",
    )
    config = _toml(settings)
    settings["toml_sha256"] = hashlib.sha256(config.encode()).hexdigest()
    (run / "train.toml").write_text(config)
    if local:
        shutil.copyfile(trainer, run / trainer.name)
    else:
        (run / "train.sbatch").write_text(_batch_script(settings, run))
    _write_json(run / "run.json", settings)
    print(
        json.dumps(
            {
                "gpus": settings["gpus"],
                "global_batch": settings["global_batch"],
                "gradient_accumulation": settings["gradient_accumulation"],
                "nodes": settings["nodes"],
                "gpus_per_node": settings["gpus_per_node"],
                **(
                    {"hardware": settings["hardware"]} if "hardware" in settings else {}
                ),
                **({"launcher": "torchrun"} if local else {}),
                "status": "planned",
                "profile": settings["profile"],
            }
        )
    )


def _environment(settings, run):
    root = Path(settings["shared_root"])
    env = dict(
        os.environ,
        COSMOS_TRAINING="1",
        LD_LIBRARY_PATH="",
        PYTHONPATH=str(root / "framework"),
        PATH=str(root / "framework/.venv/bin")
        + os.pathsep
        + os.environ.get("PATH", ""),
        WANDB_MODE="disabled",
        TZ="UTC",
        NPA_WAM_RUN_DIR=str(run),
        NPA_WAM_GPUS_PER_NODE=str(settings["gpus_per_node"]),
        NPA_WAM_HARDWARE=_topology(settings)["label"],
        LIBERO_ROOT=str(root / "data/libero_10"),
        BASE_CHECKPOINT_PATH=str(root / "base-dcp"),
        WAN_VAE_PATH=str(root / "vae/Wan2.2_VAE.pth"),
        IMAGINAIRE_OUTPUT_ROOT=str(run / "output"),
        OMP_NUM_THREADS="4",
    )
    local = settings.get("launcher") == "torchrun"
    if local:
        if "torch_compile_disable" in settings:
            env["TORCH_COMPILE_DISABLE"] = (
                "1" if settings["torch_compile_disable"] else "0"
            )
        for key in list(env):
            if key.startswith(("SLURM_", "TORCHELASTIC_")) or key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
            ):
                del env[key]
    debug_file = env.get("NCCL_DEBUG_FILE")
    if debug_file and Path(debug_file).is_fifo():
        # Soperator's FIFO reader exits when the preflight's last writer closes.
        # A later trainer would block opening that FIFO; retain per-process logs.
        node = 0 if local else int(env["SLURM_NODEID"])
        env["NCCL_DEBUG_FILE"] = str(run / f"nccl-node-{node}-%p.log")
    return env


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
        ranks = ",".join(
            str(rank * settings["gpus_per_node"]) for rank in range(settings["nodes"])
        )
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
        f"--nproc_per_node={settings['gpus_per_node']}",
        f"--nnodes={settings['nodes']}",
        f"--node_rank={rank}",
        f"--master_addr={os.environ['MASTER_ADDR']}",
        f"--master_port={os.environ.get('MASTER_PORT', '29500')}",
        "--max_restarts=0",
        "-m",
        "cosmos_framework.scripts.train",
        *_native_options(settings, run),
    ]


def _local_argv(settings, run):
    python = str(Path(settings["shared_root"]) / "framework/.venv/bin/python")
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={settings['gpus_per_node']}",
        "--max_restarts=0",
        str(run / "trainer_rank.py"),
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
    if settings.get("launcher") != "torchrun":
        _verify_video_runtime(root)


def _verify_video_runtime(root):
    from torchcodec.decoders import VideoDecoder

    for camera in ("observation.images.image", "observation.images.wrist_image"):
        directory = root / "data/libero_10/videos" / camera
        videos = sorted(directory.glob("chunk-*/*.mp4"))
        if not videos:
            raise ValueError(f"missing camera videos on this worker: {camera}")
        frame = VideoDecoder(str(videos[0]), device="cpu")[0]
        if tuple(frame.shape) != (3, 256, 256):
            raise ValueError(f"incompatible decoded camera frame: {camera}")


def _verify_gpus(expected, hardware=None):
    import torch

    topology = _topology({"hardware": hardware})
    names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if len(names) != expected:
        raise ValueError(
            f"each Slurm task must see exactly {expected} {topology['label']} GPUs"
        )
    if topology is B200:
        if any("B200" not in name for name in names):
            raise ValueError(
                f"each Slurm task must see exactly {expected} NVIDIA B200 GPUs"
            )
    else:
        for name in names:
            if name not in RTX_PRO_6000["names"]:
                raise ValueError(
                    "each Slurm task must see exactly "
                    f"{expected} {topology['label']} GPUs, got {name!r}"
                )
    return {"gpu_names": names, "torch": torch.__version__, "cuda": torch.version.cuda}


def _preflight(settings, run, rank, command, env, log):
    if settings.get("launcher") == "torchrun":
        subprocess.run(
            [command[0], run / "recipe.py", "decode-worker", "--run-dir", run],
            cwd=Path(settings["shared_root"]) / "framework",
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
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


def _worker_rank(settings, run, local):
    expected = "torchrun" if local else "slurm"
    if settings.get("launcher", "slurm") != expected:
        raise ValueError(f"command does not match planned launcher: {expected}")
    if local:
        shape = (settings["nodes"], settings["gpus_per_node"], settings["gpus"])
        if shape != (1, 8, 8) or _topology(settings) is not RTX_PRO_6000:
            raise ValueError("local torchrun requires one eight-GPU RTX host")
        if any(
            (run / name).exists()
            for name in (
                "node-0.started.json",
                "node-0.log",
                "node-0.finished.json",
                "distributed-preflight.json",
                "ranks",
                "output",
            )
        ):
            raise FileExistsError(
                "local run directory already contains execution evidence"
            )
        # Claim before runtime checks so failed or concurrent attempts cannot
        # overwrite one another. Each retry starts from a fresh plan.
        with (run / "local.started").open("x"):
            pass
        return 0
    rank = int(os.environ["SLURM_NODEID"])
    if (
        int(os.environ["SLURM_NNODES"]) != settings["nodes"]
        or not 0 <= rank < settings["nodes"]
    ):
        raise ValueError("Slurm allocation differs from the planned topology")
    return rank


def _run_worker(args, local=False):
    run = args.run_dir.resolve()
    settings_bytes = (run / "run.json").read_bytes()
    settings = json.loads(settings_bytes)
    rank = _worker_rank(settings, run, local)
    worker_started = time.monotonic()
    _verify_inputs(settings, run)
    hardware = _verify_gpus(settings["gpus_per_node"], settings.get("hardware"))
    env = _environment(settings, run)
    command = (
        _local_argv(settings, run) if local else _training_argv(settings, run, rank)
    )
    record = {
        "rank": rank,
        "hardware": hardware,
        "started_unix": time.time(),
        "run_sha256": hashlib.sha256(settings_bytes).hexdigest(),
    }
    _write_json(run / f"node-{rank}.started.json", record)
    with (run / f"node-{rank}.log").open("x") as log:
        _preflight(settings, run, rank, command, env, log)
        started = time.monotonic()
        if local:
            record.update(
                preflight_seconds=started - worker_started,
                trainer_started_unix=time.time(),
            )
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


def _run_node(args):
    return _run_worker(args)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Write a fresh private run directory")
    plan.add_argument("--shared-root", type=Path, required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--name", required=True)
    plan.add_argument("--nodes", type=int, required=True)
    plan.add_argument(
        "--hardware",
        choices=("b200", "rtx-pro-6000"),
        default="b200",
        help="GPU family; rtx-pro-6000 is one node with eight RTX PRO 6000 Blackwell Server Edition",
    )
    plan.add_argument("--gpus-per-node", type=int, choices=(4, 8), default=8)
    plan.add_argument("--steps", type=int, required=True)
    plan.add_argument("--samples-per-rank", type=int, default=64)
    plan.add_argument("--launcher", choices=("slurm", "torchrun"), default="slurm")
    plan.add_argument("--global-batch", type=int, default=2048)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--profile", action="store_true")
    node = commands.add_parser(
        "run-node", help="Run one torchrun agent per allocated node"
    )
    node.add_argument("--run-dir", type=Path, required=True)
    local = commands.add_parser(
        "run-local", help="Launch the saved single-host RTX plan"
    )
    local.add_argument("--run-dir", type=Path, required=True)
    decode = commands.add_parser(
        "decode-worker", help="Decode both views in the trainer runtime"
    )
    decode.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    os.umask(0o077)
    if arguments.command == "plan":
        _plan(arguments)
    elif arguments.command == "decode-worker":
        settings = json.loads((arguments.run_dir / "run.json").read_text())
        _verify_video_runtime(Path(settings["shared_root"]))
    else:
        sys.exit(_run_worker(arguments, local=arguments.command == "run-local"))
