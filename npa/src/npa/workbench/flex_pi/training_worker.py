"""Vendor-interpreter entrypoint for the fixed public Flex-Pi training workload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

from npa.workbench.flex_pi.training_normalization import normalization_overrides
from npa.workbench.flex_pi.training_activation import (
    activation_checkpointing_overrides,
    activation_checkpointing_receipt,
)
from npa.workbench.flex_pi.training_topology import rank_command, training_topology


def _configuration(plan, root, assets):
    from flexpi.utils.config_resolvers import register_default_resolvers
    from npa.workbench.flex_pi.training_assets import _hydra_overrides

    register_default_resolvers()
    overrides = (
        _hydra_overrides(assets)
        + normalization_overrides(plan, root)
        + activation_checkpointing_overrides(
            plan["execution"].get("activation_checkpointing", "on")
        )
        + [
            f"output_dir={root}",
            "batch_size=1",
            "gradient_accumulation_steps=24",
            "num_epochs=1",
            "max_steps=null",
            "mixed_precision=bf16",
            "learning_rate=1e-4",
            "weight_decay=0.01",
            "seed=42",
            "wandb.enabled=false",
            f"model.action_dit_pretrained_path={assets / 'ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}",
            f"num_workers={plan['execution']['num_workers']}",
            "data.train._target_=npa.workbench.flex_pi.training_engine.ExactTrainingDataset",
            "data.val._target_=npa.workbench.flex_pi.training_engine.ExactTrainingDataset",
            "+npa_optimizer=" + plan["execution"]["optimizer"],
            "+npa_deterministic_training=true",
            "+npa_ddp_bucket_policy=fixed_find_unused",
            "+npa_prefetch_factor=" + str(plan["execution"]["prefetch_factor"]),
        ]
    )
    if plan.get("resume_directory"):
        overrides.append("resume=" + plan["resume_directory"])
    if plan["execution"].get("memory_fill", "on") == "off":
        overrides.append('+npa_memory_fill="off"')
    return _compose_configuration(assets, overrides)


def _compose_configuration(assets, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(assets / "upstream/configs"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _rank_main(request_path):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from flexpi.runtime import build_datasets
    from flexpi.utils import misc
    from npa.workbench.flex_pi.training_engine import VerifiedTrainer

    plan = json.loads(request_path.read_text())
    root = Path(plan["work_directory"])
    _initialize_rank()
    misc.register_work_dir(str(root))
    cfg = OmegaConf.create(plan["configuration"])
    start = time.perf_counter()
    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device=f"cuda:{torch.cuda.current_device()}",
    )
    activation = activation_checkpointing_receipt(cfg, model)
    train_ds, val_ds = build_datasets(cfg.data)
    trainer = VerifiedTrainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    initialized = time.perf_counter() - start
    result = trainer.execute(
        plan["execution"]["mode"],
        profile_resume=plan.get("resume_probe_kind") == "profile",
    )
    result["initialization_seconds"] = initialized
    result["activation_checkpointing"] = activation
    result.update(_phase_receipts(trainer, cfg, plan))
    if torch.distributed.get_rank() == 0:
        (root / "phase-result.json").write_text(json.dumps(result, allow_nan=False))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _phase_receipts(trainer, cfg, plan):
    from npa.workbench.flex_pi.training_memory import memory_fill_receipt

    result = {"runtime": _runtime_receipt(), "memory_fill": memory_fill_receipt(cfg)}
    if getattr(trainer, "_profile_receipt", None) is not None:
        result["profiling"] = {
            **trainer._profile_receipt,
            "runtime": plan["profiler_runtime"],
        }
    if plan["execution"]["mode"] == "qualify":
        result["qualification_seconds"] = trainer._qualification_seconds
    return result


def _initialize_rank():
    import numpy as np
    import torch

    _configure_determinism()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    if torch.distributed.get_world_size() != 4:
        raise RuntimeError("the frozen public contract requires exactly four ranks")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _runtime_receipt():
    import torch

    from npa.workbench.flex_pi.training_artifacts import sha256_file

    cudnn_libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcudnn" in line
        }
    )

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_libraries": [
            {"name": Path(library).name, "sha256": sha256_file(library)}
            for library in cudnn_libraries
        ],
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "compute_capability": torch.cuda.get_device_capability(),
        "world_size": torch.distributed.get_world_size(),
        "nodes": training_topology()["nodes"],
        "gpus_per_node": training_topology()["gpus_per_node"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nccl_environment": _nccl_environment(),
    }


def _configure_determinism():
    import torch
    import torch.utils.deterministic

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "deterministic training requires the pinned cuBLAS workspace"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.utils.deterministic.fill_uninitialized_memory = True


def _nccl_environment():
    return {
        key: os.environ.get(key)
        for key in (
            "NCCL_CUMEM_ENABLE",
            "NCCL_CUMEM_HOST_ENABLE",
            "NCCL_NVLS_ENABLE",
            "NCCL_IB_DISABLE",
        )
    }


def _worker_environment(assets, root):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ACCELERATE_") or key in {
            "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            env.pop(key, None)
    env["DIFFSYNTH_MODEL_BASE_PATH"] = str(assets / "models")
    env["FLEX_PI_DINO_CHECKPOINT"] = str(
        assets / "models/timm/vit_base_patch16_dinov3.lvd1689m/model.safetensors"
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["NCCL_DEBUG_FILE"] = str(root / "nccl.%p.log")
    return env


def _parent_main(request_path):
    from npa.workbench.flex_pi.training_assets import prepare_assets
    from npa.workbench.flex_pi.training_multinode import phase_barrier

    plan = json.loads(request_path.read_text())
    root = Path(plan["work_directory"])
    assets = Path(plan["asset_directory"])
    root.mkdir(parents=True, exist_ok=True)
    phase_barrier("inputs-ready")
    capability = _capability_preflight(root, assets)
    started = time.perf_counter()
    receipt = prepare_assets(assets)
    receipt["preparation_seconds"] = time.perf_counter() - started
    plan["configuration"] = _configuration(plan, root, assets)
    from npa.workbench.flex_pi.training_profiler_runtime import (
        prepare_profiler_environment,
    )

    environment, profiler = prepare_profiler_environment(
        capability,
        plan["execution"]["mode"],
        root / "profiler-runtime",
        _worker_environment(assets, root),
    )
    if profiler is not None:
        plan["profiler_runtime"] = profiler
    phase_barrier("assets-ready")
    rank_request = root / "rank-request.json"
    rank_request.write_text(json.dumps(plan))
    rank_request.chmod(0o600)
    command = rank_command("--rank-request", rank_request)
    subprocess.run(command, check=True, env=environment)
    if training_topology()["node_rank"] == 0:
        _finalize_phase(plan, root, assets, receipt, capability, request_path)


def _finalize_phase(plan, root, assets, receipt, capability, request_path):
    from npa.workbench.flex_pi.training_metrics import (
        PROFILE_UPDATES,
        summarize_measurements,
    )

    result = json.loads((root / "phase-result.json").read_text())
    if plan["execution"]["mode"] not in {"resume", "qualify"}:
        rows = [
            json.loads(line)
            for line in (root / "measurements.jsonl").read_text().splitlines()
        ]
        if plan["execution"]["mode"] == "train":
            rows = rows[:1205]
        elif plan["execution"]["mode"] == "profile-resume":
            rows = rows[:PROFILE_UPDATES]
        result["throughput"] = summarize_measurements(rows)
    result.update(receipt)
    result["capability"] = capability
    result.update(_workload_identity(plan, root, assets))
    config_bytes = json.dumps(plan["configuration"], sort_keys=True).encode()
    result["resolved_configuration_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    result["training_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("training*.py"))
    }
    result["non_comparable_to_reference"] = True
    result["reference_benchmark_beaten"] = False
    (request_path.parent / "result.json").write_text(
        json.dumps(result, allow_nan=False)
    )


def _capability_preflight(root, assets):
    output = root / "capability.json"
    command = rank_command("--capability-output", output)
    subprocess.run(command, check=True, env=_worker_environment(assets, root))
    return json.loads(output.read_text())


def _capability_rank_main(output):
    import torch

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    topology = training_topology()
    torch.cuda.set_device(local_rank)
    if rank == 0:
        print(
            json.dumps(
                {
                    "capability_phase": "initializing",
                    "nccl_environment": _nccl_environment(),
                }
            ),
            flush=True,
        )
    torch.distributed.init_process_group("nccl")
    if (
        torch.distributed.get_world_size() != 4
        or torch.cuda.device_count() != topology["gpus_per_node"]
    ):
        raise RuntimeError(
            "visible GPU allocation differs from the frozen four-rank topology"
        )
    peers = _verify_rank_placement()
    tensor = torch.empty(4 * 1024 * 1024, device=f"cuda:{local_rank}")
    started = time.perf_counter()
    for _ in range(3):
        tensor.fill_(rank + 1)
        torch.distributed.all_reduce(tensor)
        if not torch.all(tensor == 10):
            raise RuntimeError("four-rank NCCL collective produced incorrect values")
    torch.cuda.synchronize()
    if local_rank == 0:
        result = _collective_receipt(tensor, started, peers)
        output.write_text(json.dumps(result))
        print(json.dumps({"capability_preflight": result}), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _verify_rank_placement():
    import socket
    import torch

    topology = training_topology()
    local = {
        "rank": torch.distributed.get_rank(),
        "node_rank": topology["node_rank"],
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "host_sha256": hashlib.sha256(
            os.environ.get("NPA_FLEX_PI_HOST_ID", socket.gethostname()).encode()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "visible_gpus": torch.cuda.device_count(),
    }
    peers = [None] * 4
    torch.distributed.all_gather_object(peers, local)
    actual = {(row["node_rank"], row["local_rank"]) for row in peers}
    expected = {
        (node, rank)
        for node in range(topology["nodes"])
        for rank in range(topology["gpus_per_node"])
    }
    if (
        actual != expected
        or len({row["host_sha256"] for row in peers}) != topology["nodes"]
    ):
        raise RuntimeError("the four ranks do not occupy the requested distinct hosts")
    return peers


def _collective_receipt(tensor, started, peers):
    import torch

    return {
        "world_size": 4,
        "nodes": training_topology()["nodes"],
        "gpus_per_node": training_topology()["gpus_per_node"],
        "rank_placement": peers,
        "collectives_verified": 3,
        "bytes_per_rank_per_collective": tensor.numel() * tensor.element_size(),
        "seconds": time.perf_counter() - started,
        "nccl_version": torch.cuda.nccl.version(),
        "local_peer_access": [
            [
                i == j or torch.cuda.can_device_access_peer(i, j)
                for j in range(torch.cuda.device_count())
            ]
            for i in range(torch.cuda.device_count())
        ],
    }


def _workload_identity(plan, root, assets):
    from npa.workbench.flex_pi.training_artifacts import sha256_file

    configuration = dict(plan["configuration"])
    configuration = _canonical_normalization(configuration)
    for key in (
        "output_dir",
        "resume",
        "num_workers",
        "npa_prefetch_factor",
        "npa_optimizer",
        "wandb",
    ):
        configuration.pop(key, None)
    serialized = json.dumps(configuration, sort_keys=True)
    serialized = serialized.replace(str(assets), "<assets>").replace(
        str(root), "<work>"
    )
    normalization = sha256_file(root / "dataset_stats.json")
    workload = {
        "configuration": json.loads(serialized),
        "normalization_sha256": normalization,
        "source_manifest_sha256": sha256_file(
            Path(__file__).with_name("training_sources.json")
        ),
        "split_sha256": sha256_file(Path(__file__).with_name("training_split.json")),
    }
    payload = json.dumps(workload, sort_keys=True).encode()
    (root / "workload.json").write_bytes(payload)
    return {
        "normalization_sha256": normalization,
        "workload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_normalization(configuration):
    configuration = json.loads(json.dumps(configuration))
    for dataset in ("train", "val"):
        selected = configuration.get("data", {}).get(dataset, {})
        selected["pretrained_norm_stats"] = "<verified-normalization>"
    return configuration


def main():
    """Dispatch a prepared parent process or a torchrun rank.

    Args:
        None; requests are supplied through command-line arguments.
    Returns:
        None; the parent writes a JSON phase result.
    Raises:
        RuntimeError: An input, distributed, numerical or persistence gate fails.
    """
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--rank-request", type=Path)
    mode.add_argument("--capability-output", type=Path)
    mode.add_argument("--distributed-request", type=Path)
    args = parser.parse_args()
    if args.distributed_request:
        from npa.workbench.flex_pi.training_multinode import distributed_main

        distributed_main(args.distributed_request)
    elif args.capability_output:
        _capability_rank_main(args.capability_output)
    elif args.rank_request:
        _rank_main(args.rank_request)
    else:
        _parent_main(args.request)


if __name__ == "__main__":
    main()
