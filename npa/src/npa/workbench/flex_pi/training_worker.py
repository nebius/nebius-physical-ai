"""Vendor-interpreter entrypoint for the fixed public Flex-Pi training workload."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def _configuration(plan, root, assets):
    from hydra import compose, initialize_config_dir
    from flexpi.utils.config_resolvers import register_default_resolvers
    from omegaconf import OmegaConf
    from npa.workbench.flex_pi.training_assets import _hydra_overrides

    register_default_resolvers()
    overrides = _hydra_overrides(assets) + [
        f"output_dir={root}",
        "batch_size=1",
        "gradient_accumulation_steps=24",
        "num_epochs=1",
        "max_steps=null",
        "mixed_precision=bf16",
        "learning_rate=1e-4",
        "weight_decay=0.01",
        "seed=42",
        "model.mot_checkpoint_mixed_attn=true",
        "wandb.enabled=false",
        f"model.action_dit_pretrained_path={assets / 'ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}",
        f"num_workers={plan['execution']['num_workers']}",
        "data.train._target_=npa.workbench.flex_pi.training_engine.ExactTrainingDataset",
        "data.val._target_=npa.workbench.flex_pi.training_engine.ExactTrainingDataset",
        "+npa_optimizer=" + plan["execution"]["optimizer"],
        "+npa_prefetch_factor=" + str(plan["execution"]["prefetch_factor"]),
    ]
    if plan.get("resume_directory"):
        overrides.append("resume=" + plan["resume_directory"])
    with initialize_config_dir(
        config_dir=str(assets / "upstream/configs"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _rank_main(request_path):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from flexpi.runtime import build_datasets
    from flexpi.utils import misc
    from npa.workbench.flex_pi.training_engine import VerifiedTrainer

    plan = json.loads(request_path.read_text())
    root = Path(plan["work_directory"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    if torch.distributed.get_world_size() != 4:
        raise RuntimeError("the frozen public contract requires exactly four ranks")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    misc.register_work_dir(str(root))
    cfg = OmegaConf.create(plan["configuration"])
    start = time.perf_counter()
    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device=f"cuda:{torch.cuda.current_device()}",
    )
    train_ds, val_ds = build_datasets(cfg.data)
    trainer = VerifiedTrainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    initialized = time.perf_counter() - start
    result = trainer.execute(plan["execution"]["mode"])
    result["initialization_seconds"] = initialized
    result["runtime"] = _runtime_receipt()
    if torch.distributed.get_rank() == 0:
        (root / "phase-result.json").write_text(json.dumps(result, allow_nan=False))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _runtime_receipt():
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "compute_capability": torch.cuda.get_device_capability(),
        "world_size": torch.distributed.get_world_size(),
        "nccl_environment": _nccl_environment(),
    }


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
    env["NCCL_DEBUG_FILE"] = str(root / "nccl.%p.log")
    return env


def _parent_main(request_path):
    from npa.workbench.flex_pi.training_assets import prepare_assets

    plan = json.loads(request_path.read_text())
    root = Path(plan["work_directory"])
    assets = Path(plan["asset_directory"])
    root.mkdir(parents=True, exist_ok=True)
    capability = _capability_preflight(root, assets)
    started = time.perf_counter()
    receipt = prepare_assets(assets)
    receipt["preparation_seconds"] = time.perf_counter() - started
    plan["configuration"] = _configuration(plan, root, assets)
    rank_request = root / "rank-request.json"
    rank_request.write_text(json.dumps(plan))
    rank_request.chmod(0o600)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--max-restarts=0",
        "--module",
        "npa.workbench.flex_pi.training_worker",
        "--rank-request",
        str(rank_request),
    ]
    subprocess.run(command, check=True, env=_worker_environment(assets, root))
    _finalize_phase(plan, root, assets, receipt, capability, request_path)


def _finalize_phase(plan, root, assets, receipt, capability, request_path):
    from npa.workbench.flex_pi.training_metrics import summarize_measurements

    result = json.loads((root / "phase-result.json").read_text())
    if plan["execution"]["mode"] != "resume":
        rows = [
            json.loads(line)
            for line in (root / "measurements.jsonl").read_text().splitlines()
        ]
        if plan["execution"]["mode"] == "train":
            rows = rows[:1205]
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
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        "--max-restarts=0",
        "--module",
        "npa.workbench.flex_pi.training_worker",
        "--capability-output",
        str(output),
    ]
    subprocess.run(command, check=True, env=_worker_environment(assets, root))
    return json.loads(output.read_text())


def _capability_rank_main(output):
    import torch

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
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
    if torch.distributed.get_world_size() != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError(
            "capability preflight requires one node with exactly four visible GPUs"
        )
    tensor = torch.empty(4 * 1024 * 1024, device=f"cuda:{rank}")
    started = time.perf_counter()
    for _ in range(3):
        tensor.fill_(rank + 1)
        torch.distributed.all_reduce(tensor)
        if not torch.all(tensor == 10):
            raise RuntimeError("four-rank NCCL collective produced incorrect values")
    torch.cuda.synchronize()
    if rank == 0:
        result = _collective_receipt(tensor, started)
        output.write_text(json.dumps(result))
        print(json.dumps({"capability_preflight": result}), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _collective_receipt(tensor, started):
    import torch

    return {
        "world_size": 4,
        "collectives_verified": 3,
        "bytes_per_rank_per_collective": tensor.numel() * tensor.element_size(),
        "seconds": time.perf_counter() - started,
        "nccl_version": torch.cuda.nccl.version(),
        "peer_access": [
            [i == j or torch.cuda.can_device_access_peer(i, j) for j in range(4)]
            for i in range(4)
        ],
    }


def _workload_identity(plan, root, assets):
    from npa.workbench.flex_pi.training_artifacts import sha256_file

    configuration = dict(plan["configuration"])
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
    args = parser.parse_args()
    if args.capability_output:
        _capability_rank_main(args.capability_output)
    elif args.rank_request:
        _rank_main(args.rank_request)
    else:
        _parent_main(args.request)


if __name__ == "__main__":
    main()
