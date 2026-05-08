#!/usr/bin/env python3
"""profile_train.py — Training profiler for LeRobot policies.

Three measurement modes:

  wallclock   No torch.profiler, no cuda.synchronize between stages.
              CPU stages (dataloader, data_transfer) use perf_counter.
              GPU stages (forward, backward, optimizer) use cuda.Event pairs.
              ``throughput_steps_per_sec`` (from total synchronized wall time)
              is the authoritative throughput metric.

  profiler    Full torch.profiler with record_function labels and
              cuda.synchronize at every stage boundary.  Produces Chrome
              traces and per-step CSV.  Higher overhead — use for
              diagnosing *where* time goes, not *how much* time.

  inference   Forward-only latency at batch_size=1 (forced regardless of
              --batch_size).  No backward, no optimizer.  Uses cuda.Event
              for GPU timing.  Keeps policy.train() (ACT VAE fails in
              eval mode).

Integrated into the npa toolchain:
  - Called by: npa workbench lerobot profile-train --mode wallclock|profiler|inference
  - Called by: benchmark_policies.sh (RUN_TORCH_PROFILE phase, --mode=profiler)
  - Synced to VMs at: /opt/lerobot/profile_train.py

Outputs (in --output_dir):
  wallclock mode:
    - wallclock_results.json  Structured per-step + per-stage timing
    - wallclock_summary.txt   Human-readable summary

  profiler mode:
    - chrome_trace.json       Full Chrome trace for chrome://tracing
    - stage_breakdown.csv     Per-step (step, stage, cpu_time_ms, cuda_time_ms)
    - stage_summary.txt       Percentage of total step time per stage

  inference mode:
    - inference_results.json  Per-sample forward latency statistics
    - inference_summary.txt   Human-readable summary

Timing methodology notes:
  - CPU stages use time.perf_counter() — CUDA events only measure GPU-stream
    progress, so host-side dataloader wait and CPU preprocessing are invisible
    to cuda.Event.  perf_counter captures the actual host-side wait.
  - GPU stages use cuda.Event pairs read after a single final synchronize —
    no pipeline serialization during measurement.
  - ``cpu_enqueue_ms`` is perf_counter around the full step (CPU enqueue time,
    may undercount outstanding GPU work at step boundaries).
  - ``gpu_step_ms`` is the sum of GPU-stage cuda.Event times per step.
  - ``throughput_steps_per_sec`` = measured_steps / total_synchronized_wall_time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch


STAGES = [
    "dataloader_batch_fetch",
    "data_transfer_to_gpu",
    "forward_pass",
    "backward_pass",
    "optimizer_step",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile LeRobot training — wallclock or torch.profiler mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["wallclock", "profiler", "inference"], default="wallclock",
                   help="Measurement mode: wallclock (training throughput), profiler (torch stage breakdown), inference (forward-only latency).")
    p.add_argument("--policy_type", required=True)
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true", help="Apply torch.compile to the policy model.")
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    # Shared: warmup steps to discard (both modes use this)
    p.add_argument("--warmup_steps", type=int, default=10,
                   help="Warmup steps to run before timed measurement. "
                        "Automatically raised to 50 when --compile is set.")
    # Profiler-only settings
    p.add_argument("--skip_first", type=int, default=10,
                   help="(profiler mode) Profiler schedule skip_first.")
    p.add_argument("--warmup", type=int, default=5,
                   help="(profiler mode) Profiler schedule warmup.")
    p.add_argument("--active", type=int, default=50,
                   help="(profiler mode) Profiler schedule active.")
    return p.parse_args()


# ── Shared setup ─────────────────────────────────────────────────────────


def build_training_components(args):
    """Set up dataset, dataloader, policy, optimizer using lerobot-train's
    own config resolution — same code path as ``lerobot-train`` CLI."""
    import importlib
    import shutil
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.optim.factory import make_optimizer_and_scheduler

    out_path = Path(args.output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)

    policy_type = args.policy_type
    config_module_name = f"lerobot.policies.{policy_type}.configuration_{policy_type}"
    try:
        config_module = importlib.import_module(config_module_name)
    except ModuleNotFoundError:
        raise RuntimeError(f"Unknown policy type: {policy_type}")

    config_cls = None
    from lerobot.configs.policies import PreTrainedConfig
    for attr_name in dir(config_module):
        obj = getattr(config_module, attr_name)
        if not (isinstance(obj, type) and issubclass(obj, PreTrainedConfig)
                and obj is not PreTrainedConfig):
            continue
        try:
            if obj().type == policy_type:
                config_cls = obj
                break
        except Exception:
            continue
    if config_cls is None:
        raise RuntimeError(f"Could not find config class for policy type: {policy_type}")

    policy_cfg = config_cls()
    policy_cfg.push_to_hub = False

    from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.dataset_repo_id),
        policy=policy_cfg,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=str(args.output_dir),
        eval_freq=0,
        save_checkpoint=False,
        eval=EvalConfig(),
        wandb=WandBConfig(),
    )
    cfg.validate()

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = make_dataset(cfg)

    print(f"Creating policy: {args.policy_type}")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    device = torch.device(args.device)
    policy = policy.to(device)

    if args.compile:
        print("Applying torch.compile...")
        policy = torch.compile(policy)

    policy.train()

    print("Creating preprocessor...")
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=None,
        dataset_stats=dataset.meta.stats,
    )

    print("Creating optimizer and scheduler...")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    has_sampler = hasattr(cfg.policy, "drop_n_last_frames")
    if has_sampler:
        from lerobot.datasets.sampler import EpisodeAwareSampler
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        shuffle=shuffle, sampler=sampler,
        pin_memory=device.type == "cuda", drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    return dataset, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _resolve_grad_clip(args, cfg):
    if hasattr(cfg, "optimizer") and hasattr(cfg.optimizer, "grad_clip_norm"):
        return cfg.optimizer.grad_clip_norm
    return args.grad_clip_norm


# ── Wallclock mode ───────────────────────────────────────────────────────


CPU_STAGES = ["dataloader_batch_fetch", "data_transfer_to_gpu"]
GPU_STAGES = ["forward_pass", "backward_pass", "optimizer_step"]


def run_wallclock(args, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg):
    """Measure per-step wall time and per-stage time without serialization.

    Timing methodology (no cuda.synchronize between stages or steps):
      - CPU stages (dataloader_batch_fetch, data_transfer_to_gpu): ``time.perf_counter()``
        pairs, because these stages are host-bound and invisible to CUDA events.
      - GPU stages (forward_pass, backward_pass, optimizer_step): ``cuda.Event`` pairs
        recorded at stage boundaries, read once at the end after a single
        ``cuda.synchronize()``.  Events measure GPU-stream elapsed time without
        blocking the CPU pipeline.
      - Per-step: ``cpu_enqueue_ms`` is the perf_counter time for the full step
        (CPU-side enqueue time, may undercount outstanding GPU work).
        ``gpu_step_ms`` is the sum of GPU-stage cuda.Event times per step.
        ``throughput_steps_per_sec`` is derived from total synchronized wall time
        and is the authoritative throughput metric.
    """
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grad_clip = _resolve_grad_clip(args, cfg)
    dl_iter = cycle(dataloader)
    warmup = args.warmup_steps
    measure = args.steps - warmup
    if measure <= 0:
        raise ValueError(f"steps ({args.steps}) must be > warmup_steps ({warmup})")

    print(f"\nWallclock: {warmup} warmup + {measure} measured steps\n")

    # ── Warmup (untimed) ──
    for i in range(warmup):
        batch = preprocessor(next(dl_iter))
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            loss, _ = policy.forward(batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler:
            lr_scheduler.step()

    torch.cuda.synchronize()

    # ── Measured steps ──
    # CPU stages: perf_counter pairs (host-bound work invisible to CUDA events)
    cpu_stage_ms: dict[str, list[float]] = {s: [] for s in CPU_STAGES}

    # GPU stages: cuda.Event pairs (4 boundary events per step: pre-forward,
    # post-forward, post-backward, post-optimizer)
    n_gpu_boundaries = len(GPU_STAGES) + 1
    gpu_events = [[torch.cuda.Event(enable_timing=True) for _ in range(n_gpu_boundaries)]
                  for _ in range(measure)]
    cpu_enqueue_times: list[float] = []

    t_total_start = time.perf_counter()

    for i in range(measure):
        t_step_start = time.perf_counter()

        # CPU stage: dataloader_batch_fetch (host-bound)
        t_dl_start = time.perf_counter()
        batch = next(dl_iter)
        cpu_stage_ms["dataloader_batch_fetch"].append((time.perf_counter() - t_dl_start) * 1000)

        # CPU stage: data_transfer_to_gpu (host→device copy)
        t_xfer_start = time.perf_counter()
        batch = preprocessor(batch)
        cpu_stage_ms["data_transfer_to_gpu"].append((time.perf_counter() - t_xfer_start) * 1000)

        ev = gpu_events[i]

        # GPU stage: forward_pass
        ev[0].record()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            loss, _ = policy.forward(batch)

        # GPU stage: backward_pass
        ev[1].record()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)

        # GPU stage: optimizer_step
        ev[2].record()
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler:
            lr_scheduler.step()

        ev[3].record()

        cpu_enqueue_times.append(time.perf_counter() - t_step_start)

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_total_start
            print(f"  step {i + 1}/{measure}  ({elapsed:.1f}s)")

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total_start

    # ── Read GPU-stage cuda event timings ──
    gpu_stage_ms: dict[str, list[float]] = {s: [] for s in GPU_STAGES}
    for i in range(measure):
        ev = gpu_events[i]
        for j, stage in enumerate(GPU_STAGES):
            try:
                gpu_stage_ms[stage].append(ev[j].elapsed_time(ev[j + 1]))
            except RuntimeError:
                gpu_stage_ms[stage].append(0.0)

    # ── Per-step GPU time (sum of GPU stages from cuda events) ──
    gpu_step_ms_list: list[float] = []
    for i in range(measure):
        total = sum(gpu_stage_ms[s][i] for s in GPU_STAGES)
        gpu_step_ms_list.append(total)

    # ── Compute statistics ──
    def stats(vals):
        s = sorted(vals)
        n = len(s)
        return {
            "avg": sum(s) / n,
            "p50": s[n // 2],
            "p90": s[int(n * 0.9)],
            "min": s[0],
            "max": s[-1],
        }

    cpu_enqueue_stats = stats([t * 1000 for t in cpu_enqueue_times])  # ms
    gpu_step_stats = stats(gpu_step_ms_list)

    # Merge CPU and GPU stage stats under a unified dict
    all_stage_stats: dict[str, dict] = {}
    for s in CPU_STAGES:
        all_stage_stats[s] = {"timing": "cpu", **stats(cpu_stage_ms[s])}
    for s in GPU_STAGES:
        all_stage_stats[s] = {"timing": "gpu", **stats(gpu_stage_ms[s])}

    total_stage_avg = sum(st["avg"] for st in all_stage_stats.values())

    results = {
        "mode": "wallclock",
        "policy_type": args.policy_type,
        "dataset_repo_id": args.dataset_repo_id,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "compiled": args.compile,
        "warmup_steps": warmup,
        "measured_steps": measure,
        "wall_time_seconds": round(t_total, 3),
        "throughput_steps_per_sec": round(measure / t_total, 2),
        "cpu_enqueue_ms": {k: round(v, 3) for k, v in cpu_enqueue_stats.items()},
        "gpu_step_ms": {k: round(v, 3) for k, v in gpu_step_stats.items()},
        "stages": {
            s: {
                "timing_method": all_stage_stats[s]["timing"],
                "avg": round(all_stage_stats[s]["avg"], 3),
                "p50": round(all_stage_stats[s]["p50"], 3),
                "p90": round(all_stage_stats[s]["p90"], 3),
                "min": round(all_stage_stats[s]["min"], 3),
                "max": round(all_stage_stats[s]["max"], 3),
            }
            for s in STAGES
        },
        "stages_pct_of_step": {
            s: round(all_stage_stats[s]["avg"] / total_stage_avg * 100, 1) if total_stage_avg > 0 else 0
            for s in STAGES
        },
    }

    # ── Write JSON ──
    json_path = output_dir / "wallclock_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Write human-readable summary ──
    summary_path = output_dir / "wallclock_summary.txt"
    avg_step_ms = t_total / measure * 1000
    lines = [
        f"Wallclock Summary — {args.policy_type} on {args.dataset_repo_id}",
        "=" * 78,
        f"Device: {device} | Measured: {measure} steps (after {warmup} warmup)",
        f"Batch size: {args.batch_size} | num_workers: {args.num_workers}"
        + (f" | compiled: True" if args.compile else ""),
        f"Wall time: {t_total:.2f}s | Throughput: {measure / t_total:.2f} step/s"
        f" | Avg step: {avg_step_ms:.2f}ms",
        "",
        f"GPU step time (sum of GPU stages, ms):  avg={gpu_step_stats['avg']:.2f}"
        f"  p50={gpu_step_stats['p50']:.2f}  p90={gpu_step_stats['p90']:.2f}",
        f"CPU enqueue time (ms):  avg={cpu_enqueue_stats['avg']:.2f}"
        f"  p50={cpu_enqueue_stats['p50']:.2f}  (excludes outstanding GPU work)",
        "",
        f"{'Stage':<28} {'Method':>6} {'avg ms':>10} {'p50':>10} {'p90':>10} {'% of step':>10}",
        "-" * 78,
    ]
    for s in STAGES:
        st = all_stage_stats[s]
        pct = results["stages_pct_of_step"][s]
        method = st["timing"]
        lines.append(f"{s:<28} {method:>6} {st['avg']:>10.2f} {st['p50']:>10.2f} {st['p90']:>10.2f} {pct:>9.1f}%")
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<28} {'':>6} {total_stage_avg:>10.2f}")
    lines.append("")

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"  Results: {json_path}")
    print(f"  Summary: {summary_path}")


# ── Profiler mode ────────────────────────────────────────────────────────


def run_profiler(args, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg):
    """Run training loop with torch.profiler record_function labels.
    Inserts cuda.synchronize at stage boundaries — accurate stage attribution
    but serializes the GPU pipeline (higher overhead than wallclock mode)."""
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grad_clip = _resolve_grad_clip(args, cfg)

    trace_path = output_dir / "chrome_trace.json"
    csv_path = output_dir / "stage_breakdown.csv"
    summary_path = output_dir / "stage_summary.txt"

    total_profiled = args.skip_first + args.warmup + args.active
    if total_profiled > args.steps:
        args.active = max(1, args.steps - args.skip_first - args.warmup)

    prof_schedule = schedule(
        skip_first=args.skip_first, wait=0,
        warmup=args.warmup, active=args.active, repeat=1,
    )
    dl_iter = cycle(dataloader)

    def trace_handler(prof):
        prof.export_chrome_trace(str(trace_path))
        print(f"  Chrome trace exported: {trace_path}")

    print(f"\nProfiler: {args.steps} steps "
          f"(skip={args.skip_first}, warmup={args.warmup}, active={args.active})...\n")
    t0 = time.time()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule, on_trace_ready=trace_handler,
        record_shapes=False, profile_memory=True, with_stack=False,
    ) as prof:
        for step in range(args.steps):
            with record_function("dataloader_batch_fetch"):
                batch = next(dl_iter)
            with record_function("data_transfer_to_gpu"):
                batch = preprocessor(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
            with record_function("forward_pass"):
                with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                    loss, _ = policy.forward(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
            with record_function("backward_pass"):
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                if device == "cuda":
                    torch.cuda.synchronize()
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler:
                    lr_scheduler.step()
                if device == "cuda":
                    torch.cuda.synchronize()
            prof.step()
            if (step + 1) % 20 == 0 or step == 0:
                print(f"  step {step + 1}/{args.steps}  ({time.time() - t0:.1f}s)")

    wall_time = time.time() - t0
    print(f"\nTraining complete: {args.steps} steps in {wall_time:.1f}s")
    print("Extracting stage breakdown from profiler events...")

    profiled_start = args.skip_first + args.warmup

    events = prof.key_averages()
    stage_totals = {s: {"cpu_time_ms": 0.0, "cuda_time_ms": 0.0, "count": 0} for s in STAGES}
    for evt in events:
        if evt.key in STAGES:
            stage_totals[evt.key]["cpu_time_ms"] = evt.cpu_time_total / 1000.0
            stage_totals[evt.key]["cuda_time_ms"] = evt.device_time_total / 1000.0
            stage_totals[evt.key]["count"] = evt.count

    raw_events = prof.events()
    stage_invocations = {s: [] for s in STAGES}
    for evt in raw_events:
        if evt.name in STAGES:
            stage_invocations[evt.name].append({
                "cpu_time_ms": evt.cpu_time_total / 1000.0,
                "cuda_time_ms": evt.device_time_total / 1000.0,
            })

    num_profiled = min(
        args.active,
        min(len(v) for v in stage_invocations.values()) if all(stage_invocations.values()) else 0,
    )

    per_step_records = []
    for i in range(num_profiled):
        for stage in STAGES:
            inv = stage_invocations[stage]
            if i < len(inv):
                per_step_records.append({
                    "step": profiled_start + i, "stage": stage,
                    "cpu_time_ms": round(inv[i]["cpu_time_ms"], 3),
                    "cuda_time_ms": round(inv[i]["cuda_time_ms"], 3),
                })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "stage", "cpu_time_ms", "cuda_time_ms"])
        w.writeheader()
        w.writerows(per_step_records)
    print(f"  Stage breakdown CSV: {csv_path} ({len(per_step_records)} rows)")

    total_cpu = sum(t["cpu_time_ms"] for t in stage_totals.values())
    total_cuda = sum(t["cuda_time_ms"] for t in stage_totals.values())

    lines = [
        f"Stage Breakdown Summary — {args.policy_type} on {args.dataset_repo_id}",
        "=" * 78,
        f"Device: {device} | Steps: {args.steps} | Profiled: {num_profiled}",
        f"Batch size: {args.batch_size} | num_workers: {args.num_workers}",
        f"Wall time: {wall_time:.1f}s | Throughput: {args.steps / wall_time:.2f} step/s",
        "",
        f"{'Stage':<28} {'CPU ms':>10} {'CPU %':>8} {'CUDA ms':>10} {'CUDA %':>8} {'Count':>6}",
        "-" * 78,
    ]
    for stage in STAGES:
        t = stage_totals[stage]
        cpu_pct = (t["cpu_time_ms"] / total_cpu * 100) if total_cpu > 0 else 0
        cuda_pct = (t["cuda_time_ms"] / total_cuda * 100) if total_cuda > 0 else 0
        lines.append(
            f"{stage:<28} {t['cpu_time_ms']:>10.1f} {cpu_pct:>7.1f}% "
            f"{t['cuda_time_ms']:>10.1f} {cuda_pct:>7.1f}% {int(t['count']):>6}"
        )
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<28} {total_cpu:>10.1f} {'100.0':>7}% {total_cuda:>10.1f} {'100.0':>7}%")

    if per_step_records:
        lines.append("")
        lines.append(f"Per-Step Averages (from {num_profiled} profiled steps):")
        lines.append("-" * 78)
        for stage in STAGES:
            rows = [r for r in per_step_records if r["stage"] == stage]
            if rows:
                avg_cpu = sum(r["cpu_time_ms"] for r in rows) / len(rows)
                avg_cuda = sum(r["cuda_time_ms"] for r in rows) / len(rows)
                lines.append(f"  {stage:<26} CPU: {avg_cpu:>8.2f} ms/step  CUDA: {avg_cuda:>8.2f} ms/step")

    summary_text = "\n".join(lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  Stage summary: {summary_path}\n")
    print(summary_text)


# ── Inference mode ────────────────────────────────────────────────────────


def run_inference(args, dataloader, policy, preprocessor, cfg):
    """Measure single-sample forward-pass latency. No backward, no optimizer.
    Uses cuda.Event for precise GPU timing."""
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dl_iter = cycle(dataloader)
    warmup = args.warmup_steps
    measure = args.steps

    # Keep train mode — some policies (ACT with VAE) fail in eval mode on forward()
    policy.train()

    print(f"\nInference: {warmup} warmup + {measure} measured forward passes (batch_size={args.batch_size})\n")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            batch = preprocessor(next(dl_iter))
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                policy.forward(batch)

    torch.cuda.synchronize()

    # Measured passes — cuda.Event per forward
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]

    with torch.no_grad():
        for i in range(measure):
            batch = preprocessor(next(dl_iter))
            start_events[i].record()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                policy.forward(batch)
            end_events[i].record()

    torch.cuda.synchronize()

    latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(measure)]
    latencies.sort()
    n = len(latencies)

    results = {
        "mode": "inference",
        "policy_type": args.policy_type,
        "dataset_repo_id": args.dataset_repo_id,
        "device": device,
        "batch_size": args.batch_size,
        "compiled": args.compile,
        "warmup_steps": warmup,
        "measured_steps": measure,
        "mean_ms": round(sum(latencies) / n, 3),
        "p50_ms": round(latencies[n // 2], 3),
        "p90_ms": round(latencies[int(n * 0.90)], 3),
        "p95_ms": round(latencies[int(n * 0.95)], 3),
        "p99_ms": round(latencies[int(n * 0.99)], 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
    }

    json_path = output_dir / "inference_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = (
        f"Inference Latency — {args.policy_type} on {args.dataset_repo_id}\n"
        f"{'=' * 60}\n"
        f"Device: {device} | Batch: {args.batch_size} | Compiled: {args.compile}\n"
        f"Measured: {measure} forward passes (after {warmup} warmup)\n\n"
        f"  mean={results['mean_ms']:.2f}ms  p50={results['p50_ms']:.2f}ms  "
        f"p95={results['p95_ms']:.2f}ms  p99={results['p99_ms']:.2f}ms\n"
        f"  min={results['min_ms']:.2f}ms  max={results['max_ms']:.2f}ms\n"
    )
    summary_path = output_dir / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(summary)
    print(f"  Results: {json_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA not available", file=sys.stderr)
        sys.exit(1)

    # Inference mode measures single-sample forward latency — force batch_size=1.
    if args.mode == "inference" and args.batch_size != 1:
        print(f"  Note: inference mode overrides batch_size={args.batch_size} → 1")
        args.batch_size = 1

    # torch.compile triggers lazy graph compilation on the first few
    # forward/backward calls.  The default 10 warmup steps is too few for
    # full stabilization — compilations and recompilations can extend well
    # past step 10.  Enforce a minimum of 50 warmup steps when compiling.
    COMPILE_MIN_WARMUP = 50
    if args.compile and args.warmup_steps < COMPILE_MIN_WARMUP:
        print(f"  Note: --compile requires ≥{COMPILE_MIN_WARMUP} warmup steps"
              f" (was {args.warmup_steps})")
        args.warmup_steps = COMPILE_MIN_WARMUP

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    print(f"profile_train.py — mode={args.mode}" + (" [compiled]" if args.compile else ""))
    print(f"  {args.policy_type} on {args.dataset_repo_id}")
    print(f"  steps={args.steps} batch_size={args.batch_size} num_workers={args.num_workers}")
    print()

    dataset, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg = \
        build_training_components(args)

    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  params: {num_trainable:,} trainable / {num_params:,} total\n")

    if args.mode == "wallclock":
        run_wallclock(args, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg)
    elif args.mode == "inference":
        run_inference(args, dataloader, policy, preprocessor, cfg)
    else:
        run_profiler(args, dataloader, policy, optimizer, lr_scheduler, preprocessor, cfg)


if __name__ == "__main__":
    main()
