---
name: ray-train-synthetic
description: Run or review the native Ray Train synthetic distributed CUDA reference, its S3 checkpoints, optimizer recovery, and factual Rerun metrics.
---

# Native Ray Train reference

Use [the reference guide](../../../npa/workflows/workbench/ray-train-synthetic/README.md).
This is a guarded application example: native Ray Jobs delivers source and owns
submission/status/logs/stop; SkyPilot owns the development hosts and service task.
It adds no NPA Jobs wrapper, controller, CLI group, image or workflow catalog.
Use `npa.workflow` for production composition beyond this single application.

Read `health-preflight`, `gpu-selection`, `skypilot-workflows`,
`third-party-eula-preflight`, and `teardown-and-cost` before live operation.
Preflight the selected project's exact workload bucket and prefix; do not use a
different project's Terraform-state or trajectory bucket as training storage.

The pins are application Ray 2.58.0 / Train V2, Torch 2.13.0+cu130 in the
digest-pinned upstream PyTorch image, Rerun 0.31.4 and Pillow 12.3.0. The driver and every new
or restarted worker enforce the Torch pin at run time. Keep application Ray
separate from SkyPilot's management environment and reserved ports. Pure Python
source changes use native Jobs `--working-dir`; native ABI changes require a
compatible prepared environment. There is no model or external dataset download.

Preparation uses the pinned image's root account and requires a fresh owned
mode-0700 `/opt/npa-ray-train` below an `/opt` without shared write access.
Keep its environment, exports, preparation receipt, and application Ray temporary
files inside that directory. Existing directories and symlinks fail closed;
preserve failed-attempt evidence and use fresh hosting pods rather than clearing
or adopting an unknown runtime. The service validates ownership and permissions
before it starts and scopes `RAY_TMPDIR` to this application directory.

Use two B200 hosts with one GPU each for the shipped profile. Require actual
rank/world-size, CUDA device and physical-host evidence before calling a result
multi-node. `SPREAD` and a requested GPU count alone are not placement evidence.

Ray Train V2 recovers through the same `RunConfig(name, storage_path)` and worker
`get_checkpoint()`. Do not use V1 `resume_from_checkpoint`, `restore()` or
`trainer_resources`. Every worker must call `report` equally often; only rank
zero uploads checkpoint bytes. Before injected failure, wait for native
`CheckpointConsistencyMode.COMMITTED`, then prove every worker resumed at the
next optimizer step with model and momentum state restored.

S3 credentials belong in the private hosting environment on every Ray host.
Pass no keys to `S3FileSystem` itself: its serialization otherwise carries them.
Keep credentials out of source delivery, Jobs runtime-env JSON and reports.
Store checkpoints with the native Arrow filesystem; upload the final export
with its checksum manifest last and require read-after-write verification.

Validate all optimizer steps and scalar values decoded from `metrics.rrd`
against `metrics.json` using `inspect_results.py`. Independently run
`rerun rrd verify` and `rerun rrd print -vv`. After compute teardown, download
again from S3 and reload the native model/optimizer checkpoint. Synthetic
regression proves training execution and recovery, not robot-policy quality.

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_ray_train_synthetic.py -q
```

The opt-in live suite is `npa/tests/e2e/test_ray_train_synthetic_live.py`.
Its private configuration selects an already-preflighted isolated Ray Jobs
endpoint and workload S3 prefix. Cancel exact Jobs before the Sky service task
and named development cluster; retain shared infrastructure and verified artifacts.
