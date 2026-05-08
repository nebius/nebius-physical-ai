# Nebius Physical AI — CLAUDE.md

## Project Overview

Benchmarking and deployment tooling for LeRobot robotics policy training across Nebius cloud GPU types (B300, H200, L40S, RTX PRO 6000). Two systems work together:

1. **NPA CLI** (`npa`) — Typer-based Python CLI that orchestrates everything from your laptop
2. **Research scripts** — Shell/Python scripts deployed to remote VMs that do the actual GPU work

## Repository Layout

```
npa/                                  # NPA CLI package (pip-installable)
  src/npa/
    cli/
      main.py                         # Entry point: npa workbench / adapter / workflow
      workbench/lerobot.py            # All lerobot subcommands (deploy, benchmark, profile-train, etc.)
    clients/
      config.py                       # ~/.npa/config.yaml resolution
      ssh.py                          # SSHClient wrapper
      http.py                         # HTTP client for PolicyServer
      nebius.py                       # Nebius cloud bootstrap (IAM, S3, service accounts)
      storage.py                      # S3 storage client
    deploy/
      provisioner.py                  # Terraform init/apply/destroy/outputs
      configurator.py                 # App-level deploy (install lerobot, deploy server, health check)
      terraform/                      # Bundled TF files (main.tf, variables.tf, outputs.tf, cloud_init.yaml.tpl)

research/lerobot-deploy/              # Research scripts (synced to VMs at /opt/lerobot/)
  training/
    profile_train.py                  # Per-step profiler: wallclock, profiler, inference modes
    benchmark_policies.sh             # Full benchmark runner (GPU util sampling, scaling, memory)
    benchmark_metrics.py              # nvidia-smi polling, CSV sample collection, summary stats
    train.sh                          # Thin wrapper around lerobot-train with S3 upload on exit
    eval.sh                           # Thin wrapper around lerobot-eval
    validate_policies.sh              # Smoke test: train+eval ACT, Diffusion, SmolVLA on LIBERO-10
    benchmark_python/sitecustomize.py # Python startup hook for benchmark isolation
  terraform/                          # Research-specific TF (standalone deployments)
    main.tf                           # Provisions VM + syncs research scripts via null_resource
  environment.sh                      # Env var definitions
  s3_sync.py                          # S3 upload/download utility
```

## Config

Workbench configs live at `~/.npa/config.yaml`. Structure:

```yaml
projects:
  <region-alias>:
    project_id: project-...
    tenant_id: tenant-...
    region: <nebius-region>
    workbenches:
      <name>:
        gpu_platform: gpu-h200-sxm
        gpu_preset: 1gpu-16vcpu-200gb
        endpoint: http://<ip>:8080
        ssh: {host: <ip>, user: ubuntu, key_path: ~/.ssh/id_ed25519}
        storage: {checkpoint_bucket: s3://..., endpoint_url: https://...}
default_project: eu-west1
default_workbench: h200
```

Target a specific workbench: `npa workbench lerobot -p <project> -n <name> <command>`

## How to Run Testing

### 1. Deploy a GPU VM

```bash
# First time — requires project-id, tenant-id, region
npa workbench lerobot -p uk-south1 -n b300 deploy \
  --project-id project-... --tenant-id tenant-... --region uk-south1 \
  --gpu-type gpu-b300-sxm --gpu-preset 1gpu-24vcpu-346gb \
  --no-preemptible \
  -v image_family=ubuntu24.04-cuda13.0

# Redeploy app only (infra already exists)
npa workbench lerobot -p uk-south1 -n b300 deploy --skip-infra

# Redeploy infra only (app already installed via cloud-init)
npa workbench lerobot -p uk-south1 -n b300 deploy --skip-app

# Destroy
npa workbench lerobot -p uk-south1 -n b300 deploy --destroy
```

### 2. NPA Benchmark (throughput via lerobot-train)

```bash
npa workbench lerobot -p uk-south1 -n b300 benchmark \
  -r act:lerobot/pusht:200 \
  -r diffusion:lerobot/pusht:200 \
  -r smolvla:lerobot/pusht:200 \
  -w 8 -w 0 \
  --batch-size 8
```

- `-r POLICY:DATASET:STEPS` — repeatable, defines each training run
- `-w N` — repeatable num_workers values (0 = max CPUs on the VM)
- Collects system_info.txt, per-run train.log + summary.json
- Uploads results to S3 if storage is configured

### 3. Profile-Train (per-step stage breakdown)

Three modes, all via the same command:

```bash
# Wallclock mode (ground truth, no overhead)
# Uses cuda.Event pairs — no cuda.synchronize between stages
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  -r diffusion:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8

# Profiler mode (torch.profiler with Chrome traces)
# Uses cuda.synchronize at stage boundaries — higher overhead
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  --mode profiler

# Inference mode (forward-only latency, no backward/optimizer)
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  --mode inference

# With torch.compile
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r diffusion:lerobot/pusht:100 \
  --mode wallclock --compile
```

**Modes explained:**
- `wallclock` — Ground-truth throughput. CPU stages (dataloader, data_transfer) use `time.perf_counter()`. GPU stages (forward, backward, optimizer) use `cuda.Event` pairs read after a single final `torch.cuda.synchronize()`. No pipeline serialization during measurement. `throughput_steps_per_sec` (from total synchronized wall time) is the authoritative metric.
- `profiler` — Full `torch.profiler` with `record_function` labels. Produces Chrome traces and stage_breakdown.csv. `cuda.synchronize` at boundaries inflates times on some GPUs (up to 39% on B300).
- `inference` — Single-sample forward-only latency. Forces `batch_size=1` regardless of `--batch-size`. No backward, no optimizer. `cuda.Event` per forward pass. Keeps `policy.train()` (ACT VAE fails in eval mode).

**Stages measured:** `dataloader_batch_fetch` (cpu), `data_transfer_to_gpu` (cpu), `forward_pass` (gpu), `backward_pass` (gpu), `optimizer_step` (gpu)

**Timing methodology notes:**
- `cpu_enqueue_ms` = perf_counter around full step (CPU enqueue, may undercount outstanding GPU work)
- `gpu_step_ms` = sum of GPU-stage cuda.Event times per step
- CPU stages use perf_counter because CUDA events can't measure host-side dataloader wait
- `benchmark_policies.sh` times the full lerobot-train process (includes setup/teardown) — not directly comparable to profile_train.py wallclock, which times the training loop only

### 4. Other Commands

```bash
# Check VM status
npa workbench lerobot -p uk-south1 -n b300 status

# System info (nvidia-smi, lscpu, free -h, lsblk)
npa workbench lerobot -p uk-south1 -n b300 system-info

# List all workbenches
npa workbench lerobot list

# List checkpoints on VM and S3
npa workbench lerobot -p uk-south1 -n b300 list-checkpoints

# Single training run
npa workbench lerobot -p uk-south1 -n b300 train \
  --policy-type act --dataset lerobot/pusht --steps 5000 --job-name my-run

# Eval a checkpoint
npa workbench lerobot -p uk-south1 -n b300 eval \
  --checkpoint /opt/lerobot/checkpoints/my-run/checkpoints/last/pretrained_model \
  --env pusht

# Serve a checkpoint for inference
npa workbench lerobot -p uk-south1 -n b300 serve --checkpoint <path>

# POST an observation to the running server
npa workbench lerobot infer --observation obs.json
```

### 5. Research Scripts (run directly on the VM)

These are synced to `/opt/lerobot/` on the VM during deploy. You can also SSH in and run them:

```bash
# Full benchmark suite (GPU util sampling, scaling, memory profiling)
ssh ubuntu@<ip> 'bash /opt/lerobot/benchmark_policies.sh'

# Profile training directly
ssh ubuntu@<ip> 'source /opt/lerobot/venv/bin/activate && \
  python3 /opt/lerobot/profile_train.py \
    --mode wallclock --policy_type act --dataset_repo_id lerobot/pusht \
    --steps 100 --output_dir /tmp/profile-act'

# Validate all policies work
ssh ubuntu@<ip> 'bash /opt/lerobot/validate_policies.sh'
```

## Architecture Flow

```
Your laptop                          Remote GPU VM
─────────────                        ──────────────
npa workbench lerobot                /opt/lerobot/
  deploy ──────────────────────────→   Terraform + cloud-init
                                       → installs LeRobot, venv, NVIDIA EGL
                                       → syncs research scripts

  benchmark ───────────── SSH ─────→   lerobot-train (direct invocation)

  profile-train ───────── SSH ─────→   profile_train.py (wallclock/profiler/inference)

  status ──────────────── HTTP ────→   npa-lerobot-server:8080/health

  train / eval ────────── SSH ─────→   lerobot-train / lerobot-eval

  serve ───────────────── HTTP ────→   PolicyServer.load_checkpoint()
```

## Key Policies Tested

| Policy    | Type             | Default batch | Notes |
|-----------|------------------|---------------|-------|
| act       | Transformer      | 8             | VAE encoder; fails in eval mode (use train mode for inference) |
| diffusion | U-Net + DDPM     | 8             | Many small conv/attention kernels; B300 sm_103 is 2.72x slower than H200 |
| smolvla   | Vision-Language  | 4             | Largest model; GPU-bound (num_workers irrelevant) |
| vqbet     | VQ-VAE + BeT     | 8             | `VQBeTConfig.type` is a property, must instantiate to read |

## Known Gotchas

- **cuda.synchronize inflates B300**: Up to 39% overhead. Always use `--mode wallclock` for throughput numbers.
- **draccus can't resolve policy types dynamically**: profile_train.py uses `importlib.import_module` to find config classes.
- **ACT eval mode crashes**: ACT's VAE returns None for mu/log_sigma in eval mode. Keep `policy.train()` even for inference.
- **LeRobot 0.5.1 imports**: Use `lerobot.datasets.lerobot_dataset`, NOT `lerobot.common.datasets`.
- **B300/RTX6000 need CUDA 13 image**: Pass `-v image_family=ubuntu24.04-cuda13.0` when deploying.
- **Preemptible VMs can be stopped**: RTX6000 is preemptible. Redeploy with `--skip-app` then `--skip-infra` if preempted.
- **Port 8080**: Security group must allow ingress on 8080 for the PolicyServer health check.

## Testing

### Structure

Tests live in `npa/tests/`, mirroring the source tree:

```
npa/tests/
  conftest.py              # shared fixtures: tmp_workspace, sample_config, mock_ssh, mock_s3
  test_adapter.py          # adapter transform logic
  test_config.py           # config parsing
  test_clients.py          # SSH/HTTP/S3 client wrappers
  test_workflows.py        # pipeline orchestration
  test_deploy.py           # Terraform/SSH deploy
  test_server.py           # FastAPI endpoints
  cli/
    __init__.py
    test_main.py           # top-level CLI + subcommand --help smoke tests
    test_workbench_cli.py  # workbench lerobot/genesis commands
    test_adapter_cli.py    # adapter commands
    test_workflow_cli.py   # workflow commands
```

### Rules

- **No real infra in tests.** Never SSH, hit S3, call Nebius APIs, or touch GPUs. Mock everything at the call site.
- **No GPU imports.** Do not import `lerobot`, `genesis`, `torch`, or any CUDA package at module level. Use `pytest.importorskip()` if a test specifically needs them.
- **Patch at the call site**, not the definition. If `npa.cli.workbench.lerobot` imports `run_ssh_command` from `npa.clients.ssh`, patch `npa.cli.workbench.lerobot.run_ssh_command`.
- **CLI tests use `typer.testing.CliRunner`** against `npa.cli.main:app`. Mock infra below the CLI layer.
- **Run tests from `npa/`:**
  ```bash
  cd npa
  .venv/bin/python -m pytest tests/ -v --tb=short
  .venv/bin/python -m pytest tests/ --cov=npa --cov-report=term-missing
  ```
- All tests must pass before any PR. Zero tolerance for tests that hit the network.
