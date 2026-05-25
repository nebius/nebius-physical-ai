# NPA CLI Reference

Complete reference for the `npa` command-line tool. Use this to generate correct commands.

## Command Hierarchy

```
npa
├── workbench
│   ├── lerobot     # GPU policy training, eval, serving on remote VMs
│   ├── genesis     # Genesis simulation: teacher RL, demo gen, diagnosis
│   └── workflow    # Workbench multi-stage workflow orchestration
├── adapter
│   └── convert    # Sim data → LeRobotDataset v3
├── convert
├── rerun
└── demo
```

---

## Global Options

These go **before** the subcommand:

```
npa workbench lerobot -p <project> -n <name> <subcommand>
npa workbench genesis -p <project> -n <name> <subcommand>
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--project` | `-p` | str | `""` | Project alias from ~/.npa/config.yaml |
| `--name` | `-n` | str | `""` | Workbench instance name |

Both `lerobot` and `genesis` accept `-p`/`-n`. When set, `genesis` forwards the
command to the workbench VM via SSH (using the same conda env the `distill`
workflow provisions). When omitted, `genesis` commands run locally.

---

## npa workbench lerobot

### deploy

Provision a GPU VM and install LeRobot.

```bash
npa workbench lerobot -p <project> -n <name> deploy [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gpu-type` | str | gpu-h200-sxm | Nebius GPU platform |
| `--gpu-preset` | str | 1gpu-16vcpu-200gb | GPU preset |
| `--region` | str | `""` | Nebius region |
| `--project-id` | str | `""` | Nebius project ID (required on first deploy) |
| `--tenant-id` | str | `""` | Nebius tenant ID (required on first deploy) |
| `--tf-dir` | str | `""` | Custom Terraform directory |
| `--tf-var` / `-v` | str | `[]` | Extra TF variable as key=value (repeatable) |
| `--skip-infra` | bool | False | Skip Terraform, redeploy app only |
| `--skip-app` | bool | False | Skip app deploy, provision infra only |
| `--destroy` | bool | False | Destroy infrastructure and clean config |
| `--dry-run` | bool | False | Show plan without executing |
| `--checkpoint` | str | `""` | Pre-load a checkpoint after deploy |
| `--server-port` | int | 8080 | Server port on VM |
| `--preemptible` / `--no-preemptible` | bool | True | Preemptible (spot) instance |
| `--default` | bool | False | Set as default workbench |
| `--output` | text\|json | text | Output format |

**Examples:**

```bash
# First-time deploy (B300, non-preemptible, CUDA 13)
npa workbench lerobot -p uk-south1 -n b300 deploy \
  --project-id project-xxx --tenant-id tenant-xxx --region uk-south1 \
  --gpu-type gpu-b300-sxm --gpu-preset 1gpu-24vcpu-346gb \
  --no-preemptible \
  -v image_family=ubuntu24.04-cuda13.0

# Redeploy app only (infra exists)
npa workbench lerobot -p uk-south1 -n b300 deploy --skip-infra

# Redeploy infra only (app installed via cloud-init)
npa workbench lerobot -p uk-south1 -n b300 deploy --skip-app

# Destroy
npa workbench lerobot -p uk-south1 -n b300 deploy --destroy
```

---

### benchmark

Run a benchmark suite: system info collection, training runs across worker counts, S3 upload.

```bash
npa workbench lerobot -p <project> -n <name> benchmark [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--run` | `-r` | str | **required** | Training spec as POLICY:DATASET:STEPS (repeatable) |
| `--num-workers` | `-w` | int | **required** | Dataloader num_workers to test; 0 = max CPUs (repeatable) |
| `--batch-size` | | int | 8 | Batch size for all runs |
| `--output` | | text\|json | text | Output format |

**Examples:**

```bash
# Benchmark 3 policies, 2 worker counts
npa workbench lerobot -p uk-south1 -n b300 benchmark \
  -r act:lerobot/pusht:200 \
  -r diffusion:lerobot/pusht:200 \
  -r smolvla:lerobot/pusht:200 \
  -w 8 -w 0 \
  --batch-size 8

# Single policy benchmark
npa workbench lerobot -p eu-west1 -n h200 benchmark \
  -r act:lerobot/pusht:500 \
  -w 4
```

---

### profile-train

Profile training with wallclock timing, torch.profiler traces, or inference-only latency.

```bash
npa workbench lerobot -p <project> -n <name> profile-train [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--run` | `-r` | str | **required** | Training spec as POLICY:DATASET:STEPS (repeatable) |
| `--mode` | `-m` | str | wallclock | Mode: wallclock, profiler, or inference |
| `--compile` | | bool | False | Apply torch.compile to policy |
| `--num-workers` | `-w` | int | 0 | Dataloader num_workers (0 = max CPUs) |
| `--batch-size` | | int | 8 | Batch size (inference forces 1) |
| `--warmup-steps` | | int | 10 | Warmup steps before measurement |
| `--skip-first` | | int | 10 | (profiler) Schedule skip_first |
| `--warmup` | | int | 5 | (profiler) Schedule warmup |
| `--active` | | int | 50 | (profiler) Schedule active |
| `--output` | | text\|json | text | Output format |

**Modes:**
- **wallclock** -- Ground-truth throughput. No cuda.synchronize between stages. Authoritative metric: `throughput_steps_per_sec`.
- **profiler** -- Full torch.profiler with Chrome traces and stage_breakdown.csv. Adds cuda.synchronize overhead (up to 39% on B300).
- **inference** -- Forward-only latency at batch_size=1. No backward/optimizer. Keeps policy.train() (ACT VAE needs it).

**Examples:**

```bash
# Wallclock profiling
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  -r diffusion:lerobot/pusht:100 \
  --mode wallclock --batch-size 8

# Profiler with Chrome traces
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  --mode profiler

# Inference latency
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r act:lerobot/pusht:100 \
  --mode inference

# With torch.compile
npa workbench lerobot -p uk-south1 -n b300 profile-train \
  -r diffusion:lerobot/pusht:100 \
  --mode wallclock --compile
```

---

### train

Run a single LeRobot training job on the VM.

```bash
npa workbench lerobot -p <project> -n <name> train [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--policy-type` | str | **required** | Policy type (act, diffusion, smolvla) |
| `--dataset` | str | **required** | HuggingFace dataset repo ID |
| `--job-name` | str | **required** | Unique name for this training run |
| `--steps` | int | 5000 | Training steps |
| `--batch-size` | int | 8 | Batch size |
| `--env-type` | str | `""` | Environment type |
| `--env-task` | str | `""` | Environment task |
| `--num-workers` | int | -1 | Dataloader num_workers (-1 = omit, 0+ = explicit) |
| `--gpu-count` | int | 1 | Number of GPUs (uses accelerate for >1) |
| `--device` | str | cuda | Torch device |
| `--output` | text\|json | text | Output format |

**Example:**

```bash
npa workbench lerobot -p uk-south1 -n b300 train \
  --policy-type act --dataset lerobot/pusht --steps 5000 \
  --job-name my-act-run --batch-size 8
```

---

### eval

Run LeRobot evaluation on a checkpoint.

```bash
npa workbench lerobot -p <project> -n <name> eval [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-path` | str | **required** | S3 URI or Hugging Face Hub checkpoint ID |
| `--env` | str | **required** | Environment type |
| `--env-task` | str | `""` | Environment task |
| `--episodes` | int | 10 | Number of eval episodes |
| `--output` | text\|json | text | Output format |

**Example:**

```bash
npa workbench lerobot -p uk-south1 -n b300 eval \
  --input-path s3://my-bucket/checkpoints/my-run/ \
  --env pusht --episodes 50
```

---

### serve

Start or restart the PolicyServer with a checkpoint.

```bash
npa workbench lerobot -p <project> -n <name> serve [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-path` | str | **required** | S3 URI or Hugging Face Hub checkpoint ID |
| `--env-type` | str | `""` | Environment type |
| `--env-task` | str | `""` | Environment task |
| `--port` | int | 8080 | Server port |
| `--output` | text\|json | text | Output format |

---

### infer

POST an observation to the running PolicyServer.

```bash
npa workbench lerobot infer [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--observation` | file path | **required** | Path to observation JSON file |
| `--output` | text\|json | text | Output format |

---

### status

Check what's running on the VM.

```bash
npa workbench lerobot -p <project> -n <name> status [--output text|json]
```

---

### system-info

Collect hardware info (nvidia-smi, lscpu, free -h, lsblk).

```bash
npa workbench lerobot -p <project> -n <name> system-info [--output text|json]
```

---

### list

List all configured projects and workbenches.

```bash
npa workbench lerobot list [--output text|json]
```

---

### list-checkpoints

List checkpoints on VM and in S3.

```bash
npa workbench lerobot -p <project> -n <name> list-checkpoints [--output text|json]
```

---

### train-student

Train a vision-only student policy locally from a LeRobotDataset.

```bash
npa workbench lerobot train-student [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | file path | **required** | Path to local LeRobotDataset v3 directory |
| `--policy` | str | act | Policy type (act, diffusion) |
| `--epochs` | int | 100 | Training epochs |
| `--batch-size` | int | 64 | Batch size |
| `--num-workers` | int | 4 | Dataloader num_workers |
| `--device` | str | cuda | Torch device |
| `--output-dir` | str | ./checkpoints/student/ | Checkpoint output directory |
| `--output` | text\|json | text | Output format |

---

## npa workbench genesis

### train-teacher

Train an RL teacher policy with PPO in Genesis simulation.

```bash
npa workbench genesis train-teacher [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--n-envs` | int | 4096 | Parallel environments |
| `--max-iterations` | int | 500 | PPO training iterations |
| `--output` / `-o` | str | ./checkpoints/teacher/ | Checkpoint output directory |
| `--device` | str | cuda | Torch device |
| `--log-dir` | str | ./logs/teacher/ | Tensorboard log directory |
| `--seed` | int | 42 | Random seed |
| `--action-space` | cartesian\|joint | cartesian | Action space |
| `--env-override` | str | `[]` | EnvConfig override as KEY=VALUE (repeatable) |
| `--output-format` | text\|json | text | Output format |

**Example:**

```bash
npa workbench genesis train-teacher \
  --n-envs 4096 --max-iterations 500 \
  --action-space cartesian \
  --env-override friction_min=0.5 --env-override friction_max=1.2
```

---

### generate-demos

Generate camera-only demonstrations using a trained teacher.

```bash
npa workbench genesis generate-demos [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to trained teacher checkpoint |
| `--n-envs` | int | 4096 | Parallel environments |
| `--n-episodes` | int | 0 | Episodes to collect (0 = one batch) |
| `--output` / `-o` | str | ./data/demos/ | Output directory |
| `--domain-randomize` / `--no-domain-randomize` | bool | True | Domain randomization |
| `--fps` | int | 20 | Camera frame rate |
| `--seed` | int | 42 | Random seed |
| `--allow-failure-demos` / `--no-failure-demos` | bool | False | Save all episodes even with 0 successes |
| `--action-space` | cartesian\|joint | cartesian | Must match teacher training |
| `--output-format` | text\|json | text | Output format |

**Example:**

```bash
npa workbench genesis generate-demos \
  --checkpoint ./checkpoints/teacher/model.pt \
  --n-envs 4096 --n-episodes 10000 \
  --domain-randomize --fps 20
```

---

### eval-teacher

Evaluate teacher policy under held-out conditions.

```bash
npa workbench genesis eval-teacher [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | **required** | Teacher checkpoint (model.pt) |
| `--n-envs` | int | 1024 | Parallel environments |
| `--seed` | int | 7777 | Held-out seed |
| `--action-space` | cartesian\|joint | cartesian | Must match training |
| `--output-format` | text\|json | text | Output format |

---

### eval-student

Evaluate a student vision policy in Genesis simulation.

```bash
npa workbench genesis eval-student [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | **required** | Student policy checkpoint |
| `--n-envs` | int | 1024 | Parallel environments |
| `--n-episodes` | int | 1024 | Total eval episodes |
| `--output` / `-o` | str | ./eval/ | Output directory |
| `--domain-randomize` / `--no-domain-randomize` | bool | True | Domain randomization |
| `--seed` | int | 42 | Held-out seed |
| `--teacher-success-rate` | float | -1.0 | Teacher rate for distillation gap (-1 = skip) |
| `--action-space` | cartesian\|joint | cartesian | Must match demo gen/training |
| `--output-format` | text\|json | text | Output format |

---

### diagnose

Diagnose teacher policy failures: classifies failure phases, suggests fixes.

```bash
npa workbench genesis diagnose [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | **required** | Teacher checkpoint (model.pt) |
| `--n-envs` | int | 1024 | Parallel environments |
| `--n-episodes` | int | 0 | Episodes (0 = one batch) |
| `--seed` | int | 42 | Random seed |
| `--output` / `-o` | str | `""` | Save diagnosis JSON (empty = don't save) |
| `--action-space` | cartesian\|joint | cartesian | Must match training |
| `--env-override` | str | `[]` | Override as KEY=VALUE (repeatable) |
| `--output-format` | text\|json | text | Output format |

---

### tune

Auto-tune loop: diagnose, adjust config, retrain, re-diagnose.

```bash
npa workbench genesis tune [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | **required** | Initial teacher checkpoint (model.pt) |
| `--max-rounds` | int | 5 | Max diagnose-retrain iterations |
| `--retrain-iterations` | int | 100 | PPO iterations per retrain round |
| `--n-envs` | int | 4096 | Parallel envs for retraining |
| `--diagnose-n-envs` | int | 1024 | Parallel envs for diagnosis |
| `--seed` | int | 42 | Base random seed |
| `--output` / `-o` | str | ./checkpoints/tune/ | Per-round checkpoint dir |
| `--log-dir` | str | ./logs/tune/ | Tensorboard log dir |
| `--device` | str | cuda | Torch device |
| `--action-space` | cartesian\|joint | cartesian | Action space |
| `--env-override` | str | `[]` | Override as KEY=VALUE (repeatable) |
| `--min-success-rate` | float | 0.0 | Stop threshold (0.0 = any success) |
| `--output-format` | text\|json | text | Output format |

**Example:**

```bash
npa workbench genesis tune \
  --checkpoint ./checkpoints/teacher/model.pt \
  --max-rounds 5 --retrain-iterations 200 \
  --min-success-rate 0.20
```

---

## npa adapter

### convert

Convert Genesis/sim demo numpy arrays to LeRobotDataset v3 format.

```bash
npa adapter convert [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | str | **required** | Directory of episode numpy arrays |
| `--output` | `-o` | str | **required** | Output LeRobotDataset v3 directory |
| `--fps` | | int | 20 | Frame rate for video encoding |
| `--robot` | | str | franka_panda | Robot type identifier |
| `--task` | | str | Pick and place cube to target | Task description |

**Example:**

```bash
npa adapter convert \
  -i ./data/demos/ -o ./data/lerobot_dataset/ \
  --fps 20 --robot franka_panda
```

---

## npa workbench workflow

`npa workbench workflow` is the canonical Workbench workflow namespace. The
legacy `npa workflow` shim is hidden from top-level help and prints a visible
deprecation warning when invoked.

### distill

Turnkey expert distillation. Provisions an L40S VM for Genesis simulation and an
H100 VM for LeRobot training, installs runtimes, runs the 5-stage pipeline via
SSH, and transfers artifacts between VMs via S3.

```bash
npa workbench workflow distill [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--teardown` / `--no-teardown` | bool | False | Destroy both VMs after completion (even on failure) |
| `--skip-infra` / `--provision` | bool | False | Skip provisioning, resolve VMs from config |
| `--skip-setup` / `--setup` | bool | False | Skip runtime setup (conda + npa install) |
| `--n-envs` | int | 4096 | Parallel environments for simulation |
| `--teacher-max-iterations` | int | 500 | PPO iterations for teacher |
| `--student-policy` | str | act | Student policy: act, diffusion, smolvla |
| `--student-epochs` | int | 100 | Student training epochs |
| `--student-batch-size` | int | 64 | Student batch size |
| `--eval-n-episodes` | int | 1024 | Student eval episodes |
| `--action-space` | cartesian\|joint | cartesian | Action space |
| `--output-format` | text\|json | text | Output format |

**Hardcoded infrastructure:**
- SIM VM: `l40s-distill-genesis` (gpu-l40s-a, 1gpu-40vcpu-160gb)
- TRAIN VM: `h100-distill-lerobot` (gpu-h100-sxm, 1gpu-16vcpu-200gb)
- Region: eu-north1
- Both VMs are non-preemptible

**5-stage pipeline:**
1. Train teacher (SIM VM, Genesis PPO)
2. Generate demos (SIM VM, camera rendering + domain randomization, capped at 64 envs)
3. Convert to LeRobotDataset (SIM VM, CPU-only) + S3 upload
4. Train student (TRAIN VM, LeRobot ACT/Diffusion/SmolVLA) after S3 download
5. Eval student (SIM VM, Genesis) after S3 download of student checkpoint

Teacher eval runs automatically between stages 2 and 3 to establish a held-out baseline.

**Limitation:** Does not support `--env-override`. To train a tuned teacher with
custom reward/friction settings, use individual `npa workbench genesis` commands.

**Examples:**

```bash
# Full pipeline with teardown
npa workbench workflow distill \
  --teacher-max-iterations 3000 \
  --student-policy act --student-epochs 100 \
  --action-space cartesian \
  --teardown

# Reuse existing VMs from a prior run
npa workbench workflow distill \
  --skip-infra --skip-setup \
  --teacher-max-iterations 500

# Provision + run, keep VMs alive for debugging
npa workbench workflow distill \
  --teacher-max-iterations 500
```

---

### run

Run a named workflow on existing infrastructure. Unlike `distill`, this does not
provision VMs — you manage infrastructure separately via `npa workbench lerobot deploy`.

```bash
npa workbench workflow run <WORKFLOW> [OPTIONS]
```

| Argument/Option | Type | Default | Description |
|-----------------|------|---------|-------------|
| `workflow` (arg) | str | **required** | Workflow name (currently: `distill`) |
| `--project` / `-p` | str | `""` | Project alias |
| `--robot` | str | franka_panda | Robot type |
| `--task` | str | pick_place | Task name |
| `--n-envs` | int | 4096 | Parallel environments |
| `--remote` / `--local` | bool | False | Execute on remote VMs (requires --s3-bucket) |
| `--s3-bucket` | str | `""` | S3 bucket URI (required for --remote) |
| `--sim-workbench` | str | `""` | Workbench for sim VM (Genesis stages) |
| `--train-workbench` | str | `""` | Workbench for training VM (defaults to sim) |
| `--action-space` | cartesian\|joint | cartesian | Action space |
| `--output-format` | text\|json | text | Output format |

**Examples:**

```bash
# Local distillation (single GPU machine)
npa workbench workflow run distill --n-envs 4096

# Remote distillation on pre-provisioned workbenches
npa workbench workflow run distill \
  -p eu-west1 --remote \
  --s3-bucket s3://my-bucket/workflows \
  --sim-workbench l40s --train-workbench h200
```

---

### status

Check workflow run status.

```bash
npa workbench workflow status <RUN_ID> [--output-format text|json]
```

### logs

Show logs for a workflow stage.

```bash
npa workbench workflow logs <RUN_ID> <STAGE>
```

Stages: `train_teacher`, `generate_demos`, `convert`, `train_student`, `eval_student`

---

## GPU Types and Presets

| Alias | `--gpu-type` | `--gpu-preset` | Notes |
|-------|-------------|----------------|-------|
| B300 | gpu-b300-sxm | 1gpu-24vcpu-346gb | Needs `-v image_family=ubuntu24.04-cuda13.0` |
| H200 | gpu-h200-sxm | 1gpu-16vcpu-200gb | Default for `deploy` |
| H100 | gpu-h100-sxm | 1gpu-16vcpu-200gb | Used by `distill` for training |
| L40S | gpu-l40s-a | 1gpu-40vcpu-160gb | Used by `distill` for Genesis sim |
| RTX PRO 6000 | gpu-rtx-pro-6000 | 1gpu-12vcpu-96gb | Needs CUDA 13 image; preemptible |

## Policies

| Name | `--policy-type` | Default batch | Notes |
|------|-----------------|---------------|-------|
| ACT | act | 8 | Transformer + VAE; fails in eval mode |
| Diffusion | diffusion | 8 | U-Net + DDPM |
| SmolVLA | smolvla | 4 | Vision-language; GPU-bound |
| VQ-BeT | vqbet | 8 | VQ-VAE + BeT |

## Config File

Located at `~/.npa/config.yaml`:

```yaml
projects:
  <alias>:
    project_id: project-...
    tenant_id: tenant-...
    region: <region>
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

## Common Patterns

```bash
# Turnkey distillation: provisions L40S + H100, runs full pipeline, tears down
npa workbench workflow distill \
  --teacher-max-iterations 3000 \
  --student-policy act --student-epochs 100 \
  --action-space cartesian --teardown

# Full lifecycle on a single workbench: deploy → benchmark → profile → train → eval → serve
npa workbench lerobot -p uk-south1 -n b300 deploy \
  --project-id ... --tenant-id ... --region uk-south1 \
  --gpu-type gpu-b300-sxm --gpu-preset 1gpu-24vcpu-346gb \
  --no-preemptible -v image_family=ubuntu24.04-cuda13.0
npa workbench lerobot -p uk-south1 -n b300 benchmark -r act:lerobot/pusht:200 -w 8 -w 0
npa workbench lerobot -p uk-south1 -n b300 profile-train -r act:lerobot/pusht:100 --mode wallclock
npa workbench lerobot -p uk-south1 -n b300 train \
  --policy-type act --dataset lerobot/pusht --steps 5000 --job-name act-pusht \
  --output-path s3://my-bucket/checkpoints/act-pusht/
npa workbench lerobot -p uk-south1 -n b300 eval \
  --input-path s3://my-bucket/checkpoints/act-pusht/ --env pusht
npa workbench lerobot -p uk-south1 -n b300 serve \
  --input-path s3://my-bucket/checkpoints/act-pusht/

# Manual distillation with tuned teacher (local GPU, supports --env-override)
npa workbench genesis train-teacher --n-envs 4096 --max-iterations 500 \
  --env-override friction_min=0.6 --env-override approach_weight=5.0
npa workbench genesis generate-demos \
  --checkpoint ./checkpoints/teacher/model.pt --n-envs 64 --domain-randomize
npa adapter convert -i ./data/demos/ -o ./data/lerobot_dataset/
npa workbench lerobot train-student --dataset ./data/lerobot_dataset/ --policy act --epochs 100
npa workbench genesis eval-student --checkpoint ./checkpoints/student/model.pt

# Diagnose and auto-tune a struggling teacher
npa workbench genesis diagnose --checkpoint ./checkpoints/teacher/model.pt
npa workbench genesis tune \
  --checkpoint ./checkpoints/teacher/model.pt --max-rounds 5 --min-success-rate 0.20
```
