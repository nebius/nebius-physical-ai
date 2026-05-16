# LeRobot on Nebius

Deploy a preemptible H200 GPU instance on Nebius Cloud for LeRobot training.
LeRobot is treated as an **installed dependency** (pinned PyPI version) — this
repo owns only infrastructure and orchestration.

## What you get

- Preemptible H200 NVLink instance with 141GB VRAM (`gpu-h200-sxm`) in `eu-north1`
- Default preset: `1gpu-16vcpu-200gb`
- Default image family: `ubuntu24.04-cuda12` (required for Python 3.12 / LeRobot 0.5.1)
- LeRobot installed from PyPI (pinned version, no git clone)
- S3 bucket for checkpoints and datasets
- Terraform state stored remotely in the same S3 bucket

## Prerequisites

```bash
brew install nebius/tap/nebius   # Nebius CLI
brew install jq                  # JSON parser
brew install terraform           # Infrastructure
nebius config init               # Authenticate
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519  # SSH key (if needed)
```

## Deploy

**Step 1: Set environment variables** (~30 seconds)

Get your tenant and project IDs from the Nebius dashboard or CLI:
```bash
nebius organization list      # Find your TENANT_ID
nebius project list           # Find your PROJECT_ID
```

Export credentials:
```bash
export NEBIUS_TENANT_ID='tenant-...'     # Required
export NEBIUS_PROJECT_ID='project-...'   # Required
export NEBIUS_REGION='eu-north1'         # Optional, default: eu-north1 (H200)
```

**Step 2: Bootstrap Nebius resources** (~2 minutes)

From the **repo root**, run this once:
```bash
cd /path/to/lerobot-deploy
source environment.sh
```

This automates:
- S3 bucket creation (or reuse if exists)
- Service account creation (or reuse if exists)
- Access key generation (or reuse if unexpired)
- Writing `.env` with credentials
- Configuring Terraform backend

Safe to re-run. Existing resources are reused.

**Step 3: Deploy H200 instance** (~5–10 minutes)

```bash
cd terraform
terraform init              # Setup Terraform (safe to re-run)
terraform plan -out tfplan  # Review before applying
terraform apply tfplan      # Deploy the instance
```

Watch the output for the instance IP and SSH command when complete.

**Total time: ~10 minutes** after cloud-init finishes (~5 more minutes after apply).
Works on greenfield projects with no existing VPC.

## Connect and train

**Wait for cloud-init to finish** (~5 minutes after `terraform apply`)

```bash
# Get SSH command from Terraform output
cd terraform
SSH_CMD="$(terraform output -raw ssh_command)"
echo "$SSH_CMD"  # Copy this and run it, or:
eval "$SSH_CMD"
```

**On the instance**, the environment is pre-configured:
- LeRobot: installed from PyPI in `/opt/lerobot/venv/`
- Virtual environment: auto-activated via system profile and `.bashrc`
- `.env` file: `/opt/lerobot/.env` (auto-sourced)
- S3 sync script: `/opt/lerobot/s3_sync.py`
- Training wrapper: `/opt/lerobot/train.sh`
- Validation script: `/opt/lerobot/validate_policies.sh`
- Benchmark suite: `/opt/lerobot/benchmark_policies.sh`
- System `python`/`python3` symlinked to venv (works immediately)

**Verify setup:**
```bash
# Check LeRobot is installed
python3 -c "import lerobot; print(lerobot.__version__)"

# Test S3 access
python /opt/lerobot/s3_sync.py check
```

**Start training** (automatically uploads checkpoints to S3 on exit):
```bash
# Using wrapper (recommended — auto-uploads on exit)
bash /opt/lerobot/train.sh --policy.type=act --dataset.repo_id=lerobot/pusht

# Or use LeRobot CLI directly (manual uploads needed)
lerobot-train --policy.type=act --policy.push_to_hub=false --dataset.repo_id=lerobot/pusht
```

**Evaluate a checkpoint:**
```bash
# Find latest checkpoint
ls /opt/lerobot/runs/
RUN_DIR="/opt/lerobot/runs/run-XXXXX"
lerobot-eval --policy.path="$RUN_DIR/checkpoints/last/pretrained_model"
```

**Benchmark performance:**
```bash
# Validation answers "does it run?"
bash /opt/lerobot/validate_policies.sh

# Benchmarking answers utilization, scaling, memory, bottleneck, and train/eval split
bash /opt/lerobot/benchmark_policies.sh
```

**Resume after preemption:**
```bash
# If instance is preempted, latest checkpoint is auto-uploaded to S3.
# Set RESUME=true to find the last run dir and pass --resume=true to lerobot-train:
RESUME=true bash /opt/lerobot/train.sh --policy.type=act --dataset.repo_id=lerobot/pusht
```

## Manage checkpoints via S3

LeRobot checkpoints are directories with:
- `pretrained_model/` — weights + config (needed for inference)
- `training_state/` — optimizer, scheduler, RNG (needed to resume training)
- `last` symlink — points to latest checkpoint

All auto-uploaded to S3 by the `train.sh` wrapper on exit (success, Ctrl-C, OOM, preemption).

**On the instance:**
```bash
# List uploaded checkpoints
python /opt/lerobot/s3_sync.py check                    # Status
python /opt/lerobot/s3_sync.py ls checkpoints/          # List all
python /opt/lerobot/s3_sync.py ls checkpoints/run-XXX   # One run
```

**Locally** (from repo root):
```bash
# Load .env variables
set -a; source .env; set +a

# List checkpoints
python s3_sync.py check
python s3_sync.py ls checkpoints/

# Download weights only (for inference)
python s3_sync.py download checkpoints/run-XXX/050000/pretrained_model/model.safetensors --dest model.safetensors

# Download full checkpoint (for resuming training)
python s3_sync.py download checkpoints/run-XXX/050000 --dest ./my_checkpoint
```

**Manual upload** (if not using wrapper):
```bash
# On instance:
CKPT=/opt/lerobot/runs/run-XXX/checkpoints/050000
find "$CKPT" -type f | while read f; do
  python /opt/lerobot/s3_sync.py upload "$f" --key "checkpoints/run-XXX/050000/${f#$CKPT/}"
done
```

## Upgrade LeRobot version

1. Edit `lerobot_version` in [terraform/variables.tf](terraform/variables.tf) (or pass `-var lerobot_version=0.6.0`).
   You can also change `gpu_platform` / `gpu_preset` to switch hardware.
2. `terraform apply` to recreate the instance with the new version

No merge conflicts, no broken imports — just a version bump.

## Preemptible instances

The instance can be reclaimed by Nebius at any time. The `train.sh` wrapper
uses a `trap EXIT` handler to upload the latest checkpoint to S3 regardless of
how training ends. For manual uploads:

```bash
CKPT=/opt/lerobot/runs/run-XXX/checkpoints/050000
find "$CKPT" -type f | while read f; do
  python /opt/lerobot/s3_sync.py upload "$f" --key "checkpoints/run-XXX/050000/${f#$CKPT/}"
done
```

## Teardown

**Stop the instance** (pause billing — instance can be recreated):
```bash
cd terraform
terraform destroy -auto-approve
```

**Delete the instance + S3 bucket + service account** (full cleanup):
```bash
cd terraform
terraform destroy -auto-approve

# Get IDs from env vars (if still exported) or from .env
source ../.env 2>/dev/null || true
BUCKET_NAME="${NEBIUS_S3_BUCKET:-YOUR_S3_BUCKET_3}"  # From terraform output
SA_ID="${NEBIUS_SA_ID:-...}"                         # Check nebius iam service-account list

# Delete S3 bucket (contents must be empty)
nebius storage object delete --parent-id "${NEBIUS_BUCKET_NAME}" --name "*" || true
nebius storage bucket delete --id "${NEBIUS_BUCKET_NAME}"

# Delete service account
nebius iam service-account delete --id "${SA_ID}"

# Clean up local state
rm -f .env terraform/.terraform/terraform_backend_override.tf terraform/terraform.tfstate*
```

## Troubleshooting

**Instance creation fails with "internal error" after ~5 minutes**

This is usually a quota or availability issue. H200 GPUs may not have capacity in the selected region.

*Solution:*
```bash
# Clean up
cd terraform && terraform destroy -auto-approve && cd ..

# Try a different region
export NEBIUS_REGION="eu-north2"
source ../environment.sh

# Or try H100 if capacity issue persists
cd terraform
terraform apply -var="gpu_platform=gpu-h100-sxm"

# If CPU works, contact Nebius support for H200 quota
cd terraform
terraform apply -var="gpu_platform=cpu-d3" -var="gpu_preset=cpu-d3-4vcpu-8gb"
```

**`terraform plan` asks for `ssh_cidr_block`**

This was a config bug — it's now optional and defaults to `0.0.0.0/0` (open access).

*Solution:*
```bash
cd terraform && terraform destroy -auto-approve && cd ..
source environment.sh
cd terraform && terraform plan
```

**`environment.sh` fails with "AccessKey with name already exists"**

A key from a previous run wasn't cleaned up (or Nebius API error).

*Solution:*
```bash
# Already fixed in this version — just re-run
source environment.sh
```

**SSH fails with "Permission denied"**

SSH key doesn't exist or has wrong permissions.

*Solution:*
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519  # Press enter for defaults
chmod 600 ~/.ssh/id_ed25519
cd terraform && terraform destroy -auto-approve && cd ..
source ../environment.sh
cd terraform && terraform apply
```

**Cannot connect to S3 after training**

Credentials weren't sourced properly.

*Solution:*
```bash
# On instance:
source /opt/lerobot/.env
python /opt/lerobot/s3_sync.py check

# Locally:
set -a; source .env; set +a
python s3_sync.py check
```

**`python3` cannot import `lerobot` on first SSH**

Cloud-init is still running or failed. LeRobot setup takes ~3-5 minutes after shell login.

*Solution:*
```bash
# On instance, wait for cloud-init to finish and inspect the LeRobot install log
cloud-init status --wait
tail -100 /var/log/lerobot-setup.log

# If setup failed during the pip install step, repair it in place:
sudo /opt/lerobot/venv/bin/pip install "lerobot[pusht]==0.5.1" boto3 wandb tensorboard
/opt/lerobot/venv/bin/python -c "import lerobot; print(lerobot.__version__)"
python3 -c "import lerobot; print(lerobot.__version__)"

# If the log shows a version-resolution failure, the requested PyPI version was invalid.
# This repo now defaults to 0.5.1, which exists on PyPI as of April 13, 2026.
```

If LeRobot still isn't installed, check the full logs:
```bash
tail -100 /var/log/cloud-init-output.log
```

## Project structure

```
lerobot-deploy/
  environment.sh              Nebius CLI bootstrap (IAM, S3, service account)
  s3_sync.py                  S3 helper — zero LeRobot imports (local copy)
  terraform/
    main.tf                   VPC, security group, disk, H200 instance
    variables.tf              All configurable parameters
    outputs.tf                SSH command, IP, status
    cloud_init.yaml.tpl       Cloud-init: installs lerobot, deploys wrappers
    terraform.tfvars.example  Template for manual config
  training/
    train.sh                  Wrapper around lerobot-train (local reference)
    eval.sh                   Wrapper around lerobot-eval (local reference)
    validate_policies.sh      Validation smoke and compatibility checks
    benchmark_policies.sh     Benchmark suite for utilization/scaling/memory
    benchmark_metrics.py      Metric sampler and JSON summarizer
    configs/                  Your training YAML overrides
  .env.example                Credential template
  README.md                   This file
```

The `training/` directory is the source-of-truth for the wrapper scripts.
Cloud-init deploys copies of the training scripts and `s3_sync.py` to
`/opt/lerobot/` on the VM so they're available immediately after boot.

This repo deploys a standalone VM. It does not currently configure an
InfiniBand GPU cluster or a `fabric-*` selector; in the Nebius Terraform
provider, that is modeled via `gpu_cluster`, which is a different workflow
from this single-GPU `1gpu-16vcpu-200gb` setup.

## How environment.sh works

It calls the `nebius` CLI to:
1. Validate tenant/project/region
2. Get an IAM access token
3. Check for existing VPC subnets (informational — Terraform creates its own)
4. Find (or create) an S3 bucket with versioning
5. Find (or create) a `lerobot-training` service account
6. Add the service account to the tenant's `editors` group
7. Reuse an unexpired access key, rotate an expired one, or create a new one
8. Fetch the secret via `get-secret` for new keys; read from `.env` for reused keys
9. Write a Terraform S3 backend override
10. Export all `TF_VAR_*` variables so `terraform apply` works with no `.tfvars`
11. Update `.env` — managed keys are refreshed, user-added values (HF_TOKEN, WANDB_*) are preserved

Safe to re-run. Existing resources and unexpired credentials are reused.
User additions to `.env` survive reruns. Expired keys are rotated automatically.

## Security notes

- `.env` contains credentials. Never commit it (already in `.gitignore`).
- `ssh_cidr_block` defaults to `0.0.0.0/0`. Tighten it to `"YOUR_IP/32"` if you do not want open SSH access.
- Access keys expire after 365 days. Re-run `source environment.sh` to rotate.
- Service account has `editor` permissions. Restrict further if needed.
- `LEROBOT_HOME` is **not set** — recent LeRobot releases raise a hard error if it is.
  Use `HF_LEROBOT_HOME` for cache path configuration.
