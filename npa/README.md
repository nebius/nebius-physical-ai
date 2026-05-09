# npa

`npa` is the Nebius Physical AI CLI/SDK for running a sim-to-real training loop on Nebius workbenches.

In practice it does four things:

1. Provisions and updates Nebius GPU workbenches for LeRobot with Terraform, the `nebius` CLI, SSH, and a small FastAPI policy server.
2. Runs LeRobot jobs on those workbenches: train, eval, serve checkpoints, run inference, list checkpoints, and collect benchmark/profile data.
3. Runs Genesis-side steps in a Genesis environment: train a teacher policy, generate demonstrations, and evaluate a student.
4. Converts simulation demos into LeRobotDataset v3 and orchestrates the full 5-stage distillation workflow:
   teacher train -> demo generation -> dataset conversion -> student train -> student eval.

The Python package exposes the same building blocks as importable modules: Nebius/Terraform helpers, SSH and HTTP clients, S3 storage utilities, dataset conversion, student training, and workflow orchestration.

## Install

```bash
pip install -e .
pip install -e ".[server]"   # policy server
pip install -e ".[adapter]"  # dataset conversion
pip install -e ".[genesis]"  # Genesis + distillation stages
```

Extra tools required by specific commands:

- `nebius` CLI and `terraform` for `npa workbench lerobot deploy`
- `ffmpeg` for `npa adapter convert`

## CLI layout

```bash
npa workbench lerobot ...
npa workbench genesis ...
npa adapter convert ...
npa workflow ...
```

Common examples:

```bash
# Provision or update a Nebius LeRobot workbench
npa workbench lerobot -p eu-north1 -n h200 deploy \
  --project-id project-... \
  --tenant-id tenant-... \
  --region eu-north1

# Train/eval/serve a LeRobot policy on the remote workbench
npa workbench lerobot train --policy-type act --dataset lerobot/aloha_sim_transfer_cube_human --job-name act-demo --output-path s3://my-bucket/checkpoints/act-demo/
npa workbench lerobot eval --input-path s3://my-bucket/checkpoints/act-demo/ --env aloha
npa workbench lerobot serve --input-path s3://my-bucket/checkpoints/act-demo/
npa workbench lerobot infer --observation /tmp/obs.json --output json

# Genesis-side local stages
npa workbench genesis train-teacher --n-envs 4096
npa workbench genesis generate-demos --checkpoint ./checkpoints/teacher/model.pt
npa workbench genesis eval-student --checkpoint ./checkpoints/student/checkpoints/last/pretrained_model

# Convert demos to LeRobotDataset v3
npa adapter convert --input ./runs/demos --output ./runs/dataset

# Run the full distillation workflow
npa workflow run distill --local
npa workflow run distill --remote --project eu-north1 --s3-bucket s3://my-bucket/checkpoints/
```

## Workbench Runtimes

Deploy commands support three runtime modes:

- `vm`: provisions and manages a Nebius VM with Terraform and installs the tool over SSH.
- `container`: provisions and manages a Nebius VM with Terraform, then starts the tool container over SSH.
- `byovm`: skips Terraform entirely and deploys the app to an existing SSH-accessible VM.

Use `byovm` when the VM already exists, for example for pre-provisioned
multi-GPU machines. BYOVM does not create, stop, start, resize, or destroy the
VM. A BYOVM `--destroy` only removes the local workbench entry from
`~/.npa/config.yaml`.

BYOVM requires a host and SSH key, either from flags:

```bash
npa workbench lerobot -p eu-north1 -n my-multi-gpu deploy \
  --runtime byovm \
  --host 203.0.113.10 \
  --ssh-user ubuntu \
  --ssh-key ~/.ssh/id_ed25519 \
  --gpu-count 4
```

or from `~/.npa/credentials.yaml`:

```yaml
ssh:
  host: 203.0.113.10
  user: ubuntu
  key_path: ~/.ssh/id_ed25519
```

During BYOVM deploy, `npa` probes the target with `nvidia-smi`, stores the
detected GPU count and names in `~/.npa/config.yaml`, and writes
`CUDA_VISIBLE_DEVICES` plus `NPA_GPU_COUNT` into the remote environment. Use
`--gpu-count <N>` to limit the visible devices on a larger VM.

Status and system information commands use the same saved SSH metadata:

```bash
npa workbench lerobot -p eu-north1 -n my-multi-gpu status
npa workbench lerobot -p eu-north1 -n my-multi-gpu system-info
```

The multi-GPU BYOVM pytest suite is opt-in and expects a live target:

```bash
export NPA_TEST_BYOVM_HOST=203.0.113.10
export NPA_TEST_BYOVM_SSH_KEY=~/.ssh/id_ed25519
export NPA_TEST_BYOVM_GPU_COUNT=4
export NPA_TEST_BYOVM_S3_PREFIX=s3://my-bucket/test-artifacts/
pytest tests/test_multi_gpu -m multi_gpu
```

## Config

Remote workbench commands resolve config from:

1. CLI flags
2. Environment variables
3. `~/.npa/credentials.yaml` for user-level secrets
4. `~/.npa/config.yaml` for projects and workbenches

See [`src/npa/config/sample_config.yaml`](src/npa/config/sample_config.yaml) for the expected layout.

`~/.npa/config.yaml` is machine-managed by deploy commands. Keep user tokens
out of it and store those credentials in `~/.npa/credentials.yaml`:

```yaml
tokens:
  HF_TOKEN: hf_REPLACE_ME
ngc:
  api_key: nvapi_REPLACE_ME
  # org: optional-ngc-org
  # team: optional-ngc-team
```

Standard environment variables override values in `credentials.yaml`, so this
also works for one-off runs:

```bash
export HF_TOKEN=hf_REPLACE_ME
export NGC_API_KEY=nvapi_REPLACE_ME
npa workbench cosmos deploy ...
```

For compatibility, `NGC_API_KEY`, `NGC_ORG`, and `NGC_TEAM` are also accepted
inside the legacy `tokens:` map.

Recommended permissions:

```bash
chmod 600 ~/.npa/credentials.yaml
```

If the file is readable by other users, `npa workbench ...` prints a warning.
Loaded tokens are forwarded to remote workbench SSH commands as environment
variables.

Token requirements by workbench:

- Cosmos: requires `HF_TOKEN` during deploy to download gated Hugging Face Cosmos models.
- GR00T: requires `HF_TOKEN` for gated Hugging Face GR00T models; optional `ngc.api_key` or `NGC_API_KEY` is written to the server env for NGC-backed model paths and readiness displays.
- LeRobot: may need `HF_TOKEN` for gated Hugging Face datasets or models.
- FiftyOne: may need `HF_TOKEN` for gated Hugging Face datasets.
- Isaac Lab and Genesis: no token is required by default.

Terraform remote state for managed workbenches is stored in the Nebius S3
bucket under:

```text
npa/terraform-state/<project-alias>/<workbench-name>/terraform.tfstate
```

Deploy saves the S3 backend bucket, endpoint, and access key under
`projects.<alias>.terraform_state` in `~/.npa/config.yaml` and writes that file
with `0600` permissions. Destroy reuses those exact backend credentials. If
Terraform still fails with `AccessDenied` while saving state after destroy, the
service account/access key used for `terraform_state` needs S3 `PutObject` on
`arn:aws:s3:::<bucket>/npa/terraform-state/<project-alias>/<workbench-name>/terraform.tfstate`
plus `GetObject` on that object and `ListBucket` on the bucket/prefix.

## SDK examples

```python
from pathlib import Path

from npa.adapter.sim_to_lerobot import convert
from npa.clients.http import HTTPClient
from npa.workflows.distill import run_distillation

convert(Path("./demos"), Path("./dataset"), fps=20, robot_type="franka_panda")

client = HTTPClient("http://workbench-ip:8080")
client.serve("/opt/lerobot/checkpoints/job/checkpoints/last/pretrained_model")
result = client.infer({"observation.state": [0.0] * 10})  # 9 joints + 1 gripper

run_distillation(project="eu-north1", remote=True, s3_bucket="s3://my-bucket/checkpoints/")
```

## Package map

- `npa.cli`: Typer CLI entrypoints
- `npa.clients`: Nebius, SSH, HTTP, config, and S3 helpers
- `npa.deploy`: Terraform provisioning and remote app deployment
- `npa.server`: FastAPI checkpoint-serving and inference server
- `npa.adapter`: sim demo -> LeRobotDataset v3 conversion
- `npa.genesis`: teacher training, demo generation, student evaluation
- `npa.lerobot`: local student training helpers
- `npa.workflows`: end-to-end distillation orchestration
