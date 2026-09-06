# npa

`npa` is the Nebius Physical AI CLI and Python package. Its primary command
namespace, `npa workbench`, runs robotics training, simulation, perception,
world-model generation, dataset curation, and evaluation on Nebius. Workbench
tools compose through S3 artifacts and `npa.workflow/v0.0.1` specifications
executed through SkyPilot.

Start with the [Workbench guide](../docs/workbench/README.md), choose a workload,
and follow [installation](../docs/install.md) and
[project setup](../docs/configuration.md) before provisioning its GPU resources.
The [command reference](../docs/cli/workbench.md) lists the installed tools;
`npa workbench <tool> --help` exposes each tool's actual commands.

The package also provides project provisioning, storage, artifact conversion,
viewers, and an agent interface. Python access includes typed clients, shared
implementation functions, and wrappers around CLI callbacks; available imports
and return types vary by tool. See the
[CLI / SDK / workflow walkthrough](../docs/workbench/cli-sdk-yaml-walkthrough.md)
before integrating a tool programmatically.

## Install

From the repository root, with your virtual environment active:

```bash
pip install -e npa
npa --version
npa workbench --help
```

The base package includes the non-GPU Workbench dependencies, including
FastAPI, LanceDB, and Rerun. Local engine extras such as `npa[genesis]`,
`npa[groot]`, and `npa[sonic]` are needed only when running those engines in
your Python environment; remote workloads use their container dependencies.
See [installation](../docs/install.md) for supported platforms, virtual
environments, and the separate SkyPilot environment.

Extra tools required by specific commands:

- `nebius` CLI for Serverless AI Endpoint deploys and managed Nebius deploy commands
- `terraform` for VM and container workbench deploys
- `ffmpeg` for `npa adapter convert`
- `ffmpeg` and Chrome/Chromium for `npa convert lerobot-to-mp4 --renderer rerun`
  (`NPA_RERUN_FFMPEG` and `NPA_RERUN_CHROME` may point to explicit executables)

## CLI layout

```bash
npa workbench lerobot ...
npa workbench genesis ...
npa workbench cosmos ...
npa workbench dataset ...
npa workbench workflow ...
npa workbench health ...
npa adapter convert ...
npa convert lerobot-to-mp4 ...
```

For a complete workflow, use the
[reference catalog](workflows/workbench/npa-workflows/README.md) and
[workflow guide](../docs/workbench/npa-workflow-guide.md): validate and plan the
chosen specification, prepare its data and resources, submit it, then inspect
`npa workbench workflow status`, `logs`, and `artifacts`. The
[recovery guide](../docs/workbench/troubleshooting/known-footguns.md) covers
setup and runtime failures.

The following are individual LeRobot/Genesis commands, including the older
five-stage distillation helper. Their setup and inputs are tool-specific; they
are not the canonical 14-stage Sim2Real workflow:

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
npa workbench workflow run distill --local
npa workbench workflow run distill --remote --project eu-north1 --s3-bucket s3://my-bucket/checkpoints/
```

## Workbench Runtimes

Deploy commands support these runtime modes where implemented:

- `vm`: provisions and manages a Nebius VM with Terraform and installs the tool over SSH.
- `container`: provisions and manages a Nebius VM with Terraform, then starts the tool container over SSH.
- `byovm`: skips Terraform entirely and deploys the app to an existing SSH-accessible VM.
- `serverless`: creates a Nebius Serverless AI Endpoint for a containerized serving backend. Cosmos supports this runtime first.

For Cosmos, `deploy --runtime serverless` creates the Endpoint resource with
the image, platform, preset, environment, and volumes baked into the resource.
`serve` is only a pre-warm/health operation; changing the served model or image
requires redeploying the endpoint.

```bash
npa workbench cosmos -p eu-north1 -n cosmos-sl deploy \
  --runtime serverless \
  --project-id project-... \
  --image ghcr.io/nebius/nebius-physical-ai/npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --server-port 8080 \
  --subnet-id vpcsubnet-... \
  --wait

npa workbench cosmos -p eu-north1 -n cosmos-sl status
npa workbench cosmos -p eu-north1 -n cosmos-sl serve
npa workbench cosmos -p eu-north1 -n cosmos-sl infer --prompt "A robot arm stacks colored cubes"
npa workbench cosmos -p eu-north1 -n cosmos-sl teardown --yes
```

When a Nebius project has multiple subnets, pass `--subnet-id` on serverless
deploy. Secrets should come from `~/.npa/credentials.yaml` or environment
variables; do not pass tokens as command-line arguments.

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

<a id="config"></a>

## Configuration

See [configuration](../docs/configuration.md) for project setup, credential
precedence, the credential-file layout, token access, and cross-project storage.
Use the [first-run prompts](../docs/workbench/agent-first-run.md) with a coding agent.

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

For Workbench integration, start with the documented module for your tool:

```python
import os
from npa.sdk.workbench import workflow

# Read an existing run's durable artifacts after workflow submission.
artifacts = workflow.artifacts(
    os.environ["NPA_RUN_ID"],
    workflow_s3_uri=os.environ["NPA_WORKFLOW_S3_URI"],
)
print(artifacts)
```

`npa.sdk.workbench.workflow` provides durable monitoring (`status`, `logs`,
`artifacts`, `runs`). Specification loading and planning live in
`npa.orchestration.npa_workflow`. Some tools, such as LeRobot and Genesis,
expose CLI callback wrappers under `npa.workbench`; those wrappers can print
output or raise CLI exits and do not guarantee typed response objects.
The [walkthrough](../docs/workbench/cli-sdk-yaml-walkthrough.md) explains these
differences with a detection-training service example.

Artifact utilities also have Python entry points:

```python
from npa import convert, demo, rerun

# Convert a LeRobot dataset to MP4.
convert.lerobot_to_mp4(
    input_path="s3://my-bucket/dataset/",
    output_path="trajectory.mp4",
    renderer="matplotlib",
)

# Stage demo artifacts.
demo.stage(target_bucket="customer-bucket", target_project="eu-north1")

# Share a Rerun recording.
result = rerun.host("recording.rrd")
print(f"View at: {result.share_url}")
```

Lower-level access for advanced workflows:

```python
from npa.adapter.lerobot.render import render_lerobot_to_mp4_result
from npa.clients.http import HTTPClient
```

## Package map

- `npa.cli`: Typer CLI entrypoints
- `npa.clients`: Nebius, SSH, HTTP, config, and S3 helpers
- `npa.deploy`: Terraform provisioning and remote app deployment
- `npa.server`: FastAPI checkpoint-serving and inference server
- `npa.adapter`: sim demo -> LeRobotDataset v3 conversion
- `npa.genesis`: teacher training, demo generation, student evaluation
- `npa.lerobot`: local student training helpers
- `npa.convert`, `npa.demo`, `npa.rerun`, `npa.workbench`, `npa.network`,
  `npa.workflow`: public SDK namespaces mirroring supported CLI commands
- `npa.sdk.workbench`: tool-specific clients and compatibility imports
- `npa.orchestration.npa_workflow`: specification loading, planning, execution,
  durable state, and recovery
- `npa.workflows`: workflow implementations and artifact discovery

## Developing and testing npa

To work on `npa` itself, install the dev extra into your activated venv and use
the `make` targets from the repo root:

```bash
pip install -e "npa[dev]"   # pytest, pytest-mock, pytest-cov, pytest-timeout, ruff

make test PYTHON="$(pwd)/.venv/bin/python"        # unit suite
make test-smoke PYTHON="$(pwd)/.venv/bin/python"  # onboarding CLI checks
make lint PYTHON="$(pwd)/.venv/bin/python"        # ruff
```

Use an **absolute** interpreter path: the recipes change into `npa/` before
running. Without an override, Make prefers the contributor environment
`npa/.venv/bin/python`, then `python3` on `PATH`. Live and GPU tests are
deselected from `make test`; `make test-e2e` is the explicit live-infrastructure
target and needs the relevant credentials and resources.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).
