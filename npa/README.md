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
[reference catalog](../workflows/README.md) and
[workflow guide](../docs/workbench/npa-workflow-guide.md): validate and plan the
chosen specification, prepare its data and resources, submit it, then inspect
`npa workbench workflow status`, `logs`, and `artifacts`. The
[recovery guide](../docs/workbench/troubleshooting/known-footguns.md) covers
setup and runtime failures.

## Workbench Runtimes

Choose a runtime supported by the selected tool:

| Mode | Runs on |
| --- | --- |
| Workflow | SkyPilot jobs on the configured cluster; see [Workbench setup](../docs/workbench/getting-started.md) |
| `vm` / `container` | A Nebius VM managed through Terraform and SSH |
| `byovm` | An existing SSH-accessible VM supplied by you |
| `serverless` | Nebius AI Jobs or Endpoints, where the tool supports them |

See [runtime modes](../docs/workbench/runtime-modes.md) for direct deploy,
BYOVM, and serverless examples. A mode supported by one tool does not imply
support in every other tool.

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

For artifact conversion and sharing, see the
[CLI / SDK walkthrough](../docs/workbench/cli-sdk-yaml-walkthrough.md),
[Foxglove export](../docs/workbench/foxglove-export.md), and
[Rerun sharing](../docs/workbench/rerun-sharing.md).

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

To work on `npa` itself, create the contributor environment and use the `make`
targets from the repo root:

```bash
python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install -e "npa[dev,adapter]"

make test PYTHON="$(pwd)/npa/.venv/bin/python"        # unit suite
make test-smoke PYTHON="$(pwd)/npa/.venv/bin/python"  # onboarding CLI checks
make lint PYTHON="$(pwd)/npa/.venv/bin/python"        # ruff
```

Use an **absolute** interpreter path: the recipes change into `npa/` before
running. Without an override, Make prefers the contributor environment
`npa/.venv/bin/python`, then `python3` on `PATH`. Live and GPU tests are
deselected from `make test`; `make test-e2e` is the explicit live-infrastructure
target and needs the relevant credentials and resources.
For the real Cosmos Ray batch check, set `NPA_COSMOS3_RAY_LIVE_OUTPUT_URI`
to an operator-owned S3 prefix; it has no default. The check requires an existing
authenticated GPU service and writes two synthetic images plus their provenance.
See the [Cosmos Ray live-check instructions](../docs/workbench/cosmos3-ray-serve.md)
for the remaining environment variables and the exact test command.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).
