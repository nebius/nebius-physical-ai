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

The cluster GPU smoke manages an owned SkyPilot API session through workload
cleanup; see [SkyPilot setup](../docs/orchestration/skypilot-setup.md#verify).
The [PAIDF starter guide](../workflows/guides/paidf-cosmos3.md#audit-a-completed-default-starter-run)
also provides a read-only live audit using the selected run URI, project, and
saved pre-submission UTC timestamp. Its test settings are scoped to the audit
shell and do not submit work. For task-specific augmentation, set the optional
`appearance_profiles_json` workflow config to a JSON array of coherent lighting,
background, color-grade and surface-finish profiles; its empty default retains
the starter sampler. Cosmos3's `caption_instruction` supplies `augment_subject`
as context while requiring uncertainty for unclear features; override it for
task or camera terminology. Source and generated-video caption frames are sampled
across the complete decoded clip, including both endpoints, instead of stopping
after its first eight seconds. Older caption workflows retain their default
instruction. With edge transfer, `transfer_rgb_weight` optionally adds the
complete source RGB video as a native conditioning hint, weighted relative to
edge weight 1. Its default 0 disables the hint; it never blends source pixels
into generated output. Independently, `transfer_first_chunk_conditional_frames`
defaults to 1, anchoring the first generation window to the original RGB frame.
Set it to 0 with edge transfer to allow a new appearance from the first frame;
complete source edges and generated overlap between later windows remain active.
This can also reduce object-identity preservation, so review actual paired clips.
`transfer_cfg_normalization` accepts `disabled` (the compatibility default) or
`enabled` with edge transfer. It forwards the pinned framework's `normalize_cfg`
sampling option and records the effective boolean in `transfer.json`. Compare
matched sources, prompts and seeds before adopting it; this option does not
establish geometry or contact fidelity.
The [realistic manipulation guide](../docs/workbench/guides/paidf-realistic-augmentation.md)
includes the battery dataset, configuration examples and quality review.
The [LeRobot comparison](../docs/workbench/guides/paidf-lerobot-realism.md)
extends that review to cup opening, coffee preparation and a simulated cube lift.

Extra tools required by specific commands:

- `ray[default]==2.58.0` in the NPA application environment for
  `npa workbench alpamayo2-super sweep`. The workflow renderer installs this
  dependency for the [Ray experiment templates](../docs/workbench/alpamayo2-super.md#ray-experiments).
  Use `--sample-indices`, `--seeds`, and `--diffusion-steps` for a baseline,
  or `--input-path` and `--minimum-ade` to refine a completed S3 report.
- `nebius` CLI for Serverless AI Endpoint deploys and managed Nebius deploy commands
- `terraform` for VM and container workbench deploys
- `ffmpeg` for `npa adapter convert`
- `ffmpeg` and Chrome/Chromium for `npa convert lerobot-to-mp4 --renderer rerun`
  (`NPA_RERUN_FFMPEG` and `NPA_RERUN_CHROME` may point to explicit executables)

## CLI layout

```text
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

The [Franka transfer workflow](../docs/workbench/guides/franka-rl-transfer.md)
retains invalid hosted visual judgments as failed audit evidence. Its
`npa.workflows.franka_rl visual-evaluate --prior-judgments-path` option accepts
a verified interrupted audit so saved responses are revalidated and only missing
episodes make new requests; omit the option for a fresh audit.
The workflow's `learning_recipe=adaptive-bounded-exploration` selects bounded learned joint targets and exploration,
hold-aligned rewards, richer observations, and a training-outcome curriculum.
Use `learning_recipe=joint-baseline` for the historical comparison. The module's
`prepare --learning-recipe` option seals this choice. Lift/hold thresholds stay fixed;
new runs additionally enforce measured simulation limits and a task-domain envelope.
UR10e's new sealed physics profile preserves its USD mimic-joint mechanics.

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
For a headless machine, choose
[service-account authentication or human OAuth](../docs/configuration.md#authentication-on-a-headless-machine).
Separate skills cover unattended service identities, SSH callback forwarding,
and the original private callback copy/paste helper.

Managed workbench teardown reuses the saved Terraform backend. See
[Terraform state](../docs/configuration.md#terraform-state-for-managed-workbenches)
for its storage path and permissions.

## SDK examples

Plan a workflow from Python without credentials or cloud resources. Run this
from the repository root in the environment where you installed `npa`:

```python
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec

spec = load_spec("workflows/testing/cosmos3-generate.yaml")
validate_spec(spec)
plan = build_plan(spec, run_id="demo")
for step in plan.steps:
    print(step.state, step.tool_ref)
```

Expect `generate workbench.cosmos3.generate`. The plan resolves the checked-in
example; execution still requires real input, storage, credentials, and GPUs.

After submitting a real run, read its artifacts through the monitoring SDK.
Set `NPA_RUN_ID` and `NPA_WORKFLOW_S3_URI` to the values returned by submission,
and configure the project's S3 credentials first:

```python
import os
from npa.sdk.workbench import workflow

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

make test-smoke PYTHON="$(pwd)/npa/.venv/bin/python"  # onboarding CLI checks
make lint PYTHON="$(pwd)/npa/.venv/bin/python"        # ruff
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_documentation_examples.py -q
```

For the **full unit suite**, use CPython 3.12 on Linux with `ffmpeg` and `ffprobe` available.
Some runtime tests exercise Linux `/proc` and filesystem semantics, so macOS
can run the focused checks above but does not reproduce the full Linux gate.
The interpreter must provide `os.memfd_create`; some Conda builds omit it.
Install CI's CPU checkpoint/export dependencies in this same environment:

```bash
npa/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
npa/.venv/bin/python -m pip install -e "npa[sonic]"
umask 077  # Private files are required by publication handoff tests.
PATH="$PWD/npa/.venv/bin:$PATH" NPA_REQUIRE_FFMPEG=1 \
  make test PYTHON="$PWD/npa/.venv/bin/python" PYTEST_ADDOPTS='-n auto'
```

The CPU wheel exercises real checkpoint loading without a GPU. See
[the CI environment](../.github/workflows/test.yml) for the complete coverage
gate; some optional checks also use Node, tmux, or Docker.

Pull requests receive smoke, affected subsystem tests, and security feedback.
Agent/browser changes also run Cypress before queue admission; unknown and
shared changes receive full Python 3.12 validation. The merge queue runs the
combined latest-main candidate through one dedicated browser job and focused
Python 3.10/3.14 checks alongside five duration-balanced Python 3.12 coverage
shards, then enforces the merged floor. Recognized prose-only edits retain smoke,
documentation, lint, guardrail, and security checks while skipping runtime suites.
See the [contributor CI guide](../CONTRIBUTING.md) for the conservative selection
rules and local inspection command. Scheduled/manual audits run the full suite
on all three supported versions. Every full Python 3.12 run publishes module
timings, with merged profiles from successful runs for reviewed rebalancing.
The focused compatibility check runs before CPU tensor dependencies are
installed, so async cancellation and isolated SkyPilot fixture regressions
surface before merge. Run it locally with:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_ci_workflows.py \
  npa/tests/docker/test_base_image_scan.py \
  npa/tests/orchestration/skypilot/test_workflow_logs.py \
  npa/tests/workbench/test_cosmos3_nano_video_server.py -q
```

CI uses cached uv installs constrained by `npa/ci/requirements.txt`. After changing
CI dependency inputs, run `npa/.venv/bin/python npa/scripts/ci_requirements.py
--update` with uv 0.12.5 and commit the refreshed pins. Add `--upgrade` only for an
intentional version refresh. The [contributor CI guide](../CONTRIBUTING.md#ci-dependency-setup-and-timing-reports)
also explains the automatic `ci-timing-report` job, whose summary and
JSON artifact separate runner waiting, setup, and execution for completed runs.

The required [security check](../docs/security/merge-security-gate.md) is the
single automatic candidate workflow. It runs secrets, confidentiality, source,
runtime, lint, guardrail, and test gates, and calls an always-reporting image
security workflow. Image-affecting candidates receive the deep image checks;
unrelated candidates take a verified fast path. Main and weekly audits always
scan the full image inventory.
The image security workflow scans the pinned Python base after the same OS
update and upgrade used by FiftyOne's Dockerfile. It rebuilds this local scan
target without cache so newly published security fixes are included, then fails
on fixable CRITICAL OS findings. All seven bases run inside one job with two
bounded local workers. One Trivy database download is hard-linked into isolated
worker caches, rather than using seven queued runners or a lock-contended shared
cache. This baseline check does not replace the
complete image scans required before publication.

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

## Workbench Studio

`npa studio init --directory ./my-studio` creates a portable local film editor
using the installed renderer. Create a project from your own media with
`npa studio create`, author its storyboard, then draft, narrate and render it.
Install `npa[studio]` for optional speech generation and FFmpeg separately.
See the [Studio developer flow](../docs/demos/workbench-studio/README.md) for
configuration, offline narration, artifact search and privacy boundaries.
