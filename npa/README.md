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

[flex-pi inference](../docs/workbench/flex-pi.md#cli-and-sdk) emits one JSON
document on stdout (`--output-format json`, the default); runtime diagnostics
go to stderr.

[Flex-Pi public training](../docs/workbench/flex-pi.md#public-yam-training)
uses a pinned real YAM dataset on four GPUs with effective batch 96. Run
`npa workbench flex-pi train --output-path s3://<artifact-bucket>/<run-prefix>`
inside the documented GPU workflow; `--mode profile` measures the same workload
before a complete training epoch. Outputs include full validation and a fresh
checkpoint-resume check. `--dry-run` returns the fixed contract without GPUs.
`--memory-fill` (`config.memory_fill` in workflows) defaults to `on`; `off` is
an unqualified candidate that requires verified normalization and exact parity
checks before execution.
`--activation-checkpointing` (`config.activation_checkpointing`) defaults to
`on`. Selecting `off` retains activations instead of recomputing them during
backward, using more GPU memory. Qualify each memory policy with a profile,
numerical parity and fresh resume before a full epoch; durable checkpoints
remain enabled with either policy.
`--microbatch-per-rank` (`config.microbatch_per_rank`) accepts `1` (default)
or `3`, with 24 or 8 accumulation steps respectively. Both keep effective
batch 96 and the complete 36-sample epoch tail. Larger microbatches use more
memory and can change numerical results; qualify the selected configuration
with a profile, fresh resume and held-out validation before accepting it.
`--cuda-graphs` (`config.cuda_graphs`) defaults to `off`. The experimental
`mot` option captures mixed-attention training with the pinned Torch 2.7.1
native CUDA graph API and requires `--activation-checkpointing off`. Input
preparation, noise sampling, validation and optimizer updates remain eager.
Capture failures are errors; every rank must report native forward and
backward graph replays. Qualify the GPU trace, numerical results, full held-out pass
and fresh resume before accepting a performance claim.

The four-host B300 workflow sets `config.training_nodes: "4"` and requests one
GPU on each host. `NPA_FLEX_PI_NODE_COUNT` comes from the resolved workflow
resources; the adapter checks SkyPilot's allocation and publishes only from
node zero. Provide the same original normalization and run `profile-resume`
before a full epoch. The [training guide](../docs/workbench/flex-pi.md#public-yam-training)
documents the per-host checkpoint join, runtime variables and topology caveats.

The package also provides project provisioning, storage, artifact conversion,
viewers, and an agent interface. Python access includes typed clients, shared
implementation functions, and wrappers around CLI callbacks; available imports
and return types vary by tool. See the
[CLI / SDK / workflow walkthrough](../docs/workbench/cli-sdk-yaml-walkthrough.md)
before integrating a tool programmatically.

Fleet recovery can remove a failed CPU pool without charging unchanged reserved
GPUs against free capacity again. The requested CPU count must be zero, every
other rendered capacity setting must match, and fresh provider evidence must
verify the retained pools and the removed CPU group's Terraform ownership.
Unknown identity or GPU growth still requires the normal capacity preflight.

For [Isaac Arena footage](../docs/workbench/isaac-arena.md#supported-policies),
`npa workbench isaac-arena evaluate --record-video --video-profile film`
requests native 4K capture with additional physics-frozen settling renders.
`standard` remains the default; workflows opt in with `config.video_profile`.
The updated capture runtime must be present in the selected image or a recorded
source overlay. Retained live output can be checked with
`NPA_INTEGRATION_E2E=1 NPA_ARENA_FILM_RESULT=/path/to/result.json npa/.venv/bin/python -m pytest npa/tests/e2e/test_isaac_arena_film_capture_live.py -q`.
That check reads the complete downloaded output bundle and launches no new job.

For shared Kubernetes clusters, use [team namespaces](../docs/workbench/namespaces.md) to configure
namespace selection and private SkyPilot contexts with `npa workbench namespace`.

## Install

For a persistent remote development environment, use
[`npa tools desktop`](../docs/tools/development-desktop.md). This operator tool
installs VS Code and Codex on an existing Ubuntu VM, provides a sharp browser
desktop with adjustable workspace size, and can enable authenticated HTTPS on a
public IP. Its setup and operation commands live under `npa tools`.
The browser desktop shares text with your device through copy/paste shortcuts
and **Paste** / **Copy to device** controls, with a manual clipboard fallback.
`npa tools desktop chat-setup --connect-vscode` adds authenticated mobile Codex
chat to the same HTTPS gateway; `npa tools desktop open --chat` opens it. Mobile
and VS Code share conversations, model/reasoning selections, and live activity.
The shared runtime setting covers desktop VS Code and Remote SSH clients on the VM.
Cloud chat setup also enables a persistent browser sign-in, automatic phone
routing to chat, and an installable Home Screen app using the same login.
`npa tools desktop chat-setup --local` attaches the same interface to existing Mac
sessions, including native VS Code views of mobile-owned running chats. Add
`--gateway-ssh-host <alias>` to use an existing authenticated gateway.
`npa tools desktop optimize --ssh-host <alias>` reduces desktop effects for faster
clicking and typing while preserving the running session and display resolution.

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

NPA pins its recording SDK and default hosted/share viewers to Rerun 0.38.1.
Recording inspection uses the supported streaming reader and accepts existing
0.31.4 RRD files without rewriting their contents or provenance. A recording
validator rejects ambiguous multi-recording files unless it selects each store
explicitly. For a custom viewer image, use a version that supports the producing
SDK; previously published container pins retain their recorded build versions.

## Package map

- `npa.cli`: Typer CLI entrypoints
- `npa.clients`: Nebius, SSH, HTTP, config, and S3 helpers
- `npa.deploy`: Terraform provisioning and remote app deployment
- `npa.server`: FastAPI checkpoint-serving and inference server
- `npa.adapter`: sim demo -> LeRobotDataset v3 conversion
- `npa.genesis`: teacher training, demo generation, student evaluation

  Genesis teacher training uses RSL-RL 5.5.1 actor/critic models. The loader
  retains legacy ActorCritic checkpoint support and validates saved dimensions
  before inference. ONNX export preserves RSL-RL 5 observation normalization.
  The Genesis extra pins an upstream MoviePy compatibility fix by source revision
  and archive hash so installation retains Pillow 12.3 or newer. The
  `genesis-test` extra adds CPU checkpoint regression tests to the complete test
  stage; the fast precheck does not install PyTorch. Genesis simulation remains
  in the separate `genesis` extra.
  The [Genesis skill](../skills/tools/genesis/SKILL.md#teacher-checkpoint-compatibility)
  describes the live GPU migration check and its numerical report.
- `npa.lerobot`: local student training helpers
- `npa.convert`, `npa.demo`, `npa.rerun`, `npa.workbench`, `npa.network`,
  `npa.workflow`: public SDK namespaces mirroring supported CLI commands
- `npa.sdk.workbench`: tool-specific clients and compatibility imports
- `npa.orchestration.npa_workflow`: specification loading, planning, execution,
  durable state, and recovery
- `npa.workflows`: workflow implementations and artifact discovery

## Developing and testing npa

Build actual RGB/action robot demonstrations with
[robot SDG and LeRobot export](../docs/workbench/token-factory-robot-sdg.md).
`npa workbench token-factory robot-sdg` uses S3 handoffs; the SDK's `robot_sdg`
accepts local paths. Install `npa[robot-sdg]` and `ffmpeg`. The default router
is Token Factory, and the simulator seed defaults to zero. Set
`NPA_TOKEN_FACTORY_ROBOT_SDG_LIVE=1` and `NPA_LEROBOT_PROOF_PYTHON` to a native
LeRobot 0.5.1 interpreter to run live simulation and dataset validation. Optional
`NPA_ROBOT_SDG_OUTPUT_DIR` preserves the run in a new local directory. These
environment variables are unset by default; the simulator uses no physical robot.

Build synthetic instruction datasets with the
[automatically routed Token Factory SDG pipeline](../docs/workbench/token-factory-sdg.md).
`npa workbench token-factory sdg` accepts S3 input/output paths; the SDK accepts
local files for development. Routing defaults to Token Factory's hosted Lightning
model, with Lightning/MiniMax generation and MiniMax review. Optional `--router jev`
requires a separate TypeSafe key. `NPA_TOKEN_FACTORY_SDG_LIVE=1` opts into paid live
pipeline tests; it defaults to unset.

The optional [Jev model router](../docs/workbench/jev-routing.md) selects between
eligible Token Factory text models in agent chat. `NPA_AGENT_MODEL_ROUTER=jev`
enables it during agent deployment/bootstrap; it defaults to unset.
`TYPESAFE_API_KEY` is required and can be saved in the existing private NPA
credential store. The guide includes the live evaluation script, the opt-in
`NPA_JEV_ROUTING_LIVE=1` tests, and provider-reported prefix-cache evidence.

To work on `npa` itself, create the contributor environment and use the `make`
targets from the repo root:

```bash
python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install -e "npa[dev,adapter]"

make test-smoke PYTHON="$(pwd)/npa/.venv/bin/python"  # onboarding CLI checks
make precheck  # CI pins, lint, formatting, and CI contract regressions
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_documentation_examples.py -q
```

After committing, run `git fetch origin main` and `make merge-precheck` before
pushing. This checks committed HEAD's merge with current main for conflicts and
inconsistent dependency fingerprints without modifying your index. It does not
run the full suite. Queue rejections receive a PR comment with failed jobs/steps
or timeout details; see the [merge-readiness guide](../CONTRIBUTING.md#merge-readiness-and-queue-rejections).

For the **full unit suite**, use CPython 3.12 on Linux with `ffmpeg` and `ffprobe` available.
Some runtime tests exercise Linux `/proc` and filesystem semantics, so macOS
can run the focused checks above but does not reproduce the full Linux gate.
The interpreter must provide `os.memfd_create`; some Conda builds omit it.
Install CI's CPU checkpoint/export dependencies in this same environment:

```bash
npa/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
npa/.venv/bin/python -m pip install -e "npa[sonic]"
umask 077  # Private files are required by publication handoff tests.
PATH="$PWD/npa/.venv/bin:$PATH" NPA_REQUIRE_FFMPEG=1 \
  make test PYTHON="$PWD/npa/.venv/bin/python" PYTEST_ADDOPTS='-n auto'
```

To recheck checkpoint selection from a completed GPU validation job, set
`NPA_E2E_CHECKPOINT_SELECTION_EVIDENCE_CONFIG` to an owner-only JSON file and run:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_checkpoint_selection_provider_evidence_live_e2e.py -q
```

There is no default evidence target; the live check skips without the variable.
The [test module](tests/e2e/test_checkpoint_selection_provider_evidence_live_e2e.py)
documents the required provider identity, digest-pinned image, source archive,
GPU platform/count, training-output prefix, and minimum episode count. It reads
existing resources, verifies checkpoint and source bytes, recomputes distances
from recorded final positions, and reruns selection in both candidate orders.
The input bundle must come from real policy rollouts; this check does not launch
training, establish policy quality, or claim a complete Sim2Real pipeline run.
Use `NPA_CONFIG_DIR` to select an isolated operator configuration.

The CPU wheel exercises real checkpoint loading without a GPU. See
[the CI environment](../.github/workflows/test.yml) for the complete coverage
gate; some optional checks also use Node, tmux, or Docker.

Pull requests run the full Python 3.12 suite,
dedicated Cypress job, focused Python 3.10/3.14 checks, and security gates.
Five duration-balanced coverage shards run with xdist and cached constrained
installs, then enforce the merged coverage floor. Full suites include smoke
and CLI install checks without a duplicate subsystem job. The queue verifies
this evidence against its combined latest-main candidate. Recognized prose-only
edits retain smoke, documentation, lint, guardrail, and security checks while
skipping runtime suites.
The five-minute `pr-precheck` checks dependency inputs, lint/format, guardrails, smoke and
full collection before expensive validation. The merge queue verifies fresh,
successful PR evidence for its identical Git tree and repeats secret,
confidentiality, source and dependency scans. Changed combined trees rerun
all tests, lint, guardrails and hostile-input checks; image checks rerun when
their inputs changed. Missing, failed or stale proof restores full queue
validation, so older PRs can adopt the policy without a forced branch refresh. The queue target
is ten minutes; hosted runner waiting can add delay.

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

Mocked browser checks require Google Chrome and run with
`bash npa/scripts/run_agent_cypress.sh --mock` from the repository root.
The wrapper installs the locked browser and native Codex adapter dependencies
when absent, then runs the native protocol tests and mocked desktop/agent specs.
For direct npm invocation, first run `npm ci --prefix npa/tests/browser` and
`npm ci --prefix npa/src/npa/tools/desktop/native`.
They use Chrome's software WebGL renderer for real canvas capture coverage;
Cypress 16's deprecated Electron browser cannot provide that context in CI.

CI uses cached uv installs constrained by `npa/ci/requirements.txt`. After changing
CI dependency inputs, run `npa/.venv/bin/python npa/scripts/ci_requirements.py
--update` with uv 0.12.5 and commit the refreshed pins. Add `--upgrade` only for an
intentional version refresh. The [contributor CI guide](../CONTRIBUTING.md#ci-dependency-setup-and-timing-reports)
also explains the automatic `ci-timing-report` job, whose summary and
JSON artifact separate runner waiting, setup, and execution for completed runs.
For queue rejections, follow the
[merge-readiness guide](../CONTRIBUTING.md#merge-readiness-and-queue-rejections).
The [validation concurrency policy](../CONTRIBUTING.md#validation-concurrency)
lets independent jobs use available GitHub runner capacity without shared
repository-wide job queues. Newer commits still cancel older checks of the same
PR. The final image-inventory check reports failed scans but stops when a run
is cancelled, so it cannot hold the replacement run behind an obsolete job.
Organization runner limits can cause waiting; already queued runs retain
their original workflow configuration until their branches are refreshed.
Full-suite PRs retain smoke coverage in their shards; the early precheck runs
guardrails once before those shards, and unsuccessful or cancelled shards no longer queue a coverage job.

Application and CI dependency scans reject known vulnerabilities even when the
same pin is already on `main`. Keep the AnyIO security floor at 4.14.2 or newer;
update `npa/requirements-lock.txt` and regenerate CI pins when changing package
requirements. `.github/dependabot.yml` checks Python, browser-test npm, and
GitHub Actions dependencies daily and groups version updates into one
`dependencies` PR. Review package declarations and generated locks together,
regenerate CI pins after Python input changes, and validate the combined batch. Reproduce the scan
with the [security gate instructions](../docs/security/merge-security-gate.md#reproduce-locally).

The base-image scanner uses Docker with Buildx and the checksum-verified Trivy
binary. From the repository root, `npa/.venv/bin/python
npa/scripts/scan_base_images.py --inventory
npa/docker/workbench/base-image-security.json --cache-dir <private-directory>`
scans the full inventory. `--matrix` emits every validated entry name without
starting a scan; `--entry-name <name>` scans exactly that existing entry and
rejects unknown names. These options are mutually exclusive. CI isolates each
entry on its own runner and requires every applicable result. See
[base-image scan storage and cleanup](../docs/security/image-reproducibility.md#cve-scanning).

The required [security check](../docs/security/merge-security-gate.md) is the
single automatic candidate workflow. It runs secrets, confidentiality, source,
runtime, lint, guardrail, and test gates, and calls an always-reporting image
security workflow. Image-affecting candidates receive the deep image checks;
unrelated candidates take a verified fast path. Main and weekly audits always
scan the full image inventory.
The image security workflow scans the pinned Python base after the same OS
update and upgrade used by FiftyOne's Dockerfile. It rebuilds this local scan
target without cache so newly published security fixes are included, then fails
on fixable CRITICAL OS findings. Each base has its own runner, temporary archive,
cache, and isolated builder. Owned build state is removed before the archive
scan; archives and temporary scan data are cleaned after each entry. The required
inventory aggregate fails when any applicable entry fails or does not finish.
This baseline check does not replace complete image scans before publication.

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

For the real storage-cleanup deletion check, set `NPA_STORAGE_CLEANUP_LIVE_E2E=1`
plus `NPA_E2E_PROJECT`, a private `NPA_CONFIG_DIR`, and
`NPA_STORAGE_CLEANUP_LIVE_E2E_EVIDENCE_DIR`; it has no default and deletes the
configured bucket and storage service account for real. See
[`tests/e2e/test_config_storage_cleanup_live_e2e.py`](tests/e2e/test_config_storage_cleanup_live_e2e.py)
for the full env contract and safety preconditions.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).

## Workbench specialists

Named operations can opt into `handoff_on_failure` (default `false`) to pass a
recorded terminal verification failure to the next configured model, retaining
the failed receipt and existing grants. It never retries an uncertain operation;
see the [specialist recovery guide](../docs/workbench/specialists.md).

For independently running GLM/DeepSeek agents, install `npa[agent-specialists]`
and run `npa workbench specialists --config <operator-team.json> serve`.
The [specialist guide](../docs/workbench/specialists.md) covers explicit model
endpoints, optional Jev routing, scoped workspaces, durable restart, task controls,
and the required `NPA_SPECIALISTS_TOKEN` service credential. `NEBIUS_TOKEN_FACTORY_KEY`
supplies hosted inference; `TYPESAFE_API_KEY` is needed only for optional Jev.
Configuration contains credential environment names, never credential values.
Profile `model_router: "token_factory"` uses a declared `routing_model` for one
structured classification call among `model_criteria` endpoints. It uses the
existing inference credential and requires no TypeSafe service. Set
`require_model_route: true` to block generation on unavailable or invalid routing;
default `false` records a primary-endpoint fallback. Routing usage and durable
attempts are retained separately. See [model-driven routing](../docs/workbench/specialists.md#model-driven-routing-through-token-factory).
Profile `model_router: "jev"` chooses among that profile's declared model endpoints
without changing its workspace or tool grants. Add `require_model_route: true`
to stop before generation when Jev is unavailable or abstains; the default
`false` preserves advisory fallback. See the [required Jev stack setup](../docs/workbench/specialists-jev-stack.md).
Optional `compact_context: true`
omits superseded observations from inference requests while preserving full
receipts. Observation operations can declare `wait_for` JSON states so workers
poll without routine model calls; `SpecialistTeam.wait_for_attention` lets a
coordinator wait outside its model turn and consume compact evidence reports.
The [workflow experiment runner](examples/specialists/workflows/README.md) also
offers `--coordination specialists-first` for predefined assignments with required
checks: it dispatches those workers directly and invokes Astra only for recovery
or evidence review. The default `completion` mode retains Astra planning.

## Workbench Studio

Studio `preview` and `final` can also deliver the finished MP4 with
`--output-path s3://<your-bucket>/<your-key>.mp4`. Optional `--storage-project`
selects an external NPA project alias. Uploads are verified by SHA-256 readback;
local renders and caches remain available.

`npa studio init --directory ./my-studio` creates a portable local film editor
using the installed renderer. Create a project from your own media with
`npa studio create`, author its storyboard, then draft, narrate and render it.
Install `npa[studio]` for optional speech generation and FFmpeg separately.
See the [Studio developer flow](../docs/demos/workbench-studio/README.md) for
configuration, offline narration, artifact search and privacy boundaries.

### MJLab

Install optional simulator packages with `pip install -e '.[mjlab]'` on Python
3.10–3.13, or use the dedicated MJLab GPU container recipe. `npa workbench mjlab`
provides `train`, `eval`, `export`, `list`, `status`, `system-info`, `deploy`, and
`workflow`; use `--help` for their options. Training preserves upstream defaults
unless overridden. Public handoffs are S3 URIs; `--dry-run` plans without metrics
or writes. The service requires `MJLAB_TOKEN` and `MJLAB_ALLOWED_S3_ROOTS` and
uses existing storage credentials. See the [MJLab guide](../docs/workbench/mjlab.md)
for request schemas, deployment Secrets, environment variables, validation and
image publication status.
`eval --video` also publishes `rollout.mp4` and a self-contained `rollout.html`
page alongside the measured evaluation manifest. The SDK and service expose the
same behavior with `video=True`.
