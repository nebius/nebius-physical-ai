# PAIDF with Cosmos 3: setup and run guide

Run the [Physical AI Data Factory (PAIDF) Cosmos 3 workflow](../main/paidf-cosmos3.yaml)
on Nebius AI Cloud. This guide covers account prerequisites, local installation,
project storage, Kubernetes setup, submission, and output inspection. Follow it
in a terminal or give it to a coding agent that operates your terminal.

The workflow selects a robot video, captions it with a hosted vision-language
model through Token Factory, generates appearance variants with Cosmos3-Nano
on your GPU, and evaluates them with Cosmos Evaluator. Accepted variants pass
through captioning, Cosmos Curator, and FiftyOne Brain before a final report.
If variants remain rejected after the configured refinement passes, the workflow
preserves Rerun quality evidence and stops before curation.

> **Validation scope:** Setup commands were exercised on Linux with Python 3.12
> and an existing RTX PRO 6000 Blackwell cluster. Full live validation of the
> structural-transfer workflow is in progress. See [validation](#validation)
> for completed checks and remaining limits.

## Before you start

You need:

- A Nebius AI Cloud account with billing enabled. If you are new to Nebius,
  follow the [account and billing setup](https://docs.nebius.com/signup-billing/sign-up).
- A project in the region where you will run the GPU workload. You can use a
  default project or [create a project](https://docs.nebius.com/iam/manage-projects).
  Record its project ID, tenant ID, and region from the
  [web console](https://console.nebius.com/).
- Permission to configure project storage and create a Kubernetes cluster, or
  access to an existing cluster. First-time storage provisioning requires
  project admin permission. For a shared project, have its administrator grant
  the required access before continuing.
- Hugging Face and Token Factory accounts; P4 explains the required tokens and
  model access.

You do not need a cluster in advance. S5 shows how to create one or adopt an
existing cluster. Each GPU task requests one GPU, 16 vCPUs, and 128 GiB RAM
on a single node. CPU stages request 4 vCPUs and 16 GiB RAM; reserve additional
capacity for the SkyPilot controller (2 vCPUs and 8 GiB by default), Kubernetes
overhead, and other running workloads. Nodes also need disk space for container
images and model weights. The live validation used RTX PRO 6000 Blackwell; S5
uses that platform for the new-cluster example. Check quota and available
capacity in your region before provisioning.

## P — Prerequisites

### P1. Install the base tools

On macOS with Homebrew:

```bash
brew install python@3.12 git kubectl awscli ffmpeg socat netcat jq
```

On Ubuntu 24.04, install the base packages, including its
[Python 3.12 package](https://packages.ubuntu.com/noble/python3.12) and venv module:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git curl ffmpeg socat netcat-openbsd jq
python3.12 --version
```

Also install `kubectl` using the
[Kubernetes Linux instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
and `aws` using the
[AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
For a Linux distribution without Python 3.12 packages, use
[uv's Python installation instructions](https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version)
to install version `3.12`, and verify that `python3.12 --version` succeeds.

Use Python 3.12 for both the NPA environment and isolated SkyPilot environment
in this guide. The supported
SkyPilot runtime requires Python 3.9–3.12; a newer interpreter can fail at submit.
For the new-cluster path in S5, also install Terraform 1.x using the
[platform tool instructions](../../docs/install.md#5-optional-operator-tools)
and have an SSH public key available. NPA discovers an existing key such as
`~/.ssh/id_ed25519.pub`; if you need a new key, create it with
`ssh-keygen -t ed25519` and keep the private key on your machine. Adopting an
existing cluster does not require Terraform.

### P2. Install the Nebius CLI

The recorded setup used version `0.12.254`, which is accepted by
the repository's CLI compatibility check:

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | NEBIUS_CLI_VERSION=0.12.254 bash
export PATH="$HOME/.nebius/bin:$PATH"
```

Use the version requested by NPA if its compatibility check changes. An
unsupported CLI can produce an authentication-looking error; check the version
before recreating a working profile.

### P3. Select the project and authenticate

Select your project in the Nebius web console. Copy its project ID from the
project selector and get its tenant ID and region from the project details;
see [project management](https://docs.nebius.com/iam/manage-projects#how-to-get-a-project-id).
Replace the placeholders below. `paidf` is a local profile/alias name you can
choose; it is not a cloud project ID. Keep these non-secret variables available
for the setup steps:

```bash
export TENANT_ID='<your-tenant-id>'
export PROJECT_ID='<your-project-id>'
export REGION='<your-region>'
export PROJECT_ALIAS=paidf
export NPA_NEBIUS_PROFILE="$PROJECT_ALIAS"

nebius profile create "$NPA_NEBIUS_PROFILE" \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id "$PROJECT_ID"
```

Authentication opens a browser. Sign in with the account that has access to
this project. If the named profile already exists, use
`nebius profile activate "$NPA_NEBIUS_PROFILE"` instead of creating it again.
See [CLI authentication](https://docs.nebius.com/cli/configure) for service-account
and multi-tenant setup. If you see `invalid IAM subject` or `PermissionDenied`,
check the selected account and project permissions in the web console.

### P4. Prepare Hugging Face and Token Factory access

Create a Hugging Face read token under **Settings → Access Tokens**. Review the
model terms and request access where the model page requires it:

- [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)
- [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)

The current `health access --capability cosmos3` command checks these repository
entries. Follow any additional dependency or access requirements reported by
your installed version; do not treat a token's presence as proof of model access.
See [Cosmos 3 access preflight](../../docs/workbench/cosmos3-access-preflight.md).

For Token Factory, sign in to the [Token Factory console](https://tokenfactory.nebius.com/)
and ensure your account can make API requests. Under **API keys**, create a key
and save it when displayed; it cannot be reopened later. The
[official authentication instructions](https://docs.tokenfactory.nebius.com/api-reference/introduction#authentication)
describe this process. This key is separate from your Nebius Cloud IAM token.

## S — One-time setup

### S1. Install the repository and NPA CLI

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -e npa/
npa --version
npa workbench workflow submit --help
```

The last command must show the workflow submit options before you continue.
NPA's version number alone does not identify the installed source revision.
`No such command 'submit'` is a CLI installation or command-selection problem;
changing the workflow filename cannot fix it. Follow the
[installation recovery steps](#if-submit-is-missing) below.

Run the remaining commands from this repository root. This guide uses `.venv`;
contributor validation tooling uses `npa/.venv` as described in
the [installation guide](../../docs/install.md).

#### If `submit` is missing

Check which executable your shell selects and which checkout your environment
imports:

```bash
type -a npa
.venv/bin/python -c 'import npa, sys; print(sys.executable); print(npa.__file__)'
git log -1 --format='%h %s'
.venv/bin/npa workbench workflow submit --help
```

If the explicit `.venv/bin/npa` command works, reactivate that environment and
clear the shell's cached command location:

```bash
source .venv/bin/activate
hash -r
npa workbench workflow submit --help
```

An alias or shell function named `npa` can still shadow the executable; use
`.venv/bin/npa` directly until that shell customization is corrected. If the
explicit executable also lacks `submit`, update this checkout to the intended
current repository revision and reinstall it into this environment:

```bash
.venv/bin/python -m pip install --upgrade -e ./npa
.venv/bin/npa workbench workflow submit --help
test -f workflows/main/paidf-cosmos3.yaml
```

The import path should point into this checkout's `npa/src/npa`. A different
path means you are still running a different installation. If a current checkout
still fails, retain the command, revision, and traceback for diagnosis; do not
proceed to storage or cluster setup with a missing command.

### S2. Configure the project and provision its storage

Use your active Nebius CLI profile to configure NPA and provision the bucket
and storage credentials for the selected project. First verify that the profile
can authenticate without reopening the browser:

```bash
unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN
npa workbench health preflight --checks nebius
npa configure --no-interactive --provision \
  --tenant-id "$TENANT_ID" --project-id "$PROJECT_ID" \
  --region "$REGION" --project-alias "$PROJECT_ALIAS"
npa configure --show
```

Note the bucket and storage endpoint it reports. `--provision` here performs
storage setup; cluster creation or adoption is a separate step below.

### S3. Store the service tokens

Merge this `tokens` mapping into `~/.npa/credentials.yaml`, replacing the
placeholders in your local editor. Preserve the project storage records that
`configure` created:

```yaml
tokens:
  HF_TOKEN: <your-hugging-face-read-token>
  NEBIUS_TOKEN_FACTORY_KEY: <your-token-factory-api-key>
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

Storage keys belong to the selected project's record under
`project_credentials.projects`. Use the
[credential-file reference](../../docs/credentials.yaml.example) to inspect the
shape without copying credential values into the repository or shell history.
Submit resolves the named secrets from this store or the process environment.

### S4. Check credentials and model access

```bash
npa workbench health preflight
npa workbench health access --capability cosmos3
```

Resolve failures for S3, Token Factory, and the required Hugging Face assets
before submitting. An NGC warning is not a requirement to add NGC credentials
for this Cosmos 3 path. These checks establish access; they do not prove a
successful GPU run or acceptable generated data.

### S5. Create or adopt a Kubernetes cluster

Choose one path below. For an existing cluster, use the adoption path so NPA
records its identity before considering provisioning.

#### Create a new cluster

After S4 passes, choose a cluster name and preview the topology. This example
requests one RTX PRO 6000 GPU node and one CPU node for orchestration and CPU
stages. Terraform and an SSH public key must be available as described in P1.

```bash
export CLUSTER_NAME=paidf-cosmos3
npa workbench health preflight --checks nebius
npa provision-if-absent --project "$PROJECT_ALIAS" --cluster-name "$CLUSTER_NAME" \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand \
  --dry-run --output-format json
```

Check the project, region, node types, and disk sizes in the plan. When they
match your intended setup and quota, run the same `provision-if-absent` command
without `--dry-run --output-format json`. Wait for provisioning and node health
checks to succeed. Do not proceed with a degraded cluster. This creates cloud
resources; use the [teardown guide](../../docs/teardown.md) when you finish with
a cluster you own.

Inspect the resulting cluster and load its kubeconfig:

```bash
npa cluster status --name "$CLUSTER_NAME" --project "$PROJECT_ALIAS"
export KUBECONFIG="$HOME/.npa/clusters/$CLUSTER_NAME/kubeconfig"
export KUBE_CONTEXT="$(kubectl config current-context)"
npa skypilot bind-controller \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT"
```

If NPA reports a different kubeconfig path, use that path. Continue with S6.
The [Workbench setup guide](../../docs/workbench/getting-started.md#verify-kubernetes-access)
has more detail about provisioning and cluster access.

#### Adopt an existing cluster

Open **Managed Kubernetes** in the Nebius web console for your selected project
and copy the cluster's ID and name:

```bash
export CLUSTER_ID='<your-cluster-id>'
export CLUSTER_NAME='<your-cluster-name>'
```

Fetching Kubernetes credentials alone does not register the cluster with NPA.
Fetch them, find the exact context name, then record the cluster and bind its
controller:

```bash
nebius mk8s cluster get-credentials --id "$CLUSTER_ID" --external
kubectl config get-contexts
```

```bash
export KUBE_CONTEXT='<context-from-the-previous-command>'
npa cluster kubeconfig \
  --cluster-name "$CLUSTER_NAME" \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT"
npa skypilot bind-controller \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT"
```

The Nebius CLI profile supplies cloud authentication, the NPA project alias
selects saved project credentials, and the Kubernetes context selects the
cluster. Use the values belonging to the selected project throughout.

### S6. Bootstrap and verify SkyPilot

Resolve Python from your own environment. This works on Linux and macOS without
assuming a Homebrew installation directory; `python3.12` must be on `PATH`.

```bash
npa skypilot bootstrap --python "$(command -v python3.12)"
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
export KUBECONFIG="$HOME/.npa/clusters/$KUBE_CONTEXT/kubeconfig"
npa skypilot verify --cluster "$KUBE_CONTEXT"
npa workbench workflow gpus --context "$KUBE_CONTEXT" --project "$PROJECT_ALIAS"
```

Verification must report Kubernetes enabled for this context. Copy the exact
GPU spelling from discovery; for example, an RTX PRO 6000 cluster may advertise
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`. Use it with the requested count in the run
section. See [SkyPilot setup](../../docs/orchestration/skypilot-setup.md).

### S7. Configure the AWS profile for direct artifact downloads

NPA submission resolves project storage from its credential store. The `aws`
commands in this guide use a profile named `nebius`. If you use those commands,
create the profile directory first:

```bash
mkdir -p ~/.aws
chmod 700 ~/.aws
```

Merge the selected project's access and secret keys from S2 into your local
`~/.aws/credentials`:

```ini
[nebius]
aws_access_key_id = <your-project-s3-access-key-id>
aws_secret_access_key = <your-project-s3-secret-access-key>
```

Merge the matching region and endpoint into `~/.aws/config`:

```ini
[profile nebius]
region = <your-region>
endpoint_url = https://storage.<your-region>.nebius.cloud
```

Use the exact endpoint reported by `npa configure --show`. Keep `endpoint_url`
at profile level, protect both files, and check that the AWS CLI selects the
intended profile:

```bash
chmod 600 ~/.aws/credentials ~/.aws/config
aws configure list --profile nebius
```

This profile serves the direct download commands below. NPA resolves submission
credentials from the selected project's own storage record.

### S8. Run the checkout's NPA code

Current submit stages the NPA package automatically for tasks that need it,
uses a content-addressed source prefix, and persists that reference for recovery.
Pinned workbench images also contain an NPA copy. This workflow sets
`config.source_overlay: true` so those tasks use the submitted checkout's NPA
adapters, including structural transfer and aligned evaluation. No shell
environment override is needed. The overlay retains the image's installed tool
dependencies. It also selects the current Token Factory client, which requests
direct answers for captioning and evaluator questions.

Keep the repository checkout available when you submit and use the canonical
spec at `workflows/main/paidf-cosmos3.yaml`.

## R — Run the workflow

### R1. Select the project, GPU, and caption model

In a new terminal, return to your checkout and activate `.venv`. Restore the
project alias and cluster context selected above, and set the bucket and
accelerator from your setup results:

```bash
source .venv/bin/activate
export PROJECT_ALIAS=paidf
export NPA_NEBIUS_PROFILE="$PROJECT_ALIAS"
export KUBE_CONTEXT='<your-adopted-context>'
export BUCKET='<your-configured-bucket-name>'
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
export KUBECONFIG="$HOME/.npa/clusters/$KUBE_CONTEXT/kubeconfig"
export NPA_WORKFLOW_GPU_ACCELERATOR='<discovered-gpu-name>:1'
SPEC=workflows/main/paidf-cosmos3.yaml

npa workbench token-factory models
```

Choose an available vision model for the captioning and evaluator stages:

```bash
export CAPTION_MODEL='<available-vision-model-id>'
```

The workflow currently defaults to `MiniMaxAI/MiniMax-M3`; check the model list
for availability at run time.
Do not assume a historically working model is still offered.

Workbench images use the public anonymous GHCR namespace
`ghcr.io/nebius/nebius-physical-ai` by default. No registry credential or pull
secret is needed for that mirror. Use the repository's selected image pins.

### R2. Reserve a run ID, validate, and inspect the plan

```bash
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS")"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/workflow-runs/$RUN_ID/skypilot"
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" --assume-decision promote_checkpoint \
  --var bucket="$BUCKET" --var caption_model="$CAPTION_MODEL" --json
npa workbench workflow preflight-images "$SPEC" \
  --assume-decision promote_checkpoint --project "$PROJECT_ALIAS"
```

Stop and resolve any failed check. The assumed decision previews the accepted
path and checks its downstream images; the runtime submit below follows actual
evaluator decisions. Leave image preflight enabled so incompatible images fail
before a stage starts.

Keep this SkyPilot directory for this run's submission, monitoring, resume, and
cleanup. An isolated API binds the concrete storage prefix, including the run
ID. Rerun R2 for a new experiment so it gets a fresh run ID and API directory;
reusing the previous run's API can fail before launch with `different executing
identity`. After a run finishes, follow the [controller cleanup
procedure](../../docs/teardown.md) in its original environment before moving on.
Keep the run's state and artifacts available for inspection.

### R3. Submit

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --var caption_model="$CAPTION_MODEL" \
  --runtime \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --project "$PROJECT_ALIAS" \
  --durable-s3
```

The first run needs no input preparation: submit fetches and verifies the
pinned upstream starter video, stages it for the run, and passes the source
to the Cosmos 3 workflow. It also stages source code and forwards the four
named secrets without requiring their values in the command.
Both starter variants passed the exploratory default quality gate in the
recorded live run. Review generated data separately for training suitability.
Generation and evaluator results can vary, and a new run can still be rejected.

On a cold worker, image pulls, source setup, and model downloads happen before
the payload. Runtime stages launch managed jobs and can repeat worker setup,
so even short CPU stages may take several minutes of wall time. Model downloads
can dominate GPU startup. Inspect stage logs to distinguish setup from payload
progress, and keep the submit command running so its driver can launch later
stages. R4 describes recovery if that driver is interrupted.

`--runtime` lets the orchestrator read evaluator decisions and execute real
refinement loops. This workflow declares `metadata.executionMode: runtime`, so
submission also selects this mode automatically when the flag is omitted. Keep assumed
decisions in the R2 planning command only; omit them from runtime submission
so a missing decision cannot fall back to assumed promotion. The command keeps
the workflow's configured variant count and refinement settings. A preview with
`--plan-only` uses static rendering even when `--runtime` is present; use R2 to
preview, and R3 to execute.

For your own video, add **one** of these input options to the submit command:

- `--input-video /absolute/path/source.mp4` for a local H.264 MP4.
- `--input-uri 's3://<your-bucket>/<your-prefix>/source.mp4'` for a stored MP4.

The workflow retains the selected original and prepares a constant-rate,
letterboxed reference. Native edge controls cover the complete prepared video;
generation and evaluation verify its frame count and timestamps. Inspect the
resulting action and contacts as well as the quality report. For episode/camera selection and the
generation contract, see the [Cosmos 3 workflow guide](../../docs/workbench/guides/paidf-cosmos3.md).

### R4. Monitor and recover

In another terminal, restore the same project, run ID, and other R1 variables,
then select that run's SkyPilot directory. Run these commands
separately. Stop the status watch with Ctrl-C before requesting a log tail, or
use another terminal:

```bash
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/workflow-runs/$RUN_ID/skypilot"
npa workbench workflow status "$RUN_ID" --project "$PROJECT_ALIAS" --watch
npa workbench workflow logs "$RUN_ID" --project "$PROJECT_ALIAS" \
  --stage generate-variants --no-follow
```

The runtime may submit several managed jobs or attempts. Use NPA's stage records
to identify the GPU task and each retry; do not assume one job ID or a fixed
15-row queue describes every run. Repeat the log command to retrieve a fresh
tail. The current CLI captures `--follow` output until the underlying log process
returns, so it may not display incremental output while a stage is running.

During runtime execution, the summary status and artifact index can lag the
per-wave record. For the default prefix used in this guide, read the durable
record with the AWS profile configured in S7:

```bash
aws s3 cp "s3://$BUCKET/paidf-cosmos3/$RUN_ID/npa-workflow/runtime.json" - \
  --profile nebius | jq '{status, waves: [.waves[] | {key, status}]}'
```

Each wave reports whether it is running or succeeded. A missing stage row or
`manifest_pending` in the summary does not establish that no job launched;
check this record and the stage logs before deciding to resume or submit again.

Live log lookup uses the selected SkyPilot directory and verifies its API
identity. Keep that directory tied to the run ID; finding the run in S3 does not
select its API automatically. If logs report an identity mismatch, check the
selected directory and use the durable record above while the operator
reconciles the exact controller. Do not reset shared state or cancel a running
GPU job solely because its live logs are unavailable.

An isolated API also verifies the project configuration it started with.
Staging a different checkout revision can update the project's saved source URI
and stop an existing driver with `credential configuration changed after
verification`, even when its credentials are unchanged. Finish the active run
before changing that configuration, or use separate NPA configuration stores
for independently isolated controllers. The check also covers the Nebius CLI
credential cache: another process refreshing a shared cache can interrupt the
driver. Use a dedicated operator environment for a long run and keep its
configuration stable. Preserve the run records and original authentication
state; do not edit API ownership records or assume that restarting the API
accepts changed credentials. Ask the platform operator to reconcile the exact
controller before retrying after this error.

For an interrupted launch, follow the recovery command reported by NPA. Replace
`--run-id "$RUN_ID"` with `--resume-run "$RUN_ID"` in the same submit command,
retaining the project, runtime, inputs, configuration, and secret names.
`--run-id` and `--resume-run` are mutually exclusive. Resume reconciles recorded
work; it does not turn a rejected result into an accepted one. Repeat R2 to
reserve a fresh run and API directory after changing the experiment. See the
[run lifecycle](../../docs/run-lifecycle.md).

### R5. Find and change generation and evaluation settings

The settings are in the top-level `config:` block of
[`workflows/main/paidf-cosmos3.yaml`](../main/paidf-cosmos3.yaml).
Override supported keys for one run with `--var key=value` in both the R2 plan
and R3 submit commands, or edit a copy of the spec and set `SPEC` to that copy.
Use a fresh run ID after changing inputs or settings.

| Setting | Default | What it controls |
| --- | --- | --- |
| `cosmos3_mode`, `cosmos3_checkpoint` | `video2video`, `Cosmos3-Nano` | Source-video conditioning and generation checkpoint; this composition requires `video2video`. |
| `structural_control` | `edge` | Native edge conditioning across every prepared source frame. |
| `conditioning_fps` | `24` | Preparation and generation frame rate, an integer from 10 through 30. Source duration is preserved within one output frame. |
| `transfer_chunk_frames` | `93` | Native generation window, `4k+1` frames from 9 through 297. Longer videos use overlapping windows and retain complete source coverage. |
| `control_guidance` | `1.5` | Native structural-control strength, greater than 0 and at most 10. |
| `prompt`, `negative_prompt`, `augment_subject` | See YAML | Generation intent and appearance sampling. Each effective prompt also includes source captions and the sampled appearance profile. |
| `seed`, `guidance`, `steps` | `17`, `5.0`, `24` | Generation sampling. |
| `variant_count`, `variant_parallelism` | `2`, `1` | Number of variants and concurrent generation workers, limited by visible GPUs. |
| `augmentation_seed` | `30` | Fixed appearance profiles across fresh run IDs; change it for new appearance experiments. An empty value restores run-ID sampling. |
| `refinement_iterations` | `2` | Maximum total generation/evaluation passes, including the initial pass. |
| `retry_seed_stride`, `retry_guidance_delta`, `retry_steps_delta` | `1000`, `-0.5`, `4` | Changes per retry. The second pass starts at seed `1017`, guidance `4.5`, and `28` steps. |
| `grade_threshold` | `0.2` | Exploratory evaluator and quality-gate threshold; required checks must also pass for every variant. |
| `attribute_threshold` | `0.25` | Minimum fraction of requested appearance attributes correctly recognized per variant: at least 1 of the default 4 attributes. Every question must have a valid answer. |
| `alignment_mode` | `required` | Decode and verify matching source/output timelines and generation hashes before quality scoring. |
| `caption_model` | `MiniMaxAI/MiniMax-M3` | Hosted captioning model and evaluator visual-answer model; R1 selects an available model. |
| `attribute_sample_policy` | `ranking` | Evaluator attribute-observation policy. |
| `temporal_consistency_mode`, `temporal_consistency_threshold` | `advisory`, `0.8` | Source-relative temporal diagnostic. Related `temporal_*` keys configure regions, noise floor, and blur. |
| `appearance_fidelity_mode`, `appearance_fidelity_threshold` | `advisory`, `0.8` | Protected-appearance diagnostic. Related `appearance_*` keys configure regions and tolerances. |
| `source_motion_weight` | `0.0` | Must remain zero: publish model output after guardrail processing; blending does not align motion. |
| `curator_clip_len_s`, `curator_min_clip_len_s` | `3`, `1` | Curator's target and minimum clip durations in seconds. Use a source at least one second long for the full pipeline with these defaults. |
| `curator_motion_filter` | `score-only` | Retain Curator motion measurements without discarding clips based on that diagnostic. |

The `resources.caption` entry selects the pinned CPU image with FFmpeg for
extracting accepted-video frames. Keep that image selection when adapting the
workflow; installing FFmpeg on your submitting machine does not install it in
the worker container.

For example, adding `--var seed=29 --var augmentation_seed=17` to both commands
changes the generation seed and fixes appearance sampling. This is a controlled
experiment. The reports retain the actual thresholds and all individual
attribute verdicts, including failures allowed by a reduced attribute threshold.
Reducing a quality threshold changes the acceptance criteria; it does not repair
the generated video. Model guardrails, complete evaluator responses, media
integrity and timeline alignment remain required.

The shipped `0.2`/`0.25` criteria are for exploratory pipeline runs. To require
the earlier stricter criteria, add `--var grade_threshold=0.75
--var attribute_threshold=1.0` to both plan and submit. Choose production
criteria using representative videos and human review of the resulting data.

The actual generation arguments and retry values are applied in
[`paidf_cosmos3.py`](../../npa/src/npa/workflows/paidf_cosmos3.py).
The [`tool catalog`](../../npa/src/npa/orchestration/npa_workflow/catalog.py)
maps the YAML keys to the `generate-variants` and `evaluate` commands.
Inspect `configs/manifest.json`, `configs/cosmos3-attempt.json`, and each
variant's `metadata.json` to see what was actually sampled and executed.

Preparation writes `input/original_source.mp4`, `input/source.mp4`, and
`input/timeline.json`. The prepared source uses the model's 832×480 bucket with
letterboxing and a zero-based constant frame rate. It resamples the source
timeline without stretching the action or changing its aspect ratio. Videos
must produce at least six prepared frames. The generated video must match every
prepared timestamp; the adapter never trims, stretches or blends output to force a pass.

Each variant includes `source_edges.mkv`, `transfer.json`, and alignment evidence
in `metadata.json`. The adapter verifies that the native framework loads all
control pixels unchanged, checks each effective prompt before generation, and
saves the video returned by the model's video guardrail. The evaluator independently
decodes the current source/output pair and verifies the recorded hashes before
comparing corresponding frames. `--var fps=24` and `--var num_frames=169` are
not supported controls; use the named settings above.
Native transfer uses eager execution (`native_torch_compile: false` in
`transfer.json`) because the pinned attention implementation has a Torch
compilation incompatibility for structural-control shapes.

### R6. Diagnose quality rejection and prepare the next run

`annotation requires a complete accepted evaluator disposition` means
`require-accepted-quality` refused to send unaccepted data to annotation.
Older clients could reach that guard after rejection when submitted with
`--assume-decision promote_checkpoint`. Current clients reject this execution
flag before staging or provisioning. Use the R3 runtime command
with a fresh run ID for the next experiment. With real runtime routing, a final
rejection writes quality evidence and terminates at `reject-quality`.
Changing the submit mode corrects routing; it does not improve generated videos.

First confirm the run is terminal using R4, so a refinement pass cannot replace
artifacts while you download them. Set `RUN_ID` to the failed run, retain its
submit command and overrides, and record the submitting checkout before updating
it. `npa --version` alone does not identify the source revision:

```bash
git rev-parse HEAD
git status --short
npa --version
```

The following commands use the default run prefix and S7's AWS profile. They
download the **staged source** (`input/source.mp4`) and every generated variant's
`augmented_video.mp4`, along with the reports needed to interpret them. These
are the staged source and latest published variants for this run.
Use the declared run URI if you changed `prefix`.

```bash
RUN_URI="s3://$BUCKET/paidf-cosmos3/$RUN_ID"
mkdir -p ./paidf-evidence
EVIDENCE_DIR="$(mktemp -d "./paidf-evidence/${RUN_ID}.XXXXXX")"
(
  set -e
  : "${EVIDENCE_DIR:?Evidence directory creation failed}"
  umask 077
  mkdir -p "$EVIDENCE_DIR"/{input,configs,grade,cosmos_augmented}
  chmod 700 "$EVIDENCE_DIR"
  for artifact in input/source.mp4 input/original_source.mp4 input/timeline.json input/provenance.json \
    configs/manifest.json configs/cosmos3-attempt.json \
    cosmos_augmented/manifest.json grade/quality_disposition.json \
    grade/cosmos_evaluator.json grade/decision.json; do
    aws s3 cp "$RUN_URI/$artifact" "$EVIDENCE_DIR/$artifact" --profile nebius
  done
  aws s3 cp "$RUN_URI/cosmos_augmented/" "$EVIDENCE_DIR/cosmos_augmented/" \
    --recursive --exclude '*' --include 'variant-*/augmented_video.mp4' \
    --include 'variant-*/metadata.json' --include 'variant-*/transfer.json' \
    --include 'variant-*/source_edges.mkv' --profile nebius
)
```

A missing object can identify an earlier failed stage; stop and inspect its logs
instead of treating a partial download as complete evidence. Confirm the manifest's
`run_id`, variant list, and lineage match the intended run; compare the evaluator's
`augment_uri` and disposition's `evaluator_report_uri` with that same prefix.
Also correlate the runtime waves, variant `attempt` values, and evaluator
`generated_at`: a later failed refinement can leave newer or partial videos
beside an earlier evaluator report. Do not attribute that report to the latest
videos unless the matching evaluation completed after their generation.
Keep these files private: reports and provenance can contain input content and
private resource locations. Share only through an approved private channel.

Read the aggregate rejection reasons and the individual checks:

```bash
jq '{quality_status, decision, evaluator_status, score, threshold,
     hard_checks_passed, reasons}' "$EVIDENCE_DIR/grade/quality_disposition.json"
jq '{status, passed, score, threshold, attribute_threshold, alignment_mode, clip_count, passed_clips, warnings,
     clips: [.clips[] | {clip_id, status, passed, score,
       temporal_alignment, attribute_verification, hallucination, temporal_enforced,
       temporal_consistency, appearance_enforced, appearance_fidelity, skipped}]}' \
  "$EVIDENCE_DIR/grade/cosmos_evaluator.json"
```

`status: completed` with failed checks is a quality verdict. A `degraded`, missing,
or malformed report is incomplete evidence; resolve its warnings or service errors
before interpreting its score. Acceptance requires a complete report, passing
required checks for every variant, and the recorded quality and attribute thresholds.

Compare the decoded media, without modifying the original evidence:

```bash
(
  set -e
  for video in "$EVIDENCE_DIR/input/source.mp4" \
    "$EVIDENCE_DIR"/cosmos_augmented/variant-*/augmented_video.mp4; do
    printf '\n%s\n' "$video"
    ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration:format=duration \
      -of json "$video"
    ffmpeg -v error -xerror -i "$video" -map 0:v:0 -f null -
  done
)
```

With this workflow, `temporal_alignment.status` must be `verified` for every
evaluated variant. A mismatched frame count, frame rate, timestamp or video hash
stops quality scoring. An older run with unequal durations cannot be fixed by
lowering the quality threshold; use a fresh run with the updated workflow.

Temporal and appearance diagnostics are `advisory` by default, so their failure
alone does not cause hard rejection. Read the required attribute and hallucination
checks as well as the aggregate score. Visually compare the source action,
robot/object identity, contacts, and scene continuity at corresponding times.
Changing FPS metadata, duplicating frames, or trimming the output is not proof
that the model preserved the action; timing adjustments cannot repair scene drift.

Once you have a specific input, generation, or evaluator-service correction,
reserve a new run ID with R2 and submit with R3. Preserve the original rejected
run. If you intentionally accept weaker visual matching, set the chosen
`grade_threshold` and `attribute_threshold` in both plan and submit commands.
Every variant still needs complete evaluation and verified alignment.

## Troubleshooting

Use the current setup and recovery paths below when one of these symptoms
appears.

| Symptom | What to check | Action |
| --- | --- | --- |
| `No such command 'submit'` | Executable, active environment, and installed checkout | Follow [installation recovery](#if-submit-is-missing); verify `.venv/bin/npa workbench workflow submit --help` before setup. |
| `Runtime-required workflows reject --assume-decision for execution` | Execution is using a planned decision | Use the full R3 runtime command; assumed decisions belong only in offline plans. |
| Workflow YAML does not exist | Path copied from a previous repository layout | Run from the repository root and use `workflows/main/paidf-cosmos3.yaml`. |
| Runtime status has no stage rows, or artifacts reports `manifest_pending` | Summary/index publication can lag the runtime record | Read the per-wave record in R4 and inspect stage logs before relaunching. |
| `ERROR:root:'NoneType' object has no attribute 'strip'` during an otherwise successful launch | A kubeconfig credential plugin returned a valid token with `expirationTimestamp: null` | This was nonfatal in live validation: the Kubernetes client logged an expiry parsing error after loading the token. Check the actual command exit and stage status; rerun credential preflight for authentication failures. |
| `invalid IAM subject` or `PermissionDenied` | Selected account, CLI profile, and project permissions | Verify that the account can access this project in the web console, correct its permissions, and retry P3. |
| `The active Nebius CLI profile cannot authenticate non-interactively` | CLI compatibility as well as authentication | Check the version first; install the compatible CLI from P2 before replacing the profile. |
| `legacy global storage credentials have no unique exact-project ownership` | Credentials left by an older NPA installation | Back up the local credential file, identify which project owns the keys, and reconcile against the [project credential schema](../../docs/credentials.yaml.example). Do not assign ambiguous keys to the new project. |
| Missing bootstrap-contract attestation for `npa-rerun-viewer` | Old checkout or image override | Use current supported pins and rerun `preflight-images`; retain the failing check. |
| Private-registry `403` when expecting GHCR | Explicit image/registry overrides or an older client | Use the current default public mirror and remove unintended overrides. Current runtime image selection does not inherit `NPA_REGISTRY`. |
| `No NPA cluster identity exists ... refusing controller adoption` | Cluster fetched but not adopted | Complete S5 with the exact cluster name, project alias, and context. |
| `No shared controller owner is bound` | Missing controller binding | Run `npa skypilot bind-controller` as shown in S5. |
| SkyPilot requires Python 3.9–3.12 | Interpreter used by the isolated runtime | Recreate the isolated SkyPilot environment with Python 3.12; see [SkyPilot setup](../../docs/orchestration/skypilot-setup.md). |
| `ClusterOwnerIdentityMismatchError` | Reused local/controller state from another cluster | Verify the project and context, then follow [controller setup](../../docs/orchestration/skypilot-setup.md#managed-jobs-controller). Preserve unrelated projects' state. |
| `StorageBucketGetError` or inaccessible bucket | Selected project's keys, bucket, region, and endpoint | Recheck S2/S7 and rerun credential preflight. For controller recovery, use the exact project and context in [SkyPilot setup](../../docs/orchestration/skypilot-setup.md#managed-jobs-controller). |
| `FAILED_SETUP: Forced include not found` | Incomplete manually staged source | Use current automatic source staging. For a persisted bad source URI, follow the [source-staging recovery guide](../../docs/workbench/guides/physical-ai-data-factory-deploy.md#if-submit-fails). |
| `--run-id` and `--resume-run` are mutually exclusive | Recovery command combines both options | Use only `--resume-run` for the existing run, or `prepare-run` for a fresh experiment. |
| Workflow GPU discovery says `Kubeconfig not found` | Missing NPA-managed kubeconfig | Complete S5/S6 and verify the selected context. |
| `UnsatisfiableAcceleratorError` reports no free GPU | Matching nodes may already be occupied | Check GPU discovery and active workloads. Wait for capacity or select a compatible available cluster; do not cancel another user's workload. Resume an unchanged run as described in R4. |
| Stage repeatedly recreates; `container not found` during setup | Image cannot satisfy SkyPilot bootstrap | Cancel the affected run using the [run lifecycle](../../docs/run-lifecycle.md) and correct its image. Keep image preflight enabled. |
| GPU stage recovers repeatedly; events show `Evicted`, `ephemeral-storage`, or `NodeHasDiskPressure` | GPU-node disk capacity for image layers and runtime weights | Inspect node disk pressure and available capacity, or ask the cluster administrator. Keep enough disk for image layers and runtime weights. |
| Token Factory returns `404` / model does not exist | Hosted model availability | List models again and set `caption_model` to an available vision model. |
| Evaluator reports `reasoning-only response with no visible answer` | An older bundled NPA client may be running | Update the checkout, retain `config.source_overlay: true` as in S8, and start a fresh run. Treat the original report as degraded; empty answers do not establish an attribute-quality verdict. |
| `annotation requires a complete accepted evaluator disposition` | A one-shot plan assumed promotion, or the accepted disposition is missing/incomplete | Follow [R6](#r6-diagnose-quality-rejection-and-prepare-the-next-run); retain the guard and use R3's runtime command for the next fresh run. |
| `frame_counts_match: false`, or source/output durations differ | Unequal decoded lengths and possible temporal misalignment | Probe the same run's source and variants in R6; read required checks as well as advisory diagnostics. Use R5's required alignment contract and a fresh run after changing generation settings. |
| `source_motion_weight must be 0` | Configuration copied from the old guide | Remove the old override or set it to `0.0`; publication retains model output after guardrail processing. |

For pod-level and artifact triage, see
[known Workbench issues](../../docs/workbench/troubleshooting/known-footguns.md).

## Validation

P1's Ubuntu 24.04 package commands passed in a fresh disposable container:
Python 3.12.3, virtualenv creation, pip, and all listed base tools were verified.
The separate `kubectl` and AWS installers linked from P1 were outside that check.

The portable bootstrap in S6 was exercised with Python 3.12.14 on Linux: a
fresh SkyPilot 0.12.2 installation and a second invocation that reused it both
succeeded. Credential, model-access, storage, image, source-staging, and input
preflights passed against live services. S5's new-cluster command was validated
with a live `--dry-run`; execution used an existing RTX PRO 6000 Blackwell
cluster and did not create a cluster.

The current structural-transfer validation has completed real input preparation,
configuration generation, hosted source captioning, both native GPU variants,
and Cosmos Evaluator scoring at revision `e1a6215b` on September 11, 2026.
Its 169-frame source
at 50 fps and 640×480 becomes an 81-frame reference at 24 fps and 832×480.
Duration changes from 3.38 to 3.375 seconds through frame-rate sampling;
letterboxing preserves aspect ratio and does not stretch time. Both generated
videos fully decode with the prepared reference's dimensions, frame count,
frame rate, and duration. Native control readback and text/video guardrails
passed for both variants.

Both variants passed evaluation with an aggregate score of `0.392244` against
`grade_threshold: 0.2`. Their attribute scores were `0.25` and `0.5` against
`attribute_threshold: 0.25`; all eight attribute answers were complete, with
no skipped checks or warnings. Evaluation matched the exact published video
hashes and found zero timestamp error. The accepted downstream path is still
being validated.

Matched-time visual review showed recognizable pickup/removal timing, with
gripper appearance changes, extra shadows, and strong warm coloration and
colored borders in one variant. These exploratory acceptance settings do not
certify physical fidelity or suitability for training. Review your own outputs
against the task's visual and motion requirements before using them as data.

Automated checks cover media corruption and alignment failures, native control
pixels and guardrail processing, incomplete evaluator responses, runtime
routing, every accepted variant's captions, and final artifact validation.
These checks do not substitute for a completed live run. Concurrent two-GPU
variants, longer videos requiring multiple native chunks, fresh-cluster
provisioning, and the desktop Rerun UI have not been validated by this run.

## Inspect the outputs

### Check every stage and full pipeline completion

Use the R4 runtime record to verify executed stages, then check their artifacts
under the same run prefix. A rendered plan or a Rerun file alone does not prove
that the full pipeline completed.

| Stage | Evidence to check |
| --- | --- |
| `prepare-input` | `input/provenance.json` reports `status: prepared`; the original and prepared source decode; `input/timeline.json` records timing, dimensions and hashes. |
| `generate-configs`, `annotate-original` | `configs/manifest.json` contains the requested appearance profiles; `labeled_original/captions.json` contains source captions. |
| `generate-variants` in each refinement pass | The manifest contains every requested variant; generated videos match the prepared timeline; `transfer.json` records complete edge controls and successful model guardrails. |
| `evaluate`, `quality-gate` | `grade/cosmos_evaluator.json` is complete; `grade/decision.json` agrees with its required checks and threshold. Rejected intermediate passes trigger refinement while passes remain. |
| `quality-disposition`, `visualize-quality-evidence`, `quality-route` | Final disposition and route agree; `reports/quality-evidence.rrd` contains quality evidence. A rejected run ends at `reject-quality` and skips the rows below. |
| `require-accepted-quality`, `annotate-augmented` | Accepted disposition passes the guard; `labeled_augmented/captions.json` contains nonempty captions and verified video hashes for every variant. |
| `cosmos-curate` | `curation/cosmos_curator.json` identifies the real engine and a positive `clip_count`; declared curated clips exist. |
| `curate` | `curation/report.json` identifies `curation_engine: fiftyone-brain`. |
| `visualize`, `finalize` | Final recording exists; `reports/final.json` reports completion with a positive curated count. The runtime reports success through `finalize`. |

The main variant and evaluator paths describe the latest pass; retry settings
are also retained as `configs/cosmos3-attempt-00.json`, `-01.json`, and so on.
Those attempt files do not contain complete archived videos and evaluation
reports for every pass.

Only after the runtime reports successful finalization, check the final report:

```bash
(
  set -e
  umask 077
  FINAL_DIR="$(mktemp -d "./paidf-final-${RUN_ID}.XXXXXX")"
  aws s3 cp "s3://$BUCKET/paidf-cosmos3/$RUN_ID/reports/final.json" \
    "$FINAL_DIR/final.json" --profile nebius
  jq -e '.schema == "npa.paidf.cosmos3.final.v1" and .status == "completed"
         and .variant_count > 0 and .video_bytes > 0 and .curated_clip_count > 0
         and .alignment_verified == true and .annotated_variant_count == .variant_count
         and .fiftyone_engine == "fiftyone-brain" and .has_rrd == true' "$FINAL_DIR/final.json"
)
```

A rejected run is expected to lack this final report. Check the runtime and
artifacts from your own run against every row above before treating its data as
accepted.

### Open the recording

After successful finalization, open `reports/sim2real.rrd` for the complete
accepted run, including augmented captions and both curation reports.
`reports/quality-evidence.rrd` is the earlier recording produced before curation;
it is also available after a quality rejection. List the run artifacts:

```bash
npa workbench workflow artifacts "$RUN_ID" --project "$PROJECT_ALIAS"
```

The artifact index can still be empty after the recording has been uploaded.
For the default prefix used in this guide, verify and download the known output
directly with the AWS profile from S7:

```bash
(
  set -e
  umask 077
  RRD_FILE=sim2real.rrd
  # For a rejected run or earlier quality review: RRD_FILE=quality-evidence.rrd
  RRD_DIR="$(mktemp -d "./paidf-recording-${RUN_ID}.XXXXXX")"
  RRD_URI="s3://$BUCKET/paidf-cosmos3/$RUN_ID/reports/$RRD_FILE"
  aws s3 ls "$RRD_URI" --profile nebius
  aws s3 cp "$RRD_URI" "$RRD_DIR/$RRD_FILE" --profile nebius
  rerun "$RRD_DIR/$RRD_FILE"
)
```

If you changed the workflow prefix, use its declared output URI instead. Run
the final `rerun` command in a desktop session. From a headless host, transfer
the recording to your desktop or follow the browser-viewing link below.

The NPA installation includes its pinned Rerun SDK. The recording presents the
available source/generated frames, generation prompts, captions, evaluator
decisions, and pipeline evidence. A rejected run may contain a useful recording
without accepted data or a final curation report. Check the variant MP4s and
the evaluator's individual dispositions as well as its aggregate score.

The commands above retrieve evidence from your own run. For browser viewing
or sharing, see [Rerun sharing](../../docs/workbench/rerun-sharing.md).

For project or cluster permissions, contact your administrator. For workflow
behavior and run management, continue with the
[Cosmos 3 reference guide](../../docs/workbench/guides/paidf-cosmos3.md) and
[workflow lifecycle](../../docs/run-lifecycle.md).
