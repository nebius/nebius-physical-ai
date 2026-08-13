<div align="center">

# Nebius Physical AI

**One CLI, one SDK, one workflow layer for physical-AI workloads on Nebius —
data curation, simulation, synthetic data, policy training, evaluation,
observability, and cluster orchestration.**

<img src="docs/assets/workbench-architecture.png" alt="Nebius Physical AI Workbench architecture" width="820" />

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms: macOS · Linux · WSL2](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20WSL2-lightgrey.svg)](docs/install.md)
[![Test](https://github.com/nebius/nebius-physical-ai/actions/workflows/test.yml/badge.svg)](https://github.com/nebius/nebius-physical-ai/actions/workflows/test.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Quickstart](docs/quickstart.md)** ·
**[Guides](docs/workbench/guides/README.md)** ·
**[Workbench docs](docs/workbench/)** ·
**[CLI reference](docs/cli/README.md)** ·
**[Cookbooks](docs/workbench/cookbooks/README.md)** ·
**[Contributing](CONTRIBUTING.md)**

</div>

---

## What is `npa`?

`npa` is the CLI and SDK for **Nebius Physical AI**. Workbench is its primary
solution: one command surface that composes data curation, simulation,
synthetic data, policy training, evaluation, export, observability, and
declarative workflows on Nebius object storage, orchestration, vLLM-compatible
serving, managed Kubernetes, and GPU clusters (H100 · H200 · L40S · B300 ·
RTX6000).

> Partners integrate independently. Teams assemble from open blueprints.
> Nebius owns the infrastructure layer and compute substrate.

|                                   |                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| **What you can do**               | Curate datasets · train and evaluate policies · render synthetic data · run sim-to-real loops · serve models |
| **Who it's for**                  | Robotics teams, physical-AI researchers, and partners shipping on Nebius              |
| **Where it runs**                 | Nebius S3, managed Kubernetes, and GPU clusters                                        |
| **How you extend it**             | Declarative `npa.workflow/v0.0.1` YAML specs and reusable Workbench tool refs         |

---

## Setup

Three steps take you from a clone to a real result on Nebius: install `npa`,
configure Nebius, then run your first cloud workload.

### 1. Install npa

`npa` works with **Python 3.10+** and installs editable from the clone (it is
not on PyPI):

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e npa
```

> **Managed deployments support Terraform CLI 1.x on `PATH`: verify it with `terraform version`; commands such as `npa agent fresh-setup` require it, and `pip install -e npa` does not install it. An existing 1.x minor/patch is accepted; agent bootstrap installs the tested 1.13.3 baseline only when Terraform is absent.**

Verify: `npa --version`.

> **Windows:** use **WSL2 Ubuntu**. Full per-platform steps (venv, Nebius CLI,
> WSL2): [docs/install.md](docs/install.md).

### 2. Configure Nebius

[Sign up](https://docs.nebius.com/signup-billing/sign-up) and create a
[tenant and project](https://docs.nebius.com/iam/manage-projects), install the
Nebius CLI, then run `npa configure` — it creates or reuses your Nebius CLI
profile and writes `~/.npa/credentials.yaml` + `~/.npa/config.yaml`:

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh \
  | NEBIUS_CLI_VERSION=0.12.254 bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc
npa configure
```

NPA is tested with Nebius CLI `0.12.254` (recommended) and `0.12.227`
(compatible with a warning). Other versions are blocked before provider calls
and the error prints the exact tested-version install command.

Creating a project is a privileged action outside NPA. A tenant administrator
(or another principal with permission to create projects under the tenant) can
use the pinned CLI's official `iam v2 project` surface instead of the console:

```bash
TENANT_ID="tenant-id"
PROJECT_NAME="project-name"
REGION=eu-north1
command -v jq >/dev/null

# Optional read-only parent verification before creating anything.
nebius iam v2 project list --parent-id "$TENANT_ID" --all --format json

# Creates the external project and captures its immutable ID from structured
# output (no parsing of a human table). Review tenant/name/region first.
PROJECT_JSON="$(nebius iam v2 project create --parent-id "$TENANT_ID" \
  --name "$PROJECT_NAME" --region "$REGION" --format json)"
PROJECT_ID="$(printf '%s' "$PROJECT_JSON" | jq -er '.metadata.id')"
export PROJECT_ID
test -n "$PROJECT_ID"

# Read-only identity verification.
nebius iam v2 project get --id "$PROJECT_ID" --format json

# Bind the active CLI profile, then continue with NPA under a local alias.
nebius config set tenant-id "$TENANT_ID"
nebius config set parent-id "$PROJECT_ID"
PROJECT_ALIAS="local-npa-alias"
npa configure --no-interactive --tenant-id "$TENANT_ID" \
  --project-id "$PROJECT_ID" --region "$REGION" \
  --project-alias "$PROJECT_ALIAS"
```

Project creation requires tenant-level administrative permission; the list/get
commands are read-only and are safe verification steps. The web-console path
linked above remains equivalent.

`npa configure` prompts for optional model/inference tokens and links each
one's setup guide inline. Create them step by step:
[Hugging Face](docs/workbench/huggingface-token.md) ·
[NVIDIA NGC](docs/workbench/ngc-api-key.md) ·
[Nebius Token Factory](docs/workbench/token-factory-key.md).

> **Federation profile with many tenants?** If your Nebius CLI profile has no
> `tenant-id`/`parent-id` set (common for SSO/federation logins), bind it to the
> project you want *before* `npa configure` so discovery targets the right place
> instead of listing every tenant:
> `nebius config set tenant-id <id> && nebius config set parent-id <project-id>`.
> Say **yes** to the object-storage prompt — the agent VM and the Physical AI
> Data Factory both need an S3 bucket + access key.
>
> If you already know the IDs and have a valid non-interactive Nebius profile
> or service-account credential active, skip the browser flow and tenant picker:
>
> ```bash
> npa configure --no-interactive \
>   --tenant-id "$TENANT_ID" --project-id "$PROJECT_ID" \
>   --region "$REGION" --project-alias "$PROJECT_ALIAS"
> ```
>
> No secret-value flags are accepted or shown by `npa configure --help`.
> Automation supplies values through protected environment variables and adds
> the boolean `--save-env-credentials`; NPA atomically persists only supported
> variables in its owner-only credential store. The command reuses S3 only when
> project provenance and a write/read/delete probe both verify it; otherwise it
> proposes a fresh project-scoped bucket without listing or rotating unrelated
> access keys.

Full account/credential detail: [docs/quickstart.md](docs/quickstart.md).

### 3. Run your first cloud workload

Nebius AI Cloud — GPU clusters, managed Kubernetes, and object storage — is the
substrate `npa` is built for. Sanity-check your credentials with
`npa workbench health preflight`, then launch a real GPU workload: the flagship
is [NVIDIA Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos), or
pick any robot/simulator from the [robot guides](docs/workbench/guides/README.md)
to go from a public dataset to a trained-and-evaluated policy on Nebius GPUs.

> **Just want to confirm the NPA→Nebius path works first?**
> [Nebius Token Factory](docs/workbench/token-factory.md) hosted inference is
> zero-GPU and needs only a `NEBIUS_TOKEN_FACTORY_KEY` (no cluster or GPU) — a
> cheap smoke test of your credentials, not the destination. AI Cloud GPUs are.

### The whole path, in order

Running PAIDF on Nebius needs writable storage, a cluster, an orchestrator, and
a copy of `npa` the workers can install. The browser agent is optional and is
not part of this core `set -e` path. This restart-safe shell sequence stops at
the first failed core prerequisite and defaults to on-demand capacity:

```bash
set -eu
set -o pipefail
CONTEXT=npa-cluster
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml

npa configure
# For prompt-free setup, export supported credential variables first and use:
# npa configure --no-interactive --save-env-credentials ...known project flags...
eval "$(npa configure --show --env)"
PROJECT="$NPA_PROJECT_ALIAS"
# Keep this public override after configure --env; eval may restore the saved
# project registry.
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
REGISTRY="$NPA_REGISTRY"
npa workbench health preflight
npa provision-if-absent --project "$PROJECT" --cluster-name "$CONTEXT" \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand \
  --accelerator RTXPRO6000:1 --dry-run --output-format json
npa destroy --project "$PROJECT" --all --json

npa provision-if-absent --project "$PROJECT" --skip-k8s
eval "$(npa configure --show --env)"
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
REGISTRY="$NPA_REGISTRY"
BUCKET="$NPA_BUCKET"
npa skypilot bootstrap
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" \
  --assume-decision promote_checkpoint --var bucket="$BUCKET" \
  --var n_augmentations=1 --json
# Complete every deterministic, read-only workflow gate before provisioning
# resources or allowing submit to stage the repository source.
npa workbench workflow preflight-images "$SPEC" --registry "$REGISTRY"
npa provision-if-absent --project "$PROJECT" --cluster-name "$CONTEXT" \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand \
  --accelerator RTXPRO6000:1 --gpu-readiness-timeout 900
# The accelerator-gated provisioning transaction verifies the exact
# project/context/provider cluster identity and atomically binds the shared
# jobs-controller owner before waiting for GPU readiness. No separate bind is
# needed, and an incompatible/stale owner fails the earlier dry-run/preflight
# before Terraform or source staging.

npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --registry "$REGISTRY" \
  --run-id "$RUN_ID" --runtime --var bucket="$BUCKET" \
  --var n_augmentations=1 \
  --assume-decision promote_checkpoint --infra "k8s/$CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN

printf '%s\n' "Provisioned resources: S3 at $BUCKET (write/read verified; cleanup reported separately)."
npa cluster status --project "$PROJECT"
npa workbench workflow status "$RUN_ID" --project "$PROJECT"
printf '%s\n' \
  "Running/cost-bearing when status says running: 1 cpu-d3/8vcpu-32gb node; 1 gpu-rtx6000/1gpu-24vcpu-218gb node; active PAIDF jobs." \
  "Absent: no agent VM was requested; no resources beyond the storage, cluster, and PAIDF state reported above were requested." \
  "This script performs no teardown. Exact teardown commands:" \
  "Teardown (not run): npa workflow cancel $RUN_ID --project $PROJECT" \
  "Teardown (not run): npa cluster down --project $PROJECT --context $CONTEXT --force" \
  "Teardown (not run): npa storage bucket delete --project $PROJECT --yes" \
  "Teardown (not run, after bucket): npa storage service-account delete --project $PROJECT --yes"
```

Here “restart-safe” means provisioning resumes the same secret-free operation
journal under `~/.npa/operations/`, preserves configured credentials/storage and
durable Terraform state, and prints one deterministic resume command. Source
staging and submission are content-addressed/idempotent for the explicit
`RUN_ID`. A stale or ambiguous run is never selected silently: resume it with
`--resume-run "$RUN_ID"`, or use `prepare-run` to create a distinct run.
The image check, immutable whole-path topology/quota plan, and exact project
teardown plan are intentionally read before any explicit provisioning in this
sequence. Submit repeats its deterministic checks before input/source staging,
so a missing image or identity mismatch cannot upload the 1,225-file source tree
or start a paid cluster first.

The plan treats `compute.disk.size.network-ssd` as a byte allowance, separately
from `compute.disk.count`, and prints exact `required`, `available`, and
`shortfall` values in bytes and GiB. The default whole path is 1,251 GiB of new
NETWORK_SSD capacity when nothing exists: 100 GiB for the agent root disk plus
128 GiB for the CPU node and 1,023 GiB for the GPU node. For example, 21 GiB
available is blocked with a 1,230 GiB shortfall before Terraform, networking,
the Kubernetes control plane, VMs, or disks can be created. Proven existing
resources are deducted on retries; unknown or contradictory quota evidence is
not permission to mutate.

The browser agent can be deployed independently after the core submit. Its
failure does not cancel or block PAIDF:

```bash
PROJECT="configured-alias"
RUN_ID="existing-paidf-run-id"

if ! npa agent status --project "$PROJECT" --name agent --json >/dev/null 2>&1; then
  npa agent preflight --project "$PROJECT" \
    && npa agent setup --project "$PROJECT" --name agent
fi
if npa agent status --project "$PROJECT" --name agent --json >/dev/null 2>&1; then
  npa workbench workflow load-artifact "$RUN_ID" --project "$PROJECT"
else
  printf '%s\n' \
    "Optional agent is not healthy; PAIDF remains submitted." \
    "Later, after agent recovery: npa workbench workflow load-artifact $RUN_ID --project $PROJECT"
fi
```

`provision-if-absent` now reconciles and write/read-probes S3 before it considers
Kubernetes; interrupted configuration resumes from owner-only provenance in
`~/.npa/credentials.yaml`. It never launches the cluster while required storage
is missing. The minimum runtime binding is a project-scoped NPA group with the
bucket-scoped Nebius `storage.object-editor` role. It supplies exactly
`GetObject`, `HeadObject`, `PutObject`, `DeleteObject`, and `ListObjectsV2` for
the configured bucket. Existing tenant `editors` membership remains a verified
compatibility path; NPA creates that broader grant only when the provider
explicitly reports that the narrow assignable role is unsupported and the
operator has enabled the fallback. Unreadable or insufficient IAM stops before
access-key creation or any S3 probe. Newly granted HMAC/IAM access converges through
typed, bounded retries without replacing the new identity.
The explicit fallback switch is `NPA_ALLOW_EDITORS_STORAGE_FALLBACK=1`; leave it
unset unless the provider has rejected `storage.object-editor` as unsupported.
Each custom group name includes the exact project ID, so aliases cannot collide.
Agent preflight
separately probes the exact Terraform state object and an unconditionally written,
unique sibling under its exact prefix, so a generic writable-bucket success cannot
hide state-key `403`/list failures or depend on conditional-header support. Probe
cleanup is reported independently and never replaces a more important write/read
diagnosis. Saved Object Storage HMAC
credentials—not the Nebius CLI IAM token—are then supplied to every Terraform
init/plan/apply/state/output/destroy process and never placed in argv or recovery
receipts. Readiness reports Kubernetes Ready/allocatable capacity, product-label
readiness, and SkyPilot discovery as separate evidence layers. If this operation created the cluster, a later
readiness failure rolls back only that new cluster; pre-existing shared storage,
configuration, credentials, and clusters are preserved. The command above asks for exactly one `cpu-d3` / `8vcpu-32gb` CPU
node and one `gpu-rtx6000` / `1gpu-24vcpu-218gb` RTX PRO 6000 node. On-demand is
the reliable default. Preemptible capacity is an explicit availability/cost
choice and can be reclaimed mid-run; it does not bypass hard tenant instance,
boot-disk count, NETWORK_SSD byte-capacity, or public-IP quotas. Resume reclaimed work from durable S3 artifacts.
Select it explicitly with `--preemptible` in place of `--on-demand`; the hard
quota arithmetic is unchanged.

**Easiest option: pull the OSS images from the public mirror instead of building them.**
Every workbench image is published to `ghcr.io/nebius/nebius-physical-ai` and is
anonymously pullable, so pointing at it skips image building entirely:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
```

Use your own registry when you need private or locally modified images; then the
note below applies.

**A workflow's container images are not shipped into your registry.** `npa configure`
selects (or creates) a project registry; it does not mirror workbench images into
it, so a spec that pins them (the Physical AI Data Factory pins three Cosmos
images) needs them built and pushed once per registry.
`npa workbench workflow preflight-images <spec.yaml>` reports each image as
`ok` / `not_found` / `forbidden` and prints the exact build command for anything
missing. `submit` runs the same check **before it provisions anything**, so a
registry without them costs no GPU time.

`submit` verifies these up front and prints **everything** still missing in one
list (with the command that fixes each), so you are not discovering them one
failed run at a time. `provision-if-absent` writes the cluster kubeconfig to
`~/.npa/clusters/<context>/kubeconfig` rather than merging it into
`~/.kube/config`; `submit --infra k8s/<context>` finds that file on its own, and
`kubectl` in your shell needs `export KUBECONFIG=~/.npa/clusters/<context>/kubeconfig`
(the command prints the line). Worked example end to end:
[Physical AI Data Factory runbook](docs/workbench/guides/physical-ai-data-factory-deploy.md).
For image-less stages, the same submit content-addresses the local source,
persists its exact verified S3 URI, and reuses it after interruption; no shell
`export` or separate `stage-src` command is part of the happy path.
For the shortest agent-driven setup, [copy the exact PAIDF agent
prompt](docs/workbench/guides/physical-ai-data-factory-deploy.md#run-paidf-with-a-coding-agent).

For an already configured project and provisioned cluster, this is the complete
PAIDF submit and monitor path. `status` resolves the exact run from
the selected project's receipt, canonical PAIDF prefix, or pinned managed-job
identity even while the final manifest is pending; `logs` uses the same resolver.

```bash
eval "$(npa configure --show --env)"
# Select the public mirror after eval so a saved project registry cannot replace it.
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
PROJECT="$NPA_PROJECT_ALIAS"
BUCKET="$NPA_BUCKET"
REGISTRY="$NPA_REGISTRY"
KUBE_CONTEXT="$NPA_KUBE_CONTEXT"
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"

npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --registry "$REGISTRY" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" \
  --var n_augmentations=1 --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN

MANIFEST_URI="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/npa-workflow/manifest.json"
# Normal NPA-only status lookup (no aws/sky/kubectl command is required):
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
# Explicit fallback when project storage cannot be resolved in this shell:
npa workbench workflow status "$RUN_ID" --project "$PROJECT" \
  --workflow-s3-uri "${MANIFEST_URI%/manifest.json}"
npa workbench workflow logs "$MANIFEST_URI" --project "$PROJECT" --stage finalize
# Loading is optional and requires the independently deployed healthy agent.
# Use the guarded load command in the optional-agent section above.

# After DNS/controller recovery, resume only by naming the existing ID explicitly:
npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --registry "$REGISTRY" --resume-run "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --var n_augmentations=1 \
  --assume-decision promote_checkpoint --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Every Kubernetes managed-job launch now crosses one controller-launch
transaction. NPA probes the exact selected context through the same
`KUBECONFIG` environment SkyPilot uses and requires three consecutive `/readyz`
successes spanning 10 seconds. It then reconciles the exact job name through
structured SkyPilot queue output under an owner-only logical-launch lock. A
transient controller-creation refusal is retried automatically only after exact
job absence is proven and API stability is re-established; an accepted request
is adopted by immutable job ID. Ambiguous existence blocks without duplicate
launch or name-based cancellation. JSON exposes `launch_transaction`; runtime
ledgers expose the same readiness, reconciliation, recovery, and cancellation
evidence per wave. See the [controller launch decision](docs/architecture/skypilot-controller-launch-transaction.md).

Workflow image preflight resolves each selected tag to an immutable digest. NPA
verifies first-party OCI bootstrap-contract metadata; arbitrary unattested images
receive one exact, bounded capability probe in the selected context, whose pod
must be deleted successfully. Results are cached by digest plus contract version.
First-party images cannot replace their declared user with `runAsUser: 0`.
Multi-tool workflows can pin distinct validated images with repeatable
`--image-override TOOL_REF=IMAGE`; an exact tool override takes precedence over
the optional global `--image` fallback, and the rendered task uses the digest
that preflight verified.

With no input flag, that command fetches the pinned **RoboPro Aloha-Agilex
physical robot capture**, verifies SHA-256
`caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198`, caches it
under `~/.cache/npa/physical-ai-data-factory/` (override with
`NPA_PAIDF_CACHE_DIR`), and stages it under the canonical
`physical-ai-data-factory/$RUN_ID/input/` prefix. The exact source is RoboPro
episode 000000, high camera, pinned to immutable dataset revision
`90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`: a 3.38-second 640×480 H.264 MP4
recorded during expert Aloha-Agilex teleoperation. It is CC BY 4.0; attribution,
license, immutable URL, size, media properties, and derivations are recorded in
`input/provenance.json` and the workflow/config/final manifests. NPA fetches it
at operator runtime and does not bundle the media.

Replace only the submit command's input selector as needed; selectors are
mutually exclusive and an explicit source always beats the default:

```bash
# Local H.264 MP4
npa workbench workflow submit "$SPEC" --project "$PROJECT" --run-id "$RUN_ID" \
  --runtime --var bucket="$BUCKET" --input-video ./my-capture.mp4 \
  --assume-decision promote_checkpoint --infra "k8s/$KUBE_CONTEXT"

# One S3 object (not a prefix)
npa workbench workflow submit "$SPEC" --project "$PROJECT" --run-id "$RUN_ID" \
  --runtime --var bucket="$BUCKET" \
  --input-uri s3://my-source-bucket/captures/run-42.mp4 \
  --assume-decision promote_checkpoint --infra "k8s/$KUBE_CONTEXT"

# Developers/tests only: explicitly synthetic geometric frames
npa workbench workflow submit "$SPEC" --project "$PROJECT" --run-id "$RUN_ID" \
  --runtime --var bucket="$BUCKET" --seed-fixture \
  --assume-decision promote_checkpoint --infra "k8s/$KUBE_CONTEXT"
```

Local/S3 videos are validated as decodable H.264 MP4 before image checks or
automatic provisioning. NPA then creates the exact 93-frame conditioning clip
and eight caption frames; Cosmos is invoked with mandatory
`--condition-on-input` (equivalent to `NPA_COSMOS_CONDITION_ON_INPUT=1`). Cache
hits and fetches are printed. `NPA_PAIDF_OFFLINE=1` requires a verified cache hit;
an offline miss, fetch failure, unsupported video, or digest mismatch fails
closed and never falls back to shapes. A run's committed source is immutable:
retries repair/reuse derived artifacts but never replace a user source with the
default. See [the PAIDF guide](docs/workbench/guides/physical-ai-data-factory.md#starter-input-authenticity-licensing-and-replacement)
for the source-code/model/media license boundary and full provenance fields.

JSON and text status identify every checked source. `manifest_state: pending`
requires exact submission evidence (a receipt/job/task identity). A reservation,
plan, or partial staging prefix without that evidence is `NOT_SUBMITTED` /
`PLAN_ONLY`, never `MANIFEST_PENDING`.
`VERIFICATION_UNAVAILABLE` means S3/SkyPilot/provider verification failed and is
never treated as absence. `NOT_FOUND` is emitted only after all applicable exact
sources answered authoritatively; unrelated nested S3 keys are never guessed as
runs. The owner-only local receipt at
`~/.npa/workflow-submissions/<project>/<run>.json` contains location, plan, and
job identity only—never credentials—and removes any dependency on
`NPA_SRC_S3_URI` in later shells.

Fresh submits never inherit the historical global
`~/.npa/paidf-first-run-id`. `prepare-run` writes an atomic, locked state record
scoped by stable project identity and workflow identity; an ambiguous legacy
file is warned about but never reused or deleted. Non-interactive recovery must
use `--resume-run <id>`. `--cached` is the explicit offline status/log mode and
is labeled `CACHED`; its state is not live-verified or automation-trustworthy.

The dataset view groups **Original/input** separately from
**Synthetic/augmented**, shows the source URI/kind and per-item provenance, and
identifies whether the review came from real FiftyOne Brain. PAIDF requires that
real curation stage; it fails instead of labeling a report-only summary as
FiftyOne review.

---

## Do more with npa

- **Run workbench workloads** — NVIDIA Cosmos, vlm-eval, sim2real, and more.
  Start with the [robot guides](docs/workbench/guides/README.md) and
  [Workbench Getting Started](docs/workbench/getting-started.md); the flagship
  GPU workload is
  [NVIDIA Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).
- **Deploy the self-hosted agent** — `npa agent` is a browser workbench VM. It
  builds on the setup above and additionally needs **Terraform**, an **SSH key
  pair**, a **Token Factory key**, and **writable S3** (~20 min).

### Deploy the self-hosted `npa` agent

`npa agent` is a self-hosted **browser workbench VM**: HTTPS UI with
basic-auth login, grounded chat over Nebius Token Factory
(default `nvidia/Cosmos3-Super-Reasoner`), Sim Assets + Cameras panels, an
embedded [Rerun](https://www.rerun.io) viewer for `.rrd` recordings, and
draft/validate/plan/submit endpoints for `npa.workflow/v0.0.1` specs.

After `npa configure`, deploy interactively — no project/tenant ids to type,
since it reuses the projects `configure` saved:

```bash
npa agent preflight   # includes a cleaned writable-S3 probe before any VM work
npa agent setup       # pick a configured project → deploys the VM
npa agent status --project <alias> --name agent
```

`npa agent setup` picks one of your configured Nebius projects (prompting when
you have more than one) and deploys into it. The agent VM authenticates to
Nebius AI Cloud through an **attached `npa-agent` service account** (granted the
tenant `editors` role) — it mints short-lived IAM tokens from the Nebius VM
metadata endpoint on demand, so there is **no static key** stored on the VM.

For scripted/non-interactive deploys, `npa agent fresh-setup --project <alias>
--project-id ... --tenant-id ... --region ...` is still available.

`fresh-setup` provisions the VM with Terraform, then `bootstrap` refreshes
the UI/backend/nginx layer without touching infra. Operator docs:
[skills/tools/npa-agent/SKILL.md](skills/tools/npa-agent/SKILL.md) ·
teardown/reproduce loop: [skills/workflows/agent-fresh-operate/SKILL.md](skills/workflows/agent-fresh-operate/SKILL.md).

Deploy and bootstrap are reconciled phased operations. If a client loses the
final Terraform/SSH response, repeating the exact operation adopts a matching
healthy VM or resumes its first incomplete phase; it does not replace a healthy
VM based on the lost response. Long calls emit secret-free structured heartbeats.

Setup prints four bounded phases around Terraform, SSH installation, and the
final probe; the SSH phase can be quiet for several minutes and prints a
`journalctl` diagnostic to run from another shell. Terraform's current outputs
are `platform` / `preset` (and `cpu_platform` / `cpu_preset` for the CPU-only
agent). The old `gpu_platform` / `gpu_preset` outputs remain deprecated aliases
for existing state and may contain CPU values.

The agent is tenant-aware for read-only discovery. Its **Agent access** panel
and `GET /api/access` show the running identity's effective access project by
project. Artifact search spans only buckets for which the agent can both
associate the bucket with a visible project and verify S3 object-list access.
Partial access is expected and is reported without hiding accessible projects.
Workflow submission and artifact writes/deletes remain deployment-project
scoped. Arbitrary caller-supplied S3 URIs remain configuration scoped; an exact
artifact selected from a discovered cross-project run can be read without
broadening those mutation boundaries.

This read-only tenant behavior is enforced by the agent application, not by a
structurally read-only IAM credential. Deployments may still attach a service
account with tenant-level editors-group grants; operators must treat that
credential as privileged even though cross-project mutation endpoints are not
exposed. Mutation endpoints continue to target only the configured home project.

`GET /api/artifacts/run/{run_id}` returns at most one native S3 page (up to
1,000 objects), never the entire run. A truncated response includes
`next_cursor`; consumers must repeat the request with that cursor plus the
returned `resolved_prefix` and `bucket` as `resource_bucket` until
`truncated=false`. The bundled UI follows this contract. Older consumers that
assumed a complete array must migrate to cursor following; page-local counts and
`preferred` selection describe only the returned page.

---

## Before you burn GPU-hours — preflight

A short list of things that catch first-time users mid-run. Skim before your
first GPU submit.

- **Run preflight.** `npa workbench health preflight` is a single
  PASS/WARN/FAIL/SKIP check over the credentials nearly every job needs —
  Hugging Face, NVIDIA NGC, Nebius object storage (S3), and Token Factory.
  Add `--offline` to check presence only (no network), or `--json` for
  machine-readable output. See
  [FTUE-AUDIT.md § friction 1](FTUE-AUDIT.md#friction-points-ordered).
- **GPU routing matters.** Isaac Lab needs an **RT-core** GPU (L40S / RTX
  Pro 6000), not an H100. See [docs/workbench/troubleshooting/known-footguns.md § L40S Capacity](docs/workbench/troubleshooting/known-footguns.md#l40s-capacity-is-on-demand-zero).
- **Ask the cluster what its GPUs are called.** Kubernetes names accelerators
  after node labels, so the same card can appear as `RTXPRO6000` in a spec and
  `RTXPRO-6000-BLACKWELL-SERVER-EDITION` on the cluster. Run
  `npa workbench workflow gpus --cluster <name>` once after `npa configure`; it
  prints the exact names and the `export NPA_WORKFLOW_GPU_ACCELERATOR=<name>:<qty>`
  line to use. `submit` also remaps this automatically. Note the printed
  *requestable quantity per node*: SkyPilot puts all GPUs of one task on one
  node, so `NAME:2` cannot be scheduled on a fleet of 1-GPU nodes no matter how
  many nodes you add.
- **Registry pull secrets expire silently.** A `401` on image pull usually
  means the `npa-nebius-registry` pull secret needs refreshing; a `403` means the
  credentials are valid but not permitted to pull that repository — and being able
  to list its tags does not rule that out. Kubernetes retries image pulls forever,
  so either one leaves the job in `PENDING`/`ImagePullBackOff` instead of failing.
  Run `npa workbench workflow preflight-images <spec.yaml>` to reproduce the pull
  with the run's own credentials before spending GPU time (`submit` runs it by
  default). See
  [known-footguns.md § Registry Pull Secret](docs/workbench/troubleshooting/known-footguns.md#registry-pull-secret-expires-silently).
- **Bootstrap SkyPilot with `npa skypilot bootstrap`.** It pins a kubernetes
  client SkyPilot can actually use; a newer one makes the managed-jobs controller
  reject every `pod_config` and retry forever, which looks like a hung submit.
  `npa skypilot status` reports the installed client version.
- **Prefer `npa workbench workflow submit` for multi-stage jobs.** Pass
  `npa.workflow` specs (or legacy runbooks); avoid hand-editing scheduler
  YAML. See [FTUE-AUDIT.md § friction 4](FTUE-AUDIT.md#friction-points-ordered).
- **Token Factory keys are not Nebius IAM tokens.** They start with `v1.` and
  live under `NEBIUS_TOKEN_FACTORY_KEY`. See
  [docs/workbench/token-factory.md](docs/workbench/token-factory.md).
- **Always pass `-p PROJECT -n NAME` to `<tool> status`.** Bare `status` may
  hit a stale endpoint — see the `[M] <tool> status without -p/-n` entry in
  [FIXME.md](FIXME.md).

### Tearing it back down

Teardown is an ordered sequence (cancel managed jobs → destroy the agent →
destroy the cluster → delete the bucket → remove NPA-owned storage IAM → drop
the project entry → clear local state), and missing a step leaves a hung job,
credential, or cache behind. Run `npa cleanup` for a report plus the exact
runbook. Plain `npa cleanup --yes` keeps credentials; the explicit
`npa cleanup --full --yes` scope also removes saved Hugging Face, Token Factory,
and NGC credentials, removes only exactly validated NPA Terraform caches, and
prunes an empty `~/.npa` tree. It performs a read-only storage-IAM verification
but never deletes cloud resources. Cloud deletion remains separate:
`npa storage service-account delete` removes `lerobot-training` only when the
successful create response is present in NPA's final ownership record or its
crash-safe setup journal. A display-name match, legacy ID, reused account,
conflicting record, or user-managed account is never enough. Bucket deletion
removes secret material but first writes a project-scoped, non-secret cleanup
tombstone containing immutable service-account/access-key IDs, ownership
evidence, and the storage creation outcome. That provenance survives until the
exact IAM identity is deleted or verified absent.

For one project-scoped plan, use `npa destroy --project <alias> --all`; it is
read-only unless `--yes` is supplied. Execution journals the immutable project
identity and the complete phase plan, continues independent cleanup when one
phase fails, and blocks dependent phases rather than guessing. It retains the
Nebius project by default. Explicit `--delete-project --yes` additionally deletes
the exact project ID only when one unique durable NPA provider-create record
proves ownership and strict provider inventories prove every managed child class
empty. External/shared/unproven projects, nonempty inventories, unreadable or
schema-invalid evidence, permission failures, and identity conflicts are refused.
NotFound is repeat-safe verified absence. The individual commands below remain
the exact recovery surface for a partial run. If the alias was already forgotten,
`npa destroy --receipt <id> --all --delete-project --yes` exposes only the narrow
project-deletion phase and recovers its exact project/tenant/region identity from
the durable receipt; it does not reopen a deleted Terraform backend.

Storage IAM results are explicit: verified absence/deletion exits 0; missing
trustworthy ownership or a provider/auth verification failure reports
`Partial cleanup` and exits 2. A project-scoped, non-secret
`storage_iam_verification_required` journal keeps the exact candidate visible and
blocks `--forget-project` until provider-verified absence or guarded deletion.
Do not treat exit 2 as success. `agent destroy`, `storage bucket delete`, and
`storage service-account delete` share one confirmation contract: an interactive
terminal prompts when `--yes` is absent, a non-interactive invocation refuses
with exit 1, and explicit `--yes` is required to bypass confirmation. Read-only
and `--dry-run` paths remain available without confirmation; `--json` confirmation
refusals contain one machine-readable document. The complete NPA-only sequence is:

```bash
npa workflow cancel <run-id> --project <alias> --json
npa agent destroy --project <alias> --name <name> --yes
npa skypilot cleanup-controller --project <alias> --context <context> --yes
npa cluster down --project <alias> --force
npa storage bucket delete --project <alias> --yes --wait
npa storage service-account delete --project <alias> --dry-run
# Only when the previous command reports missing ownership provenance:
npa storage service-account reconcile --project <alias> --id <exact-id> --dry-run
npa storage service-account reconcile --project <alias> --id <exact-id> \
  --reason '<legacy NPA setup evidence>' --attest-npa-created --yes
npa storage service-account delete --project <alias> --dry-run
npa storage service-account delete --project <alias> --yes
# If validation created a project-local registry, delete its exact artifact DAG
# and registry using the immutable ID/name recorded at creation:
npa registry delete --project <alias> --project-id <project-id> \
  --tenant-id <tenant-id> --id <registry-id> --name <registry-name> --yes
# NPA-created disposable projects may contain one provider-created default
# topology. This command refuses any extra, shared, or non-default topology:
npa network delete-project-default --project <alias> --project-id <project-id> \
  --tenant-id <tenant-id> --yes
# Optional and ownership-gated; omit to retain the project (the safe default):
npa destroy --project <alias> --all --delete-project --yes --json
npa cleanup --full --yes --project <alias>
npa configure --forget-project <alias>
```

`configure --forget-project` durably writes and prints an opaque receipt ID
before rewriting configuration. If teardown must resume after that point, stay
inside NPA and select the same immutable identity explicitly:

```bash
RECEIPT=<id printed by npa configure --forget-project>
npa agent destroy --receipt "$RECEIPT" --name <name> --yes
npa skypilot cleanup-controller --receipt "$RECEIPT" --context <context> --yes
npa cluster down --receipt "$RECEIPT" --context <context> --force
npa storage service-account delete --receipt "$RECEIPT" --id <exact-id> --dry-run
npa workflow cancel <run-id> --receipt "$RECEIPT" --json
# Optional, after every exact child cleanup has converged:
npa destroy --receipt "$RECEIPT" --all --delete-project --yes --json
```

Cleanup identity precedence is deterministic: exact flags, then the selected
receipt, then live configuration. Any overlapping mismatch is unsafe and fails
before provider or Terraform mutation; NPA never substitutes a default alias,
current Kubernetes context, or unrelated SkyPilot profile.

Registry and provider-default-network deletion require the same unique durable
NPA project-creation proof as project deletion. Registry teardown inventories
and removes immutable artifact IDs before deleting the exact registry. Default
network teardown accepts only one `default-network`, its one linked
`default-subnet-*`, and its provider-marked `default-security-group-*`; mixed or
additional inventory fails closed. Project deletion waits for eventual provider
absence instead of treating the first still-visible post-delete observation as
a failed mutation.

Every destructive phase writes a versioned, atomic, non-secret receipt under
`~/.npa/teardown-receipts/` before deleting the local evidence needed to audit
it. Managed jobs are checked and receipted before SkyPilot state is removed;
active or uncertain jobs preserve that state. Receipts survive project/config
removal, are not operational residue, and keep completed phases from reverting
to `unknown` on an idempotent retry. List them with `npa cleanup
--list-receipts`; prune only old, fully terminal receipts explicitly with `npa
cleanup --prune-receipts --receipt-retention-days <days> --yes`.

JSON reports expose `operational_residue_present`, `audit_receipts_retained`,
and `verification_unresolved` separately. A retained receipt alone never changes
`local_state: fully_cleaned` into `residue_present`; an unresolved action recorded
inside it remains operator action, not local operational residue.

Controller cleanup has shared blast radius. It accepts only an explicit or
unambiguously selected NPA project plus that project's exact saved context,
cross-checks immutable project/cluster identity, deletes remotely through the
SkyPilot abstraction, independently proves the controller pods absent, writes
the remote-absence checkpoint, and only then converges local SkyPilot metadata.
Authentication, RBAC, connectivity, stale, mismatched, or ambiguous identity
preserves local state for an exact retry; an unrelated current context or stale
SkyPilot profile is never a fallback.

The shared controller also has one global immutable owner. The core
accelerator-gated `provision-if-absent` transaction binds it automatically,
after the exact project/context/provider cluster identity is durable and before
GPU readiness or submission. `npa skypilot bind-controller --project <alias>
--context <context>` is therefore only for adopting an already-live cluster
outside that core flow. It performs the same provider identity checks and
rejects missing, destroyed, rolled-back, or replaced clusters. Cross-project
use is refused. `--rebind` is allowed only after the managed-job queue is proven
terminal; changing an alias for the same project/cluster IDs is not a rebind.
When an agent fails before its final config record is written, `npa agent status
--project <alias> --name <name> --json` reads the operation journal instead. It
reports the typed partial state, exact created-resource IDs and current provider
evidence, plus structured NPA-only resume/destroy commands without credentials.

Reconciliation verifies the immutable ID, expected name, project, tenant, and
selected CLI profile, then records a non-secret operator/when/reason attestation.
It never deletes IAM itself and never treats the display name as ownership. The
following `delete` still performs the existing access-key inventory and guarded
delete. Both operations are restart-safe; repeated full cleanup remains partial
while the provider state is unchanged.

`npa cluster down` uses the kubeconfig saved for the selected NPA cluster and
forces its credential plugin into non-interactive/no-browser mode for the
best-effort drain preview. It distinguishes authentication, RBAC, kubeconfig,
and API failures and still attempts Terraform destroy. For a full managed-cluster
deletion it takes one cluster-wide inventory of nodes, pods, controllers, and
PDBs with eviction-relevant selector/placement semantics. This catches system
workloads such as `cilium-operator`, CoreDNS, the CoreDNS autoscaler, and
`metrics-server`, including the common one-CPU-node-pool case where a replacement
cannot be scheduled. NPA first requests normal eviction. Only for an explicitly
confirmed full destroy whose exact NPA project/context/cluster identity is
verified may it temporarily remove those exact four `kube-system` PDBs; it
snapshots their specs and restores them if destroy aborts while the cluster
remains. Shared clusters, node-pool operations, unverified contexts, and every
user/application PDB are never weakened or force-deleted.

Ordinary cleanup deliberately leaves the invoking NPA environment alone. To
remove only a supported repository-local `.venv`, preview `npa uninstall`; the
actual deferred removal requires both `--remove-environment --yes`. A one-time
helper waits for NPA to exit and revalidates the exact path, inode, marker, and
receipt nonce before deleting it. Source, `.git`, credentials, user data, and
unrelated caches remain outside the plan.

When no cluster state/inventory and no NPA kubeconfig exist, `cluster down` is a
true no-op: it does not authenticate, initialize Terraform, download providers,
or call Kubernetes. Real Terraform runs place provider/module data in exact
NPA-owned temporary scratch and remove it on success or failure, so they do not
populate `deploy/cluster/.terraform`. `npa cleanup --full --yes` detects both a
failed scratch cleanup and the legacy source-checkout cache. A provider checksum
mismatch remains a hard failure: NPA keeps `.terraform.lock.hcl` read-only and
prints a reviewed `terraform providers lock` reconciliation command rather than
bypassing verification.

With `--receipt` or exact `--project-id/--cluster-id`, the same no-state decision
happens before Terraform: provider-verified absence exits 0, insufficient
identity fails once with the required selectors, and a present cluster without
recoverable owned state fails closed. Workflow cancellation reports
`NOT_SUBMITTED` only from durable planned/reserved evidence; if submission began
and S3 or SkyPilot verification is gone, it remains
`VERIFICATION_UNAVAILABLE` with exit 2.

For the full known-issues surface: [docs/workbench/troubleshooting/known-footguns.md](docs/workbench/troubleshooting/known-footguns.md)
and the active operational backlog in [FIXME.md](FIXME.md).

---

## Learn by doing — pick a robot

Short copy-paste walkthroughs — pick any robot or simulator. Full index:
[docs/workbench/guides/README.md](docs/workbench/guides/README.md).

| Guide                                                                                             | Robot                | Sim / engine     | Public dataset                     |
| ------------------------------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------------------- |
| [Pick-and-place with a Franka arm](docs/workbench/guides/franka-pick-and-place-genesis.md)        | Franka Emika Panda   | Genesis          | DROID (Franka)                     |
| [Teach a robot to push a T](docs/workbench/guides/pusht-sim-to-real.md)                           | sim pusher           | sim-to-real loop | `lerobot/pusht`                    |
| [Train a Reachy 2 humanoid policy](docs/workbench/guides/reachy2-lerobot-policy.md)               | Reachy 2             | LeRobot          | Pollen Robotics / LeRobot Hub      |
| [Make a Unitree G1 walk](docs/workbench/guides/g1-humanoid-walk-sonic.md)                         | Unitree G1           | MuJoCo           | NVIDIA GEAR-SONIC                  |
| [Train a quadruped to run](docs/workbench/guides/quadruped-isaac-lab.md)                          | ANYmal / quadruped   | Isaac Lab        | Isaac Lab built-in tasks           |

Want a data pipeline instead of a robot? The **NVIDIA Physical AI Data Factory**
blueprint (annotate → Cosmos Transfer augment → curate → visualize) has a
one-block copy-paste quickstart:
[docs/workbench/guides/physical-ai-data-factory-deploy.md](docs/workbench/guides/physical-ai-data-factory-deploy.md).

Longer end-to-end recipes (BDD100K + LanceDB, Isaac-Lab BYOF, LeRobot GPU
benchmarks): [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md).

---

## Workbench at a glance

Workbench is the main product surface. Every tool lives under `npa workbench`
(there is no `solutions` CLI namespace). Highlights:

- **`vlm-eval`** scores rollouts with API or self-hosted vLLM backends —
  see [`vlm-eval-single.yaml`](npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml).
- **`token-factory`** wraps Nebius Token Factory for zero-GPU inference,
  captioning, and reasoning against your own frames.
- **`health preflight`** validates HF / NGC / S3 / Token Factory credentials
  before a deploy or GPU job.
- **`sonic export`** converts locomotion checkpoints to ONNX.
- **`workflow validate-spec` / `plan-spec` / `run-spec` / `submit`** operate on
  customer-facing `npa.workflow/v0.0.1` specs — see
  [Author and submit workflows](#author-and-submit-workflows).
- **`foxglove`** packs run frames/metrics/logs into MCAP, installs the pinned
  [`@foxglove/embed`](https://docs.foxglove.dev/docs/embed/typescript-sdk) assets
  for the agent's embedded viewer, and exports or opens the canonical recording
  in Foxglove Web — see [docs/cli/foxglove.md](docs/cli/foxglove.md) and the
  [canonical export / Foxglove Web contract](docs/workbench/foxglove-export.md).
- **`trigger`** watches S3-compatible prefixes and retriggers workflows.
- **`golden-eval`** runs per-container hello-world reruns as a CI gate.
- Public GHCR releases and exact hardware-specific tags are listed in the
  [Workbench container image catalog](docs/workbench/container-image-catalog.md).
- SONIC image routing is manifest-driven — see
  [sonic-image-catalog.md](docs/workbench/sonic-image-catalog.md).

<details>
<summary><strong>Browse the full command inventory by category</strong></summary>

| Category         | Workbench commands                                                                                                                                                                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data curation    | `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list`         |
| Synthetic data   | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench cosmos2 transfer`; `npa workbench cosmos3 reason`; `npa workbench genesis generate-demos`; specs such as [`bdd100k-pipeline.yaml`](npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml) |
| Simulation      | `npa workbench isaac-lab train`, `eval`, `export-lerobot`, `export-onnx`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench sonic retargeting run`, `workflow`                                                                    |
| Eval            | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`, `workflow`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student`; `npa workbench golden-eval run`, `run-all`, `validate` |
| Robot policy    | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list`                    |
| World models    | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `finetune`, `optimize`, `autoscale`, `status`, `system-info`                                                                                                                                                                                   |
| Zero-GPU LLM    | `npa workbench token-factory caption`, `generate`, `reason`, `verify`, `models`, `workflow`, `status`                                                                                                                                                                                                    |
| Workflows       | `npa workbench workflow validate-spec`, `plan-spec`, `run-spec`, `submit`; workbench workflows under [`npa-workflows/`](npa/workflows/workbench/npa-workflows/)                                                                                                                                           |
| Observability   | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa workbench health preflight`; `npa workbench foxglove convert-run`, `inspect`, `install-sdk`, `config`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list`                                                                                       |
| Platform utils  | `npa configure` / `init`, `npa provision-if-absent`; `npa agent`, `npa skypilot bootstrap/status/verify`, `npa soperator`, `npa burst`, `npa cluster`, `npa network`, `npa adapter convert`, `npa convert lerobot-to-rrd/-mp4`, `npa viz`, `npa demo`                                                    |

</details>

Full CLI reference: [docs/cli/README.md](docs/cli/README.md).

---

## Author and submit workflows

Author pipelines as declarative `npa.workflow/v0.0.1` specs — a state graph of
Workbench `toolRef` steps with S3 handoffs, gates, and loops. The same YAML is
what you validate, plan, and submit to the cluster.

```bash
# Validate and plan (no submit)
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml
npa workbench workflow plan-spec     npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml --run-id demo

# Launch on Nebius (after npa configure)
npa workbench workflow submit npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --run-id demo --registry cr.eu-north1.nebius.cloud/<your-registry-id>

# Inspect the plan without launching
npa workbench workflow submit npa/workflows/workbench/npa-workflows/token-factory-caption.yaml \
  --plan-only --run-id demo
```

|                        |                                                                 |
| ---------------------- | --------------------------------------------------------------- |
| **Format**             | `apiVersion: npa.workflow/v0.0.1`                               |
| **CLI**                | `validate-spec` · `plan-spec` · `run-spec` · `submit`           |
| **Workbench workflows** | [`npa/workflows/workbench/npa-workflows/`](npa/workflows/workbench/npa-workflows/) |
| **Tool catalog**       | [docs/workbench/npa-workflow-tool-catalog.md](docs/workbench/npa-workflow-tool-catalog.md) |
| **Authoring guide**    | [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) |

`submit` plans the graph and launches the run. Prefer these specs for new
pipelines. Parallel fan-out and a few specialized paths remain outside
`v0.0.1` scope — see the catalog README for exceptions.

The **Sim2Real 14-stage engine** is a separate path
([skills/workbench/sim2real-engine/SKILL.md](skills/workbench/sim2real-engine/SKILL.md))
using `sim2real/runbook.yaml` plus Python stage glue.

Architecture context:
[docs/architecture/contributor-context.md](docs/architecture/contributor-context.md).

---

## Container registry

Every Workbench tool ships as a container image in a Nebius container registry —
a primary in `eu-north1` and a mirror in `us-central1`. Resolve the registry
through `npa configure` or `npa.deploy.images`; never hardcode a registry id.
The publicly redistributable subset is also mirrored to GHCR for anonymous
external pulls.

```bash
# Log Docker into the registry (tokens expire; a 401 on pull means refresh)
REGISTRY_HOST=cr.eu-north1.nebius.cloud npa/scripts/nebius_registry_docker_login.sh

# Build and push an image with the canonical tag for its tool
npa/docker/workbench/lerobot/build.sh --registry "$NPA_REGISTRY" --push

# Pull a published image without Nebius registry credentials
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
docker pull "${NPA_REGISTRY}/npa-retargeting:0.1.1"
```

| Reference | What it tells you |
| --- | --- |
| [Public Workbench image catalog](docs/workbench/container-image-catalog.md) | Exact GHCR image names, published tags, pull command, build dates, and intentional exclusions |
| [Image ↔ GPU compatibility matrix](docs/workbench/image-gpu-compatibility-matrix.md) | Every image against every Nebius GPU platform, and which cells are verified on real hardware |
| [Container packaging contract](docs/workbench/container-packaging.md) | Tiers, non-root users, ports, and redistribution classes each image must satisfy |
| [Container golden evals](docs/security/container-golden-evals.md) | The real capability test each image must pass — not an import probe |
| [Blackwell datacenter compatibility](docs/workbench/blackwell-datacenter-image-compatibility.md) | B200 / B300 build, tag, and validation runbook |
| [SONIC image catalog](docs/workbench/sonic-image-catalog.md) | Manifest-driven SONIC variant routing per GPU |
| [Image reproducibility](docs/security/image-reproducibility.md) | The two-tag strategy (`cuda12`, `cuda13-b300`) and how tags are pinned |

Every image declares a `redistribution` class in the packaging contract, which
decides whether it may leave the owning org. Public images may be mirrored to
GHCR; restricted images remain build-your-own in an operator-owned registry.
`cosmos3-serving` is currently restricted because its pinned base embeds a
runtime under NVIDIA's Deep Learning Container License. Set the class when you
add an image — the packaging-contract test fails a build that bakes a
non-redistributable runtime while claiming `public`.

---

## Validated on Nebius

Eight Workbench tools are validated end-to-end on Nebius today (LanceDB,
FiftyOne, LeRobot, Genesis, Isaac Lab, Cosmos, GR00T, SONIC). Track how each
tool scores across GPU tiers:

| Reference                                                                              | What it tells you                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [Image ↔ GPU compatibility matrix](docs/workbench/image-gpu-compatibility-matrix.md)   | Every image × every Nebius GPU platform, with the verified cells called out            |
| [B300 validation matrix](docs/b300-validation-matrix.md)                               | Which tools have passed on B300 vs which are vendor-paced or upstream-blocked         |
| [LeRobot GPU benchmarks](docs/workbench/cookbooks/lerobot-gpu-benchmarks.md)           | Steps/s throughput across H200 · B300 · L40S · RTX Pro 6000 by policy type            |
| [NVIDIA architecture coverage](docs/nvidia-platform-architecture-coverage.md)          | CUDA 12.8 x86_64 vs CUDA 13 aarch64 tool coverage                                     |
| [NPA workflow tool catalog](docs/workbench/npa-workflow-tool-catalog.md)               | Every `toolRef` you can compose in an `npa.workflow/v0.0.1` spec                       |
| [Partner roadmap](docs/architecture/partner-skills-roadmap.md)                         | NVIDIA Omniverse / NuRec / CAD-to-SimReady capabilities on the way — not yet shipped   |

---

## How it runs

Workbench runs on Nebius infrastructure: S3-compatible object storage for
artifacts, managed Kubernetes and GPU runtimes for multi-stage jobs
(H100, H200, L40S, B300, RTX6000 — validated per tool), and vLLM-compatible
endpoints for serving.

User secrets live in a versioned, exact-project map in
`~/.npa/credentials.yaml`; top-level storage fields are compatibility views of
the explicitly selected project, not host-global truth. Legacy global storage
records migrate only when their exact project ownership is provable; ambiguous
records remain unchanged and fail closed. Machine-managed config lives
in `~/.npa/config.yaml`. The repo supports multiple top-level solution
namespaces; Workbench is the current primary solution (`npa.workbench` /
`npa workbench`). Future solutions are additive and never rename or nest
Workbench.

See [solutions model](docs/architecture/solutions-model.md) ·
[CLI namespaces](docs/architecture/cli-namespaces.md) ·
[contributor context](docs/architecture/contributor-context.md).

---

## Repository layout

```text
npa/                       # Python package (CLI + SDK); install with `pip install -e npa`
  src/npa/cli/             # Typer entry point and every top-level command
  src/npa/workbench/       # Per-tool implementations (cosmos, lerobot, sonic, ...)
  workflows/workbench/
    npa-workflows/         # Workbench npa.workflow/v0.0.1 specs (author + submit these)
    sim2real/              # Staged 14-stage sim2real runbook
docs/                      # Quickstart, architecture, workbench guides, cookbooks
skills/                    # SKILL.md files for agents and contributors (source of truth)
deploy/                    # Terraform + cluster provisioning (uses Nebius solutions library)
research/                  # LeRobot deploy research (older reference)
workbench/mlflow/          # MLflow tracking-server compose stack
```

More architectural detail: [docs/architecture/contributor-context.md](docs/architecture/contributor-context.md).

---

## Documentation

| Topic                | Where to look                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Install & auth       | [docs/quickstart.md](docs/quickstart.md)                                                               |
| Workbench setup     | [docs/workbench/getting-started.md](docs/workbench/getting-started.md)                                 |
| Beginner robot guides | [docs/workbench/guides/README.md](docs/workbench/guides/README.md)                                    |
| Physical AI Data Factory | [deploy runbook](docs/workbench/guides/physical-ai-data-factory-deploy.md) (copy-paste quickstart) · [concepts](docs/workbench/guides/physical-ai-data-factory.md) |
| Cookbooks            | [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md) — includes the [BDD100K + LanceDB pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md) and [Isaac-Lab BYOF](docs/workbench/cookbooks/byof-isaac-lab/) |
| Workflow authoring   | [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) · [tool catalog](docs/workbench/npa-workflow-tool-catalog.md) |
| Container images     | [Public Workbench image catalog](docs/workbench/container-image-catalog.md) · [packaging contract](docs/workbench/container-packaging.md) |
| `npa agent`          | [skills/tools/npa-agent/SKILL.md](skills/tools/npa-agent/SKILL.md) · [agent operate](skills/workflows/agent-fresh-operate/SKILL.md) |
| Preemptible GPU VMs | [docs/workbench/preemptible-vms.md](docs/workbench/preemptible-vms.md)                                 |
| Troubleshooting      | [docs/workbench/troubleshooting/known-footguns.md](docs/workbench/troubleshooting/known-footguns.md) · [active FIXMEs](FIXME.md) · [FTUE audit](FTUE-AUDIT.md) |
| CLI reference       | [docs/cli/README.md](docs/cli/README.md)                                                               |
| Architecture        | [solutions-model.md](docs/architecture/solutions-model.md) · [cli-namespaces.md](docs/architecture/cli-namespaces.md) · [contributor context](docs/architecture/contributor-context.md) |
| Everything else     | [docs/workbench/](docs/workbench/)                                                                     |

---

## Contributing

We welcome PRs, issues, and workflow contributions.

```bash
pip install -e "npa[dev]"
make test
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the review checklist,
skill-maintenance requirements, and repo hygiene rules. New behavior should
have a matching root `skills/` entry — see [`skills/index.yaml`](skills/index.yaml).
Security disclosures: [SECURITY.md](SECURITY.md). Support and community
happen through GitHub [Issues](https://github.com/nebius/nebius-physical-ai/issues)
and [Pull Requests](https://github.com/nebius/nebius-physical-ai/pulls).

---

## License

Licensed under the [Apache License 2.0](LICENSE). Built by
[Nebius](https://nebius.com) and the physical-AI community.
