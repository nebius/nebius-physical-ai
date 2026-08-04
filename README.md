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

Running PAIDF on Nebius needs writable storage, the optional browser agent, a
cluster, an orchestrator, and a copy of `npa` the workers can install. This
restart-safe shell path uses only existing commands, stops at the first failed
prerequisite, and defaults to on-demand capacity:

```bash
set -eu
set -o pipefail
PROJECT=<alias>
CONTEXT=npa-cluster
SPEC=npa/workflows/physical-ai-data-factory.yaml

npa configure
npa provision-if-absent --project "$PROJECT" --skip-k8s
npa agent preflight --project "$PROJECT"
npa agent status --project "$PROJECT" --name agent --json >/dev/null 2>&1 \
  || npa agent setup --project "$PROJECT" --name agent
npa provision-if-absent --project "$PROJECT" --cluster-name "$CONTEXT" \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand
npa skypilot bootstrap
eval "$(npa configure --show --env)"
BUCKET="$NPA_BUCKET"
npa workbench workflow preflight-images "$SPEC"
npa workbench workflow stage-src --bucket "$BUCKET"

RUN_STATE="$HOME/.npa/paidf-first-run-id"
if [ ! -s "$RUN_STATE" ]; then
  mkdir -p "$(dirname "$RUN_STATE")"
  date -u +paidf-first-%Y%m%dt%H%M%S%NZ | tr '[:upper:]' '[:lower:]' >"$RUN_STATE.tmp"
  chmod 600 "$RUN_STATE.tmp" && mv "$RUN_STATE.tmp" "$RUN_STATE"
fi
RUN_ID="$(tr -d '\r\n' <"$RUN_STATE")"
npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --registry "${NPA_REGISTRY:-ghcr.io/nebius/nebius-physical-ai}" \
  --run-id "$RUN_ID" --runtime --resume --stage-src --var bucket="$BUCKET" \
  --var seed_default_input=true --var n_augmentations=1 \
  --assume-decision promote_checkpoint --infra "k8s/$CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN

printf '%s\n' "Provisioned resources: writable S3 at $BUCKET (write/delete verified)."
npa agent status --project "$PROJECT" --name agent
npa cluster status --project "$PROJECT"
npa workbench workflow status "$RUN_ID" --project "$PROJECT"
printf '%s\n' \
  "Running/cost-bearing when status says running: agent VM; 1 cpu-d3/8vcpu-32gb node; 1 gpu-rtx6000/1gpu-24vcpu-218gb node; active PAIDF jobs." \
  "Absent: no resources beyond the storage, agent, cluster, and PAIDF state reported above were requested; absent resources remain absent." \
  "This script performs no teardown. Exact teardown commands:" \
  "Teardown (not run): npa workbench workflow cancel $RUN_ID --project $PROJECT" \
  "Teardown (not run): npa agent destroy --project $PROJECT --name agent --yes" \
  "Teardown (not run): npa cluster down --project $PROJECT --context $CONTEXT --force" \
  "Teardown (not run): npa storage bucket delete --project $PROJECT --yes" \
  "Teardown (not run, after bucket): npa storage service-account delete --project $PROJECT --yes"
```

`provision-if-absent` now reconciles and write-probes S3 before it considers
Kubernetes; interrupted configuration resumes from owner-only provenance in
`~/.npa/credentials.yaml`. It never launches the cluster while required storage
is missing. The command above asks for exactly one `cpu-d3` / `8vcpu-32gb` CPU
node and one `gpu-rtx6000` / `1gpu-24vcpu-218gb` RTX PRO 6000 node. On-demand is
the reliable default. If on-demand capacity is unavailable, rerun the same
provision command with `--preemptible` instead of `--on-demand`; preemptible VMs
can be reclaimed mid-run, so resume from durable S3 artifacts.

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
For the shortest agent-driven setup, [copy the exact PAIDF agent
prompt](docs/workbench/guides/physical-ai-data-factory-deploy.md#run-paidf-with-a-coding-agent).

For an already configured project and provisioned cluster, this is the complete
PAIDF submit, monitor, and agent-load path. `status` discovers PAIDF's nested
durable manifest from the run ID; `logs` below also shows its exact S3 location.

```bash
eval "$(npa configure --show --env)"
SPEC=npa/workflows/physical-ai-data-factory.yaml
PROJECT="$NPA_PROJECT_ALIAS"
BUCKET="$NPA_BUCKET"
REGISTRY="${NPA_REGISTRY:-ghcr.io/nebius/nebius-physical-ai}"
KUBE_CONTEXT="$NPA_KUBE_CONTEXT"
RUN_ID="$(date -u +paidf-readme-%Y%m%dt%H%M%S%NZ | tr '[:upper:]' '[:lower:]')"

npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --registry "$REGISTRY" --run-id "$RUN_ID" --stage-src \
  --var bucket="$BUCKET" --var seed_default_input=true \
  --var n_augmentations=1 --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN

MANIFEST_URI="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/npa-workflow/manifest.json"
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
npa workbench workflow logs "$MANIFEST_URI" --project "$PROJECT" --stage finalize

# Load the final recording using the run-relative artifact contract. The auth
# file is sourced, never printed. The default deployed agent name is "agent".
AGENT_NAME=agent
AGENT_PUBLIC_URL="$(npa agent status --project "$PROJECT" --name "$AGENT_NAME" --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["public_url"])')"
source "$HOME/.npa/agents/$PROJECT/$AGENT_NAME/auth.env"
curl -skS --fail-with-body -u "$AGENT_USER:$AGENT_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"$RUN_ID\",\"key\":\"reports/sim2real.rrd\",\"prefix\":\"physical-ai-data-factory\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

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

Setup prints four bounded phases around Terraform, SSH installation, and the
final probe; the SSH phase can be quiet for several minutes and prints a
`journalctl` diagnostic to run from another shell. Terraform's current outputs
are `platform` / `preset` (and `cpu_platform` / `cpu_preset` for the CPU-only
agent). The old `gpu_platform` / `gpu_preset` outputs remain deprecated aliases
for existing state and may contain CPU values.

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
removes only bucket credentials; storage-IAM provenance survives until the exact
identity is deleted or verified absent.

Storage IAM results are explicit: verified absence/deletion exits 0; missing
trustworthy ownership or a provider/auth verification failure reports
`Partial cleanup` and exits 2. Do not treat exit 2 as success. Recover with:

```bash
npa storage service-account delete --project-id <project-id> --dry-run
npa storage service-account delete --project-id <project-id> --yes
npa cleanup --full --yes --project <alias>
```

`npa cluster down` uses the kubeconfig saved for the selected NPA cluster and
forces its credential plugin into non-interactive/no-browser mode for the
best-effort drain preview. It distinguishes authentication, RBAC, kubeconfig,
and API failures and still attempts Terraform destroy. For a full managed-cluster
deletion it relaxes only the exact `kube-system` PDBs for the NPA system add-ons
`coredns`, `cilium-operator`, and `metrics-server`; user and unknown PDBs remain
protected and are named if they can delay the drain.

When no cluster state/inventory and no NPA kubeconfig exist, `cluster down` is a
true no-op: it does not authenticate, initialize Terraform, download providers,
or call Kubernetes. Real Terraform runs place provider/module data in exact
NPA-owned temporary scratch and remove it on success or failure, so they do not
populate `deploy/cluster/.terraform`. `npa cleanup --full --yes` detects both a
failed scratch cleanup and the legacy source-checkout cache. A provider checksum
mismatch remains a hard failure: NPA keeps `.terraform.lock.hcl` read-only and
prints a reviewed `terraform providers lock` reconciliation command rather than
bypassing verification.

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
- **`foxglove`** packs run frames/metrics/logs into MCAP and installs the pinned
  [`@foxglove/embed`](https://docs.foxglove.dev/docs/embed/typescript-sdk) assets
  behind the agent's embedded Foxglove viewer — see
  [docs/cli/foxglove.md](docs/cli/foxglove.md).
- **`trigger`** watches S3-compatible prefixes and retriggers workflows.
- **`golden-eval`** runs per-container hello-world reruns as a CI gate.
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

```bash
# Log Docker into the registry (tokens expire; a 401 on pull means refresh)
REGISTRY_HOST=cr.eu-north1.nebius.cloud npa/scripts/nebius_registry_docker_login.sh

# Build and push an image with the canonical tag for its tool
npa/docker/workbench/lerobot/build.sh --registry "$NPA_REGISTRY" --push
```

| Reference | What it tells you |
| --- | --- |
| [Image ↔ GPU compatibility matrix](docs/workbench/image-gpu-compatibility-matrix.md) | Every image against every Nebius GPU platform, and which cells are verified on real hardware |
| [Container packaging contract](docs/workbench/container-packaging.md) | Tiers, non-root users, ports, and redistribution classes each image must satisfy |
| [Container golden evals](docs/security/container-golden-evals.md) | The real capability test each image must pass — not an import probe |
| [Blackwell datacenter compatibility](docs/workbench/blackwell-datacenter-image-compatibility.md) | B200 / B300 build, tag, and validation runbook |
| [SONIC image catalog](docs/workbench/sonic-image-catalog.md) | Manifest-driven SONIC variant routing per GPU |
| [Image reproducibility](docs/security/image-reproducibility.md) | The two-tag strategy (`cuda12`, `cuda13-b300`) and how tags are pinned |

Every image declares a `redistribution` class in the packaging contract, which
decides whether it may leave the owning org. All workbench images are currently
`public`; the `restricted` class is kept for the next runtime we cannot ship.
Set the class when you add an image — the packaging-contract test fails a build
that bakes a non-redistributable runtime while claiming `public`.

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

User secrets live in `~/.npa/credentials.yaml`; machine-managed config lives
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
