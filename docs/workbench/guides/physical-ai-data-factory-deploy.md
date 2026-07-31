# Deploy the Physical AI Data Factory (from zero)

A step-by-step, copy-paste runbook that takes you from an empty machine to a
running **NVIDIA Physical AI Data Factory** blueprint on Nebius + SkyPilot, with
results viewable in the NPA agent (Main stages, embedded Rerun, Voxel51/FiftyOne
curation). It complements the conceptual
[Physical AI Data Factory guide](physical-ai-data-factory.md) (blueprint mapping,
stage graph, S3 layout) — read that first if you want the "what/why"; this doc is
the "how to stand it up".

The blueprint is a single `npa.workflow/v0.0.1` spec, promoted to the top of the
workflow tree for prominence:
[`npa/workflows/physical-ai-data-factory.yaml`](../../../npa/workflows/physical-ai-data-factory.yaml).
SkyPilot is the only orchestrator; there is no OSMO and no bespoke "data factory"
tool — every stage is an existing workbench tool or a real `run.shell` step.

> **No IDs are hardcoded here.** Everywhere you see `<...>` substitute your own
> project/tenant/registry/bucket. Credentials live in `~/.npa/credentials.yaml`;
> machine-managed config in `~/.npa/config.yaml`.

---

## Quick start (copy-paste)

From a clone with `npa` installed (§1) and `npa configure` run (§2), this is the
**complete** ordered path to a stock-Cosmos demo run. Every command is needed —
skipping one is what makes a first submit fail. The run needs **no dataset**:
`seed_default_input=true` seeds its own frames for the mandatory caption stage.

```bash
# Fill BUCKET / PROJECT / KUBE_CONTEXT from ~/.npa instead of hand-substituting
# them (`npa configure --show --env` prints NPA_* assignments and no secrets;
# `npa configure --show` prints the same values in a readable block).
eval "$(npa configure --show --env)"
BUCKET="$NPA_BUCKET"
PROJECT="$NPA_PROJECT_ALIAS"
KUBE_CONTEXT="${NPA_KUBE_CONTEXT:-npa-cluster}"   # set after the cluster exists
RUN_ID="$(date -u +paidf-%Y%m%dt%H%M%sz)"

# 1. Credentials the stages need, and a GPU cluster + bucket if absent.
npa workbench health preflight
npa provision-if-absent --project "$PROJECT"          # --dry-run first to preview

# 2. SkyPilot is the orchestrator. bootstrap saves skypilot.sky_bin into
#    ~/.npa/config.yaml, so new shells resolve it with no exports.
npa skypilot bootstrap
npa skypilot verify --cluster <your-cluster-name>     # sanity-check the kubeconfig

# 3. Publish the npa package for the image-less (Token Factory / run.shell)
#    steps. `submit --stage-src` below does this inline; run it standalone to
#    reuse one copy across submits.
npa workbench workflow stage-src --bucket "$BUCKET"
export NPA_SRC_S3_URI="s3://$BUCKET/npa-src/npa/"

# 4. Secrets must be exported for --secret-env to forward them.
export NEBIUS_TOKEN_FACTORY_KEY=<your-token-factory-key>
export AWS_ACCESS_KEY_ID=<...> AWS_SECRET_ACCESS_KEY=<...>

# 5. Submit.
npa workbench workflow submit npa/workflows/physical-ai-data-factory.yaml \
  --run-id "$RUN_ID" --var bucket="$BUCKET" --var seed_default_input=true \
  --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Not sure what is still missing? `submit` checks first and prints everything at
once:

```text
Error: Cannot submit physical-ai-data-factory.yaml: missing prerequisites:
  - SkyPilot CLI is not usable (...)
      fix: run `npa skypilot bootstrap` ...
  - npa source for image-less steps (NPA_SRC_S3_URI is unset)
      fix: pass --stage-src, or set NPA_SRC_S3_URI=..., or pin --image ...
  - config.bucket is the spec placeholder 'example-bucket'
      fix: pass --var bucket=<your-bucket>
```

Add `--plan-only` to render the SkyPilot YAML without launching, or
`--skip-preflight` to bypass the checks.

Prefer to caption **real** frames? Stage them first (needs `ffmpeg` and the S3
keys / `AWS_ENDPOINT_URL` from §2), then submit **without** the flag:

```bash
INPUT="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/input"
# Synthesize with no source asset, or swap for a real clip:
#   ffmpeg -i my_clip.mp4 -vf fps=2 -frames:v 12 frame_%04d.png
ffmpeg -f lavfi -i testsrc=size=1280x720:rate=1 -frames:v 12 frame_%04d.png
aws s3 cp . "$INPUT/" --recursive --exclude '*' --include 'frame_*.png'

npa workbench workflow submit npa/workflows/physical-ai-data-factory.yaml \
  --run-id "$RUN_ID" --var bucket="$BUCKET" --stage-src \
  --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Either way the augment stage renders the bundled Cosmos example; the frames (real
or seeded) satisfy the mandatory caption stage. `seed_default_input=true` never
overwrites frames already staged under `input/`. To transfer appearance onto
**your** footage instead of stock material, stage a real `video_0.mp4` and add
`NPA_COSMOS_CONDITION_ON_INPUT=1` (see §5). Everything below is the full
explanation.

### If submit fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `missing prerequisites: ... NPA_SRC_S3_URI is unset` | image-less steps have no `npa` to install | `npa workbench workflow stage-src --bucket <b>`, or `submit --stage-src`, or pin `--image` |
| `missing prerequisites: ... SkyPilot CLI is not usable` | SkyPilot never bootstrapped, or only exported in a previous shell | `npa skypilot bootstrap` (persists `skypilot.sky_bin`) |
| `missing prerequisites: ... config.bucket is the spec placeholder` | submitting against `example-bucket` | `--var bucket=<your-bucket>` |
| `controller health check failed: ... kubeconfig ... No such file` | a cached `sky-jobs-controller-*` from another setup points at a kubeconfig that is gone | `sky status -r` (SkyPilot 0.12 rejects `sky status --all`), then `sky down sky-jobs-controller-<id>`; provision/point at a real cluster (`npa provision-if-absent`), and pass `--infra k8s/<context>` |
| `Kube context '<name>' ... is not available` | no cluster for that context: neither your kubeconfig nor `~/.npa/clusters/<name>/` has it | provision one (`npa provision-if-absent --project <alias>`, and read its warnings — it now exits non-zero when it could not) or point `KUBECONFIG` at the cluster you want; `kubectl config get-contexts` lists what is resolvable |
| `Context <name> not found ... Available contexts: []` (from SkyPilot) | an older npa left `KUBECONFIG` unset for a cluster it had provisioned | upgrade npa: `submit --infra k8s/<context>` now prepends `~/.npa/clusters/<context>/kubeconfig` itself. For `kubectl`/bare `sky`, `export KUBECONFIG=~/.npa/clusters/<context>/kubeconfig` |
| `provision-if-absent` failed on `~/.ssh/id_rsa.pub` | old default node-group key path | upgrade npa: `cluster up` now pins the first key that exists (`NPA_SSH_PUBLIC_KEY`, `id_ed25519.pub`, `id_rsa.pub`, `id_ecdsa.pub`) |
| `No images found .../input/` | the caption stage ran with an empty `input/` | stage frames, or add `--var seed_default_input=true` |

---

## 0. Prerequisites

- **Python 3.10+** and `git` on an operator/dev machine (Linux or macOS). All
  `npa`, `nebius`, and `docker` commands below run on this operator machine — the
  agent VM only receives staged env files.
- A **Nebius** tenant/project with access to: Managed Kubernetes (GPU nodes),
  Container Registry, and Object Storage (S3-compatible).
- A **Token Factory** API key (hosted VLM/LLM inference; zero-GPU stages).
- Optional: a **Hugging Face** token if you stage gated inputs/checkpoints.
- For the GPU augment stage: at least one GPU node. This blueprint is validated
  on **RTX PRO 6000 Blackwell** (`RTXPRO6000`); any SkyPilot-addressable GPU
  works if you remap the accelerator (see §6).

---

## 1. Install `npa`

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

# A dedicated venv keeps the CLI isolated. Same path as the README.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa --version
```

`pip install -e npa` installs the `npa` CLI/SDK in editable mode. Re-run it after
pulling changes that touch dependencies.

> Repo-root `.venv` is the path every user-facing doc uses. Contributor tooling
> (`docs/testing/e2e.md`, the agent skills) uses `npa/.venv` for repo validation;
> either works, but stay consistent within a checkout.

---

## 2. Credentials and config

`npa` reads two files under `~/.npa/`. Create them with `npa configure` (or hand
-write them). **Never commit these; they hold secrets.**

### 2a. `~/.npa/credentials.yaml` (secrets)

S3/object-storage keys, the Token Factory key, and optional HF token:

```yaml
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.<region>.nebius.cloud   # e.g. us-central1
  bucket: s3://<your-artifact-bucket>/
tokens:
  NEBIUS_TOKEN_FACTORY_KEY: <your-token-factory-key>
  HF_TOKEN: <your-hf-token>         # optional
ngc:
  api_key: <your-ngc-api-key>       # optional
```

This is exactly the layout `npa configure` writes — run `npa configure --show`
to print it. The service-shaped aliases `token_factory: {api_key: ...}` and
`huggingface: {token: ...}` are also accepted, but `tokens:` is canonical and
wins when both are present.

You can also point the CLI at these via environment variables when scripting:

```bash
export AWS_ACCESS_KEY_ID=<...>
export AWS_SECRET_ACCESS_KEY=<...>
export AWS_ENDPOINT_URL=https://storage.<region>.nebius.cloud
export NEBIUS_TOKEN_FACTORY_KEY=<...>
export HF_TOKEN=<...>                # optional
```

### 2b. `~/.npa/config.yaml` (machine-managed project config)

Project/tenant/registry/cluster metadata (IDs are yours; nothing is baked into
the repo):

```yaml
default_project: <alias>            # e.g. rtxpro
projects:
  <alias>:
    project_id: <nebius-project-id>
    tenant_id: <nebius-tenant-id>
    region: <region>                # e.g. us-central1
    registry_id: <container-registry-id>
    storage:
      checkpoint_bucket: s3://<your-artifact-bucket>/checkpoints/
      endpoint_url: https://storage.<region>.nebius.cloud
    kubernetes:
      kubeconfig: ~/.npa/clusters/<cluster-name>/kubeconfig
      gpu_profile: rtxpro           # optional; profile name your cluster advertises
```

Inspect the merged view any time:

```bash
npa configure --show
```

### 2c. Provision missing infra (optional)

If the bucket / cluster do not exist yet, `provision-if-absent` creates only what
is missing (dry-run first):

```bash
npa provision-if-absent --project <alias> --dry-run --output-format json
npa provision-if-absent --project <alias>            # real
```

The default cluster it provisions is the small FTUE shape — **1× GPU node
(`gpu-rtx6000`, `1gpu-24vcpu-218gb`) + 1× CPU node (`cpu-d3`, `4vcpu-16gb`),
Shared Filesystem OFF**. The Data Factory passes stage I/O over S3 URIs, so it
needs **no** Shared Filesystem (and no SFS SSD quota). For a training farm or a
shared `/mnt/data`, opt in via `deploy/cluster` tfvars/flags (see
[`deploy/cluster/README.md`](../../../deploy/cluster/README.md): `gpu_nodes_count`
/ multi-GPU `gpu_nodes_preset` / `enable_gpu_cluster`, and `enable_filestore` or
`existing_filestore`).

The container registry must be reachable for the workbench images. Point
`NPA_REGISTRY` (or the project `registry_id`) at your registry, e.g.
`cr.<region>.nebius.cloud/<registry-id>`.

---

## 3. Deploy the NPA agent

The agent is a public HTTPS workbench VM (basic-auth UI, grounded chat, embedded
Rerun, artifact browser, Voxel51/FiftyOne curation surface). See the
[`npa-agent`](../../../skills/tools/npa-agent/SKILL.md) skill for the full
operator reference; the
[`agent-fresh-operate`](../../../skills/workflows/agent-fresh-operate/SKILL.md)
skill covers teardown/fresh-setup loops.

First-time provisioning (Terraform-backed VM + long-lived `npa-agent` service
account when IAM allows). The simplest path — matching the root README — reads
the project ids/region from `~/.npa/config.yaml`, so you only pick a configured
project:

```bash
# Interactive: pick one of the projects `npa configure` saved, then deploy.
npa agent setup --name <agent-name>
```

For scripted / non-interactive deploys, pass the ids explicitly instead:

```bash
npa agent fresh-setup \
  --project <alias> --name <agent-name> \
  --project-id <nebius-project-id> --tenant-id <nebius-tenant-id> \
  --region <region>
```

Both provision the VM and bake the UI/backend. To re-bake later (e.g. after an
agent code change) without reprovisioning:

```bash
# Bake/refresh the UI + backend + nginx on the VM (~1 min; reuses auth/creds).
npa agent bootstrap --project <alias> --name <agent-name>
```

> **Deploying from behind a VPN/firewall?** `setup`/`fresh-setup` finish by
> SSHing into the new VM to wait for cloud-init. If this machine cannot reach the
> VM's public `tcp/22` (corporate VPN / split-tunnel often block it), the deploy
> times out (~4 minutes, with progress lines) and rolls the VM back with a
> one-line SSH-unreachable error. Deploy from a host with direct network access to
> the VM. `npa agent preflight` and the start of `deploy` probe outbound `tcp/22`
> and warn before a VM is created — against a recorded agent IP when you have one,
> otherwise a public host, which a split tunnel can allow while still dropping a
> fresh Nebius IP. Set `NPA_SSH_EGRESS_PROBE=<host>:<port>` to probe a host your
> network allows, or `off` to skip it.

> **Credential exposure on the agent VM.** The S3 access key and secret staged
> onto the VM are written into its cloud-init user data, which anyone with read
> access to the project can retrieve (`nebius compute instance get`). npa keeps
> those values out of the local process table (secret Terraform variables go
> through a 0600 var-file, not `-var`), but treat keys staged onto an agent VM as
> project-readable and rotate them when you destroy it. Tracked in `FIXME.md`.

> The backend and UI are **baked at bootstrap**. Any change to the agent code
> (`cli/agent.py`, `cli/agent_ui.html`, `cli/agent_chat.py`,
> `cli/agent_workflow.py`, `workflows/artifacts.py`, `workflows/data_factory_*.py`)
> requires re-running `npa agent bootstrap` to take effect on the live VM.

Verify the deploy:

```bash
NPA_AGENT_CHAT_LIVE=1 npa agent verify-live --project <alias> --name <agent-name>
```

### 3a. Auth and first connection

- Auth secrets are written to `~/.npa/agents/<alias>/<agent-name>/auth.env`
  (`AGENT_USER`, `AGENT_PASSWORD`) and are stable across redeploys.
- Public URL: `https://<public-ip>/` (self-signed cert on the VM IP).
- **On a phone:** open `https://<public-ip>/healthz` first to accept the
  self-signed certificate, then sign in at `https://<public-ip>/login-help.html`.
- Never use `localhost`/`127.0.0.1`/`:8080` — always use same-origin `/api/...`.

---

## 4. Submit the Physical AI Data Factory workflow

The blueprint lives at the promoted top-level path. Validate and plan first:

```bash
SPEC=npa/workflows/physical-ai-data-factory.yaml

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" \
  --run-id demo --assume-decision promote_checkpoint --json
```

### 4a. Stage input first (required — the run fails on empty `input/`)

The first stage, **annotate-original**, captions **image frames** from the run's
`input/` prefix. Submit without staging frames and it fails fast with:

```
No images found in s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/input/
```

…and the later augment → grade → curate → visualize → finalize stages never run
(only `configs/manifest.json` is written). This is true **even for the default
stock-Cosmos augment** (which re-renders a bundled Cosmos control example): the
augment *video* is stock, but captioning still needs real image files — **a
`.mp4` alone is not enough** for `workbench.token_factory.caption`.

> **Don't want to stage anything?** Submit with `--var seed_default_input=true`
> and the `generate-configs` stage auto-seeds `input/` with a few default
> synthetic frames (it never overwrites frames you staged yourself), so the run
> completes end-to-end with no upload. Use the real-frame staging below when you
> want the captions to describe actual footage.

Pick one `RUN_ID` and reuse it for both staging and submit so the S3 prefix and
`--run-id` match:

```bash
BUCKET=<your-artifact-bucket>
RUN_ID="$(date -u +paidf-%Y%m%dt%H%M%sz)"
INPUT="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/input"
# The aws CLI uses the S3 keys + AWS_ENDPOINT_URL you exported in §2.
```

**Minimum — 8–16 PNG/JPEG frames** (only the first `config.max_images`, default
8, are captioned; `.png`/`.jpg`/`.jpeg`):

```bash
aws s3 cp ./frames/ "$INPUT/" --recursive --exclude '*' --include '*.png'
```

**Optional — a source clip** for the Rerun viz and the condition-on-input augment
path:

```bash
aws s3 cp ./video_0.mp4 "$INPUT/video_0.mp4"   # 720p–1080p H.264/H.265, 5–15 s
```

No dataset yet? Two hermetic ways to produce captionable frames for a
stock-Cosmos-style demo end-to-end (needs `ffmpeg`), then upload them:

```bash
# (a) Extract frames from any short clip you have:
ffmpeg -i video_0.mp4 -vf fps=2 -frames:v 12 frame_%04d.png

# (b) …or synthesize frames with no source asset at all:
ffmpeg -f lavfi -i testsrc=size=1280x720:rate=1 -frames:v 12 frame_%04d.png

aws s3 cp . "$INPUT/" --recursive --exclude '*' --include 'frame_*.png'
```

Either way annotate-original gets real image files so the run proceeds while the
default augment still renders the bundled Cosmos stock example. To make augment
transform **your** footage instead (geometry/motion preserved, appearance
changed), stage a real `video_0.mp4` and submit with
`NPA_COSMOS_CONDITION_ON_INPUT=1` (see §5).

Confirm the frames landed before you submit:

```bash
aws s3 ls "$INPUT/"
```

### 4b. Submit a real run

Submit with the **same** `RUN_ID` (dynamic gate → pass `--assume-decision`):

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --stage-src \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

For a plan-only preflight without launching a GPU job, add `--plan-only` (set
`NPA_SRC_S3_URI=s3://<bucket>/npa-src/` or `--image` so the CPU tool steps
render, same as the other Token Factory specs).

---

## 5. Multi-GPU fan-out (`RTXPRO6000:N`)

The `augment` stage runs **one Cosmos Transfer 2.5 diffusion per sampled scenario
variant**. Request `N` GPUs and the stage fans the `N` variants across them (one
variant per GPU), so `N` variants complete in roughly one variant's wall-clock
instead of `N x`.

- Set the number of variants with `config.n_augmentations` (via `--var`, default
  `2`).
- Request `N` GPUs for the `gpu` resource profile. Either edit the spec's
  `resources.gpu.accelerators` to `RTXPRO6000:N`, or override at submit time
  without touching the committed blueprint using the
  `NPA_WORKFLOW_GPU_ACCELERATOR` env var (it fully replaces the accelerator
  string, count included):

```bash
# Fan 4 scenario variants across 4 GPUs on one node.
NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO6000:4 \
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-4gpu-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-artifact-bucket> \
  --var n_augmentations=4 \
  --stage-src \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

If your cluster advertises a different product name for the same GPU (e.g.
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`), pass that exact name:
`NPA_WORKFLOW_GPU_ACCELERATOR=<name>:N`. Variant parallelism is auto-detected from
the GPU count; override with `NPA_COSMOS_VARIANT_PARALLELISM`. Condition each
variant on the run's real input clip (preserve geometry/motion, change only
appearance) with `NPA_COSMOS_CONDITION_ON_INPUT=1`.

---

## 6. Real FiftyOne curation (run in the `npa-fiftyone` image)

The `curate` stage runs **real FiftyOne Brain curation** (uniqueness +
duplicate/near-dup detection + a PCA visualization) over the augmented set. Real
curation needs FiftyOne **and** a `mongod`, so it must run inside the
`npa-fiftyone` image (which now self-hosts `mongod`). On the generic CPU tier it
falls back to a report-only summary (`curation_engine=report-only`).

Run real curation standalone against a completed run's augmented output:

```bash
npa workbench fiftyone curate-augmented \
  --augment-uri s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/cosmos_augmented/ \
  --report-uri  s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/curation/report.json \
  --dedup-threshold 0.10
```

This writes an additive `fiftyone` block into `curation/report.json` (schema stays
`npa.fiftyone.curation.v1`; adds `curation_engine`, `curated_kept/dropped`, and a
`fiftyone.{brain,selection,visualization,samples}` block) that the agent's
Voxel51 tab renders. To have this happen automatically on every run, pin the
`curate` stage to the `npa-fiftyone` image in your submit (see the
[FiftyOne](../../../skills/tools/fiftyone/SKILL.md) skill and the conceptual
guide's curation section).

---

## 7. View results in the NPA agent

The agent discovers runs from its artifact bucket. If the agent's base prefix is
`checkpoints`, the run lands at
`checkpoints/physical-ai-data-factory/<run-id>/`. Open `https://<public-ip>/`,
sign in, then:

- **Main stages:** the run's stage graph (config → annotate → augment → grade →
  re-label → curate → visualize → finalize) with per-stage artifacts. Clicking a
  stage shows inline image/video thumbnails.
- **Rerun:** select the run and load it (or click `reports/sim2real.rrd` in the
  artifact browser) to open the embedded Rerun viewer with input + augmented
  frames and captions.
- **Voxel51 (FiftyOne curation):** uniqueness/kept/near-dupe badges, the source
  video sample, annotate-original captions on input frames, and the FiftyOne
  Brain PCA scatter.

Useful authenticated JSON endpoints for scripted verification:

```bash
GET  /api/artifacts/runs?prefix=physical-ai-data-factory
GET  /api/artifacts/run/<run-id>?prefix=physical-ai-data-factory
GET  /api/artifacts/provenance/<run-id>
GET  /api/fiftyone/dataset/<run-id>
POST /api/sim-viz/load-run        # {"run_id": "<run-id>"}
```

---

## 8. Where to go next

- Conceptual blueprint, stage-by-stage mapping, and S3 layout:
  [physical-ai-data-factory.md](physical-ai-data-factory.md).
- Authoring/validating `npa.workflow` specs:
  [`author-npa-workflow`](../../../skills/workflows/author-npa-workflow/SKILL.md).
- Operating the agent VM (chat, Rerun, verify-live):
  [`npa-agent`](../../../skills/tools/npa-agent/SKILL.md).
- Staged Sim2Real operations on Kubernetes:
  [`sim2real-operate`](../../../skills/workflows/sim2real-operate/SKILL.md).
- FiftyOne curation deep-dive:
  [`fiftyone`](../../../skills/tools/fiftyone/SKILL.md).
