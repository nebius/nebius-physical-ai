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

# A dedicated venv keeps the CLI isolated. The repo convention is npa/.venv.
python3 -m venv npa/.venv
source npa/.venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa --version
```

`pip install -e npa` installs the `npa` CLI/SDK in editable mode. Re-run it after
pulling changes that touch dependencies.

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
  bucket: <your-artifact-bucket>                          # no s3:// prefix
token_factory:
  api_key: <your-token-factory-key>
huggingface:
  token: <your-hf-token>            # optional
```

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
account when IAM allows), then bake the UI/backend:

```bash
# Provision the VM the first time (creates the agent record).
npa agent fresh-setup \
  --project <alias> --name <agent-name> \
  --project-id <nebius-project-id> --tenant-id <nebius-tenant-id> \
  --region <region>

# Bake/refresh the UI + backend + nginx on the VM (~1 min; reuses auth/creds).
npa agent bootstrap --project <alias> --name <agent-name>
```

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

## 4. Build and push the two Cosmos OSS images

Three stages pull a workbench image: `augment` needs `npa-cosmos2-transfer`
(already published), while `evaluate` and `cosmos-curate` need images this repo
builds. Build and push both once per branch:

```bash
REGISTRY="$(npa configure --show 2>/dev/null | grep -o 'cr\.[^ ]*' | head -1)"   # or your NPA_REGISTRY
printf '%s' "$(nebius iam get-access-token)" \
  | docker login "${REGISTRY%%/*}" -u iam --password-stdin

docker buildx create --name npa-cosmos-oss --driver docker-container   # scoped cache
for tool in cosmos-evaluator cosmos-curate; do
  docker buildx build --builder npa-cosmos-oss --push \
    -f "npa/docker/workbench/$tool/Dockerfile" \
    -t "$REGISTRY/npa-$tool:0.1.0" npa
done
docker buildx rm npa-cosmos-oss
```

Neither image carries model weights. The evaluator needs none; the curator's GPU
stages fetch theirs at run time with your Hugging Face token:

```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" -v curator-weights:/config/models \
  "$REGISTRY/npa-cosmos-curate:0.1.0" fetch-models --models split-annotate
```

Confirm all three images are pullable before spending GPU time — a missing one
surfaces late, as a stage failure:

```bash
for ref in npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z \
           npa-cosmos-evaluator:0.1.0 npa-cosmos-curate:0.1.0; do
  docker manifest inspect "$REGISTRY/$ref" >/dev/null && echo "OK   $ref" || echo "MISS $ref"
done
```

---

## 5. Submit the Physical AI Data Factory workflow

The blueprint lives at the promoted top-level path. Validate → plan → submit from
there:

```bash
SPEC=npa/workflows/physical-ai-data-factory.yaml

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" \
  --run-id demo --assume-decision promote_checkpoint --json
```

Stage input clips/frames under the run prefix your agent reads. If the agent's
base prefix is `checkpoints`, publish there directly so the run appears without a
copy:

```
s3://<your-artifact-bucket>/checkpoints/physical-ai-data-factory/<run-id>/input/
  video_0.mp4        # H.264/H.265, 720p-1080p, 5-15 s
  video_0_frame_01.png ... # extracted frames for the VLM stages
```

The CPU `run.shell` stages have no workbench image, so they install `npa` from
`NPA_SRC_S3_URI`. Stage the package **outside** the run prefix — the agent's
artifact browser lists everything under a run, and a few hundred `.py` files there
bury the stage outputs:

```bash
SRC_SHA="$(git rev-parse --short HEAD)"
export NPA_SRC_S3_URI="s3://<your-artifact-bucket>/npa-src/${SRC_SHA}/npa"
# Upload the whole installable tree, not just *.py: pyproject force-includes
# package data (solutions.toml, image manifests, golden_evals.yaml, agent_ui.html)
# and an editable install fails outright if any of them is missing.
aws s3 sync npa/src "$NPA_SRC_S3_URI/src" --exclude '*/__pycache__/*'
aws s3 cp  npa/pyproject.toml "$NPA_SRC_S3_URI/pyproject.toml"
```

Submit a real run (dynamic gate → pass `--assume-decision`):

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-artifact-bucket> \
  --var prefix="checkpoints/physical-ai-data-factory/<run-id>" \
  --registry "$REGISTRY" \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --output-format json
```

For a plan-only preflight without launching a GPU job, add `--plan-only`, and read
back the `image_id` lines to confirm the three images are pinned:

```bash
npa workbench workflow submit "$SPEC" --run-id preflight --plan-only \
  --assume-decision promote_checkpoint --registry "$REGISTRY" \
  --var bucket=<your-artifact-bucket> | grep -E 'image_id|accelerators'
```

> **Do not submit with cleared workbench images.** The submit CLI no longer treats
> `NPA_E2E_CLEAR_WORKBENCH_IMAGES` as a global override; it only clears image pins
> when you pass `--image none` explicitly. The plan-only `image_id` check above
> catches accidental unpinned submits before a run silently skips the real Cosmos
> Transfer, evaluator, or curator images.

Also `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN`: the Nebius provider prefers an
ambient (often expired) token over the fresh CLI one.

---

## 6. Multi-GPU fan-out (`RTXPRO6000:N`)

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
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

If your cluster advertises a different product name for the same GPU (e.g.
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`), pass that exact name:
`NPA_WORKFLOW_GPU_ACCELERATOR=<name>:N`. Variant parallelism is auto-detected from
the GPU count; override with `NPA_COSMOS_VARIANT_PARALLELISM`. Every managed
variant is conditioned on a supported video already uploaded beneath the run's
`input/` prefix (preserve geometry/motion, change only appearance); missing or
inaccessible input fails closed before inference.

---

## 7. Real FiftyOne curation (run in the `npa-fiftyone` image)

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

## 8. View results in the NPA agent

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

## 9. Where to go next

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
