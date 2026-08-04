# Deploy the Physical AI Data Factory (from zero)

A step-by-step, copy-paste runbook that takes you from an empty machine to a
running **NVIDIA Physical AI Data Factory** blueprint on Nebius + SkyPilot, with
results viewable in the NPA agent (Main stages, embedded Rerun, and an explicitly
provenanced dataset view backed by real FiftyOne Brain curation). It complements the conceptual
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

## Run PAIDF with a coding agent

First [configure and authenticate the Nebius CLI](https://docs.nebius.com/cli/configure)
for the target project. Create a [Token Factory project API
key](https://docs.tokenfactory.nebius.com/quickstart), a [Hugging Face read
token](https://huggingface.co/docs/hub/en/security-tokens), and an [NGC Personal
API Key](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#generating-ngc-api-keys)
with NGC Catalog access. Save each value alone in a file outside this repository
and run `chmod 600 <file>`; give the agent paths, never secret values.

Copy this prompt into your coding agent from the repository root:

```text
Use README.md and its linked PAIDF runbook to complete this task. Do not change
repository files.

The Nebius CLI is already authenticated. Configure NPA, deploy and verify the NPA
agent, provision one CPU node and one on-demand RTX PRO 6000 GPU node (use
preemptible only if needed), then run npa/workflows/physical-ai-data-factory.yaml
end to end with seeded input and one augmentation.

Tenant: <tenant-id>
Project: <project-id>
Region: <region>
Token Factory key file: </absolute/path/token-factory-key>
Hugging Face token file: </absolute/path/hf-token>
NGC API key file: </absolute/path/ngc-api-key>

Never print secrets or copy them into the repository, logs, or shell history.
Use NPA commands, continue autonomously, and report blockers and usability gaps
without fixing code. Report the agent URL, PAIDF run ID, manifest URI, and all
stage results. Leave resources running and show the NPA-only cleanup commands;
do not clean up until I ask.
```

## Quick start (copy-paste)

This is the complete clean-machine path to an input-conditioned demo run. It uses the
exact shipped spec, the public GHCR mirror unless your configured project names a
different registry, and the endpoint and secrets already stored by `npa
configure`. No stored secret is printed or manually exported. The first run
needs no dataset: `seed_default_input=true` seeds eight captionable frames, and
the GPU runner assembles those frames into the short clip that conditions Cosmos.

```bash
set -o pipefail
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e npa

# Interactive once per machine/project. This stores project, bucket, endpoint,
# S3 keys, Token Factory key, and optional HF/NGC tokens under ~/.npa/.
npa configure
eval "$(npa configure --show --env)"   # emits non-secret NPA_* assignments only

SPEC=npa/workflows/physical-ai-data-factory.yaml
PROJECT="$NPA_PROJECT_ALIAS"
BUCKET="$NPA_BUCKET"
REGISTRY="${NPA_REGISTRY:-ghcr.io/nebius/nebius-physical-ai}"
RUN_ID="$(date -u +paidf-%Y%m%dt%H%M%S%NZ | tr '[:upper:]' '[:lower:]')"

npa workbench health preflight
npa provision-if-absent --project "$PROJECT"
npa skypilot bootstrap

# Reload the kube context written by provision-if-absent, discover its actual
# accelerator spelling, and validate/plan with the real bucket.
eval "$(npa configure --show --env)"
KUBE_CONTEXT="$NPA_KUBE_CONTEXT"
npa workbench workflow gpus --context "$KUBE_CONTEXT" --spec "$SPEC"
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" \
  --assume-decision promote_checkpoint \
  --var bucket="$BUCKET" --var seed_default_input=true \
  --var n_augmentations=1 --json

# With GHCR this uses the Registry v2 anonymous token flow. With a configured
# private registry it uses the matching configured credentials and submit also
# refreshes the Kubernetes imagePullSecret before launch.
npa workbench workflow preflight-images "$SPEC" \
  --project "$PROJECT" --registry "$REGISTRY"

# --stage-src publishes npa for image-less stages. Each --secret-env NAME is
# resolved from the current environment first, then the selected project's NPA
# credentials; a missing value fails here before controller setup.
npa workbench workflow submit "$SPEC" \
  --project "$PROJECT" --registry "$REGISTRY" \
  --run-id "$RUN_ID" --stage-src \
  --var bucket="$BUCKET" --var seed_default_input=true \
  --var n_augmentations=1 \
  --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN

MANIFEST_URI="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/npa-workflow/manifest.json"
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
npa workbench workflow logs "$MANIFEST_URI" --project "$PROJECT" --stage augment

# Load the final Rerun artifact through the run-relative API contract. This
# intentionally exercises run_id + key + prefix instead of relying on an exact
# URI. Never print the sourced auth values.
AGENT_NAME=agent
AGENT_PUBLIC_URL="$(npa agent status --project "$PROJECT" --name "$AGENT_NAME" --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["public_url"])')"
source "$HOME/.npa/agents/$PROJECT/$AGENT_NAME/auth.env"
curl -skS --fail-with-body -u "$AGENT_USER:$AGENT_PASSWORD" \
  -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"$RUN_ID\",\"key\":\"reports/sim2real.rrd\",\"prefix\":\"physical-ai-data-factory\"}" \
  "$AGENT_PUBLIC_URL/api/sim-viz/load-artifact"
```

The configured S3 endpoint is selected automatically; `--s3-endpoint` is only an
explicit override. `NPA_REGISTRY` has the same precedence in `preflight-images`
and submit: an explicit `--registry` wins, then `NPA_REGISTRY`, then the selected
project's registry. Set `REGISTRY=ghcr.io/nebius/nebius-physical-ai` explicitly
to choose the public anonymous mirror even when the project has a private one.
The quick start requests one real augmentation variant for a decisive first run;
omit `--var n_augmentations=1` to use the spec's default two-variant multiply, or
raise it together with the requested GPU count for a larger batch.

Typical warm-stage times are tens of seconds for config generation, 1–3 minutes
for each Token Factory caption pass, 10–25 minutes for one Cosmos Transfer
augmentation, and 1–5 minutes for evaluation/curation/finalization. First image
pulls, checkpoint downloads, node scheduling, and jobs-controller startup can add
several quiet minutes. The commands do not impose a workflow deadline; during a
quiet period use `npa skypilot status`, the status command above (which shows the
requested accelerator), and `npa workbench workflow logs "$MANIFEST_URI"
--stage augment --follow` in another terminal.

SkyPilot may temporarily render not-yet-submitted downstream DAG rows with the
first task's CPU summary. Its human-readable queue can also show misleading
relative ages for newly started downstream tasks because those rows inherit the
managed job's submission timestamp. NPA does not rewrite SkyPilot's queue clock;
the durable manifest and stage status timestamps are the authoritative per-stage
times. The rendered augment task and NPA manifest/status retain the exact
submit-time accelerator (for example
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`).

For a first run with nothing staged, keep `--var seed_default_input=true` or the
mandatory caption stage has no frames. For real data, upload PNG/JPEG frames to
`s3://$BUCKET/physical-ai-data-factory/$RUN_ID/input/` and omit that variable;
the workflow uses those frames and never seeds over them.

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

Add `--plan-only` to render the SkyPilot YAML without launching. Do not bypass
preflight on a first run: it is what keeps registry/auth/config failures local.

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

Either way the frames satisfy the mandatory caption stage and condition real
Cosmos inference: a staged video is used directly when present, otherwise the GPU
runner assembles the frames into a temporary clip. `seed_default_input=true`
never overwrites frames already staged under `input/`. Everything below is the
full explanation.

### If submit fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `missing prerequisites: ... NPA_SRC_S3_URI is unset` | image-less steps have no `npa` to install | `npa workbench workflow stage-src --bucket <b>`, or `submit --stage-src`, or pin `--image` |
| `missing prerequisites: ... SkyPilot CLI is not usable` | SkyPilot never bootstrapped, or only exported in a previous shell | `npa skypilot bootstrap` (persists `skypilot.sky_bin`) |
| `missing prerequisites: ... config.bucket is the spec placeholder` | submitting against `example-bucket` | `--var bucket=<your-bucket>` |
| `controller health check failed: ... kubeconfig ... No such file` | a cached `sky-jobs-controller-*` from another setup points at a kubeconfig that is gone | `sky status -r` (SkyPilot 0.12 rejects `sky status --all`), then `sky down sky-jobs-controller-<id>`; provision/point at a real cluster (`npa provision-if-absent`), and pass `--infra k8s/<context>` |
| `Kube context '<name>' ... is not available` | no cluster for that context: neither your kubeconfig nor `~/.npa/clusters/<name>/` has it | provision one (`npa provision-if-absent --project <alias>`, and read its warnings — it now exits non-zero when it could not) or point `KUBECONFIG` at the cluster you want; `kubectl config get-contexts` lists what is resolvable |
| A cluster is RUNNING in the console but npa has no kubeconfig for it (interrupted provision) | `up` writes the kubeconfig only after apply finishes | `npa cluster kubeconfig --cluster-name <name> --project <alias>` adopts it (writes the kubeconfig + cluster state), or `npa cluster up` again to resume, or `npa cluster down --force` to remove it |
| `GPU quota is insufficient ...` before apply | the tenant's `compute.instance.gpu.<model>` allowance cannot cover the node group | raise the quota, or use the preemptible pool the message reports (`gpu_nodes_preemptible = true`), or pick a smaller preset/another platform |
| `Nebius refused node group ...` mid-apply | the platform rejected the node group (quota/capacity) after apply began; npa cancels rather than retrying to the Terraform timeout | fix the quota/capacity as above, then `npa cluster up` again to resume, or `npa cluster down --force` to remove the half-created cluster |
| `npa cluster status` shows a cluster RUNNING but lists a node group as `not RUNNING` | the control plane is up while that node group was never provisioned | the same quota/capacity fix; the cluster bills while it exists, so tear it down (`npa cluster down --force`) if you cannot get the nodes |
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

Explicit environment variables are optional overrides when scripting. They take
precedence over configured values, but are not required for the quick start:

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
(`gpu-rtx6000`, `1gpu-24vcpu-218gb`) + 1× CPU node (`cpu-d3`, `8vcpu-32gb`),
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
Rerun, artifact browser, Dataset & provenance curation surface). See the
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

## 4. Choose the public mirror or build into a private registry

Three stages pull a workbench image: `augment` needs `npa-cosmos2-transfer`,
`evaluate` needs `npa-cosmos-evaluator`, and `curate` needs `npa-cosmos-curate`.

The public `ghcr.io/nebius/nebius-physical-ai` mirror is anonymously pullable.
A new private project registry starts empty: `npa configure` selects or creates
it, but does not mirror images into it. Pick one path and preflight the same
registry submit will use:

```bash
REGISTRY=ghcr.io/nebius/nebius-physical-ai    # or the configured NPA_REGISTRY
npa workbench workflow preflight-images npa/workflows/physical-ai-data-factory.yaml \
  --project "$PROJECT" --registry "$REGISTRY"
```

That reports each image as `ok` / `not_found` / `forbidden` and prints the exact
build command for anything missing. `npa workbench workflow submit` runs the same
check before it provisions anything, so a missing image costs no GPU time.

For the public mirror, `ok` needs no login or build. For a private registry,
build and push what preflight reports missing (tags below track
`npa/src/npa/deploy/images.py`, which is what submit pulls):

```bash
REGISTRY="$(npa configure --show 2>/dev/null | grep -o 'cr\.[^ ]*' | head -1)"   # or your NPA_REGISTRY
printf '%s' "$(nebius iam get-access-token)" \
  | docker login "${REGISTRY%%/*}" -u iam --password-stdin

docker buildx create --name npa-cosmos-oss --driver docker-container   # scoped cache
for tool in cosmos-evaluator cosmos-curate; do
  # Tag must match npa/src/npa/deploy/images.py; submit pulls exactly that tag.
  docker buildx build --builder npa-cosmos-oss --push \
    -f "npa/docker/workbench/$tool/Dockerfile" \
    -t "$REGISTRY/npa-$tool:0.1.2" npa
done
docker buildx rm npa-cosmos-oss
```

Neither image carries model weights. The evaluator needs none; the curator's GPU
stages fetch theirs at run time with your Hugging Face token:

```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" -v curator-weights:/config/models \
  "$REGISTRY/npa-cosmos-curate:0.1.2" fetch-models --models split-annotate
```

Confirm all three images are pullable before spending GPU time — a missing one
surfaces late, as a stage failure:

```bash
for ref in npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z \
           npa-cosmos-evaluator:0.1.2 npa-cosmos-curate:0.1.2; do
  docker manifest inspect "$REGISTRY/$ref" >/dev/null && echo "OK   $ref" || echo "MISS $ref"
done
```

---

## 5. Submit the Physical AI Data Factory workflow

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
(only `configs/manifest.json` is written). Captioning needs real image files — **a
`.mp4` alone is not enough** for `workbench.token_factory.caption`. The augment
stage prefers a staged video but can assemble the caption frames into a temporary
conditioning clip.

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

**Optional — a source clip.** Cosmos conditions on it directly when present;
otherwise PAIDF encodes the staged frames as a short temporary clip:

```bash
aws s3 cp ./video_0.mp4 "$INPUT/video_0.mp4"   # 720p–1080p H.264/H.265, 5–15 s
```

No dataset yet? Two hermetic ways to produce captionable conditioning frames for
an end-to-end demo (needs `ffmpeg`), then upload them:

```bash
# (a) Extract frames from any short clip you have:
ffmpeg -i video_0.mp4 -vf fps=2 -frames:v 12 frame_%04d.png

# (b) …or synthesize frames with no source asset at all:
ffmpeg -f lavfi -i testsrc=size=1280x720:rate=1 -frames:v 12 frame_%04d.png

aws s3 cp . "$INPUT/" --recursive --exclude '*' --include 'frame_*.png'
```

Either way annotate-original gets real image files and Cosmos Transfer conditions
on them after the GPU runner assembles a temporary clip. To preserve the exact
motion of **your** footage, stage `video_0.mp4` alongside its caption frames;
managed PAIDF augmentation automatically selects it.

Confirm the frames landed before you submit:

```bash
aws s3 ls "$INPUT/"
```

### 4b. Submit a real run

Submit with the **same** `RUN_ID` (dynamic gate → pass `--assume-decision`):

```bash
npa workbench workflow submit "$SPEC" \
  --project "$PROJECT" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --stage-src \
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
  --stage-src \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

If your cluster advertises a different product name for the same GPU (e.g.
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`), pass that exact name:
`NPA_WORKFLOW_GPU_ACCELERATOR=<name>:N`. Variant parallelism is auto-detected from
the GPU count; override with `NPA_COSMOS_VARIANT_PARALLELISM`. Every managed
variant is conditioned on the run's `input/` prefix: a supported video is used
directly, or the required PNG/JPEG frames are assembled into a temporary clip
(preserve input geometry/motion, change only appearance). Missing or inaccessible
input fails closed before inference.

---

## 7. Real FiftyOne curation (run in the `npa-fiftyone` image)

The `curate` stage runs **real FiftyOne Brain curation** (uniqueness +
duplicate/near-dup detection + a PCA visualization) over the augmented set. It
uses the `workbench.fiftyone.curate_augmented` toolRef and therefore runs in the
`npa-fiftyone` image, which self-hosts `mongod`. The workflow passes
`--require-fiftyone`: a missing Brain/database runtime fails the stage instead of
publishing a report-only summary under a FiftyOne label.

Run real curation standalone against a completed run's augmented output:

```bash
npa workbench fiftyone curate-augmented \
  --augment-uri s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/cosmos_augmented/ \
  --report-uri  s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/curation/report.json \
  --curator-report-uri s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/curation/cosmos_curator.json \
  --require-fiftyone \
  --dedup-threshold 0.10
```

This writes an additive `fiftyone` block into `curation/report.json` (schema stays
`npa.fiftyone.curation.v1`; adds `curation_engine`, `curated_kept/dropped`, and a
`fiftyone.{brain,selection,visualization,samples}` block). This happens
automatically in the shipped PAIDF workflow. Standalone callers may omit
`--require-fiftyone` to request a clearly labeled report-only fallback, but the
agent will state that FiftyOne did not run rather than marketing that fallback as
FiftyOne review. See the [FiftyOne](../../../skills/tools/fiftyone/SKILL.md)
skill and the conceptual guide's curation section.

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
- **Dataset & provenance:** separate Original/input and Synthetic/augmented
  groups, explicit counts and filters, the input source kind/URI, per-sample
  provenance, uniqueness/kept/near-dupe badges, and the FiftyOne Brain PCA
  scatter. If a legacy run contains only report curation, the panel says
  “FiftyOne did not run” and does not present it as a Voxel51 review.

Useful authenticated JSON endpoints for scripted verification:

```bash
GET  /api/artifacts/runs?prefix=physical-ai-data-factory
GET  /api/artifacts/run/<run-id>?prefix=physical-ai-data-factory
GET  /api/artifacts/provenance/<run-id>
GET  /api/fiftyone/dataset/<run-id>
POST /api/sim-viz/load-run        # {"run_id": "<run-id>"}
POST /api/sim-viz/load-artifact   # {"run_id":"<run-id>","key":"reports/sim2real.rrd","prefix":"physical-ai-data-factory"}
```

---

## 8. Tear everything down

Cleanup has three owners — the agent VM, the cluster, and the storage/IAM that
`npa configure` provisioned — and each needs its own command. In order:

First, see what exists:

```bash
npa agent list                      # agents recorded under ~/.npa/config.yaml
npa cluster list                    # clusters known locally or in the project
npa storage bucket list --project "$PROJECT"
```

Then remove them:

```bash
# 1. Agent VM, its network, local record, and the IAM the deploy created for it.
npa agent destroy --project "$PROJECT" --name <agent-name> --yes

# 2. Cluster. `down` owns everything the Terraform path created — cluster, VPC,
#    subnet — and clears ~/.npa/clusters/<context>/. It reads
#    project/tenant/region from ~/.npa/config.yaml when tfvars omit them.
npa cluster down --terraform-dir deploy/cluster --project "$PROJECT" --force

# 3. Object storage. A versioned bucket cannot be deleted immediately, so this
#    schedules the purge, waits for completion, and drops the dead S3 keys.
npa storage bucket delete --project "$PROJECT" --yes --wait

# 4. Storage IAM. This deletes only the exact lerobot-training account whose
#    create-time NPA ownership record matches this project. Inspect first if wanted:
npa storage service-account delete --project "$PROJECT" --dry-run
npa storage service-account delete --project "$PROJECT" --yes

# 5. Remove the project stanza after every project-scoped cloud command has run.
npa configure --forget-project "$PROJECT"

# 6. Remove known shared-service credentials, caches, the SkyPilot venv/state,
#    and empty ~/.npa residue. Non-empty/unrelated local data is preserved.
npa cleanup --full --yes
```

Notes:

- **Which cluster verb:** `npa cluster down` is the complete teardown for a
  cluster `npa cluster up` / `npa provision-if-absent` created. `npa cluster
  destroy --name <cluster>` is the API-only path — it deletes the cluster but
  leaves the Terraform-managed `<cluster>-network` and subnet running — so use it
  for a cluster Terraform does not manage, or to clear local state for a cluster
  that is already gone.
- **Default security groups follow the parent network lifecycle.** Nebius does
  not allow direct deletion of a default security group. The agent/cluster
  teardown commands above delete the parent network only when their Terraform
  state proves NPA owns it; an existing/reused/shared network is preserved for
  its owner. Do not substitute a manual security-group delete loop.
- **Agent IAM is removed by default.** `destroy` deletes the project's
  `npa-agent` service account and access keys once no agent is left in the project
  (other agents share it, so it is kept while any remain). A deploy that rolled
  back still created them, which is why this is the default; `--keep-iam` reports
  them with the `nebius iam …` commands instead.
- **Storage IAM deletion is ownership-gated.** Configure records provenance only
  when NPA's create call made `lerobot-training`. The storage service-account
  command refuses an ID-only legacy record, a mismatched project, or an account
  configure reused. It also refuses to run while bucket credentials remain, which
  is why bucket deletion comes first. Bucket cleanup never removes IAM evidence:
  the dedicated `storage_iam` ownership record remains until the account is
  deleted or confirmed absent, even if agent bootstrap changes the generic
  `nebius.service_account_id`. Nebius CLI's raw
  `nebius iam service-account delete --id <id>` command has no `--yes` flag.
- **Do not inventory access keys as raw JSON.** The upstream list response may
  contain secret material. NPA cleanup uses CLI-side JSONPath field selection;
  for safe operator inventory use the filtered example in
  [Known footguns](../troubleshooting/known-footguns.md#raw-access-key-list-json-can-disclose-the-secret).
- **Cluster drain preview is non-interactive and ownership-aware.** `cluster down`
  uses NPA's saved kubeconfig for the selected context, disables browser auth in
  a temporary copy, and explains authentication, RBAC, kubeconfig, and API
  failures as preview-only. During a full cluster deletion it relaxes only the
  exact `kube-system` budgets for `coredns`, `cilium-operator`, and
  `metrics-server`; every user or unrecognized PDB remains untouched.

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
