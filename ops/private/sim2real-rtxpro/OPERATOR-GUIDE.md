# Sim2Real — Full Operator Guide (Mac + RTX Cluster)

**Laptop = interface only.** GPU stages run on Nebius mk8s. Artifacts land on S3.
You monitor, sync, and open Rerun locally.

**Branch:** `feat/sim2real-mandatory-stages`

**Related:** [CUSTOMER-DEMO.md](./CUSTOMER-DEMO.md) · [FRANKA-STOCK-GUIDE.md](./FRANKA-STOCK-GUIDE.md)

---

## 1. One-time setup

### Install tools (Mac)

```bash
xcode-select --install          # git, basic CLI
brew install kubectl awscli
brew install nebius/tap/nebius  # mk8s kubeconfig auth
```

### Configure NPA (no secrets in git)

```bash
npa configure
# writes ~/.npa/config.yaml (bucket, registry, k8s_context, endpoint)
# writes ~/.npa/credentials.yaml (S3, HF_TOKEN, NGC — chmod 600)
```

### Kubeconfig

```bash
nebius mk8s cluster get-credentials --context npa-rtxpro-mk8s
# → ~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
```

### Directory layout

```text
~/npa-sim2real-demo/
  run.sh                          ← copy from mac-run.sh (see below)
  nebius-physical-ai/             ← NPA checkout (feat/sim2real-mandatory-stages)
~/.npa/
  config.yaml
  credentials.yaml
  sim2real-operator.env           ← trigger URI + optional overrides (see §3)
  clusters/npa-rtxpro-mk8s/kubeconfig.resolved
```

### Operator env file (recommended)

Copy the example and edit trigger URI:

```bash
cp ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/sim2real-operator.env.example \
   ~/.npa/sim2real-operator.env
chmod 600 ~/.npa/sim2real-operator.env
```

Or generate from config:

```bash
cd ~/npa-sim2real-demo/nebius-physical-ai
./ops/private/sim2real-rtxpro/setup-local-operator.sh
# then merge ops/private/sim2real-rtxpro/env.local into ~/.npa/sim2real-operator.env
```

### Install `run.sh` (once)

```bash
cp ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/mac-run.sh \
   ~/npa-sim2real-demo/run.sh
chmod +x ~/npa-sim2real-demo/run.sh
```

---

## 2. Cosmos stack (latest pinned versions)

These match `npa/pyproject.toml` and the operator submit defaults. **You usually
do not override them** — they are set from `storage.registry` in config.yaml.

| Role | Workbench image | Tag | HF model (sibling Job) |
| --- | --- | --- | --- |
| **Stage 3 — Augment** | `npa-cosmos2-transfer` | **`2.5.0`** | Cosmos-Transfer2.5-2B (gated) |
| **Stage 8 — VLM eval** | `npa-cosmos3-reason` | **`3.0.1-genuine-sm120`** | see Reason2 + Reason3 below |
| **Reason2 sibling** | same image | **`3.0.1-genuine-sm120`** | `nvidia/Cosmos-Reason2-8B` |
| **Reason3 sibling** | same image | **`3.0.1-genuine-sm120`** | `nvidia/Cosmos-Reason2-2B` |

Full image refs (replace `<registry>` with your `storage.registry`):

```bash
<registry>/npa-cosmos2-transfer:2.5.0
<registry>/npa-cosmos3-reason:3.0.1-genuine-sm120
```

**Hugging Face:** accept licenses for Cosmos-Reason2-8B, Cosmos-Reason2-2B, and
Cosmos-Transfer2.5-2B. Mount `HF_TOKEN` via cluster secret `hf-ngc-tokens`.

**Dual VLM eval (default):** two sibling Jobs — Reason2 + Reason3 — gated by
`NPA_SIM2REAL_VLM_DUAL_REASON=1`.

Override only when pinning a hotfix:

```bash
# in ~/.npa/sim2real-operator.env
VLM_IMAGE=<registry>/npa-cosmos3-reason:3.0.1-genuine-sm120
AUGMENT_IMAGE=<registry>/npa-cosmos2-transfer:2.5.0
VLM_REASON2_MODEL=nvidia/Cosmos-Reason2-8B
VLM_REASON3_MODEL=nvidia/Cosmos-Reason2-2B
NPA_SIM2REAL_VLM_DUAL_REASON=1
```

Other sibling images (non-Cosmos):

| Stage | Image | Tag |
| --- | --- | --- |
| Orchestrator / trainer | `npa-lerobot-vlm-rl` | `0.1.0` |
| Policy rollouts | `npa-sim2real-reference-policy` | `0.1.1` |
| Held-out sim | `npa-isaac-lab` | `2.3.2.post1` |
| Held-out eval (alt) | `npa-sim2real-eval` | `0.1.1-genuine-sm120` |
| Envgen shards | `npa-sim2real-envgen` | `0.1.1` |

---

## 3. New terminal — paste once (pull + demo)

Handles: missing git, first clone, wrong branch, dirty tree, ff-only failure.

```bash
bash <<'NPA_SIM2REAL_DEMO'
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
[[ -f "${HOME}/.npa/sim2real-operator.env" ]] && source "${HOME}/.npa/sim2real-operator.env"
DEMO="${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
REPO="${NPA_SIM2REAL_REPO:-${DEMO}/nebius-physical-ai}"
BRANCH="feat/sim2real-mandatory-stages"
PASTE="${REPO}/ops/private/sim2real-rtxpro/paste-customer-demo.sh"
if [[ ! -d "${REPO}/.git" ]]; then
  GIT=""
  for GIT in "$(command -v git 2>/dev/null || true)" /usr/bin/git /opt/homebrew/bin/git; do
    [[ -n "${GIT}" && -x "${GIT}" ]] || continue
    break
  done
  [[ -n "${GIT}" && -x "${GIT}" ]] || { echo "ERROR: git not found. Run: xcode-select --install" >&2; exit 1; }
  [[ -e "${REPO}" && ! -d "${REPO}/.git" ]] && { echo "ERROR: ${REPO} exists but is not a git repo" >&2; exit 1; }
  mkdir -p "${DEMO}"
  echo "=== first-time clone ${BRANCH} -> ${REPO} ==="
  "${GIT}" clone --branch "${BRANCH}" -- https://github.com/nebius/nebius-physical-ai.git "${REPO}"
fi
[[ -f "${PASTE}" ]] || { echo "ERROR: missing ${PASTE}" >&2; exit 1; }
exec bash "${PASTE}"
NPA_SIM2REAL_DEMO
```

**Follow-up runs** (repo already present):

```bash
bash ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/paste-customer-demo.sh
```

---

## 4. Commands (`~/npa-sim2real-demo/run.sh`)

| Command | What it does |
| --- | --- |
| `./run.sh demo` | **Customer replay:** cleanup → submit (stock trigger from operator env) |
| `./run.sh cleanup` | Clear local tmp + finished K8s jobs |
| `./run.sh cleanup --run-id <id> --s3` | Also delete S3 artifact prefix |
| `./run.sh trigger` | Submit only (`WAIT=0`, prints monitor cmd) |
| `./run.sh status <RUN_ID>` | Live kubectl + S3 stage checklist (10s refresh) |
| `./run.sh sync <RUN_ID>` | Pull S3 artifacts + open Rerun |
| `./run.sh rehearsal` | Sync golden run from S3 + Rerun (no cluster) |
| `./run.sh full` | Submit + wait + sync + Rerun |
| `./run.sh help` | List commands |

### Typical customer flow

```bash
cd ~/npa-sim2real-demo
./run.sh demo
# note RUN_ID from output
./run.sh status sim2real-staged-XXXXXXXX
./run.sh sync sim2real-staged-XXXXXXXX
```

### Stock Franka trigger (no custom assets)

In `~/.npa/sim2real-operator.env`:

```bash
TRIGGER_DATASET_URI=s3://YOUR-BUCKET/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/
TRIGGER_DATASET_ID=lerobot/pusht
INNER_ITERATIONS=1
OUTER_ITERATIONS=2
# Do NOT set ASSETS_URI / SCENE_SPEC_URI — uses stock Franka + Isaac lift-cube
```

---

## 5. Stage checklist (monitor while running)

S3 prefix: `s3://<bucket>/sim2real-b/<RUN_ID>/`

| Stage | S3 marker | K8s sibling jobs |
| --- | --- | --- |
| 1 Trigger | `stage_01_trigger/trigger.json` | — |
| 2 Assets | `stage_02_assets/assets_manifest.json` | — |
| 3 Augment | `augment/cosmos2-transfer-result.json` | `s2r-cosmos-*` |
| 4–6 Envgen | `envs/raw/` | `s2r-envgen-*` |
| 7 Rollouts | policy artifacts | `s2r-policy-*` |
| 8 VLM | eval JSON under `eval/` | `s2r-vlm-*` (Reason2 + Reason3) |
| 10 Held-out | `eval/heldout/report.json` | `s2r-isaac-*` or `s2r-eval-*` |
| Report | `reports/sim2real-report.json` | — |
| Rerun | `reports/sim2real.rrd` | tier WORKS in report |

Orchestrator job name: `sim2real-<RUN_ID>`

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Usage: ./run.sh [rehearsal\|trigger\|full]` | Old wrapper — paste block above reinstalls `run.sh` |
| `git not found` | `xcode-select --install` |
| `401 ImagePullBackOff` | Submit path refreshes registry secret; re-run `./run.sh demo` |
| `job not found` + empty S3 | Run failed before start — check `./run.sh status` pod reason |
| Stuck at augment | `kubectl logs` on `s2r-cosmos-*`; verify `AUGMENT_IMAGE` tag **2.5.0** |
| Stuck at VLM | HF token + model licenses; verify `VLM_IMAGE` **3.0.1-genuine-sm120** |
| `kubectl not found` | `brew install kubectl` or fix PATH in paste block |
| `sleep`/`date` not found` | Broken PATH — use paste block (includes `/usr/bin`) |

---

## 7. Security

| Secret | Location |
| --- | --- |
| S3, HF, NGC | `~/.npa/credentials.yaml` (600) |
| Kubeconfig | `~/.npa/clusters/<context>/kubeconfig*` |
| Bucket / registry | `~/.npa/config.yaml` |

Cluster Jobs use `secretRef` only — credentials are never embedded in generated YAML.

---

## 8. Script map

| Script | Role |
| --- | --- |
| **`PASTE-NEW-TERMINAL.sh`** | Bootstrap clone + exec paste-customer-demo |
| **`paste-customer-demo.sh`** | git sync + install run.sh + demo |
| **`mac-run.sh`** | Template for `~/npa-sim2real-demo/run.sh` |
| **`run.sh`** | Operator CLI (demo/trigger/status/sync/…) |
| **`submit-k8s-staged-job.sh`** | Direct K8s submit + registry refresh |
| **`status-run-local.sh`** | kubectl + S3 monitor |
| **`cleanup-operator.sh`** | Reset for customer replay |
| **`setup-local-operator.sh`** | Generate `env.local` from config |
