# Customer Runbook — Full Sim-to-Real Pipeline (Private)

**Audience:** Customer engineers starting from **zero NPA install** on a laptop, running the **full staged pipeline** (not smoke tests) on the shared **Nebius RTX mk8s** cluster.

**Demo profile:** Industrial **UR5e** + custom scene/cameras, **LeRobot public `lerobot/pusht`** trigger data, **800 environments**.

**Your machine:** terminal + `./run.sh status` only. **GPUs, containers, and workflow code** run on Nebius.

---

## What you are running

| Layer | Where | Open source / NPA |
|--------|---------|-------------------|
| Operator CLI | Your laptop (`~/npa-sim2real-demo`) | `npa` SDK/CLI from this repo (pip install) |
| LeRobot trigger data | Your S3 bucket | Hugging Face [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) |
| Robot / scene / cameras | Your S3 bucket | MuJoCo menagerie UR5e + JSON scene-spec (seed script) |
| Policy training | Cluster GPU Job | `npa-lerobot-vlm-rl` container (LeRobot + VLM-RL) |
| Sim + augment + VLM | Cluster sibling Jobs | Isaac Lab, Cosmos transfer/reason images |
| Artifacts | S3 | `sim2real-b/<RUN_ID>/reports/sim2real-report.json`, Rerun `.rrd` |

**Pipeline stages (abbreviated):** trigger ingest → envgen (800 envs) → inner/outer policy loops → held-out Isaac eval → VLM scoring → final report.

---

## Part 0 — Prerequisites

### Nebius (provided / shared for this engagement)

- Project with **S3 bucket**, **container registry**, and **mk8s cluster** access  
- Context name for kubeconfig (example on operator VM: `npa-rtxpro-mk8s` — use **your** context string in config)  
- Registry pull secret already configured on the cluster (operator refreshes on submit)

### Local machine (Mac or Linux)

| Tool | Purpose |
|------|---------|
| `git` | Clone `nebius-physical-ai` |
| `python3` ≥ 3.10 + `venv` | NPA virtualenv (never system-wide pip) |
| `kubectl` | Optional direct job inspection |
| `awscli` | Optional S3 debugging |
| [Nebius CLI](https://docs.nebius.com/cli/install) | Auth + kubeconfig |

**RAM on laptop:** 8 GB+ free for venv + sync; pipeline memory is on cluster nodes (~1.7 TB allocatable per RTX node).

### Cluster capacity (RTX mk8s — shared)

Typical layout: **2× GPU nodes, 8 GPUs each (16 total)**, ~1.7 TB RAM per node.

| Workload | Guidance |
|----------|----------|
| **800-env demo** | One orchestrator Job + sibling GPU Jobs; fits comfortably if no duplicate runs |
| **10k validation** | Use longer timeouts; avoid overlapping 800 + 10k jobs |
| **Status check** | `./run.sh status <RUN_ID>` — no GPU on laptop |

---

## Part 1 — Install from zero (no existing NPA)

Paste once on **Mac or Linux**. Creates `~/npa-sim2real-demo/`, clones the repo, installs prerequisites, scaffolds secrets templates, builds the virtualenv.

```bash
bash <<'EOF'
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/opt/python@3.12/libexec/bin:${HOME}/.nebius/bin:${PATH}"
BRANCH="${NPA_BRANCH:-feat/sim2real-mandatory-stages}"
DEMO="${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
REPO="${DEMO}/nebius-physical-ai"
mkdir -p "${DEMO}"
if [ ! -d "${REPO}/.git" ]; then
  git clone --depth 1 --branch "${BRANCH}" \
    https://github.com/nebius/nebius-physical-ai.git "${REPO}"
fi
bash "${REPO}/ops/private/sim2real-rtxpro/first-time-setup.sh" "${REPO}"
EOF
```

**Virtualenv (auto-created):**

```text
~/npa-sim2real-demo/nebius-physical-ai/npa/.venv/bin/npa
```

Verify:

```bash
~/npa-sim2real-demo/nebius-physical-ai/npa/.venv/bin/npa --version
```

---

## Part 2 — Credentials (one time, edit — do not commit)

### 2.1 Private config templates

```bash
DEMO=~/npa-sim2real-demo
${EDITOR:-nano} "${DEMO}/private/config.yaml"
${EDITOR:-nano} "${DEMO}/private/credentials.yaml"
```

Replace every `YOUR-*` in:

- `private/config.yaml` — bucket, registry, **k8s_context**  
- `private/credentials.yaml` — S3 keys, optional `HF_TOKEN` for Cosmos VLM  

Templates: `ops/private/sim2real-rtxpro/private/*.example`

### 2.2 Kubeconfig

```bash
export PATH="${HOME}/.nebius/bin:${PATH}"
nebius profile create   # if needed
nebius mk8s cluster get-credentials --context YOUR-K8S-CONTEXT
CTX=YOUR-K8S-CONTEXT
mkdir -p ~/npa-sim2real-demo/private/clusters/"${CTX}"
cp ~/.npa/clusters/"${CTX}"/kubeconfig \
   ~/npa-sim2real-demo/private/clusters/"${CTX}"/kubeconfig
chmod 600 ~/npa-sim2real-demo/private/clusters/"${CTX}"/kubeconfig
```

### 2.3 Install private config into `~/.npa/`

Happens automatically on first `./run.sh` — or manually:

```bash
cd ~/npa-sim2real-demo && ./run.sh setup
```

---

## Part 3 — Demo run (800 env, full pipeline, industrial UR5e)

### 3.1 Seed LeRobot public data → your bucket

Downloads **`lerobot/pusht`** from Hugging Face and uploads to S3:

```bash
cd ~/npa-sim2real-demo
./run.sh seed-trigger
```

Note the printed `s3://YOUR-BUCKET/sim2real-triggers/<BATCH>/lerobot-pusht/` URI.

**Requirements on S3 tree:** `meta/info.json`, `data/*.parquet`, `videos/…` — the seed script handles this for pusht.

### 3.2 Seed industrial robot + scene assets → your bucket

On any machine with repo + credentials (laptop or operator VM):

```bash
REPO=~/npa-sim2real-demo/nebius-physical-ai
cd "${REPO}"
export CUSTOMER_TASK_ID="customer-demo-$(date -u +%Y%m%d)"
bash ops/private/sim2real-rtxpro/seed-industrial-production-assets.sh
```

Save the printed `robot_spec_uri` and `scene_spec_uri`.  
Profile template: `customer-asset-profiles/industrial.profile.example` (UR5e, conveyor/part meshes, custom cameras).

Dry-run profile resolution:

```bash
cd ~/npa-sim2real-demo
export CUSTOMER_ASSET_PROFILE=industrial
export CUSTOMER_TASK_ID=<same-as-seed>
source ~/.npa/sim2real-operator.env 2>/dev/null || true
bash nebius-physical-ai/ops/private/sim2real-rtxpro/apply-customer-asset-profile.sh
```

### 3.3 Operator env — industrial 800-env demo

```bash
cp ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/customer-demo-industrial.env.example \
   ~/.npa/sim2real-operator.env
chmod 600 ~/.npa/sim2real-operator.env
${EDITOR:-nano} ~/.npa/sim2real-operator.env
```

**Must edit:**

| Variable | Value |
|----------|--------|
| `TRIGGER_DATASET_URI` | From `./run.sh seed-trigger` |
| `CUSTOMER_TASK_ID` | Same as seed-industrial step |
| `TRIGGER_DATASET_URI` bucket | Your bucket in `private/config.yaml` |

Key defaults (already in template):

- `NPA_ENV_COUNT=800` — full pipeline at demo scale  
- `CUSTOMER_ASSET_PROFILE=industrial`  
- `NPA_SIM2REAL_SIM_BACKEND=isaac`  
- `INNER_ITERATIONS=2`, `OUTER_ITERATIONS=2`

### 3.4 Submit (cleanup + trigger)

Every new terminal:

```bash
cd ~/npa-sim2real-demo && ./run.sh demo
```

Submit **without** blocking the terminal:

```bash
cd ~/npa-sim2real-demo
export WAIT=0
./run.sh demo
```

Prints `run_id=…` and monitor commands.

### 3.5 Monitor from laptop

```bash
cd ~/npa-sim2real-demo
./run.sh status <RUN_ID>
```

Stages, K8s job state, and S3 report presence are polled here. Typical 800-env demo: **~30–90 minutes** depending on queue and stages.

Optional direct K8s:

```bash
export KUBECONFIG=~/npa-sim2real-demo/private/clusters/YOUR-K8S-CONTEXT/kubeconfig
kubectl get jobs | grep sim2real
kubectl logs job/sim2real-<RUN_ID> --tail=80
```

### 3.6 Download results + visualization

After `./run.sh status` shows completion:

```bash
cd ~/npa-sim2real-demo
./run.sh sync <RUN_ID>
```

Opens **Rerun** locally when `VISUALIZE=1` (default).  
S3 prefix: `s3://<bucket>/sim2real-b/<RUN_ID>/`

**Success criterion:** `sim2real-b/<RUN_ID>/reports/sim2real-report.json` exists on S3.

---

## Part 4 — Customize for your production data

### 4.1 Your LeRobot dataset (replace public pusht)

1. Convert or export data as **LeRobotDataset** layout (HF-compatible):  
   `meta/info.json`, `data/chunk-*/`, `videos/`  
2. Upload to **your** bucket:

   ```text
   s3://YOUR-BUCKET/sim2real-triggers/<YOUR-BATCH>/lerobot-<YOUR-TASK>/
   ```

3. Update `~/.npa/sim2real-operator.env`:

   ```bash
   TRIGGER_DATASET_URI=s3://YOUR-BUCKET/sim2real-triggers/<YOUR-BATCH>/lerobot-<YOUR-TASK>/
   TRIGGER_DATASET_ID=lerobot/<YOUR-TASK>
   ```

4. Preflight (checks S3 layout before submit):

   ```bash
   export TRIGGER_DATASET_URI=...
   bash ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/trigger-pipeline.sh
   # (preflight runs; Ctrl+C after preflight OK if you only wanted validation)
   ```

**Open-source path:** train/export with upstream [LeRobot](https://github.com/huggingface/lerobot); NPA consumes the standard layout.

### 4.2 LeRobot / trainer container

Default on cluster (from your registry):

```text
<YOUR-REGISTRY>/npa-lerobot-vlm-rl:0.1.0
```

Override in `~/.npa/sim2real-operator.env`:

```bash
TRAINER_IMAGE=YOUR-REGISTRY/npa-lerobot-vlm-rl:0.1.0
# OR your fork:
# TRAINER_IMAGE=YOUR-REGISTRY/your-org/lerobot-custom:tag
```

Orchestrator uses the same image unless `ORCHESTRATOR_IMAGE` is set. Registry must be pullable from mk8s (operator submit refreshes pull secret).

**NPA workbench (optional local test):**

```bash
source ~/npa-sim2real-demo/nebius-physical-ai/npa/.venv/bin/activate
npa workbench lerobot --help
```

### 4.3 Robot, scene, objects, cameras

Copy and edit a profile:

```bash
cp ops/private/sim2real-rtxpro/customer-asset-profiles/industrial.profile.example \
   ~/npa-sim2real-demo/private/my-assets.profile
chmod 600 ~/npa-sim2real-demo/private/my-assets.profile
```

Set in `~/.npa/sim2real-operator.env`:

```bash
CUSTOMER_ASSET_PROFILE=/home/you/npa-sim2real-demo/private/my-assets.profile
CUSTOMER_TASK_ID=your-production-batch
CUSTOMER_ROBOT_PRESET=ur5e   # or ur10e, flexiv
```

**Asset JSON examples:** `ops/private/sim2real-rtxpro/examples/customer-assets/`

| Axis | Modes |
|------|--------|
| Robot | `ROBOT_MODE=preset` + `ROBOT_PRESET` or `ROBOT_SPEC_URI` |
| Scene | `SCENE_MODE=custom` + `SCENE_SPEC_URI` |
| Object | `OBJECT_MODE=scene_spec` or `mesh` + `ASSETS_URI` |
| Cameras | `CAMERA_MODE=custom` (in scene-spec or `CAMERAS_URI`) |

Re-seed or upload URDF/MJCF/meshes to S3, then update `robot-spec.json` / `scene-spec.json`.

### 4.4 Scale after demo succeeds

```bash
# In ~/.npa/sim2real-operator.env
NPA_ENV_COUNT=10000
MONITOR_TIMEOUT_S=28800
NPA_SIM2REAL_K8S_JOB_TIMEOUT_S=32400
```

Submit a **new** run (`./run.sh demo`). Do not overlap multiple 10k jobs on the shared cluster.

---

## Part 5 — Daily command reference

| Command | When |
|---------|------|
| `cd ~/npa-sim2real-demo && ./run.sh demo` | New full pipeline run |
| `./run.sh status <RUN_ID>` | Watch progress |
| `./run.sh sync <RUN_ID>` | Pull artifacts + Rerun |
| `./run.sh seed-trigger` | Re-upload public pusht demo |
| `./run.sh cleanup --dry-run` | Preview reset |
| `./run.sh cleanup` | Clear local tmp + stale K8s jobs |

**Branch:** stay on `feat/sim2real-mandatory-stages` (or branch your operator specifies). `./run.sh` syncs repo each session.

---

## Part 6 — Troubleshooting

| Symptom | Fix |
|---------|-----|
| `first-time-setup.sh: No such file` | `git fetch && git checkout feat/sim2real-mandatory-stages` in repo |
| `template placeholders` / preflight fail | Finish `private/config.yaml` + `credentials.yaml` |
| `no LeRobot batch` / trigger preflight | Run `./run.sh seed-trigger`; verify URI in operator.env |
| `RobotSpecError` / `.xml` vs `.urdf` | Pull latest `feat/sim2real-mandatory-stages` (≥ `126d3f5`); re-seed assets |
| `ImagePullBackOff` | Re-run `./run.sh demo` (refreshes registry auth) |
| Job gone, empty S3 | Run failed — `./run.sh status` on logs; fix and `./run.sh demo` again |
| `401` / storage | Check `credentials.yaml` endpoint matches `config.yaml` region |
| Overlapping jobs | `./run.sh cleanup`; delete stray jobs: `kubectl delete job sim2real-<RUN_ID>` |

**Operator-only automation** (shared VM, not required on laptop): tmux `sim2real-converge` patch loop — see `OPERATOR-GUIDE.md`.

---

## Part 7 — Validation checklist (operator)

Before handoff, confirm on a clean laptop scaffold:

```bash
cd ~/npa-sim2real-demo
./run.sh setup
# edit private/*
./run.sh seed-trigger
# seed industrial assets + edit sim2real-operator.env
export WAIT=0
./run.sh demo
./run.sh status <RUN_ID>
# after success:
./run.sh sync <RUN_ID>
```

Scripts self-heal if branch drift deletes ops files: `ensure-converge-ops.sh` (operator VM).

---

## Related docs (same directory)

| Doc | Use |
|-----|-----|
| [QUICKSTART.md](./QUICKSTART.md) | Shorter paste blocks |
| [CUSTOMER-HANDOFF.md](./CUSTOMER-HANDOFF.md) | Public vs private file layout |
| [OPERATOR-GUIDE.md](./OPERATOR-GUIDE.md) | Mac paste terminal block |
| [customer-demo-industrial.env.example](./customer-demo-industrial.env.example) | 800-env industrial template |

**Git branch:** `feat/sim2real-mandatory-stages`  
**Do not commit:** `private/`, `~/.npa/sim2real-operator.env`, or kubeconfigs.
