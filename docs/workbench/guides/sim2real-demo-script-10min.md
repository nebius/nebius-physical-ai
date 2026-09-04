# Sim-to-Real Pipeline — 10-Minute Demo Script

> **Presentation-only legacy script.** Do not use the private operator-pack
> commands below to launch a qualification run. The sole production entrypoint
> is `npa workbench workflow submit npa/workflows/workbench/npa-workflows/sim2real.yaml --runtime`; follow
> [the canonical operator guide](sim2real-workflow.md). Pre-staged artifacts here
> are for a timed presentation, never live-run evidence.

**Audience:** Platform + robotics stakeholders  
**Duration:** 10 minutes (includes ~30 s buffer)  
**Data types & artifacts:** [sim2real-data-contracts.md](./sim2real-data-contracts.md)  
**Runtime:** pre-staged presentation artifacts; canonical live runs use the
adjacent workflow through the standard SkyPilot runtime

Replace placeholders before rehearsal:

| Placeholder | Example (operator pack) |
| --- | --- |
| `<cluster>` | from `~/.npa/config.yaml` (`storage.k8s_context`) |
| `<bucket>` | from `~/.npa/config.yaml` (`storage.bucket`) |
| `<prefix>` | `sim2real-b` |
| `<registry>` | from `~/.npa/config.yaml` (`storage.registry`) |
| `<pre-staged-run-id>` | completed golden run on your cluster bucket |
| `<live-run-id>` | From `submit-k8s-staged-job.sh` output |

S3 artifact root (canonical):

```text
s3://<bucket>/<prefix>/<run-id>/
```

---

## Pre-demo checklist (do this before the room)

1. **Pre-stage a golden run** — Sync the validated run tree for offline walkthrough:

   ```bash
   <private-operator-pack>/sim2real-rtxpro/prestage-offline-run.sh <pre-staged-run-id>
   # -> /tmp/sim2real-prestage/<run-id>/
   rerun /tmp/sim2real-prestage/<pre-staged-run-id>/reports/sim2real.rrd
   ```

   S3 canonical path: `s3://<bucket>/<prefix>/<pre-staged-run-id>/reports/sim2real.rrd`
2. **Start a live job early** — 15–30 min before showtime:

   ```bash
   export KUBECONFIG=~/.npa/clusters/<cluster>/kubeconfig
   npa workbench workflow submit \
     npa/workflows/workbench/npa-workflows/sim2real.yaml \
     --runtime --resume --run-id <live-run-id> <operator-vars-and-secrets>
   ```

3. **Optional: pre-stage a loop-back run** — Second run where `eval/gold-heldout/outer-XX/report.json` has `success_rate` below threshold and `outer_loop/loopback.json` exists (for held-out failure narrative).
4. **Preflight** — `npa workbench health preflight` (PASS/WARN/FAIL on HF, NGC, S3, Token Factory).
5. **Open tabs:** S3 browser (pre-staged root), terminal (`kubectl logs -f`), Rerun viewer with pre-staged `.rrd`.

---

## Stage → artifact map (14 stages)

Use this table as the backbone of the demo. Every row is a file or prefix you can open live or from pre-staged S3.

| # | Stage | What happens | NPA artifact (under run root) | Live vs pre-staged |
| --- | --- | --- | --- | --- |
| 1 | **Trigger** | Task-aligned Isaac seed manifest validated; run ID resolved | `stage_01_trigger/trigger.json`, `task-dataset-manifest.json` | Pre-staged: static. Live: first standard-runtime state |
| 2 | **Sim assets** | Normalize task, stock/BYO robot, object, physics, cameras, and success contract | `stage_02_assets/task-contract.json`, `consumed_*_spec.json` | Contract digest is authoritative in every downstream scenario |
| 3 | **Augment** | Real Cosmos Transfer output becomes scenario and Cosmos-Reason lineage | `augment/manifest.json` | State PPO does not falsely claim pixels as direct observations |
| 4 | **Env generation + curation** | Generate configs; reject invalid/unreachable/duplicate cases and measure coverage | `envs/raw/`, `envs/manifest/curation-manifest.json` | Show accepted/rejected counts, reasons, and strata |
| 5 | **Train / validation / gold split** | Deterministic disjoint stratified split | `envs/train/`, `envs/validation/`, `envs/gold-heldout/`, `envs/manifest/split-manifest.json` | Point at leakage proof and difficulty coverage |
| 6 | **Feature lineage** | Records that state PPO consumes scenario configs, while token/pixel artifacts remain reporting/VLM inputs | `tokens/manifest.json` | Quick JSON peek at explicit consumer fields |
| 7 | **Action rollouts** | Exact checkpoint across at least 64 curated scenarios with 32 decision/event samples each | `actions/train/outer-01/iter-01/rollout-*/` | Show applied config digests and simulator reach/contact/grasp/lift/place state |
| 8 | **VLM critique** | Dual Cosmos-Reason per-step labels calibrate against simulator truth | `vlm_eval/train/outer-01/iter-01/<rollout-id>.json` | Show confidence/disagreement plus bounded shaping |
| 9 | **RL signal + trainer** | Dense simulator reward + bounded VLM shaping → real 2,000-iteration RSL-RL PPO per resumed pass; fixed validation ranks checkpoints | `training_signal/train/...`, `byo-trainer/.../ppo-telemetry.json`, `checkpoints/validation-selection/` | Show nonzero temporal credit, PPO curves, and selection proof |
| 10 | **Gold held-out eval** | The admitted Isaac workflow task loads the validation-selected checkpoint | `eval/gold-heldout/outer-XX/report.json` | **Live:** strict success is 5 cm plus stable placement; 10/15/20 cm remain diagnostics |
| 11 | **Threshold gate** | Compare strict stable-placement `success_rate` to threshold (`0.50`) | `outer_loop/decision.json`; passing gates promote, while every real best checkpoint remains honestly packaged | **Fallback story** (below) |
| 12 | **Real-world validation (BYO seam)** | Documented external stub | `stage_12_external_validation/external_stub.json` | Always SEAM; customer hook point |
| 13 | **Retrigger** | Next dataset batch → back to Stage 1 | `stage_13_retrigger/retrigger.json` | Loop-of-loops metadata |
| 14 | **Rerun visualization** | Local `.rrd` timeline of rollouts, critiques, and held-out scores | `reports/sim2real.rrd` | Uploaded with the run tree when the `stage_14_rerun_viz` tier is **WORKS** |

**Cross-cutting artifacts** (show at end):

- `state/workflow_state.json` — outer loop reads measured strict validation/gold success and `final_decision`; no synthetic quality bump
- `reports/sim2real-report.json` — E2E summary (`outer_loop.latest_decision`, `inner_loop.reward_trend`)
- `reports/sim2real.rrd` — Rerun timeline (rollouts, critiques, held-out scores); uploaded with `--upload-artifacts` when `stage_14_rerun_viz` tier is **WORKS**

**Rerun / S3 gap (know before Monday):**

| `stage_14_rerun_viz` tier | `reports/sim2real.rrd` on S3 | Demo action |
| --- | --- | --- |
| **WORKS** | Present (~80 KB+) | `rerun reports/sim2real.rrd` after sync |
| **WARN** | Absent | `rerun-sdk` missing in orchestrator; show report JSON + say viz skipped |
| **SEAM** | Absent | `NPA_SIM2REAL_RERUN=0`; intentional disable |

Canonical URI is in `artifact_uris()` as `stage_14_rerun_viz_rrd`. The full run tree upload includes `.rrd` when emitted; there is no separate upload step.

---

## Minute-by-minute script

### 0:00–0:45 — Hook

> "This is a **14-stage, inspectable sim-to-real loop**: real robot data triggers the run, simulation generates environments, a VLM critiques rollouts, we convert critique into RL signal, train, evaluate on held-out sim, and either promote a checkpoint or loop back. Every stage writes JSON to object storage — no black box."

Show one slide or browser tab: the 14-state graph from [sim2real-architecture.md](./sim2real-architecture.md).

### 0:45–1:30 — One compositional workflow

> "One standard workflow owns every state, parallel lane, loop decision, retry,
> and durable S3 checkpoint."

```bash
# What runs on cluster (abbreviated)
npa workbench workflow submit npa/workflows/workbench/npa-workflows/sim2real.yaml \
  --runtime --resume --run-id <run-id> <operator-vars-and-secrets>
```

Mention: RTX PRO/L40S placement, queue admission, and retries remain visible to
the standard SkyPilot runtime; no stage adapter creates a hidden Job.

### 1:30–3:00 — Stages 1–6 — **pre-staged S3**

Open `s3://<bucket>/<prefix>/<pre-staged-run-id>/`.

1. **Stage 1** — `stage_01_trigger/trigger.json`: trigger dataset URI, run ID.
2. **Stage 2** — `consumed_scene_spec.json` + `consumed_robot_spec.json`:
   - *Talking point:* "Stage 2 is **WORKS** with stock Franka + Isaac tabletop when asset URIs are empty. BYO meshes and UR/Flexiv URDF are documented seams — failed BYO loads fail loud, no silent stock fallback."
3. **Stages 3–6** — Walk `augment/`, `envs/raw`, `envs/manifest`,
   `envs/{train,validation,gold-heldout}`, and `tokens/` in ~60 s.

> "Preamble finishes in one CLI call; state lands in `workflow_state.json` before any GPU-heavy work."

### 3:00–5:30 — Stages 7–9 (inner loop) — **mix live + pre-staged**

**Preferred live moment:** `kubectl logs -f job/sim2real-<live-run-id>` while outer-iteration runs.

| Show | Path / command |
| --- | --- |
| Rollout frames | `actions/train/outer-01/iter-01/rollout-0000/` |
| VLM Job spawned | `kubectl get jobs -l sim2real.local/run-id=<live-run-id>` |
| Critique schema | `vlm_eval/.../rollout-0000.json` — `npa.sim2real.vlm_eval.v2` |
| Trainer evidence | `inner_loop/outer-01/evidence.json` — policy delta, `reward_trend` |

> "Inner loop is VLM → signal → trainer, repeated `INNER_ITERATIONS` times. Policy,
> VLM, PPO, and held-out evaluation run in their admitted workflow GPU tasks
> and images are registry-qualified."

If live job not ready: stay on pre-staged paths — same filenames.

### 5:30–7:00 — Stage 10 (held-out) — **live if possible**

Open `eval/gold-heldout/outer-XX/report.json`:

```json
"schema": "npa.sim2real.heldout_eval.v1",
"success_rate": 0.625,
"per_env": [ ... ]
```

> "Held-out evaluation runs in the admitted Isaac Lab workflow task on an
> RT-core GPU and loads the validation-selected immutable checkpoint."

**Live:** watch held-out Job complete; **fallback:** pre-staged report (see below).

### 7:00–8:00 — Stage 11 (threshold) — **promote vs loop-back**

Open `outer_loop/decision.json`:

```json
"success_rate": 0.625,
"threshold": 0.50,
"decision": "promote_checkpoint"
```

If promoting, flash `checkpoints/candidate/candidate.json`.

**Held-out failure fallback (presentation):**

| Situation | What to show | What to say |
| --- | --- | --- |
| Live held-out **below threshold** | Pre-staged `<loop-back-run-id>`: `outer_loop/loopback.json` + later outer-iteration artifacts | "Stage 11 writes `loopback.json`; later PPO passes resume the selected checkpoint. Quality is always measured strict success—there is no synthetic bump." |
| Live Job **still running** at minute 7 | Pre-staged held-out + decision; keep live logs in split screen | "Artifacts are deterministic in shape; this run is mid–Stage 10." |
| Live Job **failed** (OOM, pull, etc.) | Full pre-staged tree + `sim2real-report.json` | "Orchestrator fails loud — no silent reference fallback in production bucket mode. We inspect the last good staged run." |
| Local / unit path (optional footnote) | N/A | "Without a bucket, VLM and held-out use in-process reference payloads — that's for tests, not this cluster demo." |

Command to narrate loop-back:

```bash
# canonical proof uses OUTER_ITERATIONS=3 and keeps gold held out until the final pass
grep -E 'outer=|decision=' /tmp/sim2real-cluster/<live-run-id>.log
```

### 8:00–8:45 — Stages 12–13 (external seam and retrigger record)

1. `stage_12_external_validation/external_stub.json` — real-robot validation hook.
2. `stage_13_retrigger/retrigger.json` — restarts Stage 1 only when verified new real failure data or corrected scenario data exists; otherwise records why no retrigger occurred.

> "Stage 12 is the designed external real-world validation seam. Stage 13 is a
> data gate, not an automatic retrigger.
> Stage 2 BYO robot/scene paths are optional; stock assets are production-ready."

### 8:45–9:30 — Report + Rerun (the payoff)

```bash
# From synced run dir or S3 download
jq '{decision: .outer_loop.latest_decision, reward_trend: .inner_loop.reward_trend}' \
  reports/sim2real-report.json
rerun reports/sim2real.rrd
```

Walk Rerun entities (~30 s):

- `rollouts/.../camera` + `critique` text overlay
- `signal/reward` timeseries
- `heldout/scores`

> "One `.rrd` ties every stage together for debugging — same artifact tree uploaded with `--upload-artifacts` when `stage_14_rerun_viz` tier is WORKS. Check the tier in `sim2real-report.json` → `components` if the file is missing on S3."

```bash
# Verify tier before opening Rerun
jq '.components[] | select(.name=="stage_14_rerun_viz") | {tier, message}' \
  reports/sim2real-report.json
# Sync from S3 if needed:
# aws s3 cp s3://<bucket>/<prefix>/<run-id>/reports/sim2real.rrd reports/
```

### 9:30–10:00 — Close

> "Fourteen stages, every artifact addressable on S3, three BYO seams for assets / real-world eval / retrigger, and one ordinary compositional workflow submitted through the standard runtime. Questions?"

---

## Demo commands (copy-paste)

```bash
# Preflight (credentials)
npa workbench health preflight

# Submit live run
export KUBECONFIG=~/.npa/clusters/<cluster>/kubeconfig
INNER_ITERATIONS=3 OUTER_ITERATIONS=3 ROLLOUT_COUNT=64 STEPS_PER_ROLLOUT=32 \
  VALIDATION_ENV_COUNT=64 HELDOUT_ENV_COUNT=64 SUCCESS_THRESHOLD=0.50 \
  <private-operator-pack>/sim2real-rtxpro/submit-k8s-staged-job.sh

# Monitor
<private-operator-pack>/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<live-run-id>

# Canonical URIs from code (optional)
npa/.venv/bin/python -c "
from npa.workflows.sim2real_loop import build_config_from_env, artifact_uris
import json, os
os.environ['NPA_SIM2REAL_RUN_ID'] = '<pre-staged-run-id>'
os.environ['NPA_SIM2REAL_BUCKET'] = '<bucket>'
os.environ['NPA_SIM2REAL_PREFIX'] = '<prefix>'
print(json.dumps(artifact_uris(build_config_from_env()), indent=2))
"
```

---

## Anticipated Q&A (15 s each)

| Question | Answer |
| --- | --- |
| Why a standard workflow? | The runtime owns every loop, wave, retry, and durable S3 checkpoint without a private controller. |
| What if I bring my own VLM/trainer/eval? | `BYO_*_COMMAND` envs; fails loud if output schema wrong — no silent reference swap. |
| Why Isaac here? | The canonical task contract and gold evaluation use Isaac Lab on RTX PRO 6000/L40S. |
| Why threshold 0.50? | It is the canonical promotion gate; diagnostic 10/15/20 cm rates never lower the strict 5 cm stable-placement requirement. |
| Where is viz? | Stage 14 emits non-empty `reports/sim2real.rrd` and `.mcap`; it fails closed if required evidence cannot be encoded. |

---

## Rehearsal timing notes

- **Tight on time:** Skip Stages 3–6 deep dive; show trigger + env split + jump to Stage 8 critique JSON.
- **Live job finishes early:** End on live S3 tree; use pre-staged only for loop-back example.
- **Never stall on GPU:** Pre-staged S3 is the source of truth for artifact shapes; live logs are optional spice.
