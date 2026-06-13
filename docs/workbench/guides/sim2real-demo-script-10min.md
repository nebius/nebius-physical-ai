# Sim-to-Real Pipeline — 10-Minute Demo Script

**Audience:** Platform + robotics stakeholders  
**Duration:** 10 minutes (includes ~30 s buffer)  
**Branch under review:** `feat/sim2real-mandatory-stages` (stacked on mandatory-stages base PR)  
**Runtime:** Direct Kubernetes staged job on RTX PRO cluster (SkyPilot bypass)

Replace placeholders before rehearsal:

| Placeholder | Example (operator pack) |
| --- | --- |
| `<cluster>` | `npa-rtxpro-mk8s` |
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
   ./ops/private/sim2real-rtxpro/prestage-offline-run.sh <pre-staged-run-id>
   # -> /tmp/sim2real-prestage/<run-id>/
   rerun /tmp/sim2real-prestage/<pre-staged-run-id>/reports/sim2real.rrd
   ```

   S3 canonical path: `s3://<bucket>/<prefix>/<pre-staged-run-id>/reports/sim2real.rrd`
2. **Start a live job early** — 15–30 min before showtime:

   ```bash
   export KUBECONFIG=~/.npa/clusters/<cluster>/kubeconfig
   INNER_ITERATIONS=1 OUTER_ITERATIONS=2 \
     ./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh
   # note run_id= from output; monitor with monitor-k8s-job.sh
   ```

3. **Optional: pre-stage a loop-back run** — Second run where `eval/heldout/report.json` has `success_rate` below threshold and `outer_loop/loopback.json` exists (for held-out failure narrative).
4. **Preflight** — `npa workbench health sim2real --checks all` (PASS on S3, tokens, GPU count).
5. **Open tabs:** S3 browser (pre-staged root), terminal (`kubectl logs -f`), Rerun viewer with pre-staged `.rrd`.

---

## Stage → artifact map (13 stages)

Use this table as the backbone of the demo. Every row is a file or prefix you can open live or from pre-staged S3.

| # | Stage | What happens | NPA artifact (under run root) | Live vs pre-staged |
| --- | --- | --- | --- | --- |
| 1 | **Trigger** | LeRobot dataset path consumed; run ID resolved | `stage_01_trigger/trigger.json` | Pre-staged: static. Live: written in `preamble` first seconds |
| 2 | **Sim assets (BYO seam)** | Customer mesh/SceneSpec **or** documented stub | `stage_02_assets/external_stub.json` or consumed spec | **SEAM fallback:** no `ASSETS_URI` → stub status `documented_external_stub`, component `SEAM`. With Stage 2 assets wired: consumed spec + provenance |
| 3 | **Augment** | Cosmos transfer manifest (reference image) | `augment/manifest.json` | Pre-staged OK; live identical shape |
| 4 | **Env generation (raw)** | Synthetic env batch | `envs/raw/manifest.json` | Show env count in manifest |
| 5 | **Train / held-out split** | 80/20 split | `envs/train/manifest.json`, `envs/heldout/manifest.json` | Point at held-out count (demo: 4) |
| 6 | **Token manifest** | Stage-A-compatible reference tokens | `tokens/manifest.json` | Quick JSON peek |
| 7 | **Action rollouts** | Reference policy rollouts (in orchestrator) | `actions/train/outer-01/iter-01/rollout-*/` | **Live highlight:** sibling path not used; files appear during outer loop |
| 8 | **VLM critique** | Cosmos-Reason sibling Job on GPU image | `vlm_eval/train/outer-01/iter-01/<rollout-id>.json` | **Live:** `kubectl get jobs -l run-id=<live-run-id>`; **pre-staged:** open one critique JSON |
| 9 | **RL signal + trainer** | Critique → reward signal → policy update | `training_signal/train/...`, `inner_loop/outer-01/evidence.json` | Show `reward_trend` in evidence |
| 10 | **Held-out eval** | Genesis or Isaac Lab sibling Job | `eval/heldout/report.json` | **Live:** longest stage; default Isaac task `Isaac-Lift-Cube-Franka-v0` when `sim_backend=isaac` |
| 11 | **Threshold gate** | Compare `success_rate` to threshold (demo: 0.45) | `outer_loop/decision.json`; promote → `checkpoints/candidate/candidate.json`; fail → `outer_loop/loopback.json` | **Fallback story** (below) |
| 12 | **Real-world validation (BYO seam)** | Documented external stub | `stage_12_external_validation/external_stub.json` | Always SEAM; customer hook point |
| 13 | **Retrigger** | Next dataset batch → back to Stage 1 | `stage_13_retrigger/retrigger.json` | Loop-of-loops metadata |

**Cross-cutting artifacts** (show at end):

- `state/workflow_state.json` — bash loop reads `current_quality`, `final_decision`
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

> "This is a **13-stage, inspectable sim-to-real loop**: real robot data triggers the run, simulation generates environments, a VLM critiques rollouts, we convert critique into RL signal, train, evaluate on held-out sim, and either promote a checkpoint or loop back. Every stage writes JSON to object storage — no black box."

Show one slide or browser tab: pipeline diagram from [sim2real-architecture.md](./sim2real-architecture.md) (preamble → outer loop → finalize).

### 0:45–1:30 — Three entry points, one Python module

> "SkyPilot runbook, direct Kubernetes submit, and staged CLI subcommands all call the same code: `preamble` → bash outer loop → `finalize`."

```bash
# What runs on cluster (abbreviated)
python3 -m npa.workflows.sim2real_loop preamble ...
python3 -m npa.workflows.sim2real_loop outer-iteration ... --outer-iteration 1
python3 -m npa.workflows.sim2real_loop finalize ... --upload-artifacts
```

Mention: RTX PRO cluster uses `submit-k8s-staged-job.sh` because SkyPilot kube context mismatches — same stages, direct Job.

### 1:30–3:00 — Stages 1–6 (preamble) — **pre-staged S3**

Open `s3://<bucket>/<prefix>/<pre-staged-run-id>/`.

1. **Stage 1** — `stage_01_trigger/trigger.json`: trigger dataset URI, run ID.
2. **Stage 2** — Either consumed assets **or** `external_stub.json`:
   - *Talking point:* "Stage 2 is a **BYO seam**. Reference runs continue with a documented stub (`SEAM`); production teams drop URDF/mesh + SceneSpec here — no silent fallback to stock geometry."
3. **Stages 3–6** — Walk `augment/`, `envs/raw`, `envs/train`, `envs/heldout`, `tokens/` in ~60 s.

> "Preamble finishes in one CLI call; state lands in `workflow_state.json` before any GPU-heavy work."

### 3:00–5:30 — Stages 7–9 (inner loop) — **mix live + pre-staged**

**Preferred live moment:** `kubectl logs -f job/sim2real-<live-run-id>` while outer-iteration runs.

| Show | Path / command |
| --- | --- |
| Rollout frames | `actions/train/outer-01/iter-01/rollout-0000/` |
| VLM Job spawned | `kubectl get jobs -l app=npa-sim2real,run-id=<live-run-id>` |
| Critique schema | `vlm_eval/.../rollout-0000.json` — `npa.sim2real.vlm_eval.v1` |
| Trainer evidence | `inner_loop/outer-01/evidence.json` — policy delta, `reward_trend` |

> "Inner loop is VLM → signal → trainer, repeated `INNER_ITERATIONS` times. Rollouts stay in-process; VLM and held-out eval spawn **sibling GPU Jobs** when a bucket is configured."

If live job not ready: stay on pre-staged paths — same filenames.

### 5:30–7:00 — Stage 10 (held-out) — **live if possible**

Open `eval/heldout/report.json`:

```json
"schema": "npa.sim2real.heldout_eval.v1",
"success_rate": 0.625,
"per_env": [ ... ]
```

> "Held-out runs on a **pluggable sim backend**. Isaac Lab headless is the platform default (`Isaac-Lift-Cube-Franka-v0`); Genesis remains supported. The sibling Job uses `heldout_backend_image()` — Isaac image for RT-core GPUs, eval image for Genesis."

**Live:** watch held-out Job complete; **fallback:** pre-staged report (see below).

### 7:00–8:00 — Stage 11 (threshold) — **promote vs loop-back**

Open `outer_loop/decision.json`:

```json
"success_rate": 0.625,
"threshold": 0.45,
"decision": "promote_checkpoint"
```

If promoting, flash `checkpoints/candidate/candidate.json`.

**Held-out failure fallback (presentation):**

| Situation | What to show | What to say |
| --- | --- | --- |
| Live held-out **below threshold** | Pre-staged `<loop-back-run-id>`: `outer_loop/loopback.json` + second outer iteration artifacts | "Stage 11 writes `loopback.json` — pipeline returns to Stage 7. Bash runs up to `OUTER_ITERATIONS`; quality bumps +0.12 each pass so the loop can converge." |
| Live Job **still running** at minute 7 | Pre-staged held-out + decision; keep live logs in split screen | "Artifacts are deterministic in shape; this run is mid–Stage 10." |
| Live Job **failed** (OOM, pull, etc.) | Full pre-staged tree + `sim2real-report.json` | "Orchestrator fails loud — no silent reference fallback in production bucket mode. We inspect the last good staged run." |
| Local / unit path (optional footnote) | N/A | "Without a bucket, VLM and held-out use in-process reference payloads — that's for tests, not this cluster demo." |

Command to narrate loop-back:

```bash
# submit script uses OUTER_ITERATIONS=2 so a failed first pass can retry
grep -E 'outer=|decision=' /tmp/sim2real-cluster/<live-run-id>.log
```

### 8:00–8:45 — Stages 12–13 (finalize seams)

1. `stage_12_external_validation/external_stub.json` — real-robot validation hook.
2. `stage_13_retrigger/retrigger.json` — next LeRobot drop in trigger path restarts Stage 1.

> "Stages 2, 12, and 13 are **documented seams** — the reference pipeline proves the contract; customers swap their validation and asset pipelines without forking the orchestrator."

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

> "Thirteen stages, every artifact addressable on S3, three BYO seams for assets / real-world eval / retrigger, staged CLI for CI and human gates, one module behind runbook and direct K8s submit. Questions?"

---

## Demo commands (copy-paste)

```bash
# Preflight
npa workbench health sim2real \
  --s3-bucket <bucket> \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --checks all

# Submit live run
export KUBECONFIG=~/.npa/clusters/<cluster>/kubeconfig
INNER_ITERATIONS=1 OUTER_ITERATIONS=2 ROLLOUT_COUNT=2 HELDOUT_ENV_COUNT=4 \
  SUCCESS_THRESHOLD=0.45 \
  ./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh

# Monitor
./ops/private/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<live-run-id>

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
| Why staged subcommands? | Bash (or human) gates between preamble, each outer iteration, and finalize — same state file the runbook uses. |
| What if I bring my own VLM/trainer/eval? | `BYO_*_COMMAND` envs; fails loud if output schema wrong — no silent reference swap. |
| Genesis vs Isaac? | `--sim-backend genesis\|isaac`; held-out sibling image switches via `heldout_backend_image()`. |
| Why threshold 0.45 in cluster submit? | Demo-scale held-out with small env count; raise for production. |
| Where is viz? | Stage 14 (`stage_14_rerun_viz`) in `finalize` — emits `reports/sim2real.rrd` when Rerun is available; tier **WORKS** / **WARN** / **SEAM** in report JSON. |

---

## Rehearsal timing notes

- **Tight on time:** Skip Stages 3–6 deep dive; show trigger + env split + jump to Stage 8 critique JSON.
- **Live job finishes early:** End on live S3 tree; use pre-staged only for loop-back example.
- **Never stall on GPU:** Pre-staged S3 is the source of truth for artifact shapes; live logs are optional spice.
