# Sim2Real — Stock Franka Demo (Operator Guide)

**Audience:** RTX PRO demo operators on a **Mac laptop** (interface only). GPU work runs on
Nebius mk8s; artifacts land on S3; you monitor, sync, and open Rerun locally.

**Related:** [CUSTOMER-DEMO.md](./CUSTOMER-DEMO.md) · [sim2real-customer-assets.md](../../../docs/workbench/guides/sim2real-customer-assets.md) · [sim2real-workflow.md](../../../docs/workbench/guides/sim2real-workflow.md)

---

## What “stock Franka” means

When you **do not** set `ASSETS_URI`, `SCENE_SPEC_URI`, or `ROBOT_SPEC_URI`, Stage 2
materializes built-in sim assets:

| Piece | Value | Notes |
| --- | --- | --- |
| **Robot** | Franka Panda (`stock_franka`, preset `franka`) | 7 arm + 2 gripper joints, EE link `hand` |
| **Scene** | Stock tabletop + red lift cube (`stock_tabletop`) | Isaac builtin `lift_cube`, no mesh upload |
| **Sim task** | `Isaac-Lift-Cube-Franka-v0` | Default backend `isaac` (RT-core held-out) |
| **Cameras** | Overhead workspace + EE-mounted wrist | 640×480 each |
| **Trigger** | LeRobot dataset on S3 (Stage 1) | **Only customer upload required** for a real run |

You supply **teleop / demo video as LeRobot** (`lerobot/pusht` is the usual smoke id).
The pipeline does **not** poll S3 — you upload the batch, then explicitly trigger.

---

## Your local layout (Tim’s Mac)

| Item | Path |
| --- | --- |
| Operator pack | `~/npa-sim2real-demo` (private repo) |
| NPA platform | `~/npa-sim2real-demo/nebius-physical-ai` |
| Operator env | `~/.npa/sim2real-operator.env` |
| Config (no secrets) | `~/.npa/config.yaml` |
| Secrets | `~/.npa/credentials.yaml` (chmod 600 — never paste in chat) |
| Kubeconfig | `~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved` |
| K8s context | `npa-rtxpro-mk8s` |
| Nebius CLI | `~/.nebius/bin/nebius` (profile `npa-mk8s`) |

Shell exports on every terminal:

```bash
export PATH="${HOME}/.nebius/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"
export KUBECONFIG="${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved"
```

---

## Run modes

### A. Rehearsal — no cluster (~30 s)

Sync a **completed** golden run from S3 and open Rerun. No Franka sim on the laptop.

```bash
cd ~/npa-sim2real-demo
./run.sh
# Genesis golden: RUN_ID=rtxpro-staged-2x2-20260613t011356z ./run.sh
# Isaac golden:   RUN_ID=rtxpro-isaac-2x2-20260613t043658z ./run.sh
```

| Run ID | Backend | Notes |
| --- | --- | --- |
| `rtxpro-staged-2x2-20260613t011356z` | genesis | Full 2×2, ~13 min (default `./run.sh`) |
| `rtxpro-isaac-2x2-20260613t043658z` | isaac | Full 2×2, 10K envs, ~9 min |


### B. Full pipeline — stock Franka on cluster

**Submit** (canonical — uses direct K8s Job, not SkyPilot):

```bash
cd ~/npa-sim2real-demo/nebius-physical-ai
export TRIGGER_DATASET_URI=s3://YOUR-BUCKET/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/

./npa/.venv/bin/npa workbench workflow submit \
  npa/workflows/workbench/sim2real/runbook.yaml \
  --tool sim2real \
  --run-id "sim2real-staged-$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')" \
  --var "NPA_SIM2REAL_TRIGGER_DATASET_URI=${TRIGGER_DATASET_URI}" \
  --var "INNER_ITERATIONS=1" \
  --var "OUTER_ITERATIONS=2"
```

**Monitor live stages:**

```bash
./npa/.venv/bin/npa workbench workflow status <RUN_ID> --tool sim2real --watch
```

Private operator wrapper (preflight + submit + wait + sync + Rerun):

```bash
cd ~/npa-sim2real-demo
export TRIGGER_DATASET_URI=s3://YOUR-BUCKET/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/
./run.sh trigger
```

### C. Sync + Rerun after a completed run

```bash
cd ~/npa-sim2real-demo
RUN_ID=<your-run-id> ./run.sh sync
# or
RUN_ID=<your-run-id> SUBMIT=0 ./ops/private/sim2real-rtxpro/run-demo.sh
```

Local sync dir default: `/tmp/sim2real-demo/<RUN_ID>/`

---

## Pipeline stages (13 + report)

One K8s Job (`sim2real-<RUN_ID>`) runs the orchestrator; GPU sibling Jobs handle augment,
policy rollouts, VLM, envgen shards, and held-out Isaac eval.

| # | Stage | What happens (stock Franka) | Key artifact (under `sim2real-b/<RUN_ID>/`) |
| --- | --- | --- | --- |
| 1 | Trigger | Consume LeRobot prefix | `stage_01_trigger/trigger.json` |
| 2 | Assets | Stock Franka + lift cube | `stage_02_assets/consumed_robot_spec.json`, `consumed_scene_spec.json`, `assets_manifest.json` |
| 3 | Augment | Cosmos Transfer sibling Job | `augment/cosmos2-transfer-result.json`, `augment/frames/` |
| 4 | Envgen raw | Indexed GPU shards → raw env JSONL | `envs/raw/` |
| 5 | Env split | Train / held-out split | `envs/train/`, `envs/heldout/` |
| 6 | Tokens | Token manifest for trainer | `tokens/manifest.json` |
| 7 | Rollouts | Reference policy sibling Job | `actions/train/` |
| 8 | VLM eval | Cosmos3-reason per rollout | `vlm_eval/train/` |
| 9 | RL signal | Signal JSON + in-process trainer | `training_signal/train/` |
| 10 | Held-out | Isaac Lab Franka lift eval | `eval/heldout/report.json` |
| 11 | Threshold | Promote or loop back | `outer_loop/decision.json` |
| 12–13 | Finish | External validation stub + retrigger | `stage_12_*`, `stage_13_*` |
| — | State | Cross-stage memory | `state/workflow_state.json` |
| 14 | Rerun viz | Timeline recording | `reports/sim2real.rrd` |
| — | E2E report | Stage scorecard | `reports/sim2real-report.json` |

Default demo depth: `INNER_ITERATIONS=1`, `OUTER_ITERATIONS=2` (operator pack). Stages 7–9
repeat per inner iteration; stage 11 may loop outer iterations.

---

## How to view stage progress (local laptop)

You can inspect progress **while the cluster job runs** (S3 + kubectl) or **after sync**
(local files + Rerun). Replace `<RUN_ID>` with your run id.

### 1. Find the active job

```bash
grep -E 'run_id=|job=' /tmp/sim2real-demo/submit.log | tail -5

kubectl --context npa-rtxpro-mk8s get jobs -n default --sort-by=.metadata.creationTimestamp \
  | grep sim2real | tail -5
```

### 2. Monitor until complete

```bash
cd ~/npa-sim2real-demo/nebius-physical-ai
./npa/.venv/bin/npa workbench workflow status sim2real-staged-<RUN_ID> --tool sim2real --watch
```

One-shot (no watch):

```bash
./npa/.venv/bin/npa workbench workflow status <RUN_ID> --tool sim2real
```

Operator shortcut:

```bash
OPS=~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro
${OPS}/status-sim2real-run.sh <RUN_ID>
```

Legacy job monitor (K8s job only, no stage checklist):

```bash
${OPS}/monitor-k8s-job.sh sim2real-<RUN_ID>
```

Logs:

- `/tmp/sim2real-cluster/sim2real-<RUN_ID>-monitor.log`
- `/tmp/sim2real-cluster/sim2real-<RUN_ID>.log`

Optional tmux: `tmux attach -t sim2real-cluster-live`

### 3. Tail orchestrator logs (live stage lines)

```bash
POD=$(kubectl --context npa-rtxpro-mk8s get pods -n default \
  -l job-name=sim2real-<RUN_ID> -o jsonpath='{.items[0].metadata.name}')
kubectl --context npa-rtxpro-mk8s logs -n default "${POD}" -f --all-containers
```

### 4. S3 artifact tree (best live stage signal)

Uses credentials from `~/.npa/credentials.yaml` via AWS CLI:

```bash
BUCKET=YOUR-BUCKET   # or from ~/.npa/config.yaml storage.bucket
ENDPOINT=https://storage.eu-north1.nebius.cloud
RUN_ID=<RUN_ID>

# What stages have landed?
aws s3 ls "s3://${BUCKET}/sim2real-b/${RUN_ID}/" \
  --endpoint-url "${ENDPOINT}" --recursive | tail -30

# Stage checklist (run repeatedly while job is active)
for path in \
  stage_01_trigger/trigger.json \
  stage_02_assets/assets_manifest.json \
  augment/cosmos2-transfer-result.json \
  envs/raw/ \
  envs/train/ \
  actions/train/ \
  vlm_eval/train/ \
  eval/heldout/report.json \
  state/workflow_state.json \
  reports/sim2real-report.json; do
  aws s3 ls "s3://${BUCKET}/sim2real-b/${RUN_ID}/${path}" \
    --endpoint-url "${ENDPOINT}" >/dev/null 2>&1 \
    && echo "OK  ${path}" || echo "—   ${path}"
done
```

**Interpretation:**

- Stuck after `augment/` → check sibling Job `s2r-cosmos-*` or augment image pull
- Stuck after `stage_02` with no `envs/raw/` → envgen shards (`s2r-envgen-*`) — check `kubectl get pods`
- `actions/` then `vlm_eval/` appear per inner iteration
- `eval/heldout/report.json` → Isaac Franka held-out complete

### 5. GPU sibling Jobs (parallel work on cluster)

```bash
kubectl --context npa-rtxpro-mk8s get jobs,pods -n default | grep -E 's2r-|sim2real-<RUN_ID>'
```

Typical siblings for stock Franka:

| Job prefix | Stage |
| --- | --- |
| `s2r-cosmos-*` or augment image | 3 Augment |
| `s2r-envgen-raw-shard-*` | 4 Envgen |
| `s2r-policy-*` | 7 Rollouts |
| `s2r-vlm-*` | 8 VLM |
| `s2r-heldout-*` / Isaac image | 10 Held-out |

### 6. Workflow state + stage scorecard (after partial sync)

Sync just enough to read JSON locally:

```bash
OPS=~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro
${OPS}/prestage-offline-run.sh <RUN_ID> /tmp/sim2real-demo/<RUN_ID>
```

Then:

```bash
RUN=/tmp/sim2real-demo/<RUN_ID>

# Cross-stage memory
jq '{status, quality: .current_quality, outer: .next_outer_iteration, stages: [.stage_records[] | {name: .stage, tier: .tier}]}' \
  "$RUN/state/workflow_state.json"

# Franka / stock asset confirmation
jq '{robot: .status, preset: .robot_preset, name: .name}' \
  "$RUN/stage_02_assets/consumed_robot_spec.json"
jq '{scene: .status, task_objects: [.scene_spec.objects[].name]}' \
  "$RUN/stage_02_assets/consumed_scene_spec.json"

# Inner loop evidence (per outer iteration)
jq '{reward_trend, final_quality}' "$RUN/inner_loop/outer-01/evidence.json" 2>/dev/null

# Held-out Franka lift scores
jq '{success_rate, n_envs: (.per_env | length)}' "$RUN/eval/heldout/report.json" 2>/dev/null

# Threshold decision
jq '{decision, success_rate, threshold}' "$RUN/outer_loop/decision.json" 2>/dev/null

# E2E stage tiers (WORKS / WARN / SEAM)
jq '.components[] | {name, tier, summary}' "$RUN/reports/sim2real-report.json" 2>/dev/null
```

Confirm Rerun stage:

```bash
jq '.components[] | select(.name=="stage_14_rerun_viz") | {tier, summary}' \
  "$RUN/reports/sim2real-report.json"
```

Tier **WORKS** → open `reports/sim2real.rrd`. Tier **WARN** / **SEAM** → no `.rrd` (not an upload bug).

### 7. Rerun viewer (visual stage timeline)

```bash
cd ~/npa-sim2real-demo
RUN_ID=<RUN_ID> ./run.sh sync   # ensures .rrd is local

# or manually:
rerun /tmp/sim2real-demo/<RUN_ID>/reports/sim2real.rrd
```

Rerun shows rollout frames, VLM critiques, reward curves, and held-out scores across
inner/outer iterations — the fastest way to **see** stages 7–11 after sync.

---

## Quick copy-paste: monitor unknown run

```bash
export PATH="${HOME}/.nebius/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"
export KUBECONFIG="${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved"

kubectl --context npa-rtxpro-mk8s get jobs -n default --sort-by=.metadata.creationTimestamp \
  | grep sim2real | tail -5

JOB=sim2real-<RUN_ID>   # from submit.log
~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/monitor-k8s-job.sh "${JOB}"
```

---

## Stock Franka vs BYO (when you outgrow the demo)

| Variable | Stock demo | Customer production |
| --- | --- | --- |
| `TRIGGER_DATASET_URI` | Required (LeRobot) | Required (LeRobot) |
| `ASSETS_URI` / `SCENE_SPEC_URI` | **Unset** (stock cube) | Custom meshes or SceneSpec JSON |
| `ROBOT_SPEC_URI` / `ROBOT_PRESET` | **Unset** (Franka) | `ur5e`, `flexiv`, etc. + URDF |
| `NPA_SIM2REAL_SIM_BACKEND` | `isaac` (default) | Usually `isaac` |

See [sim2real-customer-assets.md](../../../docs/workbench/guides/sim2real-customer-assets.md)
for BYO robot and scene upload contracts.

---

## Troubleshooting (stock path)

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Preflight: `no LeRobot batch` | Empty or partial S3 prefix | Use stock trigger URI or finish upload |
| `trigger.json` shows `assets_uri: ""` | Expected for stock Franka | `consumed_robot_spec.json` → `stock_franka` |
| Stuck at augment | Sibling Job failed / S3 contract | `kubectl logs` on augment pod; `augment/cosmos2-transfer-result.json` on S3 |
| Stuck at envgen | Image pull or shard failure | `kubectl describe pod` for `s2r-envgen-*`; registry auth |
| No `.rrd` after success | Rerun disabled or SDK missing in image | `stage_14_rerun_viz` tier in report JSON |
| Trigger URI ignored | Stale operator pack | `git pull` private repo + `./setup.sh` (env precedence fix) |

---

## Expected runtime (stock demo, 2× RTX 6000 Pro)

Rough order of magnitude with default operator knobs (`INNER_ITERATIONS=1`,
`OUTER_ITERATIONS=2`, envgen sharded):

| Phase | Typical duration |
| --- | --- |
| Stages 1–3 (trigger, assets, augment) | 1–3 min |
| Stage 4 envgen (16 shards, parallelism 2) | 30–90 min |
| Stages 5–6 | 5–15 min |
| Inner loop × N (policy + VLM + trainer) | 15–45 min per inner |
| Held-out Isaac (8 envs) | 10–25 min |
| Finalize + upload | ~5 min |

**Total:** ~1.5–3 h for a full cluster run. Rehearsal (`./run.sh` sync-only) is ~30 s.
