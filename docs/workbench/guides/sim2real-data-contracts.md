# Sim-to-Real — Data types & artifact contracts

**Canonical reference** for formats, schemas, and S3 layout in the 14-stage sim-to-real
loop. Other guides link here instead of duplicating tables.

---

## Doc map (read in this order)

| Doc | Use when you need… |
| --- | --- |
| **[sim2real-workflow.md](./sim2real-workflow.md)** | Run the loop: quickstart, CLI, local smoke |
| **This file** | What each artifact *is* (LeRobot vs NPA JSON vs media) |
| **[sim2real-customer-assets.md](./sim2real-customer-assets.md)** | What the customer uploads (robot, scene, trigger) |
| **[sim2real-architecture.md](./sim2real-architecture.md)** | Standard-runtime control flow, parallel waves, loops, and resume |
| **[sim2real-demo-script-10min.md](./sim2real-demo-script-10min.md)** | Presentation walkthrough |

**Code:** `npa/src/npa/workflows/sim2real/workflow_stage.py`,
`workflow_io.py`, `sim2real_envgen.py`, and `sim2real_assets.py`.

---

## Not everything is LeRobot

The loop uses **four format families**. Stage 1 accepts a task-aligned seed
dataset; it is LeRobot-native only when that dataset's explicit task contract
matches the selected task.

| Family | Used for | Examples |
| --- | --- | --- |
| **Task seed dataset** | Task-aligned demonstrations or simulator captures that **trigger** the run | Manifest + action/camera trajectory data under `NPA_SIM2REAL_TRIGGER_DATASET_URI` |
| **NPA JSON schemas** | Workflow records, env catalogs, eval reports | `npa.sim2real.*.v1` (see table below) |
| **Simulation media** | Rollout frames, augment output | `.ppm`, `.png`, `.mp4` under `actions/`, `augment/frames/` |
| **CAD / scene files** | BYO objects and robots (not LeRobot) | OBJ/STL/GLB/PLY/USD meshes; URDF/MJCF/USD for arms |

**Common mistake:** `TRAIN_ENVS_URI` is **not** LeRobot data. It is a workflow-written
**synthetic env shard** (JSONL of `npa.sim2real.raw_env.v1` records) used by Stage 7
policy rollouts.

---

## Customer input vs workflow output

| URI / artifact | Provider | Format | Stage |
| --- | --- | --- | --- |
| `NPA_SIM2REAL_TRIGGER_DATASET_URI` | Customer/operator | Task-aligned Isaac seed dataset plus `task-dataset-manifest.json`; PushT is incompatible with the default Franka lift task | 1 |
| `ASSETS_URI`, `SCENE_SPEC_URI` | Customer (optional) | Meshes + optional `npa.sim2real.scene_spec.v1` | 2 |
| `ROBOT_SPEC_URI`, `ROBOT_PRESET` | Customer (optional) | `npa.sim2real.robot_spec.v1` or preset name | 2 |
| `train_envs_uri` / `validation_envs_uri` / `gold_heldout_envs_uri` | **Workflow** | Curated, disjoint, stratified NPA scenario JSONL with task/config digests | 4–6 |
| `actions/train/…` | Workflow / policy job | Rollout dirs + `npa.sim2real.action_rollout.v1` | 7 |
| `vlm_eval/…` | Workflow / hosted Cosmos3 evaluator | `npa.sim2real.vlm_eval.v3` | 8 |
| `training_signal/…` | Workflow | `npa.sim2real.rl_signal.v1` | 9 |
| `eval/validation/outer-XX/iter-YY/report.json` | Workflow / eval job | Validation-only checkpoint comparison | 9 |
| `eval/gold-heldout/outer-XX/report.json` | Workflow / eval job | Final untouched gold evaluation, `npa.sim2real.heldout_eval.v1` | 10 |
| `reports/sim2real-report.json` | Workflow | `npa.sim2real.e2e_report.v1` | finalize |

---

## Stock smoke path vs customer embodiment

| Path | Robot | When |
| --- | --- | --- |
| **Stock smoke** | NPA default Franka (`ROBOT_PRESET` empty → `franka`) | Platform validation only; **not** the customer's arm |
| **Customer production** | `ROBOT_PRESET=ur5e` / `flexiv` + **`ROBOT_SPEC_URI`** (articulated URDF) | Real embodiment; held-out eval fails loud if URDF missing |

Franka appears in docs as a **zero-upload reference**, not because customers standardize on Franka.

---

## JSON schema catalog (by stage)

Every JSON artifact should include a top-level `"schema"` string. Constants live in
`sim2real_loop.py` unless noted.

### Preamble (Stages 1–6)

| Schema | Typical path | Stage | Notes |
| --- | --- | --- | --- |
| `npa.sim2real.trigger.v1` | `stage_01_trigger/trigger.json` | 1 | Points at the task-aligned trigger URI and its verified seed provenance |
| `npa.sim2real.task_contract.v1` | `stage_02_assets/task-contract.json` | 2 | Normalized task/data/embodiment/physics/camera/success contract and digest |
| `npa.sim2real.consumed_scene_spec.v1` | `stage_02_assets/consumed_scene_spec.json` | 2 | Stock or BYO scene after materialization |
| `npa.sim2real.consumed_robot_spec.v1` | `stage_02_assets/consumed_robot_spec.json` | 2 | Stock Franka or BYO / preset metadata |
| `npa.sim2real.stock_scene_spec.v1` | (embedded in consumed scene) | 2 | Stock-only wrapper |
| `npa.sim2real.stock_robot_spec.v1` | (embedded in consumed robot) | 2 | Stock-only wrapper |
| `npa.sim2real.scene_spec.v1` | envgen / BYO input | 2–6 | Scene composition for envgen |
| `npa.sim2real.robot_spec.v1` | BYO `ROBOT_SPEC_URI` | 2 | Articulated robot definition |
| Cosmos transfer manifest | `augment/manifest.json` | 3 | From `cosmos_split`; includes `augmented_frames_uri` |
| `npa.sim2real.augmented_frame.v1` | `augment/frames/frame-*.json` | 3 | Per-frame augment record |
| `npa.sim2real.augmented_frames.v1` | `augment/frames/index.json` | 3 | Frame index |
| `npa.sim2real.raw_env.v1` | `envs/raw/*.jsonl` (sharded) | 4 | One record per synthetic env |
| `npa.sim2real.raw_env_shard_summary.v1` | shard summary JSON | 4 | Envgen shard metadata |
| `npa.sim2real.env_manifest.v1` | `envs/raw|train|heldout/manifest.json` | 4–6 | Env lists for split dirs |
| `npa.sim2real.env_split.v1` | split sidecars | 6 | Train vs held-out partition |
| `npa.sim2real.curation_manifest.v1` | `envs/manifest/curation-manifest.json` | 4–5 | Rejections, deduplication, coverage, and Stage 3 lineage consumption |
| `npa.sim2real.split_manifest.v2` | `envs/manifest/split-manifest.json` | 5 | Uploaded disjoint split metadata and leakage evidence |
| `npa.sim2real.tokens.v2` | `tokens/manifest.json` | 6 | Honest lineage/reporting-only consumer declaration for state PPO |
| `npa.sim2real.workflow_state.v1` | `state/workflow_state.json` | all | `train_envs_uri`, `heldout_envs_uri`, quality, decisions |

**10K env note:** `NPA_ENV_COUNT=10000` creates **10K JSON env records** (~8K train /
~2K held-out). They are **not** 10K Isaac Sim instances at envgen time.

### Inner loop (Stages 7–9)

| Schema | Typical path | Stage | Notes |
| --- | --- | --- | --- |
| `npa.sim2real.action_rollout.v1` | `actions/…/rollout-*/manifest.json` | 7 | Steps, actions, camera frame names |
| `npa.sim2real.reference_actions.v1` | policy job output | 7 | Reference policy contract |
| `npa.sim2real.actions_summary.v1` | policy job summary | 7 | |
| `npa.sim2real.policy_image_contract.v1` | policy job metadata | 7 | |
| `npa.sim2real.vlm_eval.v3` | `vlm_eval/…/*.json` | 8 | Per-rollout Cosmos3 critique with provider/backend, request IDs, tokens, latency, retries, and authoritative response cost or explicit null; v1/v2 remain read-compatible only in archived/legacy readers |
| `npa.sim2real.rl_signal.v1` | `training_signal/…/*.json` | 9 | Converted RL training signal |
| `npa.sim2real.inner_loop_evidence.v1` | `inner_loop/outer-XX/evidence.json` | 9 | Reward trend, trainer deltas |

Rollout **frames** (not JSON): `camera-NNN.ppm` (or paths listed in manifest).

### Outer loop & finalize (Stages 10–14)

| Schema | Typical path | Stage | Notes |
| --- | --- | --- | --- |
| `npa.sim2real.heldout_eval.v1` | `eval/gold-heldout/outer-XX/report.json` | 10 | Strict 5 cm stable placement, distance diagnostics, decomposed/per-stratum metrics, checkpoint SHA, and exact applied config digests |
| `npa.sim2real.threshold_decision.v1` | `outer_loop/decision.json` | 11 | `promote_checkpoint` vs `loop_back_to_inner_loop` |
| `npa.sim2real.candidate_checkpoint.v1` | `checkpoints/candidate/` | 11 | Exact best real checkpoint is packaged even below threshold; only passing gates set `deployable_policy=true` |
| `npa.sim2real.loopback.v1` | `outer_loop/loopback.json` | 11 | On loop-back |
| `npa.sim2real.external_stub.v1` | `stage_12_external_validation/external_stub.json` | 12 | BYO seam |
| `npa.sim2real.retrigger.v1` | `stage_13_retrigger/retrigger.json` | 13 | Next trigger metadata |
| `npa.sim2real.e2e_report.v1` | `reports/sim2real-report.json` | finalize | Full run summary + component tiers |

**Binary viz:** `reports/sim2real.rrd` (Rerun; not JSON).

---

## BYO component I/O (shell hooks)

When `BYO_*_COMMAND` is set, commands read/write paths from env vars and must emit
schemas above on stdout files:

| Hook | Reads | Writes |
| --- | --- | --- |
| `BYO_VLM_COMMAND` | Rollout dir + manifest | `npa.sim2real.vlm_eval.v2` |
| `BYO_SIGNAL_CONVERTER` | VLM eval JSON | `npa.sim2real.rl_signal.v1` |
| `BYO_TRAINER_COMMAND` | Signal batch JSON | Trainer update JSON (see policy_container) |
| `BYO_EVAL_COMMAND` | Held-out env list | `npa.sim2real.heldout_eval.v1` |
| `BYO_POLICY_COMMAND` | `NPA_SIM2REAL_TRAIN_ENVS_URI` | Rollout dirs or summary JSON |

---

## S3 layout

```text
# INPUT (customer)
s3://<bucket>/sim2real-triggers/<run-id>/<task>/           # task-aligned seed dataset + manifest
# explicit public preset: <task> = public-franka-lift; staged at runtime, never vendored

# OPTIONAL BYO (customer)
s3://<bucket>/sim2real-assets/<task>/                       # meshes, scene-spec.json, robot-spec.json

# OUTPUT (per run)
s3://<bucket>/<prefix>/<run-id>/
  state/workflow_state.json
  stage_01_trigger/trigger.json
  stage_02_assets/consumed_scene_spec.json
  stage_02_assets/consumed_robot_spec.json
  augment/manifest.json
  augment/frames/
  envs/raw/                    # JSONL shards when NPA_ENV_COUNT>0
  envs/train/                  # ~80% train shard
  envs/validation/             # checkpoint selection only
  envs/gold-heldout/           # untouched until final Stage 10
  envs/heldout/                # compatibility alias for downstream consumers
  envs/manifest/{curation-manifest.json,split-manifest.json}
  tokens/manifest.json
  actions/train/outer-XX/iter-YY/rollout-*/
  vlm_eval/train/outer-XX/iter-YY/
  training_signal/train/outer-XX/iter-YY/
  inner_loop/outer-XX/evidence.json
  eval/validation/outer-XX/iter-YY/report.json
  eval/gold-heldout/outer-XX/report.json
  outer_loop/decision.json
  stage_12_external_validation/external_stub.json
  stage_13_retrigger/retrigger.json
  reports/sim2real-report.json
  reports/sim2real.rrd          # when Rerun tier WORKS
  component-io/<component>/     # sibling K8s job scratch
```

Prefix default: `sim2real-b`. Canonical URI helpers: `artifact_uris()` in
`sim2real_loop.py`.

---

## Sim backend field (Isaac vs Genesis)

Env records and held-out eval carry `sim_backend`:

| Value | Held-out execution | RT-core GPUs |
| --- | --- | --- |
| `isaac` (default) | Sibling `npa-isaac-lab` job, `Isaac-Lift-Cube-Franka-v0` (stock) | **Yes** — Isaac Sim headless |
| `genesis` (legacy) | Sibling `npa-loop-eval` / Genesis env | CUDA sim, not Isaac RT-core path |

Genesis remains supported; Isaac is the primary backend for held-out eval on RTX PRO
class nodes.

---

## Validate schemas in tests

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_sim2real_loop.py -q
```

Inspect a live artifact:

```bash
jq '.schema, .evaluation_split, .strict_success, .decomposed_metrics, .applied_scenario_proof' eval/gold-heldout/outer-03/report.json
jq '.train_envs_uri, .validation_envs_uri, .gold_heldout_envs_uri' state/workflow_state.json
```
