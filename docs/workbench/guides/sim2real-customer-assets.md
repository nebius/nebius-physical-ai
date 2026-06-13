# Sim2Real — Customer asset handoff (Sereact)

Per-category plan for initial stock setup; custom UR/Flexiv OBJ uploads later.

| Category | Plan | Formats |
| --- | --- | --- |
| **1. Robot / embodiment** | **Stock** (Franka Panda built-in) | Later: custom URDF/MJCF + meshes, or USD |
| **2. Manipulated objects** | **Stock** for now | Later: OBJ/STL/GLB/PLY/USD per object + dimensions/mass |
| **3. Scene / environment** | **Stock** (table + simple bins) | Later: fixture meshes or SceneSpec JSON / USD layout |
| **4. Cameras / sensors** | **Stock** (workspace + wrist defaults) | Later: custom poses/intrinsics in SceneSpec |

Wire custom uploads via `ASSETS_URI`, `SCENE_SPEC_URI`, `ROBOT_PRESET` / `ROBOT_SPEC_URI` on workflow submit.

---

## Stage 2 implementation (sim assets)

Stage 2 is no longer a documented stub. `run_assets_stage()` materializes:

- `stage_02_assets/consumed_scene_spec.json` — stock tabletop (Genesis or Isaac) or BYO mesh/SceneSpec
- `stage_02_assets/consumed_robot_spec.json` — Franka preset, UR/Flexiv preset metadata, or BYO RobotSpec JSON

Those URIs flow into envgen (`build_envgen_scene_spec`) and each env record carries an `embodiment` block (`robot_preset`, `robot_spec_uri`, `sim_backend`, cameras).

**Customer trigger vs train envs:** supply `NPA_SIM2REAL_TRIGGER_DATASET_URI` (LeRobot data). `NPA_SIM2REAL_TRAIN_ENVS_URI` is written by the workflow after envgen (~8K train shard) — not the robot asset.

---

## Production handoff scorecard (13-step Sereact story)

Tier key: **WORKS** = executable on Nebius today; **PARTIAL** = orchestrated but not full vendor fidelity; **SEAM** = documented plug point, not live integration.

| Step | Sereact stage | NPA fit | Notes |
| --- | --- | --- | --- |
| 1 | LeRobot trigger | **WORKS** | S3 URI consumed at submit |
| 2 | LanceDB curation | **SEAM** | Trigger path only; no LanceDB stage |
| 3 | Cosmos augment | **WORKS** | Cosmos Transfer 2.5 sibling K8s job (PR #110) |
| 4 | Lightwheel / sim assets | **PARTIAL** | Stock SceneSpec + Franka; BYO mesh/SceneSpec/RobotSpec; not Lightwheel catalog |
| 5 | 10K envgen | **WORKS** | `NPA_ENV_COUNT=10000` via `sim2real_envgen` |
| 6 | 80/20 split | **WORKS** | `NPA_TRAIN_FRACTION=0.8`; state carries `train_envs_uri` / `heldout_envs_uri` |
| 7 | Cortex / policy actions | **WORKS** | Swappable `POLICY_IMAGE` K8s job; `BYO_POLICY_COMMAND` seam |
| 8–9 | VLM + RL trainer | **WORKS** | Cosmos3 Reason + LeRobot VLM-signal trainer on cluster |
| 10 | Held-out eval | **PARTIAL** | Isaac Lab or Genesis rollouts; not Lightwheel eval harness |
| 11 | Threshold gate | **WORKS** | Promote vs loop-back |
| 12 | Real-world validation | **SEAM** | `stage_12_external_validation/external_stub.json` |
| 13 | Retrigger | **SEAM** | Record only; no auto S3 watcher |

**Overall:** ~**80%** as an NPA orchestration framework customers can run on RTX PRO; ~**20%** as the full Sereact vendor stack (LanceDB, Lightwheel, Cortex-native, real-world loop).

**PR stack:** [#109](https://github.com/nebius/nebius-physical-ai/pull/109) staged runbook + K8s ops; [#110](https://github.com/nebius/nebius-physical-ai/pull/110) mandatory stages + Stage 2 asset materialization (this branch).
