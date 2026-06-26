# Sim2Real Custom-Asset Test Results

_Run 2026-06-26 on `npa-rtxpro-mk8s` (RTX PRO 6000 Blackwell), Isaac image
`npa-isaac-lab:2.3.2.post1`, against the PR #147 BYO-robot seam
(`feat/sim2real-byo-robot-task`). Every cluster verdict below is from a real
GPU sibling Job — submit → wait → read pod logs + S3, not a dry-run._

**Question:** does the sim2real workflow handle CUSTOM sim assets — not the
default MultiColorCube manipuland, not the stock Franka arm? **Short answer:** a
custom **object** USD works end-to-end today; a custom **robot** does **not**
(it has no env plumbing to pass a robot USD, and even when one is forced in, the
Franka Lift task's sensors/actions/rewards are hardcoded to Franka link names and
break); a custom **scene** is not loadable at all.

## Results table

| Asset type | What was tested | Verdict | Evidence | What a customer would need |
|---|---|---|---|---|
| **Custom OBJECT USD** (YCB `004_sugar_box.usd`, a non-default rigid-ready asset) | Trained a short policy (`iters=30`, `num_envs=256`) with `NPA_BYO_ISAAC_OBJECT_USD=<sugar_box>`, then evaluated the trained checkpoint rolling the same custom object | **WORKS** (end-to-end mechanism) | Train job `s2r-byo-isaac-train-caobj-20260626t160513`: `train.py … env.scene.object.spawn.usd_path=<sugar_box>`; **no** "Failed to find a rigid body"; `TRAIN_RC=0`; `model_29.pt` → `s3://lerobot-ccc9d3c7/sim2real-b/caobj-20260626t160513/byo-trainer/…/model_latest.pt` (+`checkpoints/`). Eval job `s2r-byo-isaac-eval-caobj-20260626t160513`: `EVAL_OBJECT_USD_APPLIED=<sugar_box>`, `rollout_ok`, 16 eps, **mean object_goal_distance = 0.3787 m, min = 0.2171 m**, success@0.15 = 0.00 (poor because 30 iters is a probe, **not** an asset failure) | A rigid-ready USD: collision geometry + a `RigidBodyAPI` + mass (the YCB `Axis_Aligned_Physics/*.usd` set qualifies). Just set `NPA_BYO_ISAAC_OBJECT_USD` (+ optional `NPA_BYO_ISAAC_OBJECT_SCALE`). For a *good* policy, train far more than 30 iters. |
| **Custom non-Franka ROBOT** (UR10, `Robots/UniversalRobots/ur10/ur10.usd`) via PR #147 seam | (a) Off-cluster: drive the documented trainer env path (`NPA_BYO_ROBOT_TASK=1` + `NPA_SIM2REAL_ROBOT_PRESET=ur10e`). (b) On-cluster: force the faithful UR10 `robot_spec` (exactly what the seam would emit if it could stage a UR USD) into the in-container `register()` → articulation swap → USD-live assert → rsl_rl | **FAILS** (reveals a real gap) | (a) **Plumbing gap (deterministic):** `_resolve_byo_robot_spec` reads only preset/source; UR/Flexiv presets carry **no** USD `robot_uri` → `usd_dest=''` → trainer prints "no stageable USD; articulation will NOT be swapped" → `robot_articulation_overrides={}` → `register()` no-op → **stock Franka trains**. (b) **In-container, attempt 1** (`…train-carobot-20260626t160707`, job `failed=1`): `register()` OK → `NPA-Lift-Cube-ur10-v0` registered, **UR10 USD loaded** (no USD/rigid-body error), scene built (256 envs), sim started → `ValueError: ImplicitActuatorCfg has set both 'effort_limit_sim'(87.0 Franka) and 'effort_limit'(330.0 UR10)`. **Attempt 2** (effort_limit_sim patch applied; `…train-carobot2-20260626t161111`, job `failed=1`): past actuators → `ValueError: Failed to create frame transformer for frame 'None' with path '/World/envs/env_.*/Robot/panda_link0'. No matching prims were found.` | (1) Real env plumbing to pass a robot USD (URDF→USD conversion, or a `robot_spec_uri`/preset that actually carries a `usd_path`). (2) `register()` must also retarget the **`ee_frame` FrameTransformer** prim paths + target frames, the **action term** (arm joint names + gripper), and the **reward** ee/gripper references to the customer robot — today it swaps only the articulation USD, actuators, and joint init, so `ee_link='tool0'` is recorded but never applied. (3) Use `effort_limit_sim` for implicit actuators. (4) Handle the action-space dim + gripper mismatch (Franka 7 arm + 2 gripper vs UR10 6 + none). |
| **Custom SCENE** (`simready://…` or a table/background USD) | Inspection only (no run) | **FAILS** (not loadable today) | `DEFAULT_SCENE_CATALOG` in `sim2real_envgen.py` is `simready://warehouse/tabletop_v1`, `…/bin-picking_v1`, `…/conveyor_cell_v1` — placeholder scheme strings, **never** resolved/downloaded to a USD (no resolver: `grep simready` finds no http/s3/.usd/fetch). Trainer/eval expose **no** scene/table override — only `env.scene.object.spawn.usd_path` (the manipuland). | A real scene-USD intake + resolver, plus a task that composes the customer scene (table/background) instead of the Lift task's built-in ground/table. |

## How far is a real custom robot? (gap (a), measured)

The PR #147 seam (the change that "begins to close" gap (a)) gets a non-Franka
arm **further than expected** but not to a trainable task. Concretely, in
sequence:

1. **No env path to supply the robot USD.** The trainer resolves a robot only
   from a preset/source, and the UR/Flexiv presets have no `usd_path`. Through the
   documented interface, "BYO robot" still trains a Franka. *(off-cluster,
   deterministic)*
2. **Actuator-cfg conflict.** Forcing a UR10 USD in: `register()` registers the
   variant and the **UR10 USD loads and builds a 256-env scene** — then Isaac 2.x
   rejects `effort_limit != effort_limit_sim` on the Franka preset's implicit
   actuators. *(one-line fix: set `effort_limit_sim`)*
3. **Franka-hardcoded task scaffolding.** Past the actuator fix, the Lift task's
   `ee_frame` FrameTransformer (and, beyond it, the action term and the
   gripper-based lift reward) reference Franka prim names (`panda_link0`,
   `panda_hand`) that don't exist on a UR10 → `No matching prims were found`.
   `register()` swaps the body but not the task's Franka-specific sensors,
   actions, or rewards.

So the seam proves the **articulation can be swapped into a registered variant
and the USD loads in-sim**, but a non-Franka robot **cannot reach RL training**
on the Lift task: the task scaffolding is Franka-specific. This confirms gap (a)
in `CAPABILITY.md` — for training, BYO-robot is today a mechanism/provenance
seam, not a working custom-robot pipeline.

## Reproduction

- Driver: `scratchpad/ca_driver.py` (builds manifests from the PR #147 worktree
  via `build_isaac_job_manifest` / `build_isaac_eval_job_manifest`, mirrors
  `eval_sweep.py`).
- Faithful UR10 spec: `scratchpad/ca_robot_ur10.json` (generated from the `ur10e`
  preset via `robot_spec_payload`, `robot_source=byo_usd`).
- Object test: `ca_driver.py train <RID> <sugar_box_usd> -` then
  `ca_driver.py eval <RID> <sugar_box_usd> -`.
- Robot test: `ca_driver.py train <RID> - ca_robot_ur10.json`.
- Asset URLs HEAD-checked 200 OK on `omniverse-content-production` (sugar_box,
  ur10/ur5e/ur10e, kinova jaco2).

_The effort_limit_sim patch used to reach break (3) was an exploratory probe and
was reverted; the committed module reflects the shipped PR #147 state._
