# BYO-Robot Task Registration (Isaac-Lab)

_Status: design + opt-in scaffolding + task retargeting. Default behavior is
byte-for-byte unchanged; the BYO-robot training path is gated behind
`NPA_BYO_ROBOT_TASK=1`. `register()` now swaps the articulation **and** retargets
the Franka-hardcoded ee_frame / action / command names onto the customer robot;
the remaining blocker for a real custom robot is task/reward semantics (a gripper
for the lift task) — see "Task retargeting" and `custom-asset-test-results.md`._

## Problem

The Sim2Real RL inner loop trains by submitting an Isaac-Lab sibling Job that runs
the stock RSL-RL trainer against `Isaac-Lift-Cube-Franka-v0`
(`byo_isaac_trainer.py`), and evaluates against `EVAL_TASK`
(`byo_isaac_eval.py`). A customer `robot_spec` / `robot_preset` is parsed,
validated, and recorded at the assets stage, and a BYO robot USD can be swapped
into the **in-process held-out eval** env (`engine._set_isaac_robot_usd`). But the
**RL trainer and the on-cluster eval never consume `robot_spec`** — they always
train and score a Franka. So "bring your own robot" is, for the part that matters
(training the policy), cosmetic.

This doc describes how a customer `robot_spec` becomes a **registered Isaac-Lab
task variant** that swaps the robot articulation into the Lift env, so the policy
is genuinely trained on the customer's embodiment.

## Why a registered variant (not a hydra override)

This mirrors the proven `isaac_physics_task.py` pattern. Two hard constraints make
plain `env.scene.robot.*` hydra overrides insufficient:

1. **Boot ordering.** `isaaclab.envs` / `mdp`, and even `gym.spec(stock_task)`,
   pull USD `pxr`, which only exists **after** `AppLauncher` boots. So building the
   robot `ArticulationCfg` (a `UsdFileCfg` spawn, actuators) must happen
   **post-boot**, inside a wrapper that boots the sim app first. A hydra arg passed
   to `train.py` is parsed pre-build and cannot construct these objects.
2. **Structured replacement.** Swapping a robot is not a scalar set: the spawn USD,
   the init joint positions (keyed by joint name), and the actuator stiffness/
   damping/effort-limit groups must all be replaced consistently. That is a config
   subclass that overrides `__post_init__`, not a flat `key=value`.

So the mechanism is: ship a small module into the Isaac container, and in a
post-boot wrapper subclass `FrankaCubeLiftEnvCfg`, override its
`scene.robot` `ArticulationCfg` from the customer `robot_spec`, register it as a
new gym id, and train/eval against that id — exactly the structure
`isaac_physics_task.register()` + `TRAIN_WRAPPER_SCRIPT` already use for the
friction/mass variant.

## robot_spec → ArticulationCfg overrides

The customer `robot_spec` (see `npa.genesis.robot_assets.RobotSpec`) carries:
`robot_source`, `name`, `ee_link`, `joint_names`, `n_arm_joints`/`n_gripper_joints`
(→ `dof_count`), `home_qpos`, `kp`/`kv` gains, `force_lower`/`force_upper`,
`local_path`/`robot_uri` (the resolved USD/URDF), `gripper_open`/`gripper_close`.

The variant maps these onto the Lift task's `scene.robot` `ArticulationCfg`:

| robot_spec field | ArticulationCfg target | Notes |
|---|---|---|
| resolved USD (`local_path`, or URDF→USD conversion) | `spawn = UsdFileCfg(usd_path=...)` | preserve `articulation_props`/`rigid_props`/`activate_contact_sensors` from the stock spawn |
| `joint_names` + `home_qpos` | `init_state.joint_pos = {name: q}` | per-joint, not positional; falls back to `{".*": 0.0}` when names are absent |
| `kp` / `kv` (per DOF) | one `ImplicitActuatorCfg` group, `stiffness`/`damping` | a single `.*` actuator group; per-joint splitting (arm vs. gripper) is a follow-up |
| `force_upper` | actuator `effort_limit` | max abs effort across DOFs (conservative single value) |
| `ee_link` | recorded; used by the eval/command frame | the Lift command frame references the ee body |

A `stock_franka` spec produces **no overrides** (the helper returns an empty dict),
so the variant degenerates to the stock task and the proven path is untouched.

## Task retargeting (the seam's core gap, now closed for links/joints)

Swapping the articulation USD is necessary but not sufficient: the stock Lift task
cfg **hardcodes Franka link/joint names** in three places that break on a different
arm (`Failed to create frame transformer ... panda_link0 ... No matching prims`).
`task_retarget_overrides(spec)` (pure, unit-tested) maps the robot_spec to those
renames, and `register()` applies them post-boot in `__post_init__`:

| Franka-hardcoded cfg | retarget source | applied to |
|---|---|---|
| `scene.ee_frame` source prim `.../Robot/panda_link0` | `base_link` | FrameTransformer `prim_path` |
| `scene.ee_frame` target prim `.../Robot/panda_hand`, name `end_effector` | `ee_link` | each `target_frames[*].prim_path` (name kept) |
| `actions.arm_action.joint_names = ["panda_joint.*"]` | `joint_names[:n_arm_joints]` | arm action joint names (or `body_name` for an IK action) |
| `actions.gripper_action` joints/open/close `panda_finger.*` | `finger`/`gripper_joint_names` + `gripper_open/close` | gripper action (only when the spec declares a gripper) |
| `commands.object_pose.body_name = "panda_hand"` | `ee_link` | command resolution body |

Every field a spec omits resolves to the **stock Franka** value, so a stock-Franka
(or field-less) spec yields the panda_* names verbatim — the Franka path is
byte-for-byte. Each term is applied defensively (guarded `hasattr`/try) and the
applied/skipped set is logged as `ROBOT_RETARGET applied=[...] skipped=[...]`, so a
cfg-shape change on a newer Isaac-Lab degrades to a recorded skip, not a crash.
`register()` also keeps `effort_limit_sim` in lockstep with `effort_limit` (Isaac
2.x rejects a mismatch on implicit actuators — the first non-Franka break).

### Honest task/robot compatibility

`task_robot_compatibility(spec, task_kind)` is a separate, pure gate: a cube-lift
(and any manipulation task built on the stock Lift reward) **requires an actuated
gripper**. A gripperless arm (a bare UR/Flexiv) cannot lift no matter how the links
are renamed, so the path reports `task_robot_compatible=false` with the reason and
the customer requirements (`ROBOT_COMPAT` / `ROBOT_TASK_INCOMPATIBLE` markers, and
a `task_robot_compatible=` field in `ROBOT_SUMMARY`) rather than training a policy
that can never succeed. This is reported, never hidden.

### Pure vs. Isaac-touching split

Following `isaac_physics_task.py`, the module is split so the data mapping is unit
tested off-GPU and only the registration touches Isaac:

- **Pure (unit-tested, no torch/isaac import at module top):**
  - `robot_spec_from_env(env) -> dict | None` — read `NPA_BYO_ROBOT_SPEC_JSON`
    (and a `robot_preset` hint); return `None` when absent → stock fallback.
  - `robot_articulation_overrides(spec) -> dict` — the articulation table above as
    a plain dict (`usd_path`, `init_joint_pos`, `stiffness`, `damping`,
    `effort_limit`, `ee_link`). Empty dict for a stock-Franka spec.
  - `task_retarget_overrides(spec) -> dict` — the link/joint rename set
    (`ee_frame_source`, `ee_frame_target`, `ee_frame_name`, `arm_joint_names`,
    `command_body_name`, `gripper`). Franka-default for omitted fields; empty for a
    stock/USD-less spec.
  - `task_robot_compatibility(spec, task_kind) -> dict` — the honest task/robot
    gate (`task_robot_compatible`, `has_gripper`, `reason`, `requirements`).
  - `module_source()` — this module's own source, for shipping into the container.
- **Isaac-touching (exercised on-cluster):**
  - `register(spec) -> task_id | None` — lazily imports `gymnasium` + the Franka
    Lift cfg, subclasses `FrankaCubeLiftEnvCfg`, applies the overrides in
    `__post_init__`, and `gym.register`s `NPA-Lift-Cube-<robot>-v0`. No-op
    (`None`) when there are no overrides.
  - `TRAIN_WRAPPER_SCRIPT` — post-boot wrapper: (1) boot `AppLauncher`,
    (2) import `isaaclab_tasks`, (3) `register(spec)`, (4) run the RSL-RL
    `OnPolicyRunner` like stock `train.py`, asserting the registered env's robot
    spawn USD matches the customer USD before training (no silent stock fallback).

## Wiring (opt-in, default unchanged)

`byo_isaac_trainer.py` (and `byo_isaac_eval.py`) gain a code path gated on
`NPA_BYO_ROBOT_TASK=1` **and** a non-stock `robot_spec` present, exactly like the
existing `NPA_BYO_ISAAC_PHYSICS` gate:

- Flag unset → current stock behavior, byte-for-byte (`train.py --task
  Isaac-Lift-Cube-Franka-v0`).
- Flag set + stock-Franka spec → overrides are empty → variant degenerates to
  stock (mechanism proven without changing the policy).
- Flag set + real BYO spec → ship `isaac_byo_robot_task` + run the post-boot
  wrapper, registering and training/eval-ing the robot-swapped variant.

The robot_spec is passed to the container as `NPA_BYO_ROBOT_SPEC_JSON` (a compact
JSON of the fields above plus the in-container USD path that the job stages from
S3), mirroring how `NPA_GEN_FRICTION`/`NPA_GEN_MASS_SCALE` carry the physics.

## Scope and explicit non-goals

This establishes the **registration seam**: robot_spec reaches training, the
variant loads the customer articulation, and training runs on it. It deliberately
does **not** solve, and must not be claimed to solve:

- **Per-joint actuator fidelity** — a single `.*` actuator group with one
  stiffness/damping/effort value is a coarse approximation; arm-vs-gripper and
  per-joint gains are a follow-up.
- **Link/joint names must be real USD prims.** Retargeting propagates the declared
  `ee_link`/`base_link`/`joint_names`, but they must exist as prims in the staged
  USD. (On the UR10 USD, `tool0` is a URDF tf frame, not a rigid-body prim;
  `wrist_3_link` is the real last link — see `custom-asset-test-results.md`.)
- **Task/reward semantics for a non-Franka arm** — link/joint *names* are now
  retargeted, but the Lift reward and the **gripper-based grasp** are still
  Franka-shaped. A gripperless arm is correctly reported `task_robot_compatible=false`
  (it physically cannot lift); a genuinely different *manipulator* needs a gripper
  declared (`gripper_joint_names` + open/close) and, for arbitrary tasks, a
  customer-declarable task + reward (gap (e)). Swapping the USD + renaming links
  does not by itself make the lift reward meaningful.
- **Sim-to-real transfer** — domain randomization of robot dynamics, a held-out
  *dynamics* eval, a robot-deployable policy export, and a real-world success
  metric are all still required (see `CAPABILITY.md`, gaps b–e).

In short: this makes BYO-robot training a **real mechanism** rather than cosmetic,
and is validated end-to-end by routing the Franka itself through the BYO path. A
true custom robot still requires a real robot USD + tuned reward + the transfer
work above.

## Custom-asset test results (2026-06-26, on-cluster)

The seam was tested on `npa-rtxpro-mk8s` with real GPU jobs. Full evidence:
[`custom-asset-test-results.md`](custom-asset-test-results.md). Summary:

- **Custom OBJECT USD — WORKS end-to-end.** A non-default rigid-ready YCB asset
  (`004_sugar_box.usd`) trains (`env.scene.object.spawn.usd_path` override,
  `TRAIN_RC=0`, checkpoint uploaded) and evals (`EVAL_OBJECT_USD_APPLIED`,
  `rollout_ok`, real `object_goal_distance`) with no rigid-body error. Set
  `NPA_BYO_ISAAC_OBJECT_USD`.
- **Custom non-Franka ROBOT (UR10) — articulation + task names now retarget; the
  remaining blocker is the gripper.** The two task-scaffolding breaks the first
  test found are fixed: `register()` keeps `effort_limit_sim` in lockstep with
  `effort_limit`, and `task_retarget_overrides` renames the `ee_frame`
  FrameTransformer + arm action + command body onto the customer's links/joints
  (`ROBOT_RETARGET applied=[ee_frame, arm_action.joint_names,
  commands.object_pose.body_name]`). On-cluster, the `panda_link0` break is gone:
  with `ee_link=wrist_3_link` the FrameTransformer and arm action **pass**, and the
  run stops only at the stock gripper action (`panda_finger.*: []` on a gripperless
  arm) — exactly what `task_robot_compatible=false` reports. A gripperless UR10
  physically cannot lift; a real custom robot needs a gripper-bearing USD with prim
  names that match the declared `ee_link`/`base_link`. (The Franka path is
  unchanged: `frnoreg-*` trains to `TRAIN_RC=0` with `ROBOT_RETARGET_PLAN={}`.)
- **Custom SCENE — not loadable.** `simready://` URIs are placeholders (never
  resolved); no scene/table USD override exists, only the manipuland object.
