# BYO-Robot Task Registration (Isaac-Lab)

_Status: design + opt-in scaffolding. Default behavior is byte-for-byte
unchanged; the BYO-robot training path is gated behind `NPA_BYO_ROBOT_TASK=1`._

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

### Pure vs. Isaac-touching split

Following `isaac_physics_task.py`, the module is split so the data mapping is unit
tested off-GPU and only the registration touches Isaac:

- **Pure (unit-tested, no torch/isaac import at module top):**
  - `robot_spec_from_env(env) -> dict | None` — read `NPA_BYO_ROBOT_SPEC_JSON`
    (and a `robot_preset` hint); return `None` when absent → stock fallback.
  - `robot_articulation_overrides(spec) -> dict` — the table above as a plain
    dict (`usd_path`, `init_joint_pos`, `stiffness`, `damping`, `effort_limit`,
    `ee_link`). Empty dict for a stock-Franka spec.
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
- **Task/reward retuning for a non-Franka arm** — the Lift reward terms and the
  action/observation contract are Franka-shaped. A genuinely different arm needs a
  validated reward and a real, well-formed robot USD (RigidBody + joints + drives);
  swapping the USD alone does not make the lift reward meaningful.
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
- **Custom non-Franka ROBOT (UR10) — does NOT reach training.** Two real breaks,
  in order: (1) the trainer has no env path to supply a robot USD (UR/Flexiv
  presets carry no `usd_path`), so the documented path trains a Franka; (2) forcing
  a faithful UR10 `robot_spec` in: the USD loads and the scene builds, then Isaac
  rejects `effort_limit != effort_limit_sim` on the Franka preset's implicit
  actuators; (3) past that, the Lift task's `ee_frame` FrameTransformer (and the
  action/reward terms) reference Franka prims (`panda_link0`/`panda_hand`) absent
  on a UR10 → `No matching prims were found`. `register()` swaps the articulation
  but not the task's Franka-specific sensors/actions/rewards — confirming gap (a):
  for training, BYO-robot is a mechanism, not yet a working custom-robot pipeline.
- **Custom SCENE — not loadable.** `simready://` URIs are placeholders (never
  resolved); no scene/table USD override exists, only the manipuland object.
