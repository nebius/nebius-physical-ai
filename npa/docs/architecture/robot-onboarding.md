# Robot Onboarding (Sim2Real)

_Status: building blocks shipped — declarative spec (B1), auto-derivation (B2),
robot-aware task config (B3), and the `onboard-robot` CLI + this doc (B4). The
Franka path is byte-for-byte unchanged; a non-Franka arm now gets an arm-scaled
Lift task instead of Franka-tuned constants. Whether a given robot **learns** is
gated by B5 (see `scratchpad/ONBOARD_DONE.md` for the live Kinova evidence)._

## Goal

Let a customer onboard **their own robot** into the Sim2Real Lift / reach / place
workflow from a single declarative file — supplying the minimum (an asset + the
task) and letting the platform derive the robot-aware config that makes the
learning signal correctly scaled for that arm.

This builds on BYO-robot task registration
(`byo-robot-task-registration.md`), which made a non-Franka arm **run**
(load USD, retarget ee_frame + arm joints + gripper, register a Lift variant).
That alone was not enough: the Lift task's **action scaling, reward thresholds,
object/goal placement, and init pose were Franka-tuned and hardcoded**, so a
different arm ran but trained against a mis-scaled signal. Onboarding makes those
robot-aware and auto-derived.

## What the customer provides vs. what is auto-derived

The contract is "supply the asset + names you know; write `auto` for the rest".

| Customer provides (minimal)                         | Auto-derived (`auto` / off-GPU heuristic / on-cluster from USD) |
| --------------------------------------------------- | --------------------------------------------------------------- |
| Robot `name`                                        | —                                                               |
| `usd_path` (or `urdf_path`); https / s3 / container | USD staging path                                                |
| `ee_link`, `base_link`                              | from the asset when set to `auto`                               |
| arm `joint_names`, `n_arm_joints`                   | from the asset when `auto`                                      |
| `gripper_joint_names`, `n_gripper_joints`           | from the asset when `auto`                                      |
| `gripper_open` / `gripper_close`                    | from finger-joint ranges (`auto`)                               |
| `home_qpos`, `kp`/`kv`/forces                       | from joint limits (`auto`)                                      |
| task `skill` (lift/reach/place)                     | —                                                               |
| `object_usd` (optional)                             | default rigid-ready MultiColorCube                              |
| `goal_pos` (or `auto`)                              | placed within the derived arm workspace                         |
| `lift_height_m`, `success_distance_m`, threshold    | sensible task defaults                                          |
| —                                                   | **action_scale** (from arm reach / joint ranges)               |
| —                                                   | **workspace_reach_m**, **object_init_range**, **goal_range**   |

The derived `action_scale`, reach, placement, gripper targets, and init pose are
exactly the robot-aware values the B3 Lift variant applies in-container, replacing
the Franka constants. The CLI prints each value with its `source`
(`explicit` / `preset` / `measured` / `heuristic`) so the customer sees what was
supplied vs. derived.

## The flow

```
minimal spec  ->  validate + derive  ->  smoke (loads/retargets/trains)  ->  full train  ->  held-out eval  ->  promote
   (YAML)          onboard-robot          onboard-robot --smoke              run / inner-loop    status            workflow
```

1. **Write the spec.** Copy
   `npa/workflows/workbench/sim2real/onboarding/robot-onboarding.template.yaml`
   and fill in your robot + task. A filled, working non-Franka reference ships at
   `kinova-jaco2.yaml` (Kinova Jaco2 J2N7S300 — 7-DoF arm, 3-finger gripper).

2. **Validate + see the derived config (offline, no GPU):**

   ```
   npa workbench sim2real onboard-robot \
     --spec npa/workflows/workbench/sim2real/onboarding/kinova-jaco2.yaml
   ```

   This validates the schema, runs the auto-derivation, prints the robot-aware
   config (value `[source]`), and **gates on task/embodiment compatibility** — a
   gripperless arm asked to lift is rejected with a non-zero exit and the exact
   requirement, never a false "ready". Add `--json` for machine-readable output.

3. **Smoke (confirm it loads, retargets, and trains on the cluster):**

   ```
   npa workbench sim2real onboard-robot --spec <spec>.yaml --smoke
   ```

   Submits a short BYO Isaac trainer Job (default 20 iters / 64 envs) with the
   derived config baked in, then prints the job name + S3 output and the `status`
   command to watch it. Requires cluster creds + `ISAAC_IMAGE` /
   `NPA_SIM2REAL_BUCKET` / `AWS_ENDPOINT_URL` in the env (the sim2real operator
   env). A failed submit exits non-zero — it never prints a false success.

4. **Full train + held-out eval.** Use the staged workflow
   (`npa workbench sim2real run` / `inner-loop`) with the same BYO-robot routing;
   monitor with `npa workbench sim2real status --run-id <id>`.

5. **Promote** the checkpoint once a held-out eval clears the threshold.

## How it is wired

- **`onboarding_spec.py` (B1)** — `npa.sim2real.onboarding.v1` schema +
  validation. Records `auto` fields for the derive layer; rejects malformed specs
  and gripperless lift/place up front.
- **`onboarding_derive.py` (B2)** — pure helpers that derive `action_scale`,
  workspace reach, object/goal placement, gripper open/close, and init pose. The
  Franka spec reproduces today's known-good constants; a different arm gets
  distinct, arm-scaled values. Heavy Isaac imports stay out of this path.
- **`isaac_byo_robot_task.py` (B3)** — the registered Lift variant applies the
  derived config (action scale, reward thresholds relative to workspace, object +
  goal placement, init pose, gripper close) instead of Franka constants. The
  Franka path is byte-for-byte unchanged (the derivation reproduces it).
- **`onboarding.py` (B4 service)** — import-light glue: `build_plan(spec)` ->
  derived config + the `NPA_BYO_ROBOT_SPEC_JSON` payload + the compatibility
  verdict; `submit_smoke_job(...)` builds the trainer manifest (robot spec +
  task config baked into the container command) and applies it via an injected
  kubectl callable (so it is unit-testable without a cluster).
- **`cli/workbench/sim2real.py` (B4 CLI)** — `onboard-robot`.

## What is turnkey vs. still per-robot

- **Turnkey:** the declarative spec, validation, off-GPU derivation + display,
  the compatibility gate, the smoke submit, and the Franka-unchanged guarantee.
- **Still per-robot / per-task:** on-cluster USD introspection refines a few
  derived values (action scale / placement) from measured joint ranges, and
  whether a given arm actually **learns** the lift may need reward shaping or
  curriculum tuning. The Kinova learning evidence + remaining tuning are tracked
  in `scratchpad/ONBOARD_DONE.md`.
