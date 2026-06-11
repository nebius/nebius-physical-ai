# Sim2Real Held-Out Eval: Pluggable Sim Backend (genesis | isaac)

The Sim2Real loop's non-VLM held-out evaluation runs a robot manipulation
rollout to produce deterministic per-environment metrics that gate the outer
loop. That rollout engine is pluggable behind a single backend selector so the
same loop can run on either of two simulators without changing the report
schema or the gate.

## Backends

- **`genesis`** — the Genesis `FrankaPickPlaceEnv` pick-and-place rollout. This
  is the original, fully-supported path and remains unchanged.
- **`isaac`** (default) — a headless Isaac Sim / Isaac Lab rollout using a
  Franka manipulation task (lift-cube by default). Requires RT-core GPUs
  (L40S / RTX Pro class); it will not run correctly on datacenter GPUs without
  RT cores.

Selection points (all map to the same config field):

- CLI flag `--sim-backend genesis|isaac`
- env `NPA_SIM2REAL_SIM_BACKEND`
- runbook YAML `NPA_SIM2REAL_SIM_BACKEND`

## Why a selector instead of replacing Genesis

The two simulators have different strengths and different licensing profiles
(below). Teams that want NVIDIA's manipulation/USD ecosystem run `isaac`; teams
that want an unrestricted, lightweight default run `genesis`. Keeping both
behind one selector means the rest of the loop — VLM eval (Cosmos-Reason),
signal conversion, training, the threshold gate, and `report.json` — is
backend-agnostic.

## Schema parity (the contract)

Both backends emit the identical `npa.sim2real.heldout_eval.v1` report:

```
per_env: [ { env_id, score, success, details{ final_target_distance,
             max_reward, steps, ... } }, ... ]
```

Plus a `sim_backend` field and the shared asset-provenance block. The
outer-loop threshold gate consumes only this schema, so swapping backends never
changes downstream behavior. Only the rollout engine differs; the VLM eval is
untouched.

## Assets and provenance (no silent fallback)

Both backends reuse the same simulator-agnostic `SceneSpec` and the same
no-fallback provenance discipline:

| Path        | Genesis `asset_source`          | Isaac `asset_source` |
| ----------- | ------------------------------- | -------------------- |
| Stock       | `primitive` / `genesis_builtin` | `isaac_stock`        |
| Custom mesh | `byo_mesh` (+ sha256)           | `byo_mesh` (+ sha256)|

For the Isaac backend, a custom mesh/URDF is imported to USD offline with Isaac
Lab's converters (`MeshConverter` / `UrdfConverter`) and then loaded into the
task scene. Every object records `{asset_source, sha256, loaded}`, and a
requested mesh that fails to import or load raises rather than substituting the
stock asset. This proves the customer asset was actually exercised.

## Robot embodiment (BYO arm, not just BYO object)

The robot is described by a simulator-agnostic `RobotSpec`
(`npa.genesis.robot_assets`), a sibling of `SceneSpec`. It selects a
`robot_source`:

| `robot_source`    | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `stock_franka`    | Default. Genesis built-in Franka Panda MJCF.         |
| `byo_urdf`        | Customer arm as a URDF (+ meshes), downloaded + hashed. |
| `byo_mjcf`        | Customer arm as MuJoCo MJCF (Genesis only).          |
| `byo_usd`         | Customer arm as USD (Isaac only).                    |
| `genesis_builtin` | An articulated robot shipped in the Genesis assets tree. |

Selection points (all alongside the object SceneSpec):

- CLI: `--robot-preset {franka|ur5e|ur10e|flexiv}`, `--robot-spec-uri <RobotSpec JSON>`,
  or `--robot-source <source>`
- env: `ROBOT_PRESET` / `ROBOT_SPEC_URI` / `ROBOT_SOURCE`
  (threaded into the component as `NPA_SIM2REAL_ROBOT_*`)

Presets ship for Franka (the current values, byte-for-byte), Universal
Robots UR5e/UR10e (6-DOF, `tool0` end-effector, no integrated gripper), and
Flexiv Rizon (7-DOF, `flange` end-effector). A customer can pick a preset or
supply a full URDF plus a minimal `RobotSpec` JSON (end-effector link name,
DOF count, gains, force ranges, home pose). The env's cached links, PD gains,
force ranges, home pose, and the IK end-effector link all come from the
`RobotSpec`; the no-spec / `stock_franka` path reproduces today's Franka
behavior exactly.

**The robot must be an articulated description (URDF / MJCF / USD).** UR and
Flexiv publish these. A plain visual mesh (`.obj` / `.glb` / `.stl` / `.ply`)
is only valid for a manipulated *object*, never the robot — supplying one as
the robot raises a clear error. A BYO robot that fails to download, validate,
or load raises rather than silently falling back to Franka, and the robot
provenance (`{robot_source, robot_uri, sha256, loaded, robot_fallback_used}`)
is written into `report.json` so a run can prove the customer arm was loaded.

Gripper control for the UR/Flexiv presets is a follow-up: their published
URDFs ship without an integrated gripper, so those presets run gripperless
(the gripper action channel is ignored) until a customer attaches and
configures one.

## Code injection into the Isaac image

The Isaac Lab image ships Isaac Sim + Isaac Lab only under its bundled
interpreter and bakes no workbench code. The held-out eval component therefore
injects the branch's workbench code into that interpreter at container start —
from an object-storage source tarball (using the pod's mounted storage
credentials) or, when the source repo is reachable, a shallow git clone — and
ensures the storage client dependency is present. This keeps the simulator
image generic and lets any branch run without rebuilding it.

## Licensing reality

- **Isaac Sim / Isaac Lab** are free for internal R&D and for customers running
  their own training and simulation. The NVIDIA AI Enterprise license is
  triggered by *delivering Isaac Sim as a hosted service / redistributing it*,
  **not** by importing custom assets or running customer simulations. Bringing
  your own meshes is a normal, unrestricted use.
- **Genesis** is Apache-2.0 with no service restriction, which is why it is the
  unrestricted default option for teams that do not need the NVIDIA ecosystem.

When packaging either backend for customers, the line to watch is hosted-service
redistribution of Isaac Sim, not asset or workload content.
