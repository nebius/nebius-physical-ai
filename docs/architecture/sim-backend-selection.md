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
