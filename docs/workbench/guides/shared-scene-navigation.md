# Shared-scene Isaac navigation

[The workflow](../../../workflows/testing/shared-scene-navigation.yaml) runs an
operator's registered Isaac Lab navigation task in its existing BYOF image:
prepare inputs → native RSL-RL train → reload checkpoint and evaluate held-out
goals. It requires an integration adapter, a self-contained warehouse USDZ, and
explicit reset/probe cases. **GPU acceptance has not been run.** Proprietary
robot and scene integration remains operator input. There is no substitute task,
CPU simulator, random-action evaluation, deployment or image build in this spec.

The public [Isaac Lab ANYmal navigation configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py)
is a useful reference for registered navigation observations, commands and
low-level control. It does not establish shared-warehouse isolation. This
workflow deliberately supplies no guessed robot internals or generic
`env_spacing=0` patch. A task that does not implement the contract fails closed.

## Execution

Use the existing [BYOF integration procedure](../cookbooks/byof-isaac-lab/README.md)
to supply a digest-pinned operator image containing its registered task and
adapter. The supported native path uses Isaac Sim 6.0.1 / Isaac Lab 3 beta,
`/isaac-sim/python.sh`, `--visualizer none`, CUDA and an RT-core GPU. The workflow
requests RTX PRO 6000; use an L40S resource profile if independently qualified.
H100/H200/B200 are not this workflow's camera runtime. Existing ClearML can
remain in the task's logging integration. Workbench owns invocation, immutable
artifact handoffs and measured evaluation; it does not replace external tracking.
The task must provide `env_cfg_entry_point` and `rsl_rl_cfg_entry_point` compatible
with native RSL-RL. Other training algorithms need an explicit integration.

Stage an operator-owned bundle in S3 containing `recipe.json` and the USDZ named
by `scene_file`. The authoritative JSON schema is generated directly from
`npa.workflows.navigation.contract.Recipe.model_json_schema()`. Input code is
trusted operator code, never an untrusted uploaded module. The bundle does not
install code or run arbitrary shell commands. `adapter_module` selects a module
already installed in the image; source resolution uses `PathFinder` without
executing module or parent-package initializers, and verifies `adapter_sha256`
before importing the task. Normal import then preserves native task registration.
The pinned immutable BYOF image is the trusted code/dependency boundary,
including package initializers and import hooks; this is not an import sandbox.
The source must remain immutable between verification and import.

```bash
npa workbench workflow validate-spec workflows/testing/shared-scene-navigation.yaml --json
npa workbench workflow plan-spec workflows/testing/shared-scene-navigation.yaml --run-id preview --json
npa workbench workflow submit workflows/testing/shared-scene-navigation.yaml \
  --runtime --stage-src --run-id "$NPA_RUN_ID" \
  --var "bucket=$NPA_S3_BUCKET" \
  --var "input_uri=$NPA_NAVIGATION_INPUT_URI" \
  --var "byof_image=$NPA_NAVIGATION_IMAGE" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`input_uri` has no default. `byof_image=tool://isaac-lab` is a planning placeholder;
preparation requires the exact `registry/name@sha256:...` value in the recipe.
The generic runtime supplies `NPA_TASK_IMAGE`; train/evaluate verify equality.
`ISAAC_LAB_PYTHON` can override the native interpreter location within an
operator image. It never falls back to a system or CPU interpreter. Configure
storage and model/asset access externally before submitting. Preparation only
checks input integrity; valid schema does not mean a task is ready to execute.

## Recipe and adapter

All recipe fields are required unless listed as optional. Unknown fields,
nonfinite values, malformed module names, unsafe relative paths and duplicate
or overlapping reset cases are rejected.

| Field | Contract |
| --- | --- |
| `schema_version` | `npa.navigation.recipe.v1` |
| `task`, `adapter_module`, `adapter_sha256`, `image` | Registered task ID, installed Python module, SHA-256 of its source and exact BYOF image digest |
| `scene_file`, `scene_sha256`, `scene_prim` | Bundle-relative `.usdz`, file digest and single absolute USD prim under `/World` |
| `sensor_mode` | `state`, `static_raycast`, or `rgbd` |
| `num_envs` | At least two independent robots; exactly this many held-out cases |
| `iterations`, `episode_steps` | Positive training iteration count and the evaluation episode horizon chosen by the operator |
| `goal_tolerance_m`, `minimum_success_rate` | Positive goal radius and qualification fraction in `(0,1]` |
| `train_cases`, `eval_cases` | Disjoint case IDs, seeds and physical reset tuples; evaluation inputs never reach the training configurator |
| `probe` | `free`, `parked`, `obstacle` cases, native `actions` sequence and absolute `tolerance` (default `1e-5`, at most `0.001`) |
| `camera` | Required only for RGB-D, described below |

Each case contains `id`, nonnegative integer `seed`, `position_m: [x,y,z]`,
`heading_rad` and `goal_m: [x,y]`, all finite world-frame SI values. Episode
resets are deterministic, including articulation joints, velocities, controller
state, sensor buffers and observation history. The adapter owns task-specific
joint configuration; the workflow never invents it. Training reset events must
sample only supplied training cases, including after native auto-resets.

The installed adapter implements:

| Callable | Required behavior |
| --- | --- |
| `configure(*, task, scene_file, scene_prim, num_envs, cases, training)` | Register/import the existing task and return its native Isaac config. Reference the exact provided USDZ once at `scene_prim`; create robot articulations in that shared world. Preserve task action/reward and registered RSL configuration. Install the supplied reset distribution. During evaluation disable automatic termination/resets and command resampling; Workbench ends measurement at goal/contact/horizon. |
| `reset(env, cases)` | Apply every supplied deterministic root pose, heading and goal, reset joints/controller/history/seed, synchronize sensor state and recompute observations. Do not advance simulation unexpectedly or change scene/isolation settings. |
| `measure(env)` | Return NumPy-compatible finite `position_m [N,3]`, `heading_rad [N]`, `goal_m [N,2]`, nonnegative `obstacle_contact [N]`, and `peer_contact [N]` from actual simulator pose/command/contact data. Contacts are magnitudes accumulated across physics substeps, including terminal transitions. Exclude floor/support contacts from obstacle collision. Zeroing contact signals is invalid. |
| `probe_mode(env)` | Context manager: disable stochastic observation corruption and automatic resets for deterministic diagnostic transitions, restore original settings afterward. Preserve collision filtering, geometry, sensor selection and control dynamics. |
| `visibility_paths(env)` | RGB-D only: return `robot_roots`, `attachment_roots`, `camera_prims`. List every robot articulation root, detached attachment subtree and one observing camera per robot, in robot order. Paths are checked on the live USD stage. |

Robot collision filtering must exclude every other robot and preserve contacts
with warehouse obstacles. Implement it in the task's existing collision-group
or pair-filter setup. The runtime independently compares the focal robot's
motion and **all returned policy/critic observation streams** with peers parked
and coincident. It rejects no-motion probes, changed observations, peer contacts
or an obstacle probe that produces no physical contact. These are finite
behavioral controls, not a proof over every possible scene state. Adapter source
and contact instrumentation still require operator review.

State observations must contain only own state, goal and static-scene features.
Static raycasters must target explicit mesh descendants of `scene_prim`, never
robot paths or a scene-wide wildcard containing robots. Unrecognized sensors
fail closed. Articulation placement and measured resets share world coordinates;
a spacing configuration is not accepted as isolation evidence.

## RGB-D: all robot geometry hidden

Set `sensor_mode=rgbd` and supply a `camera` object with:

- `camera_visibility: all_robot_geometry_hidden`.
- Integer `height` and `width` greater than one.
- `rgb_tolerance` in `[0,0.05]` for normalized RGB, and `depth_tolerance_m` in
  `[0,0.01]` for metric depth.
- `minimum_changed_fraction` in `(0,1]` for each positive-control modality.
- A nonzero local camera `translation_m: [x,y,z]` for the positive control.
- `peer_in_view`, a reset case placing peer root positions in the focal
  camera's frustum while its own pose/goal stays fixed.

**All robot bodies, including self, are absent from camera images.** Geometry
under declared robot/attachment roots, including collision meshes, is hidden
with `UsdGeom.Imageable.MakeInvisible()`. Articulation/link ancestors are never
hidden or deactivated; colliders and scale are not changed. The narrow contract
requires separate visual Gprims, and cameras must remain outside hidden Gprim
subtrees. Every scene Gprim must belong to the warehouse or a declared robot
subtree. Unlisted attachments, robot lights/effects, point instancers and
instance proxies fail closed. Author editable visual roots before cloning or
supply non-instanced geometry; this implementation rejects unsupported proxies.

The visibility mechanism follows the pinned
[Isaac Sim visibility implementation](https://github.com/isaac-sim/IsaacSim/blob/v6.0.1/source/extensions/isaacsim.core.experimental.prims/python/impl/xform_prim.py)
and [USD inherited visibility](https://openusd.org/dev/user_guides/render_user_guide.html).
The [native camera implementation](https://github.com/isaac-sim/IsaacSim/blob/v6.0.1/source/extensions/isaacsim.sensors.experimental.rtx/python/impl/camera_sensor.py)
binds a render product to its camera prim. These are source-backed mechanisms;
local USD tests do not establish renderer or physics acceptance.

Visibility is checked after explicit resets and before/after native task steps,
including automatic training resets. A reset that restores robot visibility
fails the run. The camera probe uses genuine Replicator RGB and
`distance_to_image_plane` annotators on the focal camera, with fresh synchronized
render steps, `delta_time=0`, `wait_for_render=True`, and capture-on-play disabled.
It retains actual arrays with exact dimensions, uint8 RGB input, and finite
strictly positive floating depth. Missing depth, infinity background and empty
RGB fail closed; choose a finite enclosed warehouse camera view.

Parked-peer and in-view-peer frames must agree within both tolerances. Moving
the observing camera must change enough RGB **and** depth pixels; frozen buffers
cannot pass using an adapter's `passed=True`. Actual arrays and hashes are
published. The camera control checks robot index zero while all task observation
streams are tested for peer sensitivity. There is no thousands-of-cameras
throughput claim or all-view camera acceptance claim without corresponding live
evidence.

## Artifacts, acceptance and limits

Each stage conditionally creates its S3 `claim.json` once, uploads each object
with a provider `If-None-Match` guard beneath an immutable `attempts/<uuid>/`
prefix, and compares readback with the original local hashes. Only then does it
conditionally create `completion.json`, binding the exact attempt and manifest.
Consumers require matching claim/completion records and verify those same
hashes. Duplicate or conflicting writers fail, including identical retries.
Upload/readback failure leaves an incomplete claim; use a new run prefix rather
than replacing it. The protocol prevents competing Workbench writers from
changing completed evidence; bucket owners still control out-of-band access.

The workflow declares actual `completion.json` objects as stage inputs and outputs,
so `--require-inputs` can check exact keys. Resolve each logical
stage prefix with `npa.workflows.navigation.artifacts.materialize` to obtain the
following files from its accepted attempt. Logical stages are beneath
`shared-scene-navigation/<run-id>/`:

| Stage | Evidence |
| --- | --- |
| `prepared/` | Sealed recipe, scene and input integrity record, `native_runtime_verified=false` |
| `training/` | Native `policy.pt`, `agent.json`, training record with finite parameter-update norm, isolation measurements, task/image/adapter/scene identity, runtime versions and private native log |
| `evaluation/` | Reloaded checkpoint hash, unchanged policy parameters/normalization buffers, deterministic reset hash, measured per-case goal distance, obstacle/peer contact steps, path length, success fraction, raw trajectory and repeated isolation controls |
| camera stages | Actual `camera-probes.npz` parked/peer/moved RGB and depth arrays with hashes and visibility inventory |

A policy load failure aborts; there is no random-action substitute. Goal success
requires reaching the metric goal tolerance with no prior obstacle or peer
contact. Native auto-reset during evaluation is rejected to prevent scoring a
replacement episode. A completed evaluation below `minimum_success_rate` writes
its evidence and then fails the workflow. Native failures publish a failure
record and available logs; they never create a success receipt.

The spec is registered in the GPU submit matrix with an explicit rotation skip
until operator inputs are supplied. It is a real executable path, not plan-only.
[Readiness](../../../workflows/testing/shared-scene-navigation.readiness.json)
distinguishes local checks from runtime gaps. Opt-in native acceptance runs
inside the exact authorized BYOF image with storage credentials:

```bash
NPA_INTEGRATION_E2E=1 NPA_NAVIGATION_LIVE=1 \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_shared_scene_navigation_live.py -q
```

Set `NPA_NAVIGATION_INPUT_URI`, a new `NPA_NAVIGATION_OUTPUT_URI`,
`NPA_TASK_IMAGE`, and the native interpreter override if required. This test
executes real train/evaluate stages and S3 readback. Standard workflow submission
is separately exercised through the submit matrix with the supplied input/image
variables (`NPA_NAVIGATION_INPUT_URI` and `NPA_NAVIGATION_IMAGE` are consumed by
the live matrix materializer). Local contract tests make no native GPU, proprietary-task, physical
robot or camera-scale acceptance claim. No infrastructure is deployed by the
stage adapters; resource cleanup remains with the standard workflow runtime.
