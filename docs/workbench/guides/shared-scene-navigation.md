# Shared-scene Isaac navigation

[The workflow](../../../workflows/testing/shared-scene-navigation.yaml) runs the
built-in public reference or an operator's registered Isaac Lab navigation task:
prepare inputs → native RSL-RL train → reload checkpoint and evaluate held-out
goals. It requires a self-contained collision USDZ and explicit reset/probe cases.
The public bundle builder supplies a cluttered warehouse and reviewed native task;
custom robots retain the BYOF adapter boundary. **Native 4000-robot physical controls
and 1500 PPO training iterations have completed; independently reloaded held-out
performance and video acceptance remain pending.**
Proprietary robot and scene integration remains operator input.

The reference extends the pinned public
[Isaac Lab navigation configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/anymal_c/navigation_env_cfg.py).
Its public quadruped uses the upstream pretrained low-level locomotion controller;
the high-level navigation policy is trained with native RSL-RL PPO. It adds 216
static-scene range rays, fixed goals, progress rewards and measured collision
penalties. Robot clones share one warehouse at world coordinates, with native
collision filtering and independent parked/coincident-peer and obstacle controls.
The reference demonstrates the public task, not a proprietary controller.
See the [component attribution](shared-scene-navigation.NOTICE.txt) for upstream
modules and the separate runtime/asset boundary.

## Public reference inputs

The pinned runtime is Isaac Lab `3.0.0b2.post1`, Isaac Sim `6.0.1.0`, and RSL-RL
`5.0.1`. Authorize native runtime and upstream robot/controller asset access using
the normal Isaac procedure. The task fetches public assets through upstream Isaac
paths; it does not bake them into a new image. Supply an exact runtime image digest
and the reviewed Workbench source overlay. `source_bundle_sha256` binds all
installed navigation Python module names and bytes before adapter import; it is
distinct from the image digest and is not a hash of the entire overlay archive.
Use the same source checkout to build inputs and stage runtime source.

The public recipe also pins `reference_controller_sha256` for the separate
runtime-fetched locomotion controller. The action factory verifies downloaded
bytes before any TorchScript decoding, supplies those exact verified bytes to
the upstream action implementation, and records the loaded identity. Both
comparison arms retain this pin independently of the high-level checkpoint.
The [attribution record](shared-scene-navigation.NOTICE.txt) lists its vendor URL
and public reference digest; weights are not included in this repository.

The upstream robot USD assets and recurrent actuator model retain their versioned
Isaac 6.0 asset references. This recipe does not byte-pin that complete vendor
asset closure, and the runtime image digest does not cover those fetched bytes.
Keep any cache or URL hashes observed after loading separate from the locomotion
policy's verified-before-load identity. Qualification reports must state this
boundary and reject comparisons when observed asset identities differ.

```bash
npa/.venv/bin/python -m npa.workflows.navigation.reference_bundle \
  --output-path "$NPA_NAVIGATION_BUNDLE" --image "$NPA_NAVIGATION_IMAGE" \
  --num-envs 4000 --iterations "$NPA_PPO_ITERATIONS" \
  --episode-steps "$NPA_EVALUATION_STEPS"
```

Training iterations and evaluation horizon are explicit experiment inputs. The
reference's native training episodes last 30 simulated seconds. Its generated
warehouse has real triangle-mesh floor, walls and racks, with disjoint seeded
routes through the clutter. This is a public procedural scene, not a reconstructed
customer environment. To use reconstructed geometry, supply `--scene-file` and
`--cases-file`; the latter must contain measured `train_cases`, `eval_cases`, and
`probe` objects. Geometry must be metric Z-up static triangle meshes with actual
USD collision schemas. The adapter derives its range mesh from those exact
colliders, without adding a floor or modifying collision surfaces. Choose root
heights from measured floor elevation and the public robot's stance.

The built-in task requires flat supported routes. Five native downward rays at
the center and corners of a 0.70 × 0.46 m footprint must hit the actual imported
collision mesh. Root clearance must stay within 0.25–0.85 m, the five floor
heights must differ by at most 0.15 m, and the root upright cosine must remain at
least 0.5. Missing floor, falling, excessive tilt or clearance outside that
envelope produces `physical_failure`, independently of obstacle contact. Training
penalizes and terminates those episodes; evaluation cannot score them as success.
Raw world root positions, upright cosine, clearance and support diagnostics are
retained. Custom BYOF tasks must implement their own physically valid support
definition and report it through the same measurement contract.

The contact instrumentation samples native PhysX at every physics substep. It
classifies nonvertical contact normals as obstacles, counts every base contact,
and excludes ordinary vertical foot support. Unexpected net contacts after
subtracting static-scene forces are reported as peer contacts. Values below
0.02 N are treated as numerical noise. These controls and the 4000-robot target
require real GPU validation; local tensor and USD tests do not prove runtime
throughput or learning convergence.

A native RTX PRO 6000 qualification constructed all 4000 robots in one warehouse
and passed repeated-reset and coincident-peer controls with zero measured focal
state/observation deltas and zero peer contact. The obstacle positive control
produced actual contact. Native PPO completed 1500 iterations and 48 million
control transitions in 6230.69 seconds of learning, averaging 7703.80 transitions
per second. The checkpoint changed by a finite, nonzero parameter norm and all
34,500 exported scalar observations were finite. Independent held-out performance
and video acceptance remain outstanding in the readiness record.

Native training retains `reference_checkpoint.pt` before the first update,
`policy.pt` after learning, TensorBoard logs and exported `learning-curves.json`.
The training record reports measured wall time, native control-transition
throughput and peak PyTorch CUDA memory; runtime evidence includes the actual
GPU model, physical device memory, robot population and static collider count.
`comparison.compare_reference(training_prefix, output_prefix)` reloads both
checkpoints on the identical held-out inputs and publishes the measured gain.
The initial reference is an untrained high-level navigation policy above the
public pretrained low-level controller; it is not a customer's existing policy.

Raw training curves retain upstream metric names. `Metrics/success_rate` is
updated after automatic goal resets in this reference and is not held-out success.
`Episode_Termination/base_contact` includes goal arrival and physical failure;
it is not a collision rate. Derive success, obstacle contact and physical failure
from the separately reloaded evaluation trajectories and `episode_rows` metrics.

Public reference evaluation captures the actual Isaac Replicator RGB renderer
while the focal robot follows its native policy. The video overlays measured
goal distance, contact forces and simulated time. Peer visuals are hidden only
for this recording; their physics remain active, and render steps use zero
simulation delta. Built-in evaluation enables native multi-tick and per-sensor
acceleration-structure modes through the pinned AppLauncher's `kit_args` before
Kit initializes. Training and external BYOF launch arguments are unchanged.
The camera explicitly applies Isaac's `OmniSensorAPI` with its native zero-Hz
autotrigger mode before render-product creation. Capture checks the startup modes,
temporarily enables the scheduler disabled by Lab, and restores that setting afterward.
Two frozen render
passes flush annotations. Each accepted frame must match its authored camera
pose, intrinsics and resolution, and its rational renderer time must match actual
native simulation-manager time backed by physics step events, with the same time
read independently from Fabric. The recorder uses the existing native extension
interface because pinned Lab replaces Isaac's public manager class. It never
creates or writes a renderer clock. Uncached native root transforms, root
velocities, joint state, native and Lab step counts, and clocks must remain
unchanged across rendering. The native
time origin follows the preceding controls; it is distinct from the video's
relative rollout time. Native qualification of this capture correction is pending.
Artifacts include `rendered-rollout/*.png`, frame measurements,
and `rollout.mp4` when the runtime supplies FFmpeg. The PNG sequence can be encoded
later without rerunning or reconstructing the simulation. Custom BYOF tasks
retain their own visualization integration.

The focal robot continues stepping in the video while other robots finish their
episodes. Its score stops at its first goal arrival, obstacle/peer contact, or
physical failure. Present the focal robot's scored step count and outcome beside
the full video; later frames do not change that score or the cohort metrics.

## Resume and independent checkpoint evaluation

Add `--checkpoint` when building inputs, or set optional
`initial_checkpoint: {file: baseline.pt, sha256: ...}` in a custom recipe. The
checkpoint must be a contained `.pt` file with a distinct name from the generated
`policy.pt`. Training verifies its bytes immediately before safe tensor-only
decoding and restores complete native policy, optimizer and iteration state with
strict architecture checks. `training.json.initialization` records the input
digest and initial iteration; parameter-change measurement starts after loading.

Programmatic local/S3 entrypoints are `stages.prepare(input, prepared, image)` and
`stages.run_stage("train", prepared, trained)`. For paired comparisons, construct
a separate sealed bundle with the held-out scene/cases and each arm's exact
`initial_checkpoint`, then call
`stages.run_stage("evaluate-checkpoint", prepared_eval, evaluated)`. It reloads
that checkpoint and emits `evaluation.json` and raw `trajectory.json` without
requiring a training receipt for the held-out scene. A valid low-scoring arm is
published with `passed=false`; it remains usable for a baseline/candidate
comparison. Normal workflow `evaluate` retains its qualification gate.

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
| `initial_checkpoint` | Optional contained `.pt` file and SHA-256 for actual native resume or independent checkpoint evaluation |
| `source_bundle_sha256` | Required for the built-in public reference: digest of all installed navigation modules, checked before import; optional for image-owned external adapters |
| `reference_controller_sha256` | Required only for the built-in reference: exact runtime-fetched low-level controller bytes, verified before native loading |

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
| `measure(env)` | Return NumPy-compatible finite `position_m [N,3]`, `heading_rad [N]`, `goal_m [N,2]`, nonnegative `obstacle_contact [N]`, `peer_contact [N]`, boolean `physical_failure [N]`, `upright_cosine [N]`, and nonnegative `ground_clearance_m [N]` from actual simulator pose/command/contact/support data. Physical failure includes falls and unsupported poses using the task's support definition. Contacts are accumulated across physics substeps, including terminal transitions; ordinary floor support is excluded from obstacle collision. Zeroing failure/contact signals is invalid. |
| `probe_mode(env)` | Context manager: disable stochastic observation corruption and automatic resets for deterministic diagnostic transitions, restore original settings afterward. Preserve collision filtering, geometry, sensor selection and control dynamics. |
| `visibility_paths(env)` | RGB-D only: return `robot_roots`, `attachment_roots`, `camera_prims`. List every robot articulation root, detached attachment subtree and one observing camera per robot, in robot order. Paths are checked on the live USD stage. |

Robot collision filtering must exclude every other robot and preserve contacts
with warehouse obstacles. Implement it in the task's existing collision-group
or pair-filter setup. The runtime independently compares the focal robot's
motion and **all returned policy/critic observation streams** with peers parked
and coincident, after checking a second identical parked-peer reset for
repeatability. It rejects no-motion probes, changed observations, peer contacts
or an obstacle probe that produces no physical contact. These are finite
behavioral controls, not a proof over every possible scene state. Adapter source
and contact instrumentation still require operator review.

Native stages retain compressed probe arrays and `isolation-comparisons.json`
before checking repeatability and peer isolation. These artifacts preserve all
robots' measured states and the focal robot's observations, with exact deltas
by stream and step. The public reference also records joint targets and recurrent
actuator state so a failed reset can be distinguished from peer influence.

State observations must contain only own state, goal and static-scene features.
Static raycasters must target explicit mesh descendants of `scene_prim`, never
robot paths or a scene-wide wildcard containing robots. Unrecognized sensors
fail closed. Articulation placement and measured resets share world coordinates;
a spacing configuration is not accepted as isolation evidence.

## RGB-D: all robot geometry hidden

This optional BYOF path remains unqualified on the pinned Lab preset, including
its disabled Replicator scheduler. The public range-sensor reference and its
rollout video qualification do not validate this separate RGB-D probe path or
camera-conditioned policy throughput.

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
requires reaching the metric goal tolerance with no prior obstacle contact, peer
contact or physical failure. Per-episode `physical_failure_steps` remains separate
from contact counts. Native auto-reset during evaluation is rejected to prevent scoring a
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
