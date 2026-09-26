# Shared-scene Isaac navigation

[The workflow](../../../workflows/testing/shared-scene-navigation.yaml) runs the
built-in public reference or an operator's registered Isaac Lab navigation task:
prepare inputs → native RSL-RL train → reload checkpoint and evaluate held-out
goals. It requires a self-contained collision USDZ and explicit reset/probe cases.
The public bundle builder supplies a cluttered warehouse and reviewed native task;
custom robots retain the BYOF adapter boundary. **The historical procedural
warehouse task passed its unchanged 80% success requirement: 3997 of 4000 fresh
test cases (99.925%) reached within 0.5 m of their goals after 1500 initial PPO iterations
and a further 500 iterations with a corrected arrival reward. Independently
reloaded policies, physical controls and native rendered evidence were verified.**
Proprietary robot and scene integration remains operator input.
The 4000-robot measurements use static-scene range observations; camera-conditioned
policy training at that population remains unqualified. The warehouse result used
the earlier contact metric and control protocol. The current source separately
passed reconstructed-scene physical controls and rendered baseline evaluation:
759 of 4000 cases succeeded (18.975%), below the same 80% requirement.
Reconstructed-scene policy training and improvement remain unqualified.

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

The current built-in reference uses [fresh native control processes](navigation-fresh-controls.md)
for its four physical controls before a separate training/evaluation process.
This protocol passed the reconstructed-scene qualification described below.
The historical warehouse results retain their original warm-reset protocol;
neither result establishes warm-reset correctness for reconstructed scenes.

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

The contact instrumentation samples native PhysX at every physics substep. Its
original obstacle rule is `abs(normal.z) < 0.7` or any base contact. Native foot
contacts against a triangle floor can have oblique solver normals, so the
current reference additionally checks a verified spherical foot's terrain-side
witness against the source geometry before counting those contacts as obstacles.
Base contacts and contacts on other body
parts keep the original rule. Unexpected net contacts after subtracting
static-scene forces remain peer contacts. Obstacle force is summed per robot;
both obstacle and peer measurements must exceed 0.02 N to count. The force
threshold is unchanged.

A foot contact is recognized as support only when its actual native body path
identifies one reference foot with one enabled, active and loaded spherical
collision shape in the composed USD, including instance proxies. Uniform,
positive, unsheared transforms and native single-shape identity are required.
The query radius is
that shape's resolved, positive contact offset. Its finite rest offset must be
smaller than the contact offset; zero and negative rest offsets are allowed.
Neither offset is modified. The source-surface distance from
`point - separation * sign(force) * normal` must fit inside this radius, and the
root must be above the selected source point. Contact offset defines contact
onset, not a maximum negative penetration; it does not bound the absolute native
separation. Finite inputs, nonzero signed force and a unit normal within the
recorded float32 tolerance are required. See the
[spherical foot contact contract](navigation-contact-witness.md) for the source
semantics and live identity boundary. Each native root pose is bound to the unique
USD articulation root within its robot asset, including roots nested below the
asset container. Missing, ambiguous or mismatched roots abort qualification.

The query uses the exact world-space triangle mesh already used by the static
range sensor, checked against its authored USD geometry. Every triangle within
the radius must face upward with normal z at least 0.7. Its normal's dot product
with the selected face normal must also be at least 0.7, and the local faces
must connect through edges inside the radius.
A nearby wall, underside, open or nonmanifold edge, inconsistent winding,
degenerate face, disconnected surface or more than 64 local faces preserves
obstacle classification. Feet without a verified single-shape margin keep the
original rule. Inconsistent native identities, invalid offsets or mismatched
source geometry abort qualification. The 64-face bound limits query memory;
it does not limit workload duration or simulation steps.

This correction changes the obstacle metric, which also feeds the reference's
training rewards and termination. It does not modify physics, collision shapes,
reset cases, actions or evaluation thresholds. The separate native acceptance
must demonstrate unchanged physical probe trajectories, retained obstacle
positive controls and source-bound contact evidence. CPU Warp geometry tests
and local tensor tests cover the decision and failure paths. The separate native
qualification below verifies this revision's controls and baseline evaluation;
it does not demonstrate learning convergence.

## Reconstructed-scene baseline with fresh native controls

On 26 September 2026, the current static-range reference completed four fresh
native control processes and a fifth process that reloaded the original
1500-iteration warehouse checkpoint in the reconstructed collision scene.
Every process constructed all 4000 robots in one shared physics scene. The
same-placement repeat and coincident-peer comparisons both had maximum measured
state/observation delta 0.0, with zero peer contact. The focal free-space control
traveled 3.5502147 m; the obstacle positive control measured 2475.4917 N.

Independent readback verified actual sensor/body mappings, all 16000 spherical
foot identities and resolved offsets per control process, and contact evidence
against the source triangle mesh. The 398 original solo-control robot/interval obstacle
measurements corrected to floor support retained their raw contact evidence.
All 558 non-obstacle arrays matched the original solo trace exactly. These
checks validate the revised measurement and fresh-process protocol; they do
not show that warm resets clear native solver history.

The baseline evaluation retained the full 300-step horizon and 301 rendered
frames. Every frame passed trajectory, camera, native/Fabric clock and frozen
physical-state checks; all 4000 episode outcomes were independently recomputed.
The original checkpoint reached within 0.5 m in **759 / 4000 cases (18.975%)**,
so it **failed the fixed 80% policy requirement**. This is baseline replay,
with no reconstructed-scene policy updates or improvement claim. The earlier
99.925% warehouse result concerns a different scene, cohort, checkpoint and
metric/protocol version.

This native evidence binds source commit
`fd9fc2d254621112bf45b7df15fb8a80d0aba504`, navigation module digest
`0c70c463c94a4036c7262885eb6c14395621566c102ddedadceb4fcd52b797a1`,
and acceptance receipt SHA-256
`906f3a2043bf35ad8350f63fc4ab9f977e594726b6e71e5ae861ea8930f748a8`.
It qualifies static-range controls, checkpoint reload and native rendering for
that source. Camera-conditioned policies, reconstructed-scene fine-tuning and
held-out improvement still need their own measured results.

## Historical procedural warehouse learning

A native RTX PRO 6000 qualification constructed all 4000 robots in one warehouse
and passed repeated-reset and coincident-peer controls with zero measured focal
state/observation deltas and zero peer contact. The obstacle positive control
produced actual contact. Native PPO completed 1500 iterations and 48 million
control transitions in 6230.69 seconds of learning, averaging 7703.80 transitions
per second. The checkpoint changed by a finite, nonzero parameter norm and all
34,500 exported scalar observations were finite. Independently reloaded initial
and trained checkpoints each completed a 300-step evaluation of 4000 held-out
start/goal cases in the same warehouse, with 301 native frames and an MP4 per arm.
Independent readback recomputed every episode metric, checked every frame's
trajectory, camera and native-clock binding, and decoded both complete videos.

| Measured outcome | Initial checkpoint | After 1500 PPO iterations |
| --- | ---: | ---: |
| Goal success within 0.5 m | 0 / 4000 | 11 / 4000 (0.275%) |
| Episodes with obstacle contact | 3982 | 11 |
| Episodes with peer contact | 0 | 0 |
| Episodes with physical failure | 0 | 3 |
| Median final goal distance | 17.00 m | 0.635 m |

The trained policy avoided most collisions and approached goals, but 3978
episodes timed out; 3957 finished between 0.5 and 0.7 m from the goal. It did not
meet the unchanged 80% goal-success requirement. These measurements precede the
arrival-reward correction and do not establish a usable navigation policy.

The inherited position rewards pay repeatedly near a goal, while arrival ends
the training episode. Without a terminal success reward, this creates an
incentive to remain just outside the goal radius. The corrected reference adds
an arrival reward only when distance is below 0.5 m and there is no obstacle,
peer or physical failure. Its weight of 200 yields 40 reward units after the
pinned 0.2-second native scaling. This exceeds both the maximum 30 units of
position reward over a 30-second episode and its infinite discounted value of
20 at the pinned PPO discount of 0.99, including timeout bootstrapping. The bonus
is disabled for probes and evaluation, which do not automatically reset on goal
arrival. This removes the position-reward incentive to delay success; it does
not guarantee learning. Other reward terms, the
goal radius and the qualification threshold are unchanged. A new sealed native
training experiment completed 500 continuation iterations and 16 million new
control transitions in 4027.57 seconds of learning. Independent checkpoint
decoding verified that all 52 saved tensors (274,978 elements), optimizer state
and iteration state exactly matched the original checkpoint before updates.
The resulting policy changed by a finite parameter norm of 1.92235; all 12,000
exported scalar observations were finite. Goal success was measured separately
by reloading the resulting checkpoint.

The earlier comparison informed the objective correction, so its cases are now
development data. A fresh set of 4000 start/goal cases was sealed before the
continuation, with no overlap in seeds, identifiers or actual start/goal tuples
with training and development cases. Independent comparison of the original
1500-iteration policy and corrected continuation on this same fresh cohort
produced these results:

| Fresh test outcome | Original 1500-iteration policy | After 500 continuation iterations |
| --- | ---: | ---: |
| Goal success within 0.5 m | 14 / 4000 (0.350%) | 3997 / 4000 (99.925%) |
| Episodes with obstacle contact | 5 | 3 |
| Episodes with peer contact | 0 | 0 |
| Episodes with physical failure | 0 | 0 |
| Episodes ending at the time limit | 3981 | 0 |
| Median final goal distance | 0.634 m | 0.322 m |

These fresh-cohort results, verified on 26 September 2026, retain their original
contact metric and source:
commit `8998db8aaad36d21c33a16cb625ae0ed7861c70d`, with navigation module digest
`9a0cce1c8e4b7b9cca5431d06b8cf31488b600095b0c7e7c1833e40de6403aaf`.
They have not been recomputed or relabeled as results from the geometry-aware
foot-support revision.

The baseline's 14 successes differ from its earlier 11 because this is a fresh
cohort, evaluated using the same original checkpoint. The continuation passes
the unchanged 80% requirement. Both arms used identical inputs and evaluator
source, a 0.5 m radius and a maximum 300 control steps (60 simulated seconds).
Independent readback recomputed all 8000 episode records from raw trajectories
and verified native clock, camera and frozen-state evidence for every frame.
The baseline produced 301 frames; the candidate produced 41 because every
candidate episode had ended by step 40, without shortening the declared horizon.
Focal robot 0 timed out at 0.637 m for the baseline and succeeded at step 27
(5.4 simulated seconds), 0.244 m from its goal, for the candidate. The candidate
video's remaining 13 frames are outside that robot's scored episode.

The earlier checkpoint, comparison and videos remain evidence of the original
objective. These results qualify the public procedural warehouse and static-range
task; reconstructed-scene learning, private task parity and 4000 camera-conditioned
policies still require their own native qualification. Training wall time measures
aggregate control transitions; the simulated clock is not a per-robot real-time
throughput measurement.

Native training retains `reference_checkpoint.pt` before the first update,
`policy.pt` after learning, TensorBoard logs and exported `learning-curves.json`.
The training record reports measured wall time, native control-transition
throughput and peak PyTorch CUDA memory; runtime evidence includes the actual
GPU model, physical device memory, robot population and static collider count.
`comparison.compare_reference(training_prefix, output_prefix)` reloads both
checkpoints on the identical held-out inputs and publishes the measured gain.
For a newly initialized run, the reference is an untrained high-level navigation
policy above the public pretrained low-level controller. A resumed run instead
retains the loaded checkpoint as its pre-update reference. The public
qualification uses public task checkpoints; it does not establish parity with a
customer's existing policy.

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
time origin is measured within the evaluation process; it is distinct from the
video's relative rollout time and from every control process's clock. Historical
in-process runs qualified all 301 frames in the initial-checkpoint diagnostic,
all 602 frames of the original comparison and all 342 frames of the fresh
comparison. These results establish renderer and checkpoint reload behavior for
their original source and control protocol. The separate reconstructed-scene
fresh-process qualification verified another 301 frames; its 18.975% baseline
goal success remains below the policy requirement.
Artifacts include `rendered-rollout/*.png`, frame measurements,
and `rollout.mp4` when the runtime supplies FFmpeg. The PNG sequence can be encoded
later without rerunning or reconstructing the simulation. Custom BYOF tasks
retain their own visualization integration.

The focal robot continues stepping in the video while other robots finish their
episodes. Its score stops at its first goal arrival, obstacle/peer contact, or
physical failure. Present the focal robot's scored step count and outcome beside
the full video; later frames do not change that score or the cohort metrics.
The HUD's `Failed` value reports the separate physical-support/fall indicator;
an obstacle collision can end an episode while this indicator remains false.
The trailing camera can be occluded by scene geometry near walls. Preserve those
actual frames and the scored outcome rather than interpreting visibility as a
collision or success measurement.

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

`runtime.scene_instances` counts the verified direct reference to the supplied
warehouse asset. The built-in reference separately requires exactly one active
`UsdPhysics.Scene`. For a BYOF task, the operator must verify that the actual
physics-scene and simulation-owner mapping places all robot articulations and
static colliders in one shared physics world. The asset-reference check does
not verify that mapping.

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

Registering a `Camera` and passing the auxiliary focal render probe do not
establish that the policy uses images. Camera-conditioned qualification must
separately verify the actual `Camera.data` path through observations and any
encoder to the actor inputs, including the sensor/channel-to-environment mapping
and frame freshness for every claimed camera. The current checks do not enforce
that dependency.

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


### Diagnosing reference probe contacts

The built-in reference retains `probe-<name>-contacts/index.json` and one NPZ
per executed control step alongside each physical probe trace. The index uses
`npa.navigation.probe-contact-samples.v3`. These private artifacts retain the
strongest individual contact selected by the **original** normal/base rule for
each robot during the interval, including contacts subsequently recognized as
floor support. Ties keep the first sampled contact. This bounded selection is
not a complete history of every contact.

Each sample records actual native sensor/filter paths, signed force, world
contact point and normal, separation, body/root poses and native physics event
count/time. The index keeps actual sensor order and the original classifier's
robot/body indices separately. Support classification requires these identities
to agree. Native body and root paths independently select the copied poses.

The v3 index also records resolved foot offsets, each live sphere's collider
identity and transform, the witness method, source mesh hash and topology,
query capacity and reason codes. Selected foot samples retain the actual radius,
rest offset, source face, nearest point, distance and decision reason, plus
`terrain_witness_world_m` and `support_witness_valid`. Raw point, signed force,
normal and separation remain unchanged.
`original_candidate` and `effective_candidate` distinguish the two metrics.
Non-foot samples keep explicit geometry sentinels; their original native
separation remains in `separation_m`.
Historical v1/v2 artifacts retain their original meanings and must not be
decoded as v3.

`sample_tick_original_classified_sum_n` and
`sample_tick_effective_classified_sum_n`, with their corresponding contribution
counts, describe the selected sample's actual physics tick.
`control_interval_peak_effective_sum_n` separately records every robot's interval
maximum after the unchanged force threshold and must match the probe trajectory.
The strongest individual contact and the largest summed force can occur at
different ticks. Several small contacts can exceed the threshold together, so
these samples cannot reconstruct every contributing contact or removed force.

Evidence is written before the later probe success/failure checks, including a
partial control step if native stepping raises. The recorder runs only inside
physical probes and does not change native arrays, forces or physics settings.
Independent validation must check the exact sample schema, native identities,
resolved margins, source geometry and full-population interval metrics. Native
qualification additionally checks all physical controls and rendering; a local
decoder or a contact diagnostic alone is not a successful policy evaluation.
