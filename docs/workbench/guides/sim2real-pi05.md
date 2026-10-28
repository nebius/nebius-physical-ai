# pi0.5 released-surface Sim2Real variant

`workflows/testing/sim2real-pi05.yaml` is a separate declarative adaptation of
the compositional Sim2Real workflow. It does not change
`workflows/main/sim2real.yaml`, replace its PPO policy, or compare its airborne
goal task directly with released placement.

The Isaac, OpenPI, and viewer images must bake the same NPA source commit used
by the workflow manager. Publication checks both the OCI revision label and
`NPA_IMAGE_SOURCE_SHA`; native stage setup verifies the latter before running
the baked adapter. OpenPI identifies `/opt/venv/bin/python` as its baked NPA
interpreter through `NPA_BAKED_PYTHON`.

The public artifact-access preflight runs on CPU and publishes source and image
provenance without querying GPU devices. Training and checkpoint reload retain
their required live GPU provenance.

Component records use `npa.sim2real.pi05.component_record.v1`, with GPU device
evidence required for collection, both gold evaluations, training, and checkpoint
reload. The original Sim2Real graph retains its own schema and GPU stage map.
Consumers must select the expected workflow schema explicitly; the default
canonical reader rejects π0.5 records. Both schemas require immutable image,
source, and workflow-job provenance for every executed stage.

Policy-service startup waits for both Kubernetes pod readiness and a successful
health check through the private Service address. If the pod becomes ready before
Service routing has caught up, a refused connection remains in the existing
startup polling loop. Invalid health responses and other transport errors still
fail closed; checkpoint and hardware receipts are required before startup passes.

## What the graph proves

The workflow first runs a differential-IK Franka expert in Isaac physics. The
expert must establish bilateral finger/object contact, lift and transport the
object, open the fingers, retreat, and leave the object supported and stable on
the table for three consecutive control steps. It never writes object pose or
velocity. Failed attempts and their captured frames are retained but excluded
from training.

Every admitted row binds a causal exterior image, a calibrated wrist camera
whose world pose is recomputed from the live `panda_hand` physics pose, seven
measured joints, measured gripper width in the DROID
convention (`0=open`, `1=closed`), the controller's computed absolute joint
target, and simulator timestamps for image capture, command application, and
resulting state. The exporter forms 15-step targets only within episode
boundaries, computes normalization from the train split, and writes both the
OpenPI dataset contract and native LeRobot v3 data.

After the retreat waypoint, the expert records one additional 15-step horizon
of real controls and new camera observations. This keeps the original release
and retreat observations eligible as current model inputs, rather than losing
them at the end of a demonstration. Targets are never padded or copied from a
terminal frame. The capture receipt records the added steps, and exported
sample lineage records each input's `source_phase` for coverage audits.

Camera receipts preserve the SDK capture timestamp and the episode-local
observation timestamp separately. Causal age is evaluated in the reproduced
float32 sensor-clock domain; the raw float64/float32 representation delta is
also retained. Export rejects inconsistent clock bindings and frames older than
one 15 Hz control interval rather than clamping representation offsets.

The exact Isaac Lab 3.0 camera contract uses XYZW quaternions, live pose
updates, and ProxyArray-to-tensor conversion before indexing. Its RTX backend
consumes USD camera transforms while Fabric owns the moving articulation, so
the adapter publishes the calibrated live-hand pose to both a world-pose USD
camera and CameraData before every render. The wrist uses a calibrated
hand-local side bracket at `(0.09, 0, 0.015)` m; its ROS +Z optical axis aims at
the measured finger-link reference near hand +Z, with an 18 mm focal length.
This keeps the measured default size variants' closest faces outside the 30 mm
near clip by construction, but only actual rendered object and bilateral-finger
pixels qualify it. Runtime evidence must prove the measured optical axis points
toward USD `ComputeUntransformedBound` finger geometry transformed by current Gen3
`ArticulationData.body_pos_w/body_quat_w` physics buffers, and that object and
both fingers have actual semantic pixels in the relevant
approach-through-release phases. The exterior camera is similarly derived from
a target-side oblique static look-at pose, separating the source-side object
and destination in depth while retaining side faces, rather than retaining an
ambiguous legacy quaternion. That rationale is geometric only; its object and
destination pixels still require live qualification.
Projected finger bounds are labelled only as in-frustum geometry; they are not
treated as rendered visibility and cannot satisfy the pixel gate.

Pinned RTX output represents semantic pixels as signed `int32` values whose
bits are little-endian packed RGBA codes. Each code is accepted only through an
exact, bijective `idToLabels` RGBA-to-class mapping; unmapped, malformed, or
ambiguous metadata fails closed. These codes are not advertised as numeric
class IDs and RGB color similarity is never used as semantic evidence.

Measured Franka finger separation remains a raw physics value for contact and
success metrics. The observation adapter alone saturates widths between 80 and
85 mm to DROID state `0.0` (open), matching the collector's bounded actuator
overshoot allowance and the policy path. Larger, negative, non-finite, or
out-of-range normalized states fail closed before export or normalization.

The destination is a rendered, non-colliding cyan disc centered on the declared
surface target with the same 5 cm radius as the success threshold. The prompt
names that cue, and metadata-bound semantic pixels in the exterior camera must
prove it was present in the causal source frames. It is scene geometry, not a
video overlay, and it never writes object state. The current fixed destination
can be learned during adaptation; it does not establish generalization to new
or moving destination locations.

### Default simulation asset scope

The checked-in train/validation/gold split is one Isaac multicolor-cube USD
whose runtime-composed source extent is measured as 60 mm per axis, scaled to
48, 60, and 72 mm respectively. Every run records the measured source and
effective extents and uses the effective grasp width/support height, including
for an operator-supplied USD whose geometry is centered on its rigid root,
instead of assuming a 50 mm cube. Offset-origin assets fail closed until an
explicit grasp-frame adapter is supplied. It is a
size/morphology holdout within one
asset family, not evidence for novel object categories or another robot
embodiment. All three splits use the same task-owned, single-rigid-body support
surface layout; this replaces the stock multi-body table so object/support
contact filtering is body-exact on the pinned PhysX backend. The inherited
Isaac `reset_root_state_uniform` event is explicitly rebound in both expert
and policy configs so its relative offsets produce world `x=[0.42,0.55]`,
`y=[-0.25,-0.08]`; admitted rows must fall inside those ranges. Separate seeds
exercise the configured pose, mass, and friction randomization; their old
human-readable scene labels did not create independently authored scenes and
are no longer represented as such. Source-derived asset diagrams must remain
labelled as schematics until actual simulator frames exist.

Each expert and policy episode begins with the same two 15 Hz control intervals
outside the training/evaluation rows. The receipt records both object poses,
speeds, support forces, and episode-local command/result clocks; admission
requires the second sample to be supported and below 0.03 m/s. The expert also
measures the complete two-finger link interval and its centroid in the live
`panda_hand` frame, then transforms those offsets through the commanded downward
hand orientation. This is explicitly link geometry, not a claimed pad center;
live bilateral contact is still required. Contact and lower waypoints are
size-aware: the proximal measured finger-link plane stays 8 mm above the
measured object top while the rest of the finger interval overlaps the object
side. The receipt records the measured interval, object height, clearance,
reference plane, and resulting hand-origin target. Contact and grasp phase
advancement also requires the live transformed complete-link proximal bound to
retain that 8 mm clearance; the 12 mm Cartesian convergence tolerance cannot
consume it. This remains complete-link evidence rather than a pad or palm
collision-geometry claim, so live 48/60/72 mm contact qualification remains
required without relaxing waypoint or success thresholds.
The aligned clearance trace is required again at dataset admission and at the
ordered contact/grasp samples, so older episodes cannot bypass the new phase
evidence during OpenPI or LeRobot export.
The stock task's frame sensor and differential-IK action otherwise use distinct
nonzero tool offsets from `panda_hand`. This variant explicitly sets both to the
hand origin before environment construction, applies the same sensor-frame
definition to policy evaluation, and records live hand/sensor/control-point
positions, quaternions, configured offsets, and target prim identity. It fails
before collection if those frames do not coincide.
Both pi0.5 environments also select the pinned upstream
`FRANKA_PANDA_HIGH_PD_CFG`: robot gravity is disabled and the upstream
high-PD actuator plant is shared while the expert retains relative IK and the
policy retains absolute joint targets. Collection records the live actuator
joint membership, stiffness, damping, and rigid-body gravity setting; policy
evaluation rejects a gold protocol unless its runtime plant receipt is equal.

The base policy uses the pinned Polaris config and checkpoint documented in
[the OpenPI pi0.5 guide](../openpi-pi05-polaris.md). The adapted service reconstructs
the training LoRA model, joint delta/absolute transforms, and train-only statistics.
Its `start` call requires `--dataset-uri`, `--dataset-manifest-uri`, `--training-uri`,
and `--evaluation-uri`; the workflow supplies these artifacts automatically.
The service verifies their checkpoint and dataset lineage, loads the selected
`<step>/params` directory, and checks saved `<step>/assets/droid` normalization
against the training data before starting. Its receipt records the loaded step,
model variants and dataset digest. Private checkpoint staging updates the
server's `OPENPI_CHECKPOINT_URI` environment binding to the downloaded directory.
The launcher retains the shared quoted Python command, so shell parsing cannot
strip quotes from an inserted local path. Cleanup needs no dataset or model imports.
They are evaluated with the same gold object, scenario seeds, instruction,
15 Hz absolute-joint controller, and eight-action receding-horizon prefix. A
continuous policy gripper value is explicitly thresholded at `>0.5`, matching
the upstream DROID execution example, then checked against both physical finger
controller targets. Finite model scores outside `[0, 1]` remain valid inputs to
that binary threshold and are retained without clamping; the measured
observation and training state remain strictly normalized to `[0, 1]`. The
final report permits an improvement claim only when the
two gold protocol digests and checkpoint train/reload hashes agree.

The pinned Isaac joint action is an affine bridge, `raw * scale + offset`; its
stock configuration has no raw-action clip. The adapter therefore admits any
finite inverse-affine value whose resulting absolute target stays within the
physical Franka joint limits. It rejects a configured SDK clip instead of
inventing a normalized `[-1, 1]` range or silently changing a learned target.
Each model response and action hash is recorded before controller execution,
then marked completed or failed with attempted and verified prefix lengths.
Incomplete execution remains failure evidence and is ineligible for a matched
wrist-effectiveness claim.

Each closed-loop checkpoint evaluation can run matched `fresh`, `frozen`, and
deterministic `pixel_channel_shuffle` wrist arms from the same held-out scenario
seeds, initial object poses, checkpoint, exterior/state contract, and
instruction. The legacy mode identifier flattens the complete HWC tensor and
permutes scalar pixel values across spatial and channel positions; it is a
pixel-value shuffle, not an RGB-channel-only or temporal-frame permutation.
This pipeline's private service uses an NPA-owned wrapper around
the pinned real `Policy.infer(observation, noise=noise)` API. It validates the
padded model noise shape from the selected config and returns only its hash,
shape, and sequence identity. Reports reject a matched-effectiveness claim if
those identities do not pair at every decision. Generic upstream serving stays
unchanged. The scoped wrapper requires a bearer token from an operator-managed
Kubernetes Secret; the same Secret is mounted into the Isaac evaluator, and no
token is written to the service receipt, request receipt, or image.
Pixel-value shuffle is explicitly an out-of-distribution
corruption, not an unrelated real wrist frame. The separate local held-out
diagnostic also pairs explicit noise and proves only action sensitivity and
transform/mask wiring, not helpful task use.

Closed-loop review artifacts also contain an independent static witness camera.
It is never a policy input. Its uncut video and Rerun entities bind every
pre-command observation plus a post-final-control frame to the same simulator
clock and physics trace. A latched placement claim additionally records a
15-control terminal dwell with open fingers, zero finger force, table support,
the unchanged 5 cm goal criterion, low speed, and hand retreat. This witness is
evaluation instrumentation; replayed footage must be labelled replay and cannot
replace a new policy trial.

## Required operator inputs

The checked-in spec is intentionally non-runnable until an operator supplies:

- a writable bucket and fresh prefix;
- exact immutable CPU, Isaac, and OpenPI images carrying the selected NPA
  source revision;
- the read-only Isaac cache PVC and two distinct run-scoped controller
  ServiceAccount/Role/RoleBinding sets, one each for `<run-id>-base` and
  `<run-id>-adapted`; generate each with
  `build_controller_rbac_manifests` and its exact suffixed run ID, then pass
  the resulting account names as `base_service_account` and
  `adapted_service_account`;
- an operator-managed Kubernetes Secret whose `auth_token` key authenticates
  the private policy data path;
- healthy, separately routed capacity for Isaac, OpenPI training, and the
  one-GPU policy service; and
- anonymous, generation-bound access to the exact public Polaris GCS objects,
  or the operator's own credential and exact-revision payload access if an
  explicitly gated Hugging Face artifact is selected instead.

On a fresh Kubernetes cluster, the configured SkyPilot worker ServiceAccount
must exist before image-pull preflight. SkyPilot 0.12.2 uses
`skypilot-service-account` by default and installs its standard RBAC during
native bootstrap. If that account is absent, create it in the selected target:

```bash
kubectl --context '<selected-context>' --namespace '<selected-namespace>' \
  create serviceaccount skypilot-service-account
```

The policy authentication token enters camera-bearing pods through
`valueFrom.secretKeyRef`. Submission receipts retain the Secret name and key;
the token remains in the operator-managed Secret. The ledger accepts validated
selectors and rejects malformed selectors or embedded credential fields.

The `source_sha` variable identifies the workflow consumer source. Normally it
also matches every baked runtime image. The SmolVLA interchange reference
exposes `cpu_image_source_sha`, `isaac_image_source_sha`, and
`runtime_image_source_sha` as ordinary scalar config inputs alongside the three
digest-pinned image inputs. They default to `source_sha`; when a reviewed
workflow reuses an unchanged image from an older source, override the matching
source input with that image's actual baked source SHA. The renderer resolves
those pairs into the fail-closed image-source map before provisioning. Setup
then checks each image against its own SHA without relabelling its provenance
as the newer consumer source. Conflicting identities fail rendering. Record
the operator-side submission client's revision separately and retain every
runtime image's signed source provenance.

The Isaac dependency closure remains mounted read-only. Camera-bearing jobs
separately mount a pod-local writable `emptyDir` at `/tmp/npa-isaac-state` for
Kit portable state, XDG data, logs, and shader caches. Both collector and gold
evaluator pass that validated portable root into `AppLauncher`; a path outside
the job-state mount, an escaped path, or duplicate portable-root option fails
before Kit starts.

The outer collection and evaluation commands run in the image's baked NPA
interpreter. Each Isaac child must launch through `/isaac-sim/python.sh`, which
selects the verified SDK cache interpreter and applies Kit privacy and writable
state settings. Reusing the parent's `sys.executable` bypasses that handoff and
cannot load Isaac Lab. A failed launcher stops the stage; the evaluator also
requires a complete successful report read back by its outer process.
The collector also reads each uploaded `collection.json` outside Kit rather
than parsing child stdout: Kit shutdown can exit before the CLI prints its
report. It checks the requested asset, scale, scene, episode count, and seeds,
and validates the dense causal rows of every claimed success. A zero child exit
without those durable report bytes cannot complete the collection stage.

The reference held-out model evaluation uses batch size one. This deliberately
covers every exported validation sample, including the observed 695-sample
corpus, without dropping or padding a tail batch. An operator override may use a
larger fixed batch only when it divides the held-out sample count and the
allocated device count exactly; evaluation rejects any mismatch before fetching
the trained checkpoint.

Held-out model loss compiles the checkpoint's frozen model state once and
evaluates every validation sample with the declared batch size and seed.
Training, held-out loss, and held-out action prediction report completed counts
as `npa.workbench.openpi.progress.v1` JSON records on stderr. Use the stage logs
to follow these counts; the final stdout JSON remains the stage result.

NPA adds no OpenPI/Gemma consent variable. Upstream terms remain the operator's
responsibility through use. The Isaac expert can be qualified independently
under the existing Isaac product routing.

### Smallest exact-source image set

The 14 states require three distinct exact-source images, not five builds. The
spec binds every CPU role to one `cpu_image`; use the `rerun-viewer` digest for
that value. The image carries the controller/Kubernetes closure, PyArrow
and FFmpeg dataset conversion closure, and Rerun finalization closure. Use the
`isaac-lab` digest only for the three camera-bearing physics states, and the
`openpi` digest for artifact preflight, training, held-out evaluation, and the
private model Deployment. Their build recipes are respectively
`npa/docker/workbench/rerun-viewer/Dockerfile`,
`npa/docker/workbench/isaac-lab/Dockerfile` (or its `build.sh` wrapper), and
`npa/docker/workbench/openpi/Dockerfile`.

After those exact-source bytes pass the secure development-image workflow, use:

```bash
--var cpu_image='<rerun-viewer@sha256:digest>' \
--var isaac_image='<isaac-lab@sha256:digest>' \
--var runtime_image='<openpi@sha256:digest>'
```

This reuse is a dependency-closure choice, not evidence that any image or the
whole workflow has passed live qualification. The OpenPI recipe gates all
model-side module entrypoints in its upstream environment and keeps the minimal
Rerun worker isolated; the Rerun Viewer recipe gates all three CPU roles. A
developer-venv import does not substitute for those Docker `RUN` checks. The
Isaac recipe separately runs the real msgpack/WebSocket image codec and creates,
decodes, and removes a synthetic MP4/Rerun pair in its own interpreter. The
workflow's non-Kit evaluator parent also reads back the declared report and
rejects a missing, failure-schema, or incomplete result even if Kit exits zero.

The service receipt records both the upstream base configuration and the actual
selected LoRA configuration. The final comparison admits their different names
only with proof of the selected load, the expected LoRA architecture and action
horizon, and saved normalization verified against training-only data. Source,
image, task protocol, camera, noise, physics, and checkpoint checks still apply.
The report exposes both configuration names and rejects unsupported comparisons.

## One or two eight-GPU RTX PRO 6000 nodes

The default spec is **not** an RTX-only qualification. It routes OpenPI
training and serving to one B200 (`sm_100`) and Isaac rendering to one RTX PRO
6000 (`sm_120`). The OpenPI and Isaac images, compiled probes, CUDA/JAX
dependencies, scheduler aliases, and Kubernetes product labels must all be
qualified again for RTX PRO 6000 before execution can be called supported.

One eight-GPU RTX PRO 6000 node has enough GPU slots for the graph's current
shape: training can use up to eight visible GPUs after the base service has
been cleaned up, while each closed-loop evaluation needs one policy-service
GPU plus one Isaac GPU concurrently. CPU, RAM, cache/PVC, image pullability,
and actual scheduler placement still require preflight. Training, serving, and
Isaac counts are intentionally separate; setting one shared count to eight
would make the service reserve the whole node and starve its concurrent Isaac
client.

An RTX assessment render can use these explicit overrides after immutable
SM120-compatible images are available:

```bash
--var training_gpu_type=RTXPRO6000 \
--var training_gpu_count=8 \
--var training_expected_gpu_type='RTX PRO 6000' \
--var training_expected_compute_capability=12.0 \
--var training_num_nodes=1 \
--var train_fsdp_devices=8 \
--var service_gpu_count=1 \
--var service_expected_gpu_type='RTX PRO 6000' \
--var service_expected_compute_capability=12.0 \
--var service_gpu_node_selector_value=RTX6000 \
--var isaac_gpu_type=RTXPRO6000
```

`openpi.training.sharding.make_mesh` can mesh the eight visible devices on one
host. This variant now fails closed if `SKYPILOT_NUM_NODES` or JAX reports more
than one host because it does not initialize multi-host JAX. A second 8-GPU
node can provide scheduling isolation or spare capacity, but setting
`training_num_nodes=2` does not create valid 16-GPU training. Use the separate
full-DROID multi-host recipe for that topology.

Validate and render without provisioning:

```bash
npa workbench workflow validate-spec workflows/testing/sim2real-pi05.yaml
npa workbench workflow plan-spec workflows/testing/sim2real-pi05.yaml \
  --run-id '<run-id>' --json --check-render \
  --var bucket='<owned-bucket>' --var source_sha='<full-source-sha>' \
  --var cpu_image='<immutable-rerun-viewer-image>' \
  --var isaac_image='<immutable-image>' --var runtime_image='<immutable-image>' \
  --var isaac_cache_pvc='<read-only-cache-pvc>'
```

Run health and exact access checks before submission. Use the standard runtime
submit path; the spec declares service start, Isaac clients, normal-path cleanup,
durable S3 handoffs, and resume behavior. Its top-level `onFailure` group attempts
both exact-owned service cleanups after a terminal payload failure and records a
cleanup failure separately without replacing the original error. A terminally
failed run whose cleanup completed requires a new run ID; `--resume` is reserved
for an interrupted non-terminal driver and adopts its durable waves. Do not use
local `run-spec --execute` as a substitute for a SkyPilot/Kubernetes acceptance
run.

The Isaac collector fails before Kit startup unless the container can load the
injected NVIDIA GLX/EGL libraries and `vulkaninfo` enumerates an NVIDIA physical
device. A passing CUDA compute probe alone does not qualify RTX rendering.
Startup/reset tracebacks are uploaded before the app is closed, and a successful
process or Kubernetes status is never accepted without a valid collection.
The spec applies `config.isaac_runtime_class` (default `nvidia`) only to the
camera-bearing Isaac resource. Override it only with a runtime handler qualified
for that same graphics path; CPU and OpenPI service resources do not inherit it.

## Outputs and claim boundary

The graph declares physics collection reports, dense OpenPI and LeRobot data,
the trained checkpoint manifest, exact reload evaluation, base/adapted gold
reports, MP4, Rerun recording, and a final factual report. MP4 and Rerun are
decoded after writing. Successful orchestration is reported separately from
task success.

Physical robot deployment remains an explicit operator seam. DROID-shaped
Franka observations are useful for integration, but neither simulation success
nor Polaris loading establishes safe zero-shot transfer to an unqualified robot.

## Policy and embodiment interchange

The 14-stage collect → export → baseline → native adaptation → exact reload →
same-task gold → cleanup → factual proof structure is a useful reusable design.
The original pi0.5 implementation is not itself embodiment-neutral: Panda joint
names, two physical finger joints, DROID aliases and padding, a calibrated Panda
hand camera, and OpenPI's JAX/LoRA/noise protocol remain in its adapter.

The interchange boundary keeps those details out of the shared contract:

- the robot profile binds an immutable canonical robot bundle, ordered physical
  arm and gripper joints, limits, home pose, base/end-effector/finger links,
  controller representation, semantic gripper scalar, and calibrated cameras;
- the rigid-object profile binds exact licensed USD bytes, units, collision,
  inertia, material, root/reference/grasp transforms and grasp extents; and
- the policy profile binds timestamped camera roles, ordered measured state and
  units, exact action representation/names/units/frame, gripper semantics,
  horizon, cadence, normalization, checkpoint, and backend capabilities.

Each substitution changes one named profile and its adapter while retaining the
standard workflow runtime:

| Swap | Change | Held fixed | Pre-submit rejection |
| --- | --- | --- | --- |
| Policy | `policy_backend` and that backend's export/train/reload/serve adapters | robot, object, cameras, task scorer and sealed evaluation cohort | unsupported action names/order/units/frame, horizon, controller mapping or reload binding |
| Robot/controller | resolved robot bundle plus `robot_profile` | policy contract, rigid-object contract and task protocol where compatible | joint/link mismatch, unsupported gripper, missing limits/home pose, or controller representation without an implementation |
| Object | sealed rigid-asset bundle plus `object_profile` grasp/reference transforms | robot, sensors, policy and placement-event scorer | articulated/deformable asset, missing collision/inertia/material data, or incompatible grasp width |
| Sensors | calibrated camera entries in `robot_profile` | remaining robot morphology, task and policy backend | unknown mount link, stale capture/state clocks, missing mask/layout, or absent exterior/wrist role |
| Data/evaluation | backend-native exporter and sealed cohort identities | physical controller semantics and scorer for a comparison | train/evaluation lineage overlap, non-training normalization, or incomparable task protocol |

The checked-in non-Franka reference requests 12 CPUs for each Isaac stage. That
value includes controller headroom and is intentionally below the observed
node's allocatable ceiling; a rendered plan must still be checked before every
submission. Runtime resource overrides may narrow a request, but cannot change
robot, object, policy, cohort, or scorer identity.

`sim2real-policy-interchange-smolvla.yaml` is the model-neutral 16-stage
reference. After initial profile validation, it creates or exactly adopts
run-owned policy and Isaac cache PVCs, an authentication secret and
role-specific service RBAC, then warms the digest-pinned Isaac runtime. Its
terminal stage deletes only those exact owned resources after the report is
durable. Interrupted execution adopts the same identities and rejects foreign
or drifted resources. The middle stages use
real LeRobotDataset materialization and native SmolVLA
training, processor-bound reload and typed serving. OpenPI/DROID translation,
padding, MessagePack/WebSocket and explicit sampler-noise coupling remain only
inside the pi0.5 adapter. SmolVLA instead declares its native action queue and
requires independently seeded repeated paired task trials. The reference
declares `smolvla_action_selection: uniform_temporal_ensemble`: each control
uses the mean of only time-aligned predictions from the current and still-live
prior native chunks, and the current fresh observation always contributes
action zero. `first_action` remains an explicit negative-baseline option; the
two protocols must never be combined under one result label.

The clean-start lifecycle is implemented but has not yet passed a complete live
run or interrupted-run resume. This reference is therefore not yet a supported
turnkey default. The normal product
surface will be the standard `npa workbench workflow submit ... --runtime
--resume` path after the clean-start image, profile, storage, private service,
strict held-out task-success, cleanup, and interrupted-run resume checks have
all passed in one source-bound run. Until then, operators must treat it as a
qualification workflow rather than a supported one-command adaptation recipe.

`sim2real-policy-interchange-non-franka.yaml` is the second complete 14-stage
acceptance graph. It collects successful geometry-aware teacher episodes for
all three splits, converts only the training split to LeRobotDataset, runs base
and adapted policies against the same sealed physical protocol, performs native
SmolVLA training and processor-bound reload, and retains cleanup and factual
finalization as explicit stages. Its Isaac bridge derives the action dimension
from the declared robot profile, rejects reordered or missing joints and links,
maps the declared semantic gripper value to physical driver joints, republishes
the moving wrist camera from its calibrated end-effector mount before every
capture, and records actual controller targets and causal timestamps. The
unchanged event order is contact → grasp → lift → transport → release → support
contact → stable placement. A complete action trace proves execution validity;
only all ordered measured events prove task success. One paired execution is
not treated as a statistically powered improvement comparison.

Current support matrix:

| Combination | Source contract | Native execution | Task success | Claim |
| --- | --- | --- | --- | --- |
| Franka + cube-size family + pi0.5 | qualified | completed | adapted fresh 1/8; baseline provenance invalid | exercised only; no valid improvement or wrist-benefit comparison |
| Franka + cube-size family + SmolVLA | implemented; exact-source DEV image bytes independently verified | native task pending | unmeasured | unsupported pending qualification |
| UR10e/Robotiq + offset-origin spool + SmolVLA | implemented six-axis/tool/calibration bridge; exact-source DEV image bytes independently verified | first conversion returned success without its declared manifest; successor makes the hidden conversion failure observable | unmeasured | unsupported pending successful conversion and full run |
| Historical UR10e/Robotiq + spool + PPO | exercised on native Isaac | 0/1024 lifts | failed | diagnostic only; privileged PPO state is not a VLA input |
| Articulated/deformable props, mobile bases, non-parallel-jaw grippers, Cartesian policies without qualified IK | rejected or unimplemented | not run | unmeasured | unsupported |

Cube sizes are one shape family, not object-category generalization. Simulation
does not qualify physical-robot deployment, and model-specific imitation losses
must not be used to rank different policy backends. Update this table only from
immutable source/image/robot/object/dataset/checkpoint/seed evidence and retain
failed combinations rather than converting unavailable metrics to zero.

## Policy-camera evidence and recovery ownership

Closed-loop review artifacts distinguish source sensor pixels from the pixels
actually consumed by the policy. The lossless policy-input archive retains the
workspace and wrist arrays before video encoding, including frozen or
pixel-value-shuffled wrist interventions. The review MP4 and Rerun timeline show
the consumed workspace and wrist inputs beside both independent witness views,
with observation, command and resulting-state clocks plus the applied action.
Video remains a visualization; the array and per-query hashes are authoritative.

Nominal successful demonstrations do not establish learned recovery behavior.
The recovery contract therefore classifies command/measurement blockage,
motion without task progress and poor workspace/wrist visibility separately.
Any view-seeking fallback is a distinct `recovery_planner` action owner. It must
bind the resolved robot joint order and limits, calibrated camera extrinsics and
collision model, and it may not release an active grasp. The frozen baseline
declares recovery disabled. Enabling a planner is a separate one-component
experiment and its actions must never be attributed to the VLA.
