# pi0.5 released-surface Sim2Real variant

`workflows/testing/sim2real-pi05.yaml` is a separate declarative adaptation of
the compositional Sim2Real workflow. It does not change
`workflows/main/sim2real.yaml`, replace its PPO policy, or compare its airborne
goal task directly with released placement.

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

The base and adapted policies use the pinned OpenPI source, Polaris config, and
checkpoint documented in [the OpenPI pi0.5 guide](../openpi-pi05-polaris.md).
They are evaluated with the same gold object, scenario seeds, instruction,
15 Hz absolute-joint controller, and eight-action receding-horizon prefix. A
continuous policy gripper value is explicitly thresholded at `>0.5`, matching
the upstream DROID execution example, then checked against both physical finger
controller targets. The final report permits an improvement claim only when the
two gold protocol digests and checkpoint train/reload hashes agree.

Each closed-loop checkpoint evaluation can run matched `fresh`, `frozen`, and
deterministic `pixel_channel_shuffle` wrist arms from the same held-out scenario
seeds, initial object poses, checkpoint, exterior/state contract, and
instruction. This pipeline's private service uses an NPA-owned wrapper around
the pinned real `Policy.infer(observation, noise=noise)` API. It validates the
padded model noise shape from the selected config and returns only its hash,
shape, and sequence identity. Reports reject a matched-effectiveness claim if
those identities do not pair at every decision. Generic upstream serving stays
unchanged. The scoped wrapper requires a bearer token from an operator-managed
Kubernetes Secret; the same Secret is mounted into the Isaac evaluator, and no
token is written to the service receipt, request receipt, or image.
Pixel/channel shuffle is explicitly an out-of-distribution
corruption, not an unrelated real wrist frame. The separate local held-out
diagnostic also pairs explicit noise and proves only action sensitivity and
transform/mask wiring, not helpful task use.

## Required operator inputs

The checked-in spec is intentionally non-runnable until an operator supplies:

- a writable bucket and fresh prefix;
- exact immutable CPU, Isaac, and OpenPI images carrying the selected NPA
  source revision;
- the read-only Isaac cache PVC and scoped service account/RBAC;
- an operator-managed Kubernetes Secret whose `auth_token` key authenticates
  the private policy data path;
- healthy, separately routed capacity for Isaac, OpenPI training, and the
  one-GPU policy service; and
- anonymous, generation-bound access to the exact public Polaris GCS objects,
  or the operator's own credential and exact-revision payload access if an
  explicitly gated Hugging Face artifact is selected instead.

The Isaac dependency closure remains mounted read-only. Camera-bearing jobs
separately mount a pod-local writable `emptyDir` at `/tmp/npa-isaac-state` for
Kit portable state, XDG data, logs, and shader caches. Both collector and gold
evaluator pass that validated portable root into `AppLauncher`; a path outside
the job-state mount, an escaped path, or duplicate portable-root option fails
before Kit starts.

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
durable S3 handoffs, and resume behavior. The current workflow interpreter has
no `finally`/`on_failure` transition, so an ungracefully lost downstream Isaac
Job does not prove service teardown. Before a live service run, operate a scoped
owner-only failure monitor that invokes the exact cleanup toolRef, or first add
and validate a real failure transition. Do not use local `run-spec --execute` as
a substitute for a SkyPilot/Kubernetes acceptance run.

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
