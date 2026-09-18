# Antioch + OpenPI live example

This public-source project runs a real Isaac Sim Franka scene in an Antioch
livestream and sends its two current 224x224 camera frames and robot state to an
external OpenPI pi0.5 DROID policy. It applies only validated finite `[15, 8]`
chunks, enters safe hold on stale/malformed/unsafe responses, and reconnects with
bounded exponential backoff. Antioch telemetry and a viewport overlay report the
live counters; neither is reconstructed from a recording.

An independent 1280x720 RTX camera records the real robot and tabletop at the
same completed render steps. The viewer gives this native HD view the largest
pane and keeps both policy inputs visible alongside it. `showcase-frames.zip`
retains JPEG frames, exact producer markers, simulation timestamps, and checksums
for video assembly after execution. JPEG compression is the only image treatment;
the recording camera does not supply policy observations or advance physics.
Black-only or stale recordings fail the recording check. The camera framing and
rendering cost still require live validation on the selected GPU.

The client uses a 90-second response-age safety deadline because a cold request
can take tens of seconds even though warmed supported-GPU requests are normally tens of
milliseconds. The reviewed `pi05_droid_jointpos_polaris` output contract is seven
absolute arm joints plus one DROID gripper-position command. The pickup scenario
selects Isaac's native Franka **Robotiq 2F-85** accessory and the DROID reset
posture. Seven arm joints and the active `finger_joint` are bound by name;
PhysX controls the passive mimic joints. The observed gripper angle is normalized
from zero to pi/4 into DROID's `0=open, 1=closed` convention. Model commands are
binarized at 0.5 and mapped back to that actuator angle. The older communication
scenario retains the stock Panda fingers and their inverse opening mapping.
Raw out-of-distribution joint/gripper
counts are reported separately from Franka-limit and per-target-step safety
projections. Each receding-horizon query executes five returned targets at 15 Hz of simulation time. Wall time remains the authority for transport staleness. The
observation-to-action loop is best-effort and not hard real time.

The manipulation scene is a lit tabletop with a reachable red cube and an open
Franka in the DROID reset posture. Both policy views use Isaac Sim 6's supported
`isaacsim.sensors.experimental.rtx` authoring/runtime split: an independent
`RtxCamera(tick_rate=15.0)` and `CameraSensor(annotators=["rgb"])` per view.
After scene reset, the scenario commits timeline play with Isaac's public app
utility. Each control tick then uses exactly one `world.step(render=True)` for
physics, the streamed viewport, and the attached policy-camera render products
before reading both sensors. It does not add a second Replicator orchestrator
step, which can invalidate the RGB render-var lifecycle. Acquisition consumes the copied
numpy/Warp result and producer metadata from `get_data("rgb")`. Each
sample copies the completed sensor buffer into scenario-owned memory; a public
clock attached to that sensor's render product supplies the exact producer marker.
The pickup uses native 320x180 RGB inputs, resized to 224x126 and centered in a
224x224 black-padded tensor, preserving the wide cameras' aspect ratio. Exposure
and contrast checks inspect only the content rows, so padding cannot disguise a
blank or overexposed sensor. Target resolution checks use the exact model pixels.

The fixed exterior camera views the table over the robot's shoulder. A fixed
wrist camera inherits the Robotiq body transform; it never tracks the cube.
Extrinsics and optics follow the public
[NVIDIA RoboLab DROID reference](https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/robolab/robots/droid.py),
with a -90-degree local-Y basis correction for Isaac's native wrist accessory.
No RoboLab robot assets are redistributed. Diagnostic renders verify the initial
and approach framing; pickup success still requires the complete live policy run.
Both cameras must initially contain the target and grasp region geometrically,
and both must visibly resolve the red target before inference.

The renderer uses fixed exposure (ISO 100, shutter 50, f-number 4, automatic
exposure disabled), lower light intensities, and a dark ground plane. These are
scene-specific authored starting settings, not a claim of live calibration.
The **actual RGB arrays** must pass brightness, contrast and framing checks:
mean intensity above 5 and at most 220; variance above 25; 95th–5th percentile
range at least 32; and at most 60% near-white pixels (mean RGB above 240).
The initial red-target proxy must occupy at least 64 pixels and span at least
8 pixels in both dimensions in each view. This color mask is not semantic
segmentation. During motion, the target must remain resolved in at least one
view, except during a measured bilateral grasp with commanded closure. Freshness, distinct-view and
exterior-frustum checks still apply. A missing wrist target alone does not stop
motion when the exterior view remains useful. Rejected images remain inspectable;
no postprocessing brightens or recolors policy inputs to make the gate pass.

The checked-in project ID is deliberately unusable. The cluster-native controller
creates a private runtime copy with an assigned Antioch project ID, starts the
supported sim service, and copies a 0600 run bundle into the running sim service
with `antioch service cp`. The bundle contains the cluster-local policy gateway
CA/API key/endpoint plus a separate short-lived CA, certificate, key, and API key
for the service-side bridge. Credentials are never passed through scenario
parameters, Kubernetes arguments/annotations, Git, or images.

The sim declares an Antioch-managed port that is reachable only at the adapter
pod's localhost while services are up. A bounded authenticated WSS rendezvous runs in
the persistent `sim` service. The streamed scenario connects to its `simulation`
role first; the same Kubernetes pod's bounded relay connects to its
`operator` role and only then connects to the selected policy gateway by verified
WSS through a ClusterIP Service on port 443. Both controller and relay are
containers in one MK8s pod, so the operator VM is not in the frame/action path.
This double-WSS route is not a public unauthenticated proxy. Both legs reconnect
independently and the relay writes only fixed counters and error classes to its
private state file.

The project Dockerfile adds only pinned `msgpack` and `websockets` wire-protocol
dependencies to Antioch's version-matched Isaac Sim base. The small local codec
is adapted from the pinned Apache-2.0 OpenPI client and rejects object arrays;
neither OpenPI model code nor weights are included in the sim image.
The controller copies the reviewed scenario, codec, and bounded WSS bridge through
supported `service cp` and verifies their readability before dispatch. Because the
current CLI restricts copies to `/workspace/project`, it uploads there and uses
`service exec` to install the private bundle as a 0600 generation under `/tmp`.
The controller builds one immutable project revision and starts that exact revision
as the project session.

The default cluster scenario is `openpi_franka_pickup_v3`. Two valid replies
establish communication only. Pickup requires at least 5 cm of measured approach,
force on **both** fingers, and 5 cm of cube lift held continuously for one
**simulation second** while the gripper is measurably closed from its open width.
The closure check accommodates the 7 cm cube; it does not incorrectly require
a finger gap below 4 cm. Both initial action segments must execute before task
success can be reported.

`control_steps` is the finite trial length in applied policy targets, default
450 (30 nominal seconds of target intervals, excluding inference waits).
Exhaustion without physical pickup produces a failed task with saved evidence.
Cold streamed-renderer startup has a separate 600-second readiness bound: shader
initialization can block a rendered step for several minutes. Until both native
policy buffers and producer clocks appear, acquisition checks every completed
render rather than waiting for the next 15 Hz sample. Startup duration and its
named check are persisted. Merely producing pixels does not authorize inference;
the same consecutive freshness, image-quality and target-visibility gates apply.
After startup, the existing 90-second transport safety interval also bounds continuous camera
unavailability or lack of applied control progress, yielding an explicit failure instead of waiting for
the outer platform timeout. The final target receives a full control interval
of physics before termination. The controller verifies the persisted physical
checks, quality measurements and matching evidence archive before accepting a
pickup. A failed finite attempt is never automatically replaced by another scenario.

`openpi_franka_mk8s_live_v2` remains available as an explicit finite communication proof.
It requires two replies containing finite `[15,8]` pi0.5 action arrays and now
executes both five-target segments before exiting. It is excluded from the default
pickup suite. Those measurements and named checks remain on the completed Antioch record.
A clean child exit without the exact persisted passed record fails closed and is not renewed.
The controller preserves its stop marker across restarts. A fresh attempt needs
a fresh adapter identity; it cannot reuse retained runtime state to launch another proof.

For read-only verification of a saved proof, set
`NPA_ANTIOCH_COMPLETED_SCENARIO_ID` to its run identifier and
`NPA_ANTIOCH_MK8S_RUNTIME_CONFIG` to the owner-only runtime configuration.
Run only `test_completed_poc_checks_persist_after_session_release` from
`npa/tests/e2e/test_antioch_mk8s_live_e2e.py` with the documented live-test gates;
the identifier has no default and this test never deploys another scenario.

The supervisor verifies every private bundle file and atomically swaps one complete
generation into place after session replacement. The pod controller directly owns
both `antioch service ports --bind sim.policy-relay=127.0.0.1:18444 --serve sim`
and `antioch scenario run --stream --verbose` as foreground children. Supported
structured `scenario list`, `session status`, and `service ps` must agree on the
exact current session; the simulator process and session must be ready. A child
exit, mismatched session, unhealthy process, or stale observation revokes readiness.
Failure recovery cancels only the exact scenario, proves stable absence, rebuilds an
immutable revision when the session is lost, re-stages source and credentials, and
fails closed for either finite evaluation identity. Ambiguous ownership fails closed.

Mission Control's livestream state is independent of policy-camera readiness.
The scenario waits in safe hold for both RTX render products to return distinct,
advancing RGB frames; it never treats a viewer connection or the control-loop
counter as a camera producer clock. The supported lifecycle follows the reviewed
`antioch-sim==0.4.236` and `isaac-sim-6.0.1` runtime identity: timeline play is
committed once, then every sensor read follows a completed rendered world step. See the
[compatibility matrix](../../../docs/workbench/antioch.md#live-camera-compatibility-contract)
before changing either pin.

`openpi_franka_pickup_v3` uses a versioned remote scenario identity so an
already-published definition cannot mask a new camera/action contract. It dispatches
the default instruction `pick up the red cube`
and records only the non-sensitive `red_cube_pickup` task label in proof telemetry.
It records both current cameras in a default side-by-side Rerun layout, per-view
luminance/variance/dynamic range, cube-frustum evidence, rendered exterior cube
pixels, pair difference, typed camera/action rejection and projection reasons,
latency percentiles, every Franka joint, and the
rendered robot's USD link transforms in Rerun. Those transforms drive generated
volumetric link, joint, base, palm, and finger primitives, so the live 3D view is
recognizably Franka-shaped instead of a thick line strip. The actual Isaac render
remains visible in the exterior and wrist camera panes. No Isaac or Franka mesh
bytes are copied into telemetry, source, or the image.

All scenario telemetry is organized under the single `openpi-live` Rerun entity
root. Antioch's logger receives that root and resolves relative entities beneath
it; every authored blueprint origin uses the same resolver for the camera panes,
3D scene, camera/decision/grasp metrics, Franka joint plots, and policy errors.
The control loop retains full-rate safety calculations and durable acceptance
metrics, while viewer scalars and generated scene geometry are grouped at a
5 Hz display cadence. Exterior images, wrist images, and non-image display data
use three fixed latest-only publisher lanes with one replaceable pending sample
per lane. JPEG encoding or Rerun transport can therefore delay or drop an older
display sample without blocking simulation or control, growing memory, or hiding
the newest current frame. Begin/ok/error markers and an independent loop
heartbeat identify the exact encode/transport boundary in preserved logs. This
is observability isolation, not a hard real-time guarantee.

Pickup evidence is physical rather than inferred from action issuance: live Isaac
poses report end-effector approach and distance, a tracked rigid-contact view reports
cube-to-finger contact force, and the cube pose reports lift relative to its initialized
tabletop height. Acceptance requires at least 5 cm of lift held with gripper contact
and measurable closure for at least one continuous simulation second.

The source is original Apache-2.0 NPA example code. Isaac Sim is supplied by the
Antioch-managed runtime under the operator-accepted NVIDIA terms. OpenPI source and
the pi0.5 checkpoint remain governed by their own runtime contracts; no model
weights or proprietary simulator payloads are present here.

Every run attaches `policy-evidence.zip`. It contains exact client request
MessagePack bytes, lossless exterior/wrist PPM images with pixel hashes, raw
numeric action arrays before projections (`.npy`, never pickle), and a JSONL
control trace. Request/response identities join applied targets to pre-action
joint state and subsequent measured physics poses and contacts. A SHA-256 manifest
covers each file. The request payload saved here is the exact payload passed to
the WSS client, not a claim of an independent policy-server receipt. Rejected
camera samples are included on failure. The live Rerun display can drop stale
viewer updates; this archive is the full control evidence and distinguishes
raw actions, commanded targets and measured state.

For independent, read-only pickup verification, use
`NPA_ANTIOCH_COMPLETED_PICKUP_SCENARIO_ID` with
`test_completed_pickup_checks_and_evidence_persist` in the live test module.
It does not deploy or dispatch. Local fake-simulator tests exercise loop ordering,
physics-time holds, input rejection and byte-preserving evidence, but do not prove
rendered appearance or policy task success. Those require a fresh live run.
