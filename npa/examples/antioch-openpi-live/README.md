# Antioch + OpenPI live example

This public-source project runs a real Isaac Sim Franka scene in an Antioch
livestream and sends its two current 224x224 camera frames and robot state to an
external OpenPI pi0.5 DROID policy. It applies only validated finite `[15, 8]`
chunks, enters safe hold on stale/malformed/unsafe responses, and reconnects with
bounded exponential backoff. Antioch telemetry and a viewport overlay report the
live counters; neither is reconstructed from a recording.

The client uses a 90-second response-age safety deadline because a cold request
can take tens of seconds even though warmed supported-GPU requests are normally tens of
milliseconds. The reviewed `pi05_droid_jointpos_polaris` output contract is seven
absolute arm joints plus one DROID gripper-position command. The simulator uses the
DROID reset posture and maps its finger joints into DROID's `0=open, 1=closed`
observation convention. The inverse actuator mapping sends `0` to two 4 cm-open
Isaac finger joints and `1` to closed joints; the model output is binarized at 0.5
as in the upstream DROID deployment example. Raw out-of-distribution joint/gripper
counts are reported separately from Franka-limit and per-target-step safety
projections. Five returned targets are applied at a nominal 15 Hz. The
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
sample asks `CameraSensor` to fill a documented `(224, 224, 3)` uint8 CPU buffer
and immediately copies it into scenario-owned memory
instead of exposing a mutable or device-backed view to downstream code; a
public clock attached to that sensor's render product supplies the exact marker
when the RGB annotator omits it. The wide exterior camera has explicit optics
and frames the complete tabletop manipulation region. The wide wrist camera is
calibrated once from the measured stock-Franka hand and fingertip transforms,
then its fixed tool-frame extrinsics are re-applied before every rendered step so
it follows the live gripper without looking through the fingers. Flat, black,
malformed, stale, non-finite, geometrically irrelevant, or mutually duplicated
camera pairs are never sent to policy inference. The exterior gate also requires
rendered red-cube pixels; both views require the known cube center inside their
current optical frustum. Every accepted pair receives monotonically increasing
pair and render identities carried unchanged through its one policy request and
response evidence.

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

The scenario is a finite communication proof. It returns successfully only after
two policy requests used distinct advancing, nonblack exterior and wrist camera
pairs and both responses were validated as finite `[15,8]` pi0.5 action arrays.
Those measurements and named checks remain on the completed Antioch record after
the controller retires the session and the adapter is scaled down. A clean child
exit without that exact persisted passed record fails closed and is not renewed.
The controller preserves its stop marker across container restarts so shutdown
cannot silently dispatch another proof. A fresh attempt needs a fresh adapter identity.

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
starts one successor with capped backoff. Ambiguous ownership fails closed.

Mission Control's livestream state is independent of policy-camera readiness.
The scenario waits in safe hold for both RTX render products to return distinct,
advancing RGB frames; it never treats a viewer connection or the control-loop
counter as a camera producer clock. The supported lifecycle follows the reviewed
`antioch-sim==0.4.236` and `isaac-sim-6.0.1` runtime identity: timeline play is
committed once, then every sensor read follows a completed rendered world step. See the
[compatibility matrix](../../../docs/workbench/antioch.md#live-camera-compatibility-contract)
before changing either pin.

`openpi_franka_mk8s_live_v2` uses a versioned remote scenario identity so an
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
and closure for at least one continuous second.

The source is original Apache-2.0 NPA example code. Isaac Sim is supplied by the
Antioch-managed runtime under the operator-accepted NVIDIA terms. OpenPI source and
the pi0.5 checkpoint remain governed by their own runtime contracts; no model
weights or proprietary simulator payloads are present here.
