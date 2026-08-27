# LeIsaac browser teleoperation

For the measured split control/video transport, fallback behavior, latency
instrumentation, and security model, see
[LeIsaac low-latency browser transport](guides/leisaac-transport-latency.md).

NPA exposes [LightwheelAI/LeIsaac](https://github.com/LightwheelAI/leisaac)
as a persistent agent-UI tab for a selected session. Live controls report an
explicit unavailable/reconnect state, while the immutable S3 episode browser
remains usable without a running simulator. The integration runs upstream
LeIsaac v0.4.0 at commit
`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`, the real
upstream `SO101Keyboard`. The checked-in registry advertises only the two tasks
at that commit that use this exact single-arm control and asset path:

- `LeIsaac-SO101-PickOrange-v0`;
- `LeIsaac-SO101-LiftCube-v0`.

A fresh session defaults to the real `LeIsaac-SO101-LiftCube-v0` task with the
runtime-fetched built-in SO-101 follower robot, built-in table/lift-cube scene,
and upstream `SO101Keyboard` driven by the browser keyboard. The tab shows all
four choices before any upload and labels the keyboard as the default test
device. Built-ins are reported as runtime assets, never as uploaded S3 bundles.

## Live view and recording modes

The authoritative live-view modes are `single_fast` and `dual_slow`. A fresh
session uses **Fast single**: one full-width, rotatable primary viewport on a
stable Kit camera path. With the default **Primary only** recording schema this
does not schedule, read back, encode, relay, decode, or paint secondary frames.
**Dual view — slower** is an explicit opt-in. It serializes a distinct overview
camera through the same fully awaited viewport, keeps causal primary work at
higher priority, and bounds overview capture to 2.5 FPS. Mode requests are
latest-value/coalesced and are acknowledged as applied only after active GPU
capture and encoder work reach a safe boundary; changing modes does not restart
Isaac, reset the task, alter camera orientation, or remount the tab.

Display mode and episode schema are intentionally independent. Episodes may use
`primary_only` or `primary_and_secondary`. The latter can keep overview capture
active while Fast single remains displayed and is therefore labeled as a
performance cost. Camera membership and both mode values are frozen in episode
provenance when recording starts; changes take effect only while the recorder
is idle. A primary-only episode contains one real track and never fabricates an
overview track.

Inspect the machine-readable source of truth with
`npa workbench leisaac list-tasks --output json`. A session runs one environment
at a time. `--num-envs` is intentionally restricted to `1`; collect named
sequential instances with stable `--environment-id`, `--environment-index`, and
`--seed` values into the same dataset prefix instead of presenting unsupported
parallel control routing.

## What makes the tab appear

The browser mounts the `LeIsaac` tab and its readiness panel as soon as the UI
is wired, before remote capability or artifact discovery. With no registered
runtime, session-only configuration, livestream, motion/orbit, and recorder
widgets are not rendered. The available **Retry readiness** action is safe and
does not launch infrastructure; the panel instead shows the non-secret
operator prerequisites and a placeholder Workbench CLI launch template. It
never accepts an EULA or creates a session from browser state.

A live simulator is optional; an agent-relay launch registers its run through
the agent's authenticated, certificate-pinned HTTPS API. The browser checks
that independently registered LeIsaac run. An unrelated Rerun, artifact, or Voxel51
selection is never treated as the default LeIsaac capability and cannot delay
the tab's first paint. The
backend discovers exactly one `reports/leisaac-session.json` artifact, validates its schema,
run/task/device, fixed transport endpoints, expiry, source commit, and
digest-pinned image, registry fingerprint, task/environment identity, and S3
dataset destination, then verifies the live service's matching one-way nonce
attestation. The raw nonce never appears in `/status` or the browser. Any
absent, stale, malformed, unreachable, mismatched, or non-ready live session
keeps live controls out of the DOM without removing safe dataset surfaces.
When a registered manifest has a configured dataset, immutable episode browsing
and checksum-validated bundle list/upload remain available; upload is explicitly
storage-only and never claims that the bundle is applied. Runtime configuration
and recording are not available in this state.
Selecting another live LeIsaac run
updates the registered capability; switching unrelated artifact runs does not
discard it.

The browser receives no service nonce or agent credential. The live Isaac Sim
5.1 path returns same-origin, authenticated `/api/leisaac/frame.jpg`,
`/api/leisaac/input`, and `/api/leisaac/view` routes. The agent resolves only the selected run's
validated capability, authenticates to the private service with the hidden
session nonce, rejects malformed JPEG bytes and unknown controls, and relays
the response through HTTPS. The service reports accepted inputs separately
from inputs consumed by upstream `SO101Keyboard`, so live validation proves
that controls reached the simulator rather than only the public API.

Even after health and attestation are ready, the page keeps motion, orbit,
recorder, view-mode, recording-camera, and bundle apply/reset actions locked
until **Connect teleoperation** obtains a run/client-bound controller lease.
Each locked widget is associated with the visible lease prerequisite. A second
authenticated browser can still inspect status and immutable data, but receives
`controller_busy` and cannot mutate the runtime until the current controller
disconnects. Disconnect, transport recovery, and lease loss immediately return
those controls to the locked state.

The measured public profile uses an authenticated same-origin WebSocket for
reliable ordered control and an independent bounded binary WebSocket for
full-quality dual-camera JPEG. Real profiling rejected both full-quality JPEG
and small reliable controls over this deployment's TURN route; those optional
WebRTC data-channel endpoints remain compatibility surfaces, not the preferred
path. Every control adapter terminates in the same runtime sequence ledger,
applied-ack service, and cancellation-safe disconnect-release transaction.
Direct SO-101 actions for a real browser gamepad or a custom action source use
that same path. It is not a raw public device port: HTTPS authentication, a
run/client-bound session, bounded messages, reconnect handling, and
accepted/applied acknowledgements still apply. The
exact action message is:

```json
{
  "v": 1,
  "type": "action",
  "run_id": "SELECTED_RUN",
  "client_id": "custom-device-instance",
  "seq": 1,
  "device": "custom-so101",
  "action": [0, 0, 0, 0, 0, 0, 0, 0],
  "client_mono_ns": 0,
  "client_wall_ns": 0
}
```

`action` must be exactly eight finite numbers in `[-1, 1]`; device is exactly
`browser-gamepad` or `custom-so101`. Sequence reuse with different content,
gaps, stale history, unknown fields, non-finite values, and unsupported devices
are rejected. The simulator consumes a direct action once, clears it, and emits
the existing simulator-step application acknowledgement. This keeps a lost
release or disconnected custom source from leaving persistent motion.

The NVIDIA livestream WebRTC path remains available as a compatibility transport. In that
mode, the no-store status response includes one derived session-scoped TURN
credential, signaling uses `/api/leisaac/signal`, and the hash-pinned NVIDIA
browser client is proxied through an authenticated route.

The supported transport is `agent-relay`. It keeps the session nonce, control
messages, status, and relayed frames behind the agent's authenticated HTTPS
origin. The historical `public-load-balancer` transport is rejected before EULA
or infrastructure mutation: its S3 manifest deliberately omits the nonce and
has no secure credential-reinjection path, so browser discovery cannot complete
without publishing a secret. Status and destroy retain compatibility only for
cleaning up historical resources.

`agent-relay` consumes no additional public IPv4 allocation: Kubernetes uses a
private `ClusterIP` service, the saved
NPA agent runs a hardened systemd relay, and a non-GPU sidecar in the simulation
pod initiates an authenticated WSS backhaul through nginx `443` to it. A
digest-pinned coturn sidecar shares the simulator's pod network for the
compatibility WebRTC path. The
backhaul uses the agent's existing basic-auth credential, pins the public HTTPS
certificate SHA-256, and authenticates again with a random session nonce. The
relay binds status to `127.0.0.1:48080`, signaling to `127.0.0.1:49100`, and
its raw backhaul socket to `127.0.0.1:48081`. Status, signaling, and public
UDP `3478` TURN control datagrams use the authenticated WSS backhaul. The
coturn allocation's private UDP `47999-48015` relay range and Isaac Sim's
UDP `47998`
media peer communicate directly inside the shared pod network namespace.
Only explicit operator CIDRs can reach public UDP `3478`; UDP
`47999-48015`, the
GPU pod, and the GPU node remain private. WebRTC sessions force
`iceTransportPolicy=relay`. TURN long-term authentication,
one session-scoped user with a bounded 16-allocation quota, and the exact
security-group rule prevent the public control relay from acting as an open
proxy. The agent and cluster may
remain in separate VPCs because their only cross-VPC path is the pod-initiated
WSS backhaul; no GPU-node ingress or host port is required. The
backhaul script, agent auth, certificate hash, and nonce are mounted into the
pod through a Kubernetes Secret. The UI and TCP APIs remain behind nginx HTTPS
and basic authentication; port `8787`, `8080`, `49100`, cluster ports, and the
GPU pod are never publicly reachable.

## Runtime and licensing

`npa-leisaac` derives from the digest-pinned public runtime-fetch
`npa-isaac-lab:2.3.2.post1` image. Its compatibility set is:

- Isaac Sim `5.1.0.0`;
- Isaac Lab `2.3.2.post1` and source commit `37ddf626…`;
- LeIsaac `0.4.0` / commit `1651c321…` (upstream requests Isaac Lab 2.3.0;
  NPA uses the compatible patched 2.3.x release and validates the real task);
- NVIDIA Omniverse WebRTC streaming client `5.6.0` for the retained
  compatibility path, the version documented by
  NVIDIA's [web viewer sample](https://github.com/NVIDIA-Omniverse/web-viewer-sample)
  for Kit 107.3.1+ and compatible with Isaac Sim 5.1. Its pristine
  runtime-fetched JavaScript is hash-verified, then receives one exact
  transport-only patch so a numeric signaling host on port 443 selects WSS.
  Both source and served hashes are recorded in provenance. The browser still
  requests `forceWSS` as defense in depth for clients that expose that option.

The image bakes only Apache-2.0 LeIsaac source and OSS dependencies. The live
JPEG data-channel peer uses pinned `aiortc==1.15.0` (BSD-3-Clause); it adds no
proprietary payload and does not change the runtime-fetch/EULA boundary or the
built-image payload scan. The
unlicensed optional Feetech SDK used by physical leader hardware is not
redistributed; an explicit packaging-only patch removes that dependency edge,
and this browser service uses upstream's software keyboard path with a narrow,
fail-safe integration patch that publishes readiness only after the real task
reset and a non-empty primary RTX frame. Overview readiness is required only
while the applied display or recording mode schedules that camera. It drains a bounded, validated input queue
inside upstream `SO101Keyboard`, applies each press for eight simulation steps,
and records the consumed-input count separately. Browser teleoperation uses
the RTX viewport rather than policy camera tensors, so the same patch removes
the two unused tiled-camera sensors, their observation terms, and the now-unused
front-camera randomizer to avoid unnecessary camera/DirectGpu work on `sm_120`.
Physics for this single interactive environment runs on `cuda:0` with Fabric,
and RTX rendering, viewport capture, and JPEG production remain active on the
selected RT-core GPU. After environment construction, browser mode raises Isaac
Lab's steady-state render interval and makes the 60 Hz rate limiter sleep
without its upstream unconditional render side effect. Explicit renders remain
immediate whenever a background or causal capture is due, so control cadence is
independent of idle rendering without changing physics cadence. The browser
path also disables Isaac Lab's texture-loading wait: the
headless session uses the active viewport rather than RTX camera observations,
and the default asset-loading loop does not terminate reliably on this path.
The pod requests 16 CPU cores and may use up to 32 so
the USD-backed first reset is not throttled by the previous eight-core limit. The
runtime uses one stable primary camera and, only when requested, alternates the
distinct overview path through the same fully awaited Kit viewport; this avoids
unsafe concurrent shared-renderer capture while preserving two real views.
Recorder capture groups wait only for the cameras frozen into that episode's
schema. The public binary envelope identifies the camera without base64, and
bounded latest values coalesce obsolete work. Drag/touch/wheel orbit commands
update the interactive primary camera; robot keys remain scoped to its focus and
orbit gestures never become robot actions. The
session supervisor starts Kit in an isolated process session with closed stdin
so HTTP-service signal handling cannot interfere with upstream teleoperation.
The browser service uses the explicit launch seed (default `42`) and reports it
with the stable environment identity in `/status` and every recorded frame.
On a cold pod, liveness remains healthy while the supervised simulator process
is alive, including during the licensed runtime fetch and first reset. Readiness
and `/status` remain unavailable until the real reset and required non-empty
camera frames are ready; a failed or exited simulator still fails liveness so
Kubernetes can
restart it while preserving the pod-local `emptyDir` caches.
The exact patch is commit-locked in the image build and named in runtime
provenance. It
uses the shared Isaac acceptance contract before any download. The single public
input defaults to `ACCEPT_EULA=Y`; affirmative legacy values normalize to `Y`,
recognized negative or empty values opt out before download, and other values
fail distinctly. The runtime derives its vendor launcher variable internally.
Only after that preflight are Isaac, the NVIDIA client, the SO101 asset, and both
scene assets fetched into mounted caches. The
assets and client are hash-verified and recorded in runtime `provenance.json`;
EULA acceptance and proprietary bytes are never baked into an image.

## Launch

Rendering requires an RT-core GPU. The current Kubernetes launcher supports and
hard-selects only the RTX PRO 6000 pool; L40S has suitable RT cores in general,
but is not an advertised LeIsaac route until this exact image and launcher are
validated there. The launcher never routes this path to H100/H200.
The image must be pinned by digest and at least one public `/32` operator source
range must be provided. The session has no implicit lifetime;
an operator may add `--expires-at` as an explicit security policy, otherwise
the live service lifecycle controls tab availability.
Before applying the deployment, `launch` refreshes the selected Kubernetes pull
secret with a newly minted Nebius IAM token and verifies that the secret exists.
If credential minting or the secret apply fails, launch stops before scheduling
the GPU workload instead of relying on a warm node image cache.

Deploy or re-bootstrap the agent through the supported lifecycle command. A
fresh deployment provisions a Nebius public IP; re-bootstrap resolves the
existing VM's current public IP from provider state and persists the canonical
customer URL. The operator-facing endpoint is always
`https://<agent-public-ip>/` when public HTTPS is enabled.

```bash
npa agent fresh-setup --project PROJECT_ALIAS --name AGENT_NAME \
  --project-id PROJECT_ID --tenant-id TENANT_ID --region REGION
npa agent status --project PROJECT_ALIAS --name AGENT_NAME --json
```

The public endpoint terminates HTTPS in nginx. `/healthz` is the intentionally
unauthenticated liveness probe; the UI, API, LeIsaac client, and WebSocket
signaling relay require the agent's basic-auth credentials. The FastAPI backend
listens only on VM loopback and is never exposed directly. Accept the
self-signed certificate for the public IP, then verify the endpoint from the
operator host:

Deploy and re-bootstrap also remove a dedicated legacy `allow-npa-*` rule for
the internal backend port if an older deployment left one behind. NPA refuses
to rewrite an unmanaged or mixed-purpose rule and fails closed instead. HTTPS
ingress is ensured through the existing agent security group; this path does
not broaden SSH or publish the backend listener.

```bash
AUTH_SECRET_PATH=/secure/path/reported-by-agent-deploy
source "${AUTH_SECRET_PATH}"
AGENT_URL="$(npa agent status --project PROJECT_ALIAS --name AGENT_NAME --json \
  | npa/.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["public_url"].rstrip("/"))')"
curl -sk "${AGENT_URL}/healthz"
curl -sk -u "${AGENT_USER}:${AGENT_PASSWORD}" "${AGENT_URL}/api/health"
```

```bash
# ACCEPT_EULA defaults to Y for Isaac-backed routes. To opt out, set it to N.
# On shared operator hosts, select the Nebius/Kubernetes access profile used by
# the command. The repository-selected public GHCR image needs no pull secret.
export NPA_NEBIUS_PROFILE=agent-sa

npa workbench leisaac launch \
  --run-id leisaac-teleop-example \
  --image ghcr.io/nebius/nebius-physical-ai/npa-leisaac@sha256:DIGEST \
  --context YOUR_KUBECTL_CONTEXT \
  --source-range OPERATOR_PUBLIC_IP/32 \
  --transport agent-relay \
  --agent-project PROJECT_ALIAS \
  --agent-name AGENT_NAME \
  --task LeIsaac-SO101-LiftCube-v0 \
  --environment-id table-a \
  --environment-index 0 \
  --seed 42 \
  --num-envs 1 \
  --output-path s3://BUCKET/datasets/leisaac-demo \
  --manifest-prefix s3://BUCKET/checkpoints
```

`--output-path` is always a dataset prefix. `--manifest-prefix` is the
capability publication prefix; it may also be an exact
`.../reports/leisaac-session.json` leaf. The deprecated `--artifact-uri` alias
retains those exact leaf semantics, fixing the older duplicated
`leisaac-session.json/<run>/reports/leisaac-session.json` behavior without
hiding already-written historical objects from discovery.

Launch waits are bounded without limiting the workload lifetime. The defaults
allow 10 minutes for a historical LoadBalancer address and four hours for the
Isaac Deployment to become ready. Operators with a known slower control plane
may set positive `NPA_LEISAAC_EXTERNAL_IP_TIMEOUT_SECONDS` and
`NPA_LEISAAC_READY_TIMEOUT_SECONDS` values to any finite positive number of
seconds; timeout errors include the last Kubernetes provider or rollout
observation and normal launch rollback still runs.

Launch, destroy, and the live lifecycle proof share a per-run exclusion lock.
Before any mutation they verify that the current Kubernetes identity can
`get`, `create`, `update`, and `delete` Secrets in the selected namespace. This
is the same resource kind used by LeIsaac relay and recorder credentials, but a
cleanup-only custom role may need those verbs added; a failed preflight names
the missing verbs and leaves all LeIsaac resources unchanged.

Replacing the NPA agent does not require launching a second simulator. Preserve
the exact session-manifest URI before destroying the old agent, deploy the
replacement under the same saved agent identity, then reconnect the one
existing workload:

```bash
npa workbench leisaac reconnect-agent \
  --run-id EXISTING_RUN_ID \
  --manifest-uri s3://BUCKET/PREFIX/EXISTING_RUN_ID/reports/leisaac-session.json \
  --agent-project PROJECT_ALIAS --agent-name AGENT_NAME \
  --context KUBECTL_CONTEXT --namespace EXISTING_NAMESPACE
```

`reconnect-agent` verifies the private, single-replica, NPA-owned topology and
the exact existing run-scoped EULA values before mutation. It rotates only the
relay Secret and Deployment nonce, updates exact agent ingress and the existing
session manifest, and waits for authenticated relay attestation. It does not
create a Deployment, change the image/task/dataset, or record a new EULA
acceptance. The exact manifest URI is intentionally required so replacement
does not broaden discovery across the agent bucket.

`agent-relay` resolves the agent IP from live provider state and refuses a
stale saved address, missing SSH key or agent auth, unrestricted source range,
TLS certificate mismatch, invalid session nonce, or a second active relay
session. The supported deployment uses a digest-pinned, non-root coturn sidecar
and exposes no coturn port from the GPU cluster.
The historical `--transport public-load-balancer` value is rejected. Use
`agent-relay`; the old status/destroy branches exist only so operators can
inspect and remove resources created by an earlier release.

For a release-gated existing session, the mutating lifecycle proof exercises
the exact authenticated browser control/video sockets, restarts the relay while
a control is held, lets a bounded relay credential expiry fire, proves the old
controller lease resumes with no held keys before any pod restart, proves the
old credential cannot authenticate after rotation, and restores the same
Deployment with a fresh session credential and a new immutable capability
generation. During the authentication proof it temporarily scales that same
Deployment to zero so acceptance of the new nonce and denial of the stale nonce
are tested without an already-connected sidecar. It does not create a workload
or modify the original capability or any immutable dataset object. Run it only
when the operator has authorized
restart/restore of that exact existing run; its evidence file is atomically
written mode `0600` and contains no credentials, cookies, or provider IDs:

```bash
npa/.venv/bin/python npa/scripts/verify_leisaac_relay_lifecycle_live.py \
  --project PROJECT_ALIAS --name AGENT_NAME \
  --context KUBECTL_CONTEXT --namespace EXISTING_NAMESPACE \
  --run-id EXISTING_RUN_ID --evidence /secure/evidence/relay-lifecycle.json
```

This is an interactive, lifecycle-bearing service rather than a finite batch
stage, so it is intentionally launched and destroyed through the Workbench
lifecycle command, not represented as an `npa.workflow` step that would report
completion while the browser session still needs to remain alive.

Reload the agent UI after launch, open `LeIsaac`, and choose **Connect
teleoperation**. No run-ID entry is required. Click the simulation to focus it.
Controls are the upstream bindings: `W/S`, `A/D`, `Q/E` translate; `J/L`,
`K/I` rotate; `U/O` open/close the gripper. Episode state is explicit: **Start
episode**, **Mark success** or **Mark failure**, then **Finalize & upload**.
Finalize is disabled until an outcome is selected. Upload errors stay visible
and do not discard the pod-local episode; **Finalize & upload** becomes a retry
when the prior conditional publication attempt failed. Marking an outcome
freezes the episode boundary before upload.

The **Custom assets and devices** area accepts a bounded set of USD, Python,
and JSON files. Paths are relative POSIX paths with no traversal, duplicates,
or unsupported suffixes; each file and the canonical bundle carry SHA-256, and
objects are written below the selected dataset's immutable `bundles/` prefix.
A robot/scene entrypoint must be USD. A device entrypoint is a JSON mapping for
`npa.leisaac.so101-device.v1`: driver `custom-so101`, the exact action order
`x,y,z,roll,pitch,yaw,shoulder_pan,gripper`, and an integer update rate from 1
through 120 Hz. Uploaded Python is parsed but never executed:
only docstrings, approved Isaac Lab/LeIsaac/typing/numpy imports, and literal
declarations are accepted. Calls, functions, classes, attribute writes, unsafe
imports, and executable statements are rejected.

Selecting a bundle is an authenticated runtime operation, not a cosmetic UI
preference. It requires the active controller lease; upload/list alone never
selects or applies a bundle. The runtime downloads the manifest and each file directly from the
selected dataset prefix, revalidates canonical and per-file SHA-256 values,
materializes only below a digest-named private cache, and rejects a kind
mismatch. It refuses to restart while a recording is active. Otherwise it
passes the verified robot and scene USD entrypoints to the pinned task config,
passes the device descriptor as provenance for the validated direct-action
channel, terminates only the supervised simulator child, and starts it again.
**Reset to built-in defaults** clears all three uploaded overrides as one
authenticated operation and restarts the same registered task on its task-aware
built-ins. The scoped selection is persisted with run, dataset, task, and
registry identity. After a rollout, a mismatch is reported as pending rather
than restored automatically: a browser must reconnect, obtain the controller
lease, and explicitly apply the selection. A stale selection is never carried
into another dataset.
The tab remains present with an explicit reconnect state until both distinct
RTX viewports are nonblank. Runtime health and episode provenance expose the
selected names and digests without exposing storage credentials. Custom robot
assets must preserve the SO-101 joint/prim contract and custom scenes must
preserve the selected task's expected object prim names; incompatible USD fails
the real task startup rather than silently falling back to stock assets.

The authenticated UI polls authoritative recorder status continuously and
recovers the same state after a refresh or reconnect. Its transition contract
is:

| State | Enabled controls | Meaning |
| --- | --- | --- |
| `idle` | **Start episode** | No active episode; outcome and finalize controls explain what must happen first. |
| `recording` | **Mark success**, **Mark failure** | An active episode exists and synchronized frame count advances. |
| `outcome-pending` | outcome controls, **Finalize & upload** | The selected outcome is persisted and the local episode is frozen, but nothing is uploaded implicitly. |
| `uploading` | none | Records, raw frames, and H.264 video are being published. |
| `upload-failed` | **Finalize & upload** | Retry the same episode without creating a second commit. |

Every browser transition carries an idempotency request ID. The session server
atomically reserves one pending command before appending it to the simulator
queue, so double-clicks and concurrent requests cannot enqueue duplicate
episode commits. The final idle status exposes the immutable dataset version,
episode index, and episode-commit URI.

## Immutable episode browser

The persistent **Episodes** area uses only the selected manifest's configured
S3 dataset prefix. Version and episode listings are bounded and paginated; no
bucket-wide list or local-file fallback is used. Filters cover task,
environment, outcome, date, robot, scene, and device. Finalize returns the exact
immutable version and episode, so **View uploaded episode** opens it immediately
without relying on `latest.json` or a page reload.

Video is served through an authenticated same-origin streaming route. The
backend validates the committed size/checksum metadata, forwards bounded full,
open-ended, and suffix byte ranges to S3, returns `206` with exact
`Content-Range`, and returns `416` for invalid bounds. It never buffers a whole
MP4 in FastAPI and never returns a presigned URL. H.264 files are encoded with
`+faststart`, so browser metadata, seeking, and resume do not require the final
object bytes first.

The player renders the committed metadata and checksums, immutable dataset
version, task/environment/outcome, start/end/duration/frame count, success and
reset markers, and robot/scene/device/bundle provenance. Its scrubber aligns
video time with action, state, reward, marker, and source timestamps. Playback
rate, next/previous episode, keyboard controls, retry, and records/metadata
downloads remain available. Two-camera commits use synchronized workspace and
overview tracks from the same capture groups. Older one-camera commits show an
explicit single-camera fallback. Unrecognized committed artifacts remain
visible as downloads instead of disappearing.

**Describe this frame** captures the pixels from the video element currently
displayed in the episode player and submits them with the visible episode and
timeline metadata. If capture fails or the frame is blank, the UI reports that
failure and does not claim the agent saw pixels.

## LeRobot demonstration dataset

Every captured JPEG comes from the real RTX viewport and is paired with the
most recent completed real `env.step`. Its Parquet record contains the real
six-joint observation, exact eight-dimensional action passed to `env.step`,
reward, terminated/truncated/done values, simulator step, seed,
task/environment identity, source-frame SHA-256, and monotonic
and wall-clock nanosecond timestamps. The fixed 16 FPS `timestamp` field is the
LeRobot video clock; the original clocks remain separate audit features.

Finalization encodes each synchronized camera as an H.264 faststart MP4 and
uploads the ordered raw JPEGs and a unique raw episode bundle. Their per-camera
per-frame hashes are tied to
the records and commit, so frame/action alignment remains independently
auditable after lossy H.264 encoding. A conditional
`commits/episode-NNNNNN.json` object makes the
episode durable before a new immutable `versions/vNNNNNN-UUID/` LeRobot tree is
published. A crash can leave an unreferenced bundle but cannot overwrite a
completed commit or prior version. The next session resumes numbering from the
commit objects and can add a different supported task/environment to the same
prefix. `latest.json` is only a pointer; version contents are immutable.

The output targets `LeRobotDataset` 0.5.1 / format `v3.0`. Download one
immutable version and validate it with the supported loader (Python 3.12+):

```bash
python3.12 -m venv /tmp/lerobot-051
/tmp/lerobot-051/bin/pip install 'lerobot==0.5.1'
/tmp/lerobot-051/bin/python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(repo_id="local/leisaac", root="/path/to/version")
print(len(dataset), dataset.meta.total_episodes, dataset.meta.video_keys)
PY
```

## PAIDF appearance augmentation

Export a finalized episode directly from its immutable S3 version; no local
operator download is needed:

```bash
npa workbench leisaac export-paidf \
  --dataset-uri s3://BUCKET/datasets/leisaac-demo/versions/v000002-UUID \
  --episode 0 --run-id paidf-leisaac-001 \
  --output-path s3://BUCKET/physical-ai-data-factory/paidf-leisaac-001 \
  --output json
```

The result reports the exact runnable
`npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml` command with
`NPA_COSMOS_CONDITION_ON_INPUT=1`. The workflow invokes the real
`workbench.cosmos2.transfer_execute` toolRef with
`--condition-on-input --execute`; a manifest-only stub is not accepted.
Source dataset/version/episode/task/environment/checksums are written to
`input/leisaac-lineage.json` and carried into the PAIDF final report. Config
generation detects that lineage and uses an orange-pick or cube-lift
appearance-only prompt instead of the blueprint's default cloth-folding scene.
An agent-scoped base prefix is allowed before the exact
`physical-ai-data-factory/<run-id>` suffix.

After a real run, materialize one variant:

```bash
npa workbench leisaac materialize-paidf \
  --dataset-uri s3://BUCKET/datasets/leisaac-demo/versions/v000002-UUID \
  --episode 0 \
  --paidf-run-uri s3://BUCKET/physical-ai-data-factory/paidf-leisaac-001 \
  --variant 0 --output-path s3://BUCKET/datasets/leisaac-derived \
  --output json
```

Materialization requires `mode=cosmos_transfer2.5_gpu`, `status=executed`, and
`input_conditioned=true`, decodes a nonblank augmented clip, and rejects any
frame-count or timestamp difference greater than 1 ms. Only then does it copy
the parent Parquet labels byte-for-byte and replace the selected visual object
in a new immutable version. Cosmos Transfer appearance conditioning preserves
the demonstrated motion structurally; it is not action augmentation and adds
no robot state, reward, label, or task-success evidence.

## Status and cleanup

```bash
npa workbench leisaac status --run-id leisaac-teleop-example --context YOUR_KUBECTL_CONTEXT
npa workbench leisaac destroy --run-id leisaac-teleop-example --context YOUR_KUBECTL_CONTEXT
```

Destroy removes only that run's transient Deployment, Services, relay Secret,
and dedicated recorder credential Secret. Completed S3 episodes, dataset
versions, and PAIDF evidence are preserved. For an
agent-relayed run it reads the owning agent and source CIDRs from Kubernetes
metadata, stops only the matching relay unit, and deletes only the matching
relay and any compatibility TURN unit, and deletes only the matching
NPA-managed UDP `3478` rule (plus a legacy `47999` rule when an older
session recorded one). It preserves the S3
manifest/log/evidence record.
Once the service is gone, live health fails and the agent UI replaces live
controls with the readiness panel while keeping immutable S3 episodes and
downloads available.

Durable validation evidence and screenshots live under
`docs/evidence/leisaac/`.
