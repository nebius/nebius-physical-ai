# Antioch Workbench integration

The Antioch adapter is a CPU-only control-plane service. It stages immutable
projects from S3, submits scenarios or suites with Antioch's supported structured
CLI, reconciles retries, collects checks/logs/results/Rerun files, and optionally
publishes a strict LeRobotDataset v3 for offline policy training. Antioch executes
simulation on its managed infrastructure; this image contains no simulator.

## Authentication and runtime boundary

Install Antioch's CLI normally and authenticate once as the operator. Confirm the
existing session without printing identity data:

```bash
export NPA_ANTIOCH_ACCEPT_TERMS=YES
npa workbench antioch terms-preflight --output json
npa workbench antioch health --output json
```

`NPA_ANTIOCH_ACCEPT_TERMS=YES` is an exact, explicit attestation that the
operator reviewed the [Antioch Terms of Service](https://antioch.com/terms)
(version dated 2026-02-28) for the scoped use of `antioch-sim==0.3.63` and the
Antioch Service. Any customer MSA or order form remains controlling. Other
spellings fail closed. The adapter records only the agreement name, public URL,
version, scope, and accepted boolean in durable operation state; it never stores
the environment value in an image, cache, project, dataset, or credentials file.

Do not copy the Antioch config into an image. For Kubernetes, create a secret from
the existing config out of band and mount it read-only with `deploy
--antioch-config-secret`; create a separate secret whose `token` key protects the
adapter HTTP API. Create another runtime-only secret whose `accepted` key is the
exact value `YES`, and pass its name through `--terms-acceptance-secret`. The
deploy command prints secret *names*, never values.
Provide S3 credentials through a pre-created `--s3-credentials-secret`, or omit
that option when the pod workload identity supplies S3 access. The adapter's
self-contained resolver uses `--storage-endpoint` and leaves credentials to
boto's workload-identity chain; it does not import host-only NPA configuration
modules. The optional secret uses
the ordinary `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
optional `AWS_SESSION_TOKEN` keys.

The public adapter pins `antioch-sim==0.3.63` and its reviewed SHA-256. On first
use it fetches the wheel directly from the vendor's PyPI delivery into
`NPA_ANTIOCH_RUNTIME_CACHE`, verifies it, and installs it in that writable volume.
`NPA_ANTIOCH_RUNTIME_OFFLINE=1` fails closed when the cache is cold. Neither the
wheel nor runtime cache belongs in the adapter image. The operator's direct
delivery and use remain subject to the operator's Antioch/NVIDIA terms.

On a host, the default cache is `$XDG_CACHE_HOME/npa/antioch` or
`~/.cache/npa/antioch`. The adapter container explicitly keeps its writable
`/workspace/.cache/npa/antioch` mount. The virtual environment is created at its
final versioned path under a file lock and publishes `.complete` last; moving a
prepared virtualenv is invalid because its generated command shebangs are absolute.

Today the CLI session is personal OAuth stored in Antioch's config directory.
That is suitable for a human-operated smoke, but not a production unattended
identity. Production deployment should use an Antioch service identity when the
vendor exposes one; until then, token expiry requires an operator to refresh the
mounted session. The adapter never initiates interactive login.

## Immutable input and S3 output

`--input-path` names a prefix containing:

```text
project-manifest.json
project.tar.gz
```

The manifest uses schema `npa.antioch.project.v1` and records archive name, size,
SHA-256, source name/revision/license/digest, and asset hashes. Extraction rejects links,
traversal, device nodes, credentials, key files, and projects without exactly one
`antioch.yaml`. The adapter rewrites only its project id to a deterministic value
derived from workflow run and state identities.

`--output-path` must be a run-scoped S3 prefix. Durable state is under `_control/`.
Collected bytes are under `artifacts/<scenario-run>/`, normalized training data
under `dataset/`, and the versioned manifest at `manifests/v1.json`. `_SUCCESS.json`
is created immutably only after every preceding write and checksum check succeeds.
Consumers must gate on that marker. Never reuse one output prefix across unrelated
workflow runs.

## Operations and recovery

```bash
npa workbench antioch submit --input-path s3://BUCKET/input \
  --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --robot-type ROBOT \
  --task "TASK DESCRIPTION" --suite SUITE --output json
npa workbench antioch status --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
npa workbench antioch resume --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
npa workbench antioch cancel --output-path s3://BUCKET/runs/RUN/simulation \
  --workflow-run RUN --state-id simulate --output json
```

`run` is submit, monitor, and collect. A conditional S3 claim and deterministic
Antioch project id ensure a pod retry reconnects rather than creating another
billable suite. `reconcile` repairs the submission-to-state crash window. Terminal
state is immutable; `resume --rerun-terminal` rejects an in-place rerun and directs
the caller to use a new state identity. HTTP 429 and
5xx failures are retryable; authentication failures, malformed JSON, conflicting
identity, invalid artifacts, and schema failures are terminal. Cancel is
idempotent. Cancelling a completed, failed, or already-cancelled operation is a
no-op that preserves its status and immutable completion/dataset records. Cancel
Operation failures retain the CLI's retryable/terminal classification in both the
returned error envelope and durable operation state. Cancel test work before
releasing any machine it used.

Collection uses a durable S3 compare-and-swap lease refreshed across download,
checksum verification, conversion, upload, manifest, and final-state publication.
Concurrent collectors are excluded while an owner is active. Exceptions clear
ownership and record a retryable error; after a crash, an expired owner can be
adopted with the same identities. Deterministic immutable manifests and completion
markers make retry and resume safe.

The sanitized operation record contains the vendor run id. Open that run in the
Antioch Mission Control console using the authenticated account; never paste a
signed console URL into logs, manifests, issues, or pull requests. If a supported
CLI response supplies a non-signed console URL, the adapter may expose its
redacted form. It does not construct undocumented Rome URLs.

## Continuing OpenPI live demonstration

`npa/examples/antioch-openpi-live` is a separate live-control example, not an
offline dataset claim. It renders a real Franka scene and two current policy
cameras, sends observations through an authenticated TLS connection to the
persistent OpenPI service, requires exact finite `[15, 8]` action chunks, and
applies at most five validated targets per observation at a nominal 15 Hz. Joint
limits, a per-target delta bound, a response-age deadline, and reconnect backoff
fail closed into safe hold. Report measured rates and latency; this is not hard
real-time control.

The live viewer includes the normal streamed Isaac viewport, current camera images
in Antioch telemetry, and counters for observation sequence/time, requests, round
trips, latency, action shape/index, safe hold, reconnects, and safely applied
targets. These are emitted by the running scenario, not inferred from an `.rrd`.
The scenario uses one `openpi-live` telemetry root: Antioch's logger resolves all
relative entities below it, and the Rerun blueprint uses those exact resolved
origins for both cameras, the 3D scene, metrics, Franka joint plots, and errors.
The versioned `openpi_franka_mk8s_live_v2` remote scenario identity prevents a
previously published definition from masking the schema-2 camera/action contract.
Its default dispatched instruction is `pick up the red cube`; public proof telemetry
uses only the non-sensitive `red_cube_pickup` label.

The two 224x224 policy cameras use Isaac Sim 6's supported
`isaacsim.sensors.experimental.rtx` API: separate `RtxCamera` authoring objects
and `CameraSensor` runtime objects with an explicit `rgb` annotator. The scenario
authors a nonzero 15 Hz sensor tick rate and calls Isaac Replicator's documented
blocking `orchestrator.step(delta_time=0.0, pause_timeline=False,
wait_for_render=True)` hook immediately before sampling. This explicit scheduling
boundary does not assume that Antioch's scenario lifecycle autoplays independent
RTX render products. `world.step(render=True)` continues to advance physics and
the streamed viewport; the orchestrator step completes the policy-camera capture.
The scenario refuses an unreviewed Antioch SDK/engine identity or a missing or
incompatible orchestrator hook before policy control begins.

The scenario reads `(data, info)` from
`get_data("rgb", out=cpu_rgb)`, using the documented uint8 RGB shape and safely
copies the CPU result
to host memory before publication, and
uses sensor metadata or the public exact render-product clock for advancement.
Policy remains in safe hold until both independent views are valid, distinct,
and advancing; viewer state and control-loop iterations are not producer clocks.

### Live camera compatibility contract

| Surface | Reviewed contract | Upgrade treatment |
| --- | --- | --- |
| Antioch SDK/CLI | `antioch-sim==0.3.63`; public scenario surface exposes `scenario`, `ScenarioRun`, `Logger`, `world`, and `engine` | Exact runtime check; any other version is unsupported until reviewed. |
| Antioch engine | `antioch-engine/isaac-sim-6.0.1:0.3.63` / engine identity `isaac-sim-6.0.1` | Exact runtime check; do not substitute a newer engine under the old scenario. |
| Isaac camera | `isaacsim.sensors.experimental.rtx.RtxCamera` + `CameraSensor`, documented uint8 RGB CPU output, immediate scenario-owned copy, nonzero sensor tick | Capability is exercised through public Isaac Sim 6 APIs. |
| Render advancement | synchronous `omni.replicator.core.orchestrator.step` with `wait_for_render=True` | Missing or incompatible signatures fail clearly; no implicit autoplay fallback. |
| Current Antioch direction | The public Antioch site advertises higher-level `Simulation` and `antioch.sensors.RgbCamera` authoring | Directional drift evidence only. It is absent from the installed 0.3.63 public exports and is not a migration specification. |

The Isaac contracts above are documented in the official [Isaac Sim 6 camera
guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html)
and [Replicator workflow guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html).
The higher-level Antioch surface is visible on the [public Antioch
site](https://antioch.com/), but public marketing alone is insufficient to
change this adapter.

To upgrade safely, obtain versioned vendor SDK and engine documentation, inspect
the installed public exports without reading auth state, update this boundary
and its dependency-injected scheduler tests, then repeat the full local gates.
Keep the PR draft until one authorized live run on the exact proposed versions
shows both producer clocks advancing beyond their initial value, at least 120
valid camera pairs, at least 100 successful policy round trips, at least 500
applied targets, and every remaining acceptance threshold below. Do not infer
camera readiness from a healthy service, viewer state, loop count, or scheduler
call alone.

The scenario keeps safety calculations and durable acceptance counters at control
cadence, but groups Rerun scalars and generated scene geometry at a documented
5 Hz display cadence. Fixed latest-only exterior-image, wrist-image, and display
publisher lanes each retain at most one replaceable pending sample. Slow JPEG
encoding or viewer transport cannot block the simulation/control loop or a policy
safe hold; image begin/ok/error markers and a separate loop heartbeat make a later
stall boundary explicit. These are bounded observability semantics, not a hard
real-time claim.

The OpenPI bootstrap is publicly installed as `npa-openpi-live-deploy` (implemented
by `npa.workflows.byof.openpi_live`): a single B200
Deployment with readiness/liveness, `Recreate` rollout semantics, and a PVC-backed
runtime checkpoint cache. Only a bounded TLS WebSocket gateway is exposed; an
API-key Secret and TLS Secret are generated per live deployment, while the raw
policy and diagnostic ports remain outside the Service and blocked by ingress
policy. Kubelet probes reach those health ports only from exact discovered node
InternalIP host routes. The target Cilium enforcement preserves those node sources,
but standard Kubernetes NetworkPolicy cannot distinguish kubelet from another
host-network process on the same enumerated node; this is the narrow remaining
traffic tradeoff and does not admit ordinary workload-pod sources. The checkpoint,
keys, CA private material, credentials, and simulator
payload never enter the public image or project source.

The steady-state deployment is MK8s-native. `live-k8s-deploy` reads one
operator-owned mode-0600 runtime file and reconciles a two-container adapter
Deployment in the `workbench` namespace. The controller container runs only
supported `antioch services build|up|exec|cp|down` operations and `antioch
scenario run --stream --verbose`. The supported service tunnel binds on pod
localhost. A bounded relay container in that same pod network namespace connects
the tunnel's authenticated WSS operator role to a CA-verified, authenticated
ClusterIP OpenPI Service. The operator VM launches and observes this Deployment
but carries no camera frames, policy messages, or actions.

The private runtime file contains Kubernetes coordinates and paths to the
existing Antioch config, assigned project-id file,
and retained OpenPI objects. Those values do not appear in CLI arguments or
ordinary output. The deployer stages them as owner-labelled Kubernetes Secrets,
rotates the policy gateway certificate for its `.svc` DNS name, and copies Secret
files through a root-only init container into a memory-backed 0600 volume owned
by the non-root runtime uid. Terms values, API keys, certificates, OAuth state,
and project identity never enter a ConfigMap, image, manifest output, annotation,
or log. Terms acceptance is read only from the exact
`NPA_ANTIOCH_ACCEPT_TERMS=YES` value in the deploy process environment; it has no
runtime-config file field and is never persisted by NPA on the operator VM.

The policy Service is `ClusterIP` by default. Its NetworkPolicy permits the WSS
gateway only from the exact adapter identity and permits health ports only from
the enumerated kubelet node addresses. The adapter denies all ingress and limits
egress to cluster DNS, the selected policy pods, and vendor service ports only.
Antioch SaaS has no stable destination CIDRs, so TCP 443 and 8443 (plus supported
service SSH on TCP 22) necessarily use `0.0.0.0/0`. "Restricted traffic" here
means port- and direction-restricted, not destination-restricted; a manifest
assertion pins that limitation. A former
owned LoadBalancer may remain temporarily for rollback; run
`live-k8s-finalize-cutover` only after sustained acceptance to remove that exact
public Service. The retained B200 Deployment and checkpoint PVC are reused, not
duplicated.

`live-k8s-stop` waits for sanitized controller evidence written only after the
exact scenario is terminal or stably absent and the service is down, then scales
the Deployment to zero. Missing/malformed evidence, `cleanup_failed`, timeout,
or forced/SIGKILL termination remains unproven and is never reported as stopped.

```bash
npa workbench antioch live-k8s-deploy \
  --runtime-config /path/to/private-runtime.json --output json
npa workbench antioch live-k8s-status \
  --runtime-config /path/to/private-runtime.json --output json
```

The runtime schema is `npa.antioch.mk8s-live-config.v1`; the checked-in example
uses only placeholders. `workflow_run` plus `state_id` derive every adapter
identity, so independent Antioch stages cannot collide. `adapter_image` must be
an immutable digest. Deployment, status, stop, and cutover-finalization refuse
unowned objects.

Scenario timeout is a finite platform boundary, so the supervisor renews
indefinitely until explicitly stopped. Renewal resets the simulated episode and
briefly interrupts the viewport; it is continuous service supervision, not one
immortal scenario process.
The supervisor atomically rechecks and re-stages the private client bundle after
container recreation, and rebuilds the machine-local service image after a machine
recycle before dispatching another scenario. If a dead pod leaves the exact project's
machine assignment bound to its former local SSH client, the replacement cancels the
exact live run, releases only that project assignment through the supported CLI, and
retries the service build before dispatch. Typed retryable control-plane failures during
service build or startup remain in the same controller startup and use capped backoff;
fatal errors still fail immediately. A Mission Control stream in `ready` state is
published but waiting for an authenticated viewer; do not describe it as actively
viewed until the viewer connects and the first rendered frame advances.
The controller owns `antioch scenario run --stream --verbose` as its direct
foreground child and drains its output in-process. A remote run cannot be adopted
after that child exits: the daemon session heartbeat belongs to the departed CLI.
The controller cancels only the matching project-scoped run, proves stable exact
absence through supported `scenario list` and `machine status` JSON, and then starts
one successor. This prevents both a stale run with no client heartbeat and duplicate
stream dispatch.

Adapter readiness is an exact daemon-ownership contract, not a process or open-port
check. The controller continuously reconciles the project-scoped scenario inventory
with the supported structured `machine status` contract. Readiness requires Rome's
`runtime_status.guest_state` to be healthy and freshly observed, a fresh direct-daemon
`runtime` observation, matching Rome/direct exact stream ownership, exactly one
`antioch scenario run` session lease, process and stream leases, and a live direct
child whose parent is container PID 1. State uses versioned JSON and owner-only atomic replacement. Local and
status-command readers base64-frame the file bytes to prevent transport-level JSON
coercion and retry a bounded number of transient empty/partial reads. Kubelet probes
use the same fail-closed state predicates over pod-local HTTP because exec-probe RPC
failures are not application-health evidence; ingress is restricted to the configured
kubelet CIDRs and the health ports have no Service. A
missing schema, malformed value, wrong identity, stale heartbeat, absent stream owner,
missing vendor session, unhealthy/stale Rome observation, child exit, or unreadable
state revokes readiness. Converged loss terminates the exact child process group,
cancels the exact run, rebuilds and re-stages after recycle when needed, and starts one
successor with capped backoff. Ambiguous ownership fails closed. The operator/Codex
process is not part of this supervision and may exit after handoff.

Before a retained live run is accepted, require at least 930 seconds (15 minutes
plus a 30-second margin), 120 valid camera pairs, 100 successful policy round trips,
and 500 applied targets; at
least 90% of policy requests must succeed. Camera luminance mean must exceed 5
and variance 25 for both views, both cumulatively across accepted pairs and on the
current pair. The schema-2 camera proof requires one validated pair for every policy
request; rejected startup or flat pairs are counted but never enter the accepted
minimum. Latency must have p95 at most 2 seconds and
p99/max at most the 90-second stale-response bound, with at most five reconnects.
No malformed, non-finite, wrong-shaped, joint-limit, gripper-range, or joint-step
action may be applied; the live numeric metric contract must also report horizon
15, dimension 8, and finite=true. The current scene must be a lit tabletop with the
DROID reset posture, open gripper, reachable red cube, exterior view, and hand-mounted
wrist view. The action adapter uses absolute seven-joint targets in Franka order and
DROID's `0=open, 1=closed` gripper convention; raw distribution mismatches remain
separate from safety-projection counters. Acceptance additionally requires measured
end-effector approach to within 12 cm, supported Isaac cube-to-finger contact evidence,
at least 5 cm of cube lift from initialized tabletop height, and at least one continuous
second of lifted contact with a closed gripper. Action issuance alone is not success.
These thresholds are fixed before live execution and are not reduced after observing a
run.

## Policy data contract

Arbitrary logs or telemetry are not training data. Every collected `.npz` episode
must carry arrays `observation_state`, `observation_image_workspace`,
`observation_image_wrist`, `action`, `reward`, `terminated`, `truncated`,
`timestamp`, plus JSON `provenance`. Lengths must agree; timestamps must increase;
observations/actions must be finite; and exactly one of terminated/truncated must
be true on the final frame only. The action width must match `action_schema`.
The pinned LeRobot ACT path currently requires at least two physically meaningful
action channels; collection fails closed rather than padding or duplicating a
single-channel control.

The `npa.antioch.episode.v1` provenance includes scenario, case, seed, parameters,
engine and SDK versions, source SHA-256, asset hashes, observation/action schemas,
and FPS. Incompatible episodes fail collection, leaving no completion marker.
Validated episodes are converted by the real NPA LeRobot v3 adapter, with
`meta/antioch-provenance.json` retaining provenance. This supports static offline
imitation training. It does **not** turn the export into an online PPO/RSL-RL
environment.

Public publication runs `scan_image_antioch_payload.py` before push and against
the exact pushed bytes. It inspects OCI config, full history, and every layer for
renamed vendor distributions, native proprietary signatures, vendor auth/config
state, checkpoints, credentials/private keys, and Antioch/Isaac/Omniverse payload.
Positive and negative fixtures and CI path registration guard this enforcement.

`--robot-type` and `--task` are required at submit/run time and are bound into
the idempotent operation record before the remote run starts. Collection always
uses those immutable values. Missing metadata fails before submission; there is
no cartpole fallback and a later collector cannot silently relabel a dataset.

The executable example
`npa/workflows/workbench/npa-workflows/antioch-offline-policy-train.yaml` follows
collection with real LeRobot ACT training and publishes a genuine checkpoint.
The `workbench.antioch.run` toolRef reads its idempotency state identity from
`config.antioch_state_id`. When a workflow contains multiple Antioch stages, set
a distinct `antioch_state_id` in each state's `params`; retries of the same stage
must reuse that value together with the workflow run id.

## Security and cleanup

The adapter filters sensitive keys and bearer/JWT/signed-URL forms from CLI errors
and log objects. It never emits environment dumps, identity fields, config files,
tokens, or customer metadata. Use only synthetic/public projects for validation.
After a smoke, cancel only the run ids created for that smoke, then release only
the associated project machine if one was allocated; queued managed execution
normally requires no persistent operator machine.
