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
(version dated 2026-02-28) for the scoped use of `antioch-sim==0.4.236` and the
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

The public adapter pins `antioch-sim==0.4.236` and its reviewed SHA-256. On first
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
releasing its project session.

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
redacted form. It does not construct undocumented service URLs.

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
The versioned `openpi_franka_pickup_v3` remote scenario identity prevents a
previously published definition from masking the schema-4 camera/action contract.
Its default dispatched instruction is `pick up the red cube`; public proof telemetry
uses only the non-sensitive `red_cube_pickup` label.

The two 224x224 policy cameras use Isaac Sim 6's supported
`isaacsim.sensors.experimental.rtx` API: separate `RtxCamera` authoring objects
and `CameraSensor` runtime objects with an explicit `rgb` annotator. The scenario
authors a nonzero 15 Hz sensor tick rate, commits timeline play once, and
uses a completed `world.step(render=True)` before every sensor read. There is
one stepping owner. Adding a second Replicator orchestrator step can invalidate
the policy render products and is not part of this contract.

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
| Antioch SDK/CLI | `antioch-sim==0.4.236`; public scenario surface exposes `scenario`, `ScenarioRun`, `Logger`, `world`, and `engine` | Exact runtime check; any other version is unsupported until reviewed. |
| Antioch engine | `antioch-engine/isaac-sim-6.0.1:0.4.236` / engine identity `isaac-sim-6.0.1` | Exact runtime check; do not substitute a newer engine under the old scenario. |
| Isaac camera | `isaacsim.sensors.experimental.rtx.RtxCamera` + `CameraSensor`, documented uint8 RGB CPU output, immediate scenario-owned copy, nonzero sensor tick | Capability is exercised through public Isaac Sim 6 APIs. |
| Render advancement | timeline play committed once, followed by `world.step(render=True)` and exact producer-marker readback | No second stepping owner; a loop counter does not prove camera freshness. |
| Antioch control plane | Project revisions, project sessions, singular `service` operations, and named session routes | Re-discover the supported profile and project through structured CLI commands; never persist a service endpoint or console hostname. |

The Isaac contracts above are documented in the official [Isaac Sim 6 camera
guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_camera.html)
and [Replicator workflow guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html).
The current CLI package and its structured command help are the control-plane
contract. The [public Antioch site](https://antioch.com/) is product context, not
an API endpoint or a substitute for supported discovery.

To upgrade safely, obtain versioned vendor SDK and engine documentation, inspect
the installed public exports without reading auth state, update this boundary
and its dependency-injected scheduler tests, then repeat the full local gates.
Keep the PR draft until an authorized live run on the exact proposed versions
shows both producer clocks advancing, useful decoded camera images, executed
actions and every declared task check passing. Local unit tests do not prove
rendered appearance or policy task success. Do not infer camera readiness from
a healthy service, viewer state, loop count, or scheduler call alone.

The scenario keeps safety calculations and durable acceptance counters at control
cadence, but groups Rerun scalars and generated scene geometry at a documented
5 Hz display cadence. Fixed latest-only exterior-image, wrist-image, and display
publisher lanes each retain at most one replaceable pending sample. Slow JPEG
encoding or viewer transport cannot block the simulation/control loop or a policy
safe hold; image begin/ok/error markers and a separate loop heartbeat make a later
stall boundary explicit. These are bounded observability semantics, not a hard
real-time claim.

The OpenPI bootstrap is publicly installed as `npa-openpi-live-deploy` (implemented
by `npa.workflows.byof.openpi_live`): a single dynamically selected supported GPU
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
supported `antioch project build`, `antioch session new|status|release`,
`antioch service exec|cp|ps|ports`, and `antioch scenario list|run|cancel`
operations. Its foreground named-route bridge binds only on pod localhost. A
bounded relay container in that same pod network namespace connects the route's
authenticated WSS client role to a CA-verified, authenticated
ClusterIP OpenPI Service. The operator VM launches and observes this Deployment
but carries no camera frames, policy messages, or actions.

The private runtime file contains Kubernetes coordinates, the explicit supported
Antioch deployment profile, and paths to the existing Antioch config, assigned project-id file,
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
Antioch's current HTTPS/WSS control and session edges have no stable destination
CIDRs, so TCP 443 necessarily uses `0.0.0.0/0`. "Restricted traffic" here
means port- and direction-restricted, not destination-restricted; a manifest
assertion pins that limitation. A former
owned LoadBalancer may remain temporarily for rollback; run
`live-k8s-finalize-cutover` only after sustained acceptance to remove that exact
public Service. The retained policy Deployment and checkpoint PVC are reused, not
duplicated.

`live-k8s-stop` waits for sanitized controller evidence written only after the
exact scenario is terminal or stably absent and the project session is released,
then scales the Deployment to zero. Missing/malformed evidence, `cleanup_failed`, timeout,
or forced/SIGKILL termination remains unproven and is never reported as stopped.
Before release, the controller requires supported `session status` to match both
its exact project and the session identity it created; a replacement session is
never released as cleanup for the old controller.

```bash
npa workbench antioch live-k8s-deploy \
  --runtime-config /path/to/private-runtime.json --output json
npa workbench antioch live-k8s-status \
  --runtime-config /path/to/private-runtime.json --output json
```

The runtime schema is `npa.antioch.mk8s-live-config.v3`; the checked-in example
uses only placeholders. `antioch_deployment_profile` is required and is forwarded
to the controller as the vendor CLI's supported `ANTIOCH_ENV` selector; it has no
default because silently falling back to a different deployment can bind the wrong
project namespace. `policy_managed_by` must come from the verified retained-policy
handoff and must match the selected Deployment, authentication Secret, TLS Secret,
and NetworkPolicy before any is adopted. `antioch_config_dir` may point directly
at the current CLI's owner-only deployment profile; only `auth.json` and optional
`workspace.json` are copied into the pod. The current CLI's `.auth.lock` is ignored,
while legacy machine, tunnel, or cached endpoint state is rejected.
`workflow_run` plus `state_id` derive every adapter identity,
so independent Antioch stages cannot collide. `adapter_image` must be an immutable
digest. Deployment, status, stop, and cutover-finalization refuse unowned objects.

The default MK8s reference is `openpi_franka_pickup_v3`. It distinguishes
communication from measured manipulation: 5 cm of end-effector approach,
bilateral finger contact, and 5 cm of lift held continuously for one simulation
second are required for pickup success. Its `control_steps` parameter defaults
to 450 applied targets; exhaustion without pickup is an explicit failed result.
Camera exposure/contrast and initial target/gripper framing are separate gates.
The exact requests, lossless inputs, raw actions, applied targets and measured
physics are retained in `policy-evidence.zip` with a SHA-256 manifest. The
controller checks both the persisted verdicts and archive digest before retiring
compute. No failed finite attempt is automatically resubmitted.

The viewer leads with an independent native 1280x720 RTX recording camera and
keeps the exterior/wrist policy inputs beside it. `showcase-frames.zip` preserves
the rendered JPEG frames, producer markers, simulation timestamps and checksums
for offline video assembly. This spectator view never supplies policy inputs or
steps physics. Its recorded check requires advancing nonblack frames; physical
pickup checks remain separate. Live GPU validation must establish the final
framing, rendering performance and visual quality.

The explicit `openpi_franka_mk8s_live_v2` communication proof remains available.
It validates two finite `[15,8]` replies and executes both five-target segments
before returning. Its success does not claim a reach or pickup. Both identities
must persist as `completed/passed`; a timeout is never successful completion.
See the [example README](../../npa/examples/antioch-openpi-live/README.md) for
camera thresholds, rendering settings, episode parameters and evidence formats.

A completed controller remains live while retiring its session. Once shutdown
writes the pod's stop marker, a container restart refuses to initialize another
runtime or dispatch another scenario. The marker and prior evidence survive in
the pod's state volume; a new authorized attempt requires a fresh adapter identity.

To verify an existing record without another dispatch, set the owner-only runtime
configuration through `NPA_ANTIOCH_MK8S_RUNTIME_CONFIG` and the saved run identifier
through `NPA_ANTIOCH_COMPLETED_SCENARIO_ID`, then select
`test_completed_poc_checks_persist_after_session_release` in
`npa/tests/e2e/test_antioch_mk8s_live_e2e.py`. The existing live-test terms and
`NPA_INTEGRATION_E2E=1` gates still apply. This test independently reads the saved
checks twice; it does not deploy or start a scenario.

The supervisor atomically rechecks and re-stages the private client bundle after
session replacement. It builds an immutable project revision and starts that exact
revision through `session new`; an existing idle project session may be replaced,
but unrelated active work is never force-released. Typed retryable control-plane
failures during build or session startup remain in the same controller startup and
use capped backoff; fatal errors still fail immediately. A Mission Control stream in `ready` state is
published but waiting for an authenticated viewer; do not describe it as actively
viewed until the viewer connects and the first rendered frame advances.
The controller owns both the foreground `antioch service ports --bind
sim.policy-relay=127.0.0.1:18444 --serve sim` process and `antioch scenario run
--stream --verbose` as direct children. It drains scenario output in-process.
The controller cancels only the matching project-scoped run and proves stable exact
absence through supported `scenario list` JSON before dispatch. Failure recovery
may start one successor with capped backoff, but a verified successful proof is
terminal and never renewed. This prevents duplicate stream dispatch.

Adapter readiness is an exact session-ownership contract, not an open-port check.
The controller continuously reconciles the project-scoped scenario inventory with
supported structured `session status` and `service ps` output. Readiness requires
one running scenario whose session identity matches the current ready project
session, a healthy simulator process, and live directly owned scenario and named-route
children whose parent is container PID 1. State uses versioned JSON and owner-only atomic replacement. Local and
status-command readers base64-frame the file bytes to prevent transport-level JSON
coercion and retry a bounded number of transient empty/partial reads. Kubelet probes
use the same fail-closed state predicates over pod-local HTTP because exec-probe RPC
failures are not application-health evidence; ingress is restricted to the configured
kubelet CIDRs and the health ports have no Service. A
missing schema, malformed value, wrong identity, stale heartbeat, absent scenario owner,
mismatched or unhealthy session/service state, child exit, or unreadable
state revokes readiness. Converged loss terminates the exact child process group,
cancels the exact run, rebuilds and re-stages after recycle when needed, and fails closed for finite evaluations. Ambiguous ownership fails closed. The operator/Codex
process is not part of this supervision and may exit after handoff.

Before accepting the finite PoC, require a terminal `completed/passed` Antioch
record with the `exterior_observations_advancing_nonblack`,
`wrist_observations_advancing_nonblack`, and `pi05_responses_finite_15x8` checks all
passing. The record must show at least two observations per view, increasing first
and last accepted render sequences, camera luminance mean above 5 and variance
above 25, and at least two finite policy responses with shape `[15,8]`. Rejected
startup or flat pairs are counted but never enter the accepted
minimum. Every accepted response must arrive within the 90-second stale-response
bound.
No malformed, non-finite, wrong-shaped, joint-limit, gripper-range, or joint-step
action may be applied; the live numeric metric contract must also report horizon
15, dimension 8, and finite=true. The current scene must be a lit tabletop with the
DROID reset posture, open gripper, reachable red cube, exterior view, and hand-mounted
wrist view. The action adapter uses absolute seven-joint targets in Franka order and
DROID's `0=open, 1=closed` gripper convention; raw distribution mismatches remain
separate from safety-projection counters. End-effector approach, contact, cube lift,
and pickup hold are mandatory checks for the default pickup scenario. They remain
separate from the explicit communication-only proof. Never describe cube pickup as successful unless those independent task
measurements actually meet their declared thresholds. Action issuance alone is not
task success.

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
`workflows/testing/antioch-offline-policy-train.yaml` follows
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
the associated project session. Detached background execution is cancelled through
its exact run rather than through an unrelated interactive session.
