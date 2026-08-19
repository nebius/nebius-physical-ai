# Antioch / Isaac Sim Franka with an OpenPI B200 policy server

This path keeps rendering and inference on different GPU workloads:

The latest sanitized sustained-run measurements are in
[the continuous demo report](antioch-openpi-continuous-demo.md).

| Workload | GPU | Image and responsibility |
| --- | --- | --- |
| simulator bridge | RTX PRO 6000 (`sm_120`, RT cores) | digest-pinned `npa-isaac-lab`; runtime-fetches Isaac under the operator's NVIDIA acceptance, captures exterior and wrist RGB plus Franka state, validates and applies bounded position targets |
| policy server | B200 (`sm_100`) | digest-pinned OpenPI runtime image; an init container runtime-fetches `pi05_droid_jointpos_polaris` and its PaliGemma tokenizer only after the exact run-scoped Gemma gate, then the credential-free server serves upstream MessagePack/WebSocket on port 8000 from a read-only cache |

The policy Service is `ClusterIP`; no Ingress, NodePort, or load balancer is
created. A NetworkPolicy admits port 8000 only from the run's bridge pod. The
bridge bypasses ambient HTTP proxies only for the generated in-cluster service
and cluster-local DNS suffixes. The
Antioch configuration Secret, Isaac acceptance Secret, and optional S3 Secret
are mounted only into the simulator bridge. The policy pod receives only the
separate Gemma-terms Secret. Neither workload receives the other's credential.

## Protocol and fail-closed control

The bridge uses upstream OpenPI's MessagePack NumPy encoding without pickle.
Every request must contain exactly:

- `observation/exterior_image_1_left`: `uint8[224,224,3]`
- `observation/wrist_image_left`: `uint8[224,224,3]`
- `observation/joint_position`: finite `float32[7]`
- `observation/gripper_position`: finite `float32[1]`
- `prompt`: non-empty string

Every response must contain finite absolute targets with exact shape `[15,8]`.
The seven arm targets must be inside Franka position limits and the gripper must
be in `[0,1]`.

Continuous soft-real-time control is the default. Isaac render/physics work and
camera acquisition are cadence-scheduled on the simulator-safe thread while a
dedicated policy worker owns one persistent binary MessagePack WebSocket. The
worker permits at most one inference request in flight. Its observation queue
has capacity one: a newer completed exterior/wrist/state observation replaces a
superseded frame instead of accumulating stale work. Requests carry a local
sequence, monotonic capture time, and control epoch; responses older than the
configured age or produced across a reconnect epoch are rejected.

Validated chunks are consumed as a receding horizon. Five targets are eligible
by default, each joint changes by at most `0.08` radians per control step, and a
new valid chunk supersedes the unused tail of the old one. On underrun, timeout,
disconnect, malformed MessagePack, wrong shape, non-finite value, unsafe target,
or stale response, the bridge immediately uses the configured `hold-current` or
`no-action` behavior. It never substitutes a stale, random, clipped-through, or
best-effort policy target. Reconnect uses bounded exponential backoff and resets
the epoch before another response can execute.

These are soft-real-time semantics. Python, WebSocket, Kubernetes scheduling,
and the authenticated Antioch relay do not provide deterministic latency or a
hard-real-time guarantee. The bridge emits sanitized counts/rates and latency/
age percentiles only—never frames, prompts, endpoints, credentials, or live
infrastructure identities.

`--control-mode finite-smoke` retains the old one-observation, one-`[15,8]`
chunk diagnostic. It is a finite Job and is not the production mode.

Before Isaac starts, a non-GPU init container polls the private policy health
endpoint with bounded requests and exponential backoff. Its readiness deadline
defaults to 1,800 seconds and is configurable with
`--policy-ready-timeout-seconds`; expiry prevents simulator startup and action
application. Continuous bridge readiness is separate: a readiness marker is
published only after camera observation sequences/timestamps advance, multiple
policy round trips succeed, multiple safe targets are applied, and the minimum
sustained interval elapses. A live viewport, screenshot, PID, tunnel, or policy
health response alone is insufficient.

Some managed GPU node images expose CUDA compute but omit the Vulkan ICD and
NVIDIA GL/EGL userspace required by Isaac rendering. The bridge therefore has a
second, GPU-bound init stage. It reads the running kernel-driver version,
downloads that exact `no-compat32` driver runfile and its SHA-256 file directly
from NVIDIA under the runtime acceptance, extracts only into an operator-owned
volume, and atomically publishes a version-keyed ready tree. The bridge mounts
that tree read-only. A missing acceptance, unavailable checksum, mismatch,
partial tree, or absent Vulkan library prevents Isaac startup. This runtime
delivery does not modify the node and does not place driver userspace in the
public image.

The main streaming controls are exposed by `openpi-stack`:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--observation-hz` | 10 | requested camera/state acquisition cadence |
| `--policy-request-hz` | 2 | maximum inference request cadence |
| `--control-hz` | 10 | target-consumption/hold cadence |
| `--executed-targets-per-chunk` | 5 | receding-horizon prefix, from 1 to 15 |
| `--maximum-observation-age-seconds` | 0.75 | oldest observation accepted for a request |
| `--maximum-response-age-seconds` | 1.5 | oldest request/response accepted for control |
| `--camera-warmup-seconds` | 10 | bounded render-only wait for the first complete two-camera observation |
| `--inference-deadline-seconds` | 10 | socket/inference deadline before safe recovery |
| `--safe-hold-behavior` | `hold-current` | `hold-current` or `no-action` on underrun/failure |
| `--minimum-ready-cycles` | 3 | required round trips and applied policy targets |
| `--minimum-ready-seconds` | 5 | sustained interval before readiness |

Actual achieved rates depend on rendering, capture, policy, and transport
latency and must be read from the emitted metrics rather than inferred from the
requested values.

## Build and deployment

Build the policy image through the existing pinned `byof-openpi.yaml` path. It
pins OpenPI source revision `15a9616a00943ada6c20a0f158e3adb39df2ccac`, retains
the CUDA/JAX stack proven on B200, compiles the `sm_100` probe, and keeps the
checkpoint out of its layers. Build the bridge with the existing Isaac build
script; this change adds only the MessagePack/WebSocket client to its
resolver-closed control dependencies. Isaac and Antioch vendor bytes remain
runtime-only.

Resolve both pushed tags to registry digests, create the Secrets out of band,
and render the stack with generic, configurable selectors:

```bash
npa workbench antioch openpi-stack \
  --run-id <unique-run> \
  --policy-image '<private-registry>/openpi@sha256:<digest>' \
  --bridge-image '<private-registry>/npa-isaac-lab@sha256:<digest>' \
  --policy-terms-secret <gemma-acceptance-secret> \
  --isaac-acceptance-secret <isaac-acceptance-secret> \
  --image-pull-secret <private-registry-pull-secret> \
  --antioch-config-secret <antioch-runtime-config-secret> \
  --s3-credentials-secret <runtime-s3-secret> \
  --output-path 's3://<bucket>/<run-prefix>/bridge.json' \
  --output json
```

Create secrets from protected files or a secret-manager sync, never literal
values on a command line. The policy terms Secret is referenced only by the
single-writer cache init container. The policy server does not receive it, the
simulator never receives it, and the policy server never receives the Antioch
configuration or Isaac acceptance Secret.

### Runtime cache tiers

Without `--policy-cache-pvc`, the policy Deployment uses a named `emptyDir`.
That is the explicit node-local ephemeral tier: it survives container restarts
inside one pod but is lost with the pod or node and is never described as image
state. For production reuse, create a project-controlled PVC and pass its DNS
label with `--policy-cache-pvc <claim>`. The warmer mounts that volume
read-write; the policy server mounts the same volume read-only. A restarted
policy pod can therefore reuse a verified durable cache without redownloading.

The cache has two independently immutable GCS identities: the checkpoint tree
is keyed by provider, bucket/artifact, canonical object-generation-manifest
SHA-256, and format version; the tokenizer is keyed by provider, bucket/object,
exact generation, and format version. Population is serialized by a volume
lock, downloads into unique temporary directories, verifies the exact file set,
sizes, and upstream MD5 values, and publishes by atomic rename plus a ready
marker. The tokenizer and checkpoint normalization-assets paths expected by
upstream OpenPI are relative symlinks to their verified immutable identities,
so they remain usable when the server mount is read-only. An existing
mismatched alias is never overwritten. Partial or corrupt state fails closed
and can be rebuilt without replacing a valid identity.

Inspect the manifest, then repeat with `--apply`. Secret objects and values are
never rendered. The command rejects mutable image tags. After collecting the
report, rerun the same command with `--delete` to remove the exact rendered
policy Deployment, private Service, bridge Deployment or validation Job, and
NetworkPolicy; `--apply` and `--delete` are mutually exclusive.

## Antioch-hosted execution

`npa/examples/antioch-openpi-franka` is a thin Antioch scenario over the same
bridge function. Antioch's runner owns Kit startup; the wrapper does not fork a
second simulator or duplicate control logic. The exact NPA revision must be
installed in the private project image.

An Antioch account session is required only to allocate/start that hosted
engine and publish its managed scenario record. Select the intended organization
with supported `antioch auth switch` behavior and verify it with
`antioch auth whoami`; never inspect or copy the underlying auth file. Package
the example at the reviewed NPA revision, run suite
`openpi_franka_streaming` in the private Isaac Lab engine, and require advancing
camera observations, multiple finite `[15,8]` responses, multiple safely
applied targets, and sustained readiness. If the supported status call reports no usable session or
no assigned compatible machine, preserve the credential-free RTX/B200 result
and report that exact external assignment gate without attempting browser-token
extraction or credential recovery.

Antioch does not accept secret mounts in `antioch.yaml`. For hosted streaming,
keep the Kubernetes credential on the operator host: the example's loopback
reverse relay uses Antioch's authenticated port tunnel, and the local connector
pairs it with `kubectl port-forward` to the ClusterIP policy Service. This gives
the hosted simulator a private byte stream without copying a kubeconfig into the
Antioch service or creating an Ingress, NodePort, or load balancer. No supported
direct private Antioch-to-cluster network path is assumed by this example; use
one only when current documented Antioch APIs expose it. The authenticated
operator-host relay is the compatible fallback and its jitter/availability is
part of measured soft-real-time performance.

## Licensing and artifacts

- NPA and OpenPI source are Apache-2.0.
- The bridge image is eligible for public redistribution only because its built
  layers contain no Isaac, Omniverse Kit, Antioch SDK, checkpoint, cache, or
  credential bytes. Driver-matched Vulkan/GL userspace is delivered by NVIDIA
  into a runtime volume and is not an image layer. The image also uses distro FFmpeg instead of the separately
  licensed static executable bundled in the `imageio-ffmpeg` wheel. Publish an
  exact scanned digest only after the repository's guarded GHCR procedure.
- The OpenPI policy image is independently public-eligible only when its layers
  contain source/runtime dependencies but no pi0.5/Gemma checkpoint, model
  cache, access credential, or live infrastructure value. The operator fetches
  the checkpoint at runtime under the run-scoped Gemma acceptance.
- Isaac/Omniverse and Antioch runtime caches remain private runtime state and
  must never be committed or copied into a derived image.
- Polaris weights contain Gemma-derived material, are fetched only after the
  exact `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` runtime gate, and remain private.

Scan the built bridge with `scan_image_omniverse_payload.py`; scan the OpenPI
image with the BYOF/OpenPI built-byte checks. Acceptance changes permission to
run this workload, not redistribution rights.
