# OpenPI Franka operation

## Deployment and cache boundary

Follow `docs/workbench/antioch-openpi-franka.md` for the user workflow and
`npa/examples/antioch-openpi-franka` for the pinned hosted scenario.

- Resolve the B200 policy and RTX adapter/runtime-fetch images to immutable
  digests. Pin the hosted scenario's NPA source revision separately.
- Warm the exact OpenPI checkpoint and tokenizer at runtime under the operator's
  accepted upstream terms. Key cache identities by provider, artifact, immutable
  generation/revision, and format. Serialize writers, download to unique
  temporary paths, verify files/checksums, atomically publish a ready marker,
  and refuse partial or mismatched identities.
- Default to pod/node-local ephemeral cache. Use a durable PVC only when
  configured, and prove a restarted policy pod reuses it. The warmer alone gets
  entitlement and read-write access; the server gets the verified cache
  read-only and no entitlement.
- Test cold population, concurrent writers, warm reuse, corruption refusal and
  recovery, revision separation, missing entitlement, and read-only serving.

## Private cross-GPU transport

Keep the B200 policy endpoint as ClusterIP port 8000. Use an ingress policy that
allows only the RTX bridge. For Antioch-hosted execution, bind the relay backend
to the service interface reached only through Antioch's declared authenticated
port, keep its OpenPI frontend on loopback, and use `policy_tunnel_connector.py`;
do not expose the policy publicly or copy Kubernetes credentials to Antioch.

Tunnel readiness requires all of:

1. supported Antioch service/API state is ready;
2. the B200 `/healthz` endpoint passes through the local port-forward;
3. the connector completes a session carrying repeated valid OpenPI
   handshakes/requests;
4. exterior and wrist camera sequence/timestamps advance over a sustained
   interval;
5. multiple exact finite `[15,8]` chunks are observed; and
6. multiple rate-limited targets are safely applied while physics and viewport
   frames continue advancing.

An empty accepted tunnel session commonly means the hosted frontend opened and
closed before the local connector reached the policy, the service port was not
converged, or one side restarted. Keep the policy port-forward alive, re-check
service/API and health state, then reconnect with bounded exponential backoff.
Do not loop without a cap, bypass auth, scrape cookies, or declare success from
an open socket.

## Distinguish the four rates

- **Console video** proves that Antioch can stream a viewport to a viewer. It
  does not prove the bridge captured policy observations.
- **Observation streaming** is the completed exterior+wrist+state acquisition
  sequence. Under backpressure, only the newest completed observation is kept.
- **Policy cadence** is bounded by one in-flight request over one persistent
  binary MessagePack WebSocket. Requested frequency is not achieved frequency.
- **Physics control** consumes the current validated receding-horizon chunk and
  safely holds on underrun. It continues independently of policy latency.

Python, WebSocket, Kubernetes, and the authenticated relay provide soft-real-
time operation only. Never claim a hard-real-time rate or deterministic bound.
Declare readiness only from advancing camera timestamps, multiple successful
round trips, multiple safe applications, and a sustained interval. Publish the
measured observation/control FPS, inference latency, frame/response age, drops,
underruns, reconnects, rejections, and safe-target counts without frames,
prompts, endpoints, or infrastructure identities.

For stalled frames, first distinguish an active console viewport from advancing
camera sequence. Check the supported service/API state and camera callback, then
allow only a bounded render-only warmup for an initially empty camera tensor;
do not send observations or apply actions until both frames are complete. Then
check the observation drop count. For stale actions, compare observation/response age
to the configured maxima and confirm the reconnect epoch advanced. For tunnel
jitter, keep latest-observation semantics, reduce requested policy cadence if
needed, and measure—never queue stale frames. Any timeout, disconnect, malformed
reply, unsafe target, queue underrun, or epoch mismatch must enter the configured
hold/no-action behavior until a fresh validated chunk arrives.

## Compatibility failures to handle explicitly

- **Single environment camera batches:** accept ordinary RGB frames or the
  leading one-environment dimension; reject other rank/shape rather than
  silently squeezing arbitrary input.
- **Hosted viewport:** Antioch owns Kit startup. Run with authenticated streaming
  enabled, require an active viewport, render and advance the supported Kit app
  loop until the capture callback completes, and fail on timeout or malformed
  RGBA bytes. Do not create a second SimulationApp.
- **Standalone cameras:** launch Isaac with camera support and configure exterior
  and wrist sensors inside the same one-environment scene.
- **Compute-only managed driver:** CUDA readiness does not prove Vulkan
  readiness. Require an NVIDIA ICD plus a real `vulkaninfo` renderer probe. If
  the node runtime omits graphics userspace, use the repository's simulator-only
  init stage: it fetches the exact running-driver `no-compat32` runfile and
  upstream SHA-256 directly from NVIDIA under runtime acceptance, atomically
  publishes an immutable volume tree, and mounts it read-only into the bridge.
  Never install it on the host, copy it into an image, substitute a different
  driver series, or treat `nvidia-smi` as rendering evidence.
- **Franka assets:** probe the runtime-advertised immutable asset root. If its
  Franka sentinel is unpublished, use only the reviewed published compatibility
  root after probing it. Rewrite both module asset constants and any task config
  imported before the rewrite. Never guess a mutable asset URL or fall back to a
  local untracked asset.
- **Imports:** keep Isaac/control imports lazy so CPU render/CLI paths remain
  importable. Keep the policy health helper isolated from dataset, manager,
  storage, and optional control dependencies.
- **Identity:** pin hosted source, image digest, adapter version, engine, SDK, and
  asset compatibility independently in evidence. Record the driver-matched
  graphics runtime identity separately. An image version is not the runtime,
  engine, or driver-userspace version.

## Control safety and evidence

Continuous soft-real-time streaming is the production default. The finite
one-observation/one-chunk path is an explicit smoke only. Validate observation
keys, two `uint8[224,224,3]` images, seven joints, one
normalized gripper value, and a bounded prompt before serialization. Validate
the response as finite absolute targets shaped `[15,8]`; enforce Franka joint
limits, gripper `[0,1]`, per-step joint delta, execution-step cap, observation
and response age, connect and inference timeouts, and bounded reconnect backoff.
Use a capacity-one latest-observation queue, capacity-one response handoff, a
bounded action horizon, bounded metric windows, and a local sequence/timestamp/
epoch. Reset the epoch on reconnect so a pre-disconnect reply cannot execute.

Timeout, malformed MessagePack, wrong shape/dtype, non-finite or out-of-range
values, stale replies, camera failure, rate-limit failure, underrun, and
exhausted reconnects must apply no new policy target and must safely hold/stop.
Previously safe applications remain evidence; never overstate this as zero total
targets after a later stream fault. Capture sanitized camera shapes/backend,
compute capability, action shape, achieved rates/latencies, rejection counts,
executed-target count, asset identity, and fail-closed status. Never record
frames containing customer data without explicit artifact authorization.
