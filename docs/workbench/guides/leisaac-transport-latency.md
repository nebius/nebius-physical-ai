# LeIsaac low-latency browser transport

This runbook describes the measured transport behind the public agent's LeIsaac
keyboard teleoperation. It is deliberately separate from the recorder state
machine: video congestion must not delay controls, and transport changes must
not change LeRobot, MP4, S3, task/environment, or PAIDF lineage contracts.

## Measured original path

The original public path was:

1. a browser keyboard event or `<img>` refresh;
2. public HTTPS and Basic Auth at nginx;
3. an HTTP FastAPI agent adapter;
4. synchronous HTTP over the agent's loopback relay;
5. an authenticated WebSocket backhaul to the Kubernetes pod;
6. the LeIsaac session HTTP server and JSONL simulator input queue;
7. the RTX PRO 6000-backed simulator/viewport;
8. a JPEG file, then the same relay, agent, nginx, browser decode, and paint.

The browser scheduled the next JPEG request 150 ms after the preceding image
loaded. A press and release were separate HTTP requests. The runtime used an
HTTP/1.0 server that closed each request. The measured source viewport produced
11.28 FPS, while only 1.58 FPS reached browser paint under the control workload.
This is evidence of serialization and repeated connection/request work. It is
not evidence that FastAPI request dispatch by itself was the bottleneck.

The before trial used one public run/task/environment and 160 HTTP control
acknowledgements, 80 event-to-aggregate-application observations, 100 UI frames,
100 direct frame fetch/decode/paint samples, 2,031 pod frame observations, and
30 cross-host clock brackets:

| Metric | n | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| control request/ack RTT | 160 | 195.3 ms | 340.3 ms | 440.5 ms |
| event to observed simulator aggregate | 80 | 534.4 ms | 839.2 ms | 1,042.9 ms |
| encode-commit to browser paint estimate | 100 | 642.8 ms | 834.4 ms | 935.6 ms |
| frame bytes | 100 | 47,278 | 47,495 | 47,578 |

The old protocol had no per-control sequence/application timestamp and no frame
capture timestamp. Therefore the second row is an observation upper bound. The
third row joins the exact JPEG SHA-256 to the pod's encode-commit mtime, but its
cross-host wall offset has ±536.0 ms best-probe uncertainty and capture started
before that mtime. Do not treat it as exact capture age. The new protocol carries
monotonic stage timestamps and frame identity so its after trial is more precise.

## Design

The measured public RTX path uses two independent, authenticated, same-origin
WebSockets so video backpressure cannot delay reliable controls:

- `control`: bounded JSON messages with a random per-page client ID, monotonic
  sequence, press/release event, client monotonic/wall timestamps, and separate
  accepted and simulator-applied acknowledgements. The shared ledger enforces
  ordering and idempotency, retains a bounded replay window, and releases held
  controls on disconnect. The same ledger accepts an exact eight-value direct
  SO-101 action for browser gamepads and the declarative custom-device contract;
  video and S3 work never share its socket or queue.
- `video`: a bounded binary WebSocket carries a 128-byte envelope followed by
  JPEG bytes. It carries frame
  sequence, capture/encode wall and monotonic timestamps, runtime/agent stage
  timestamps, byte length, drop count, and SHA-256. Runtime and agent relays
  each retain only the latest bounded value per camera and credit receipt before
  accepting another frame. The browser retains one replaceable decode candidate
  per camera and skips a decoded frame only if a newer candidate for that camera
  arrived before paint. A one-bit flag routes workspace and overview to separate
  canvases, and fair selection prevents one viewport from starving the other.

FastAPI routes remain adapters. Ordering, message/frame limits, binary framing,
backpressure, and counters live in the shared `leisaac_transport` module shipped
to both the agent and runtime. File reads, simulator input writes, applied-ack
scans, storage discovery, health calls, and recorder operations cross an async
thread boundary instead of blocking an ASGI event loop.

The browser uses reliable WebSocket control plus binary WebSocket video for the
measured public profile. The optional authenticated WebRTC data-channel routes
remain available for compatibility experiments, but real full-quality JPEG and
control trials over this deployment's TURN path were slower. If the WebSocket
contract fails, the UI uses an explicit `JPEG polling · fallback` indicator.
Reconnect retains sequence state, resends only idempotent
unacknowledged controls, and uses application ping/pong samples to estimate
clock offset and uncertainty. The HTTP frame/input routes remain authenticated
and tested.

The simulator keeps the 60 Hz real-time limiter, CUDA PhysX, Fabric, full
1280×720 quality-82 capture, and explicit rendering for every due background or
causal frame. Browser mode removes the unused policy camera tensors, raises the
steady-state Isaac Lab `render_interval` only after environment construction,
and disables the rate limiter's redundant render side effect. Thus control and
physics do not pay for an idle viewport render, while the first post-apply
primary capture still advances the real renderer immediately. `single_fast` is
the default and uses only the stable interactive primary camera unless the
episode schema explicitly requests both cameras. `dual_slow` serializes the
overview camera at 2.5 FPS only after higher-priority causal and final-orbit
primary work. Runtime, relay, and browser queues retain bounded latest values;
the requested and scheduler-applied mode revisions remain separate until the
GPU/readback/encoder boundary is safe.

Security properties:

- nginx Basic Auth remains the public authentication boundary;
- because the browser WebSocket constructor cannot attach an authorization
  header, the Basic-authenticated API mints a 120-second `Secure`, `HttpOnly`,
  `SameSite=Strict` cookie scoped only to `/api/leisaac/transport`; FastAPI
  verifies its HMAC, expiry, run ID, and nginx-attested client address before
  resolving or contacting a runtime;
- control and video cookies are single-use. The pinned NVIDIA signaling client
  may consume its cookie twice: once for the initial connection and once for
  its single internal reconnect. A further or app-level reconnect must obtain a
  fresh same-origin cookie, so the 120-second credential is not an unconstrained
  replay window;
- no password, nonce, or long-lived secret appears in a WebSocket URL;
- the SDP offer is a bounded exact JSON object accepted only from the
  Basic-authenticated same HTTPS origin with the explicit control/CSRF header;
  neither SDP nor ICE credentials are logged;
- the public adapter requires HTTPS, exact `Origin == Host`, one exact
  subprotocol, one bounded `run_id` query parameter, a valid selected manifest,
  and matching runtime health;
- the runtime requires its per-session nonce, exact run ID, and exact
  subprotocol or offer schema; it is not an arbitrary upstream proxy;
- compatibility WebRTC data channels are TURN-only in the browser, the peer
  stays in the private GPU pod, and the image exposes no additional port;
- Uvicorn and the shared parser bound message/frame size and queue depth;
- metrics contain only fixed counter names and no run/client/secret labels.

## Primary sources and implications

- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/) and
  [FastAPI WebSocket testing](https://fastapi.tiangolo.com/advanced/testing-websockets/):
  use native ASGI WebSocket endpoints and deterministic `TestClient` coverage.
- [FastAPI concurrency guidance](https://fastapi.tiangolo.com/async/): synchronous
  file, HTTP, storage, and simulator boundaries must not run on the event loop.
- [Starlette WebSockets](https://www.starlette.io/websockets/): accept/receive/send
  through the ASGI WebSocket state machine and handle disconnect explicitly.
- [Starlette responses](https://www.starlette.io/responses/) and
  [FastAPI streaming responses](https://fastapi.tiangolo.com/advanced/stream-data/):
  HTTP streaming is supported, but a single response does not provide the
  bidirectional ordered-control protocol needed here.
- [Uvicorn settings](https://github.com/Kludex/uvicorn/blob/main/docs/settings.md)
  and [deployment guidance](https://www.uvicorn.org/deployment/): bound WebSocket
  message/queue sizes, configure ping intervals/timeouts, and disable compression
  for already-compressed JPEG payloads.
- [nginx WebSocket proxying](https://nginx.org/en/docs/http/websocket.html) and
  [proxy module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html):
  explicitly forward `Upgrade`/`Connection`, use HTTP/1.1, disable proxy/request
  buffering, and choose timeouts compatible with heartbeat traffic.
- [MDN WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket),
  [`binaryType`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/binaryType),
  and [`bufferedAmount`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/bufferedAmount):
  request `arraybuffer` binary delivery and never use an unbounded browser send
  backlog as flow control.
- [MDN `createImageBitmap`](https://developer.mozilla.org/en-US/docs/Web/API/Window/createImageBitmap),
  [`requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame),
  and [WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API):
  decode the bounded latest compressed frame off the DOM image-loader path and
  count presentation only in a paint callback. WebCodecs remains a future codec
  option because the current runtime produces independently decodable JPEGs and
  the measured change did not require a new browser codec dependency.
- [MDN `createDataChannel`](https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/createDataChannel),
  [`maxRetransmits`](https://developer.mozilla.org/en-US/docs/Web/API/RTCDataChannel/maxRetransmits),
  [aiortc API](https://aiortc.readthedocs.io/en/latest/api.html), and the
  [W3C WebRTC recommendation](https://www.w3.org/TR/webrtc/): the evaluated
  compatibility route can bound SCTP reliability independently from control,
  but the measured TURN path is not selected merely because it is UDP-based.
- [FFmpeg format options](https://ffmpeg.org/ffmpeg-formats.html) and
  [libjpeg-turbo documentation](https://libjpeg-turbo.org/Documentation/Documentation):
  use MP4 `+faststart` for Range-based episode playback and the system's
  optimized JPEG implementation for low-copy independent live frames; already
  compressed payloads do not use WebSocket per-message deflate.
- [WHATWG WebSockets](https://websockets.spec.whatwg.org/) and
  [RFC 6455](https://www.rfc-editor.org/info/rfc6455): WebSocket messages retain
  their order, while application-level sequence IDs still provide idempotency,
  reconnect recovery, and simulator-application evidence.
- [Python monotonic clocks](https://docs.python.org/3/library/time.html#time.monotonic)
  and [browser high-resolution time](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API/High_precision_timing):
  calculate same-host intervals with monotonic clocks; report an explicit
  offset/RTT uncertainty when comparing wall clocks across machines.
- [Isaac Lab `SimulationCfg`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg)
  and [AppLauncher](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html):
  use the supported `cuda:0` device, preserve camera-enabled initialization,
  and change the render interval only after the environment has initialized.

The bottleneck attribution and split control/video transports are measured design
inferences. The cited sources establish protocol/runtime behavior, not the
performance of this deployment.

## Validation procedure

Run the protocol/runtime tests and the browser transport tests:

```bash
npa/.venv/bin/python -m pytest -q \
  npa/tests/cli/test_leisaac_transport.py \
  npa/tests/cli/test_agent_leisaac.py \
  npa/tests/workbench/test_leisaac.py
(cd npa/tests/browser && npm run cy:mock)
```

For before/after comparison, use the same public URL, task, named environment,
seed, GPU, control count, frame count, resource-monitor interval, and browser
viewport. Preserve raw JSON. Report sample counts, p50/p95/p99, error rate,
source and delivered FPS, frame bytes, explicit drop/coalesce counters, network
deltas, and CPU/RSS. Verify accepted controls equal simulator-applied controls.

Then perform a real recorder cycle (`start`, control activity, outcome,
`finalize`), verify the unique S3 version/commit and H.264 MP4, load the dataset
with LeRobot 0.5.1, capture the live transport/latency/teleoperation/recorder UI,
and leave a detached health monitor running.
