# Cosmos3-Nano video runtime details

[Deployment and client guide](README.md) · [Recorded measurements](measurements.md)

## Model and frame contract

The [official Nano model card](https://huggingface.co/nvidia/Cosmos3-Nano/blob/7a312c868bcce8e40b3eb40861300a9d0ba3fde1/README.md)
was checked before generation. It currently specifies **5–400 output frames**,
up to **five conditioning input frames**, and output at the requested fps. The
often-quoted 4 fps recommendation applies to reasoner inputs. Only BF16 is
officially tested. This recipe deliberately retains a stricter **300-frame
per-request ceiling**.

The [baked diffusion pipeline](https://github.com/vllm-project/vllm-omni/blob/9c1b7504b178afcf541867c1a2d30db48c69cda8/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)
rounds non-transfer frame counts upward to `4k+1`; 300 would become 301. At 24 fps
and 832×480, the rollout requests **297, 297, and 137 frames**. The first request
is text-to-video. Subsequent requests upload the previous segment as MP4 and
set `condition_frame_indexes_vision: [0, 1]` and `condition_video_keep: last` in
`extra_params`. These are latent indices: two latent positions represent five
pixel frames, which remain clean conditioning through denoising.

Stitching discards the duplicated five-frame prefix from each continuation,
then trims one final frame: `297 + 292 + 132 - 1 = 720`, exactly 30 seconds.
There is no blending or interpolation to hide discontinuities. The client
retains the chunks and eight-frame contact sheets around joins at 12.375 and
24.5417 seconds. Generation uses 35 steps, guidance 6, flow shift 10, and a
4096-token sequence setting inside `extra_params` (the image's HTTP handler
does not parse a top-level `max_sequence_length` form field).

## Source-conditioned visual augmentation

Continuation preserves a generated tail and extends time. Visual augmentation
instead transforms an existing video's appearance while conditioning every
output interval on the corresponding original frames. The earlier continuation
benchmark is evidence for continuation and concurrency, not augmentation quality.

`nano-video-augment` accepts a real S3 source MP4 at **832×480, 24 fps**, with at
least six frames. It rejects invalid hashes, incomplete decoding, changing
dimensions, incorrect timestamps and unsupported controls before GPU work. Output
duration matches the complete source; no silent resampling or truncation occurs.
The CLI and SDK share one client and the server's request/report contract.

The installed upstream `make_edge_control` computes Canny edges from each
original source interval. Controls are stored as lossless FFV1 videos and must
round-trip pixel-exactly through the actual transfer loader. No extra learned
preprocessor checkpoint is needed. The first interval uses zero RGB conditioning
frames. Later intervals upload the preceding **augmented five-frame tail** for
continuity while using edges from the matching **original** interval for motion.
This is the image's structural transfer path, not continuation prefix conditioning.

With the default 121-frame windows, a 720-frame source uses intervals starting
at frames **0, 116, 232, 348, 464, 580 and 696**. Their lengths are six windows
of 121 frames and one of 24. The final model window is 25 frames; native padding
is trimmed back to 24. Removing the repeated five-frame prefixes yields
`121 + 5×116 + 19 = 720` frames, without blending or interpolation. All seven
requests run sequentially on one selected replica. The service retains the
original 16-replica least-outstanding plus FIFO policy and holds its GPU lock
until accepted generation finishes, including repeated client cancellations.

`--chunk-frames 297` uses three source intervals starting at **0, 292 and 584**,
with lengths **297, 297 and 136**. The last model window is 137 frames and is
trimmed to the source length. Dropping the two five-frame prefixes gives
`297 + 292 + 131 = 720` frames, with joins at frames 297 and 589. Longer windows
reduce the number of independently sampled boundaries; they do not remove the
need to inspect each join or guarantee better motion.

The default transfer parameters are 35 steps, text guidance **3**, control
guidance **1.5**, flow shift **10**, medium Canny thresholds **100/200**, BF16,
TP=1, shared control/target temporal positions, and a 4096-token limit. These
start from the installed single-edge transfer defaults; increasing guidance or
steps is not inherently a quality improvement. Steps and text guidance also
respect the actual video API's upper bounds of 200 and 20. There is no generic
`strength`, `control_weight`, arbitrary server-side file path, or unchecked
extra-parameter bag.

Use a detailed scene description, preferably a structured JSON object covering
subjects, setting, lighting, materials, camera motion and temporal continuity,
following [NVIDIA's transfer guidance](https://github.com/NVIDIA/Cosmos/tree/main/cookbooks/cosmos3/generator/transfer).
The exact submitted and effective positive, negative and system prompts are
retained per interval. The installed formatter rewrites JSON duration to the
integer local model-window duration even when metadata templates are disabled;
the retained source-frame map supplies the exact global timestamps.

```bash
npa workbench cosmos3 nano-video-augment \
  --input-path "$NPA_COSMOS3_AUGMENT_INPUT_URI" \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" \
  --prompt "$NPA_COSMOS3_AUGMENT_PROMPT" \
  --negative-prompt "$NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT" \
  --seed 42 --num-inference-steps 35 --guidance-scale 3 \
  --control-guidance 1.5 --flow-shift 10 --edge-threshold medium \
  --chunk-frames 121 --output-format json

# Retrieve completed work or retry artifact downloads/publication; never generates again.
npa workbench cosmos3 nano-video-augment-recover \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" --output-format json
```

SDK entries are `npa.sdk.workbench.cosmos3.nano_video_augment` and
`nano_video_augment_recover`. The authenticated API accepts multipart `/run`
with exactly `request` (JSON) and `input_reference` (the complete MP4). Existing
JSON continuation requests remain supported. `GET /result?request_id=...`
retrieves durable state through any replica on the shared output filesystem.
A missing result after an interrupted POST is ambiguous and never authorizes an
automatic generation retry.

Artifacts keep **`input.mp4`**, **`augmented.mp4`** and the synchronized, labeled
**`comparison.mp4`** separate, alongside source controls, augmented RGB tails,
chunk outputs, exact requests, hashes and measurements. The comparison places
the actual source on the left. S3 reservation and publication use conditional
immutable writes with hash-verified readback. Recovery accepts existing objects
only when their bytes match and can finish publication without a serving token
when the completed local result is already verified.
If artifact retrieval fails after generation, recovery uses authenticated GETs
to retrieve the same completed request before validating and publishing it;
the original submission marker remains unchanged.

Technical validity is separate from visual quality. A changed hash does not
prove meaningful augmentation. Evaluate requested appearance change, lighting,
materials/reflections, identity, source motion, wheel contact and every actual join
against a rubric fixed before candidate scoring. Record agent/VLM judgments and
their sampling limitations separately; do not present them as a standardized
quality score or claim eight-way augmentation from the continuation benchmark.

## Infrastructure and image

Use NPA `cluster up` or the NPA fleet mk8s backend with a task-owned project in
`us-central1`, 16 `gpu-b200-sxm` GPUs, managed CUDA 13 drivers,
a CPU pool, and shared filesystem storage. Ray uses 16 one-GPU worker pods,
so the cluster may provide two eight-GPU nodes or a validated mix of B200 node
sizes. Keep NPA's quota, fabric,
stabilization and per-node CUDA validation enabled. If using reserved capacity,
supply its exact group through owner-only runtime configuration and require
`STRICT` reservation placement. Follow the [fleet instructions](../../../skills/tools/fleet/SKILL.md)
and [driver strategy](../../../docs/workbench/mk8s-gpu-driver-strategy.md).

The head explicitly advertises zero Ray GPUs and hides NVIDIA/CUDA devices.
Its required affinity selects nodes without `nvidia.com/gpu.count`; verify that
the NPA CPU pool has no such label and every validated GPU node has a positive
count before applying the manifest. Worker device visibility stays under the
Kubernetes device plugin and Ray's one-GPU assignment.

For a mixed cluster, declare the planned per-node GPU counts through
`GpuHealthConfig.expected_gpu_counts`. Its optional `nvswitch_gpu_counts` subset
identifies node sizes that require fabric checks. Every multi-GPU SXM/NVL size
must be included. An explicitly unattached one-GPU guest can report fabric as
not applicable while still passing driver, device-plugin and CUDA checks.
The mixed live health test verifies the declared total, node-count distribution,
stability and CUDA on every GPU ordinal before serving begins.

The Dockerfile extends `vllm/vllm-omni:cosmos3`, pinned to manifest digest
`sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.
The Dockerfile pins that index's Linux AMD64 manifest,
`sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c`.
The extension supplies Ray Serve, FFmpeg and the NPA adapters. It preserves the
diffusion engine and adds no model weights, task or customer data, credentials
or acceptance state. The inherited image contains public vendor test fixtures
and example media. Its vendor runtime makes this an **operator-private** image;
the NPA public publisher excludes it. Build and push only to the operator's own
registry, then deploy the verified immutable image digest.

The staging Job is the single writer of the shared model cache. It downloads
the pinned Nano revision once, verifies all diffusion and VAE tensor dtypes as
BF16, hashes the staged files, and atomically publishes `READY.json`. Replicas
mount the cache read-only and run with Hugging Face offline mode. Their vLLM
processes preserve `--init-timeout 1800`. The synchronous HTTP generation route
has no additional workload deadline. Each replica runs one complete rollout
at a time; Ray's custom router considers all replicas at one rank and chooses
the least outstanding queue, with capacity rejection protecting concurrent
admission.

The router uses Ray's public FIFO fulfillment mixin to keep pending requests
reachable when another scheduler has already consumed their routing metadata.
FIFO selects the pending request; fresh queue snapshots across all replicas
select the least outstanding replica. The router follows Ray's normal retry backoff.
It disables Ray 2.56's queue-length cache because the cached-success path can
spin after out-of-order assignments and accumulate background probes in the
head proxy. Every selection still compares the complete replica rank; strict
admission prevents two requests from occupying the same replica.

The adapter writes an explicit single diffusion stage configuration with
`model_config.sound_gen: false`, BF16 and TP=1, then passes it through
`--stage-configs-path`. This supported legacy flag is needed for the pinned
engine: its single-stage diffusion fallback drops `--stage-overrides`, and the
checkpoint's sound dimensions would otherwise enable an unstaged audio
tokenizer. The image's CPU validation resolves the actual engine configuration
and checks that sound remains disabled; CLI parsing alone is insufficient.

## Deploy and operate

1. Verify provider credentials with `npa workbench health preflight --checks
   nebius --json`, and anonymously probe the pinned checkpoint payload using
   the command below before provisioning. Verify the selected image and storage
   before starting generation. The broader `health access --capability cosmos3`
   also checks gated guardrail weights; this recipe disables guardrails and does
   not fetch those weights. Check their actual upstream access separately if
   enabling a guardrail-dependent route.
2. Provision the NPA mk8s cluster and shared filesystem. Install KubeRay operator
   **1.7.0** with its namespace watch restricted to `workbench`. Resolve all
   manifest placeholders from owner-only configuration. Keep the operator's
   standard namespace-scoped Secret permissions: `rayClusterConfig.authOptions`
   enables token authentication, and KubeRay creates a separate management Secret,
   injects it into the head and workers, and authenticates its dashboard requests.
   Do not override `RAY_AUTH_*` in the pod templates or reuse the inference token.
3. Apply [shared-pvc.yaml](shared-pvc.yaml) after NPA installs its shared-filesystem
   CSI driver. Create the API token Secret and operator-private registry pull
   Secret, then run the weight staging
   Job to completion. Run the RayService only after the immutable cache is ready.
4. Require all 16 model replicas to be healthy, confirm B200 placement, and
   verify both the API and Ray dashboard reject unauthenticated requests. Expose access through an
   authenticated private route or a local port-forward.
5. Set `NPA_COSMOS3_VIDEO_ENDPOINT`, `NPA_COSMOS3_VIDEO_TOKEN` and
   `NPA_COSMOS3_VIDEO_RECOVERY_DIR` outside Git. Configure the normal NPA S3
   endpoint and credentials in the client process. Use distinct artifact
   prefixes for each batch, separate from any agent trajectory dataset.

Ray management credentials grant code execution and are for administrators only.
KubeRay's [token authentication integration](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/kuberay-auth.html)
uses its existing namespace-scoped RBAC; inference clients need only the separate
API token. Management authentication does not encrypt cluster traffic. Keep Ray
ports on the trusted private cluster network and use an authenticated encrypted
administrative tunnel when accessing the dashboard remotely. The application
builder refuses a cluster without explicit token authentication. For standalone
containers, the default `--serve` launcher creates a fresh owner-only management
credential before starting local Ray, disables the dashboard and removes the
credential on shutdown. Docker health checks use the authenticated inference
endpoint and do not need access to the management credential.

The payload preflight reads one byte from a diffusion shard at the exact model
revision. It uses no Hugging Face token and does not stage the checkpoint:

```bash
npa/.venv/bin/python - <<'PY'
from urllib.request import Request, urlopen

revision = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
shard = "transformer/diffusion_pytorch_model-00001-of-00007.safetensors"
url = f"https://huggingface.co/nvidia/Cosmos3-Nano/resolve/{revision}/{shard}"
with urlopen(Request(url, headers={"Range": "bytes=0-0"})) as response:
    if response.status != 206 or len(response.read(1)) != 1:
        raise RuntimeError("Exact-revision Nano payload access was not verified")
print("Anonymous pinned Nano payload access: PASS")
PY
```

For `proxy_location: HeadOnly`, route port 8000 through KubeRay's stable
`cosmos3-nano-video-head-svc`, for example
`http://cosmos3-nano-video-head-svc.workbench.svc.cluster.local:8000` inside the
cluster, or a local port-forward to that Service. KubeRay updates this Service's
head-only selector when the active cluster changes. The generated
`cosmos3-nano-video-serve-svc` is unused: KubeRay 1.7 always labels worker pods
as serving endpoints, while these workers have no HTTP proxy. The explicit
worker readiness probe checks Ray's native health endpoint on port 52365;
all 16 model replicas must separately pass application readiness. Custom
`serveService.spec.selector` values are
[overwritten by KubeRay](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/controllers/ray/common/service.go#L216).

```bash
npa workbench cosmos3 nano-video-batch --concurrency 1 \
  --output-path "$NPA_COSMOS3_SINGLE_OUTPUT_URI"
npa workbench cosmos3 nano-video-batch --concurrency 8 \
  --output-path "$NPA_COSMOS3_FANOUT_OUTPUT_URI"
```

The SDK entry is `npa.sdk.workbench.cosmos3.nano_video_batch`. The authenticated
service accepts `/run`; `/artifacts/<request_id>/<filename>` serves the retained
files. CLI and SDK verify returned hashes, fully decode all three chunks and the
stitched MP4, and
publish immutable S3 objects with read-after-write verification. Publication
failure retains the local recovery copy; **do not repeat GPU generation to
retry an upload**.
