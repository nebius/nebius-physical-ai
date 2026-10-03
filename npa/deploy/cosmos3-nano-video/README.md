# Cosmos3-Nano: 30-second video service

[Workbench docs](../../../docs/workbench/README.md) · [Runtime details](runtime-details.md) · [Recorded measurements](measurements.md)

This operator recipe serves diffusion video generation through vLLM-Omni and
Ray Serve. It runs **16 replicas on 16 B200 GPUs**, with one GPU and TP=1 per
replica. Use it for 30-second continuation or source-conditioned appearance
augmentation. Audio is disabled and this recipe explicitly runs with
**guardrails off** (`--no-guardrails`).

For the guarded Cosmos Framework generation path, start with
[Cosmos 3 generation](../../../docs/workbench/cosmos3-generate.md) or
[native Cosmos Ray Serve](../../../docs/workbench/cosmos3-ray-serve.md).
They use different images and APIs.

## Before provisioning

| Requirement | What to verify |
| --- | --- |
| Compute | Task-owned Nebius Kubernetes cluster with 16 B200 GPUs, a CPU pool, and validated managed CUDA 13 drivers |
| Shared storage | A verified RWX filesystem and `csi-mounted-fs-path-sc`; the example PVC requests 512 GiB |
| Image | Validated immutable digest of your **operator-private** image; this image is excluded from the public NPA publisher |
| Model | Anonymous payload access to the pinned Cosmos3-Nano checkpoint; see the [payload probe](runtime-details.md#deploy-and-operate) |
| Control plane | KubeRay operator 1.7.0 watching `workbench`; its managed Ray credentials remain separate from the inference token |
| Client | Installed NPA, project-scoped S3 credentials, a private service route, and a writable private recovery directory |

Run `npa workbench health preflight --checks nebius --json` before provisioning.
Follow the [GPU driver procedure](../../../docs/workbench/mk8s-gpu-driver-strategy.md)
and [runtime infrastructure requirements](runtime-details.md#infrastructure-and-image).
Keep reservation identifiers and registry coordinates in private configuration.

## Deploy in order

The manifests are templates. `kubectl apply` does **not** expand their `${...}`
placeholders. Render private copies with these exact values before applying:

| Variable or Secret | Use |
| --- | --- |
| `NPA_COSMOS3_VIDEO_IMAGE` | Operator-private `registry/image@sha256:...` |
| `NPA_COSMOS3_SHARED_PVC` | Bound RWX claim; the supplied manifest names it `cosmos3-nano-video-shared` |
| `NPA_B200_GPU_PRODUCT` | Exact `nvidia.com/gpu.product` label on validated B200 workers |
| `cosmos3-nano-video-registry` | Registry pull Secret in `workbench` |
| `cosmos3-nano-video-auth`, key `token` | Separate inference API token in `workbench` |

1. Provision and validate compute, shared filesystem, and KubeRay. Confirm the
   CPU head can select nodes without `nvidia.com/gpu.count`.
2. Create the two Secrets and apply [shared-pvc.yaml](shared-pvc.yaml). Confirm
   the claim is `Bound` before starting the staging Job.
3. Apply your rendered [weights-job.yaml](weights-job.yaml). Require completion
   and its verified `READY.json`; this Job is the only model-cache writer.
4. Apply your rendered [rayservice.yaml](rayservice.yaml). Require all **16**
   model replicas healthy and correctly placed on B200 GPUs.
5. Verify authenticated inference readiness, then connect through a private
   route or the loopback forwarding command below.

Inspect the selected context's resources:

```bash
kubectl -n workbench get pvc cosmos3-nano-video-shared
kubectl -n workbench get job cosmos3-nano-video-weights
kubectl -n workbench logs job/cosmos3-nano-video-weights
kubectl -n workbench get rayservice cosmos3-nano-video
```

Use the stable **head** Service: this recipe sets `proxy_location: HeadOnly`,
so worker endpoints in the generated Serve Service do not host the HTTP proxy.
Keep this terminal open:

```bash
kubectl -n workbench port-forward service/cosmos3-nano-video-head-svc 8000:8000
```

The [deployment details](runtime-details.md#deploy-and-operate) cover Ray
management authentication, model staging, service routing, and readiness.
A running pod or live HTTP process does not establish that all models are ready.

## Configure the client

Set these variables in your private shell or secret store:

| Variable | Value |
| --- | --- |
| `NPA_COSMOS3_VIDEO_ENDPOINT` | `http://127.0.0.1:8000` when using the tunnel above |
| `NPA_COSMOS3_VIDEO_TOKEN` | Token from the inference Secret |
| `NPA_COSMOS3_VIDEO_RECOVERY_DIR` | Private local directory outside the checkout |
| `NPA_COSMOS3_SINGLE_OUTPUT_URI` | Fresh run-scoped S3 destination for continuation |

Also configure NPA's S3 endpoint and credentials. Use a new output prefix for
each generation; existing immutable artifacts must not be overwritten.

## Generate a continuation video

```bash
npa workbench cosmos3 nano-video-batch --concurrency 1 \
  --output-path "$NPA_COSMOS3_SINGLE_OUTPUT_URI"
```

Expect three generated chunks and a stitched MP4 with **720 frames, 24 fps,
832×480, and 30 seconds**, plus hashes and join contact sheets. The client fully
decodes the media and verifies publication by reading S3 objects back.
Inspect both joins; successful decoding does not prove seamless visual motion.
See [frame arithmetic](runtime-details.md#model-and-frame-contract).

The SDK entry is `npa.sdk.workbench.cosmos3.nano_video_batch`. For concurrent
requests, use a fresh batch destination and the desired `--concurrency`; the
[recorded eight-request run](measurements.md#measured-b200-acceptance) reports
its specific timing and overlap, not sustained saturation throughput.

## Augment an existing video

Provide an S3 MP4 at **832×480 and 24 fps**, with at least six frames. This route
preserves the complete source duration and does not silently resample or trim
it. Set the input/output URIs and your positive/negative scene descriptions:

```bash
npa workbench cosmos3 nano-video-augment \
  --input-path "$NPA_COSMOS3_AUGMENT_INPUT_URI" \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" \
  --prompt "$NPA_COSMOS3_AUGMENT_PROMPT" \
  --negative-prompt "$NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT" \
  --seed 42 --num-inference-steps 35 --guidance-scale 3 \
  --control-guidance 1.5 --flow-shift 10 --edge-threshold medium \
  --chunk-frames 121 --output-format json
```

Inspect `input.mp4`, `augmented.mp4`, and `comparison.mp4`. The comparison shows
**source on the left, augmentation on the right**. Review source motion,
identity, lighting, materials, and every join; changed bytes alone do not prove
a useful augmentation. See the [control and chunk contract](runtime-details.md#source-conditioned-visual-augmentation)
and [recorded augmentation assessment](measurements.md#measured-full-source-augmentation).

## Recover and finish

After an interrupted augmentation download or publication, recover the same
request instead of generating again:

```bash
npa workbench cosmos3 nano-video-augment-recover \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" --output-format json
```

Recovery verifies existing bytes and retrieves completed artifacts through the
authenticated API when needed. Missing state after an interrupted POST is
ambiguous; preserve the recovery directory and inspect it before any new request.

Save and verify S3 outputs before removing the serving deployment. Stop new
requests, let accepted work finish, close the port-forward, and remove only the
RayService, staging Job, Secrets, storage, and infrastructure you own. Shared
filesystem and cluster deletion are separate lifecycle steps; see
[teardown](../../../docs/teardown.md).

<a id="model-and-frame-contract"></a>
<a id="source-conditioned-visual-augmentation"></a>
<a id="infrastructure-and-image"></a>

## Detailed contracts and evidence

The former detailed sections are preserved in [runtime-details.md](runtime-details.md).

<a id="measured-b200-acceptance"></a>
<a id="measured-full-source-augmentation"></a>

Historical latency, memory, concurrency, visual review, and image-scan results
are preserved in [measurements.md](measurements.md). They apply to the recorded
image and deployment; later source changes have not received a new GPU validation
through this documentation update.
