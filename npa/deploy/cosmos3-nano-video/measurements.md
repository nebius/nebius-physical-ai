# Cosmos3-Nano video measurements

[Deployment and client guide](README.md) · [Runtime contract](runtime-details.md)

## Measured B200 acceptance

The measurements below are retained evidence from the September 6 deployment.
Subsequent PR maintenance reconciled current main and hardened failed-start
cleanup and loopback readiness against ambient proxies. Those lifecycle changes
have CPU regression coverage; they have not been rebuilt into a newly validated
GPU image. These measurements apply to the previously tested artifacts.

After adding augmentation and shared-server cancellation handling, acceptance was rerun on the updated image: one full 30-second continuation video through the SDK, then eight complete continuation requests concurrently through the CLI. All nine clips passed full decoding at 832×480, 24 fps and 720 frames. The two batches published 13 and 97 immutable objects respectively, with read-after-write hash verification. Models were already initialized and BF16 weights prestaged before timing.

The updated-image single-request batch took **155.85 s**; the eight-request batch took **161.23 s**. These batch times include generation, downloads and local validation, and exclude S3 publication. The complete sequential acceptance test case, including both batches and their S3 publication/readback, took **323.87 s**; pytest reported **324.88 s** for the full invocation. Deployment, model initialization and prestaging are excluded.

| Request | Chunk 1 (s) | Chunk 2 (s) | Chunk 3 (s) | Server total (s) | Client total (s) | Peak allocator reserved (MiB) | Peak device used (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 59.89 | 60.04 | 22.46 | 149.99 | 155.82 | 37408.00 | 38316.00 |
| C1 | 59.60 | 60.45 | 22.74 | 150.82 | 155.66 | 38220.00 | 39130.00 |
| C2 | 60.74 | 61.29 | 23.07 | 154.47 | 161.14 | 37408.00 | 38316.00 |
| C3 | 60.12 | 60.68 | 22.98 | 152.85 | 160.05 | 37408.00 | 38316.00 |
| C4 | 59.18 | 59.83 | 22.40 | 150.07 | 154.55 | 37408.00 | 38316.00 |
| C5 | 60.91 | 61.07 | 22.79 | 153.94 | 161.08 | 37408.00 | 38316.00 |
| C6 | 60.00 | 60.62 | 23.01 | 152.24 | 158.89 | 37408.00 | 38316.00 |
| C7 | 60.48 | 60.95 | 22.89 | 153.46 | 161.01 | 37408.00 | 38316.00 |
| C8 | 59.64 | 60.59 | 22.95 | 152.18 | 159.24 | 37408.00 | 38316.00 |

S1 is the single request; C1–C8 belong to the concurrent batch. Chunk time is the synchronous diffusion HTTP round trip inside the replica. Server total includes generation, full decode, stitching, seam extraction and GPU sampler teardown. Client total additionally includes routing, artifact downloads and client validation; it excludes S3 publication.

Both memory columns use MiB. Despite its name, the engine’s `X-Peak-Memory-MB` header divides bytes by `1024**2` and reports the peak CUDA allocator reserved pool. Device usage comes from the assigned B200’s `nvidia-smi memory.used` sampled every 0.5 seconds; this includes total device residency and can miss shorter spikes. These are separate measurements.

The eight-request batch completed **8/8** videos on **eight distinct replicas**, with **eight overlapping rollout intervals** and **eight overlapping diffusion chunk requests**. These results cover one eight-request batch on the 16-replica deployment; sustained saturation throughput was not measured.

After the timed acceptance, the exact deployed image separately passed the repository B200 checker with Torch 2.11.0+cu130 and native `sm_100` support. Its shipped golden-evaluation command also generated a complete three-chunk rollout, and all four MP4s independently decoded successfully. All 16 serving replicas and their pods remained unchanged. This separate check ran alongside the serving model; its timing and memory are excluded from the production measurements above.

The original pre-augmentation acceptance also remains valid: its single and eight-request batches took **153.61 s** and **163.04 s**, respectively, and all nine videos and 110 published objects passed validation. Those measurements use the earlier image.

During that original run, across 75 successful CPU-head samples taken every 5 seconds, proxy RSS peaked at **214.68 MiB** (private USS **111.47 MiB**) and whole-head cgroup usage peaked at **1639.55 MiB**. The same proxy process and head pod remained present, with zero sampled OOM events. This observation window includes baseline and completion handoff; it is separate from generation latency, and five-second samples can miss brief spikes.

Separate independent reviews of the original nine clips and all 18 joins in the updated-image nine-clip acceptance found no obvious hard cuts or scene/vehicle identity resets in the eight-frame contact sheets. Small wheel, reflection and edge changes remain visible around some joins. Static contact sheets do not establish imperceptible motion continuity; the stitch uses direct concatenation without blending.

The private image scan retained 218 HIGH and six unfixed CRITICAL findings, with zero fixable CRITICAL findings and zero detected secrets. The targeted dependency updates remove the base image’s fixable CRITICAL findings. One inherited package metadata mismatch remains: `nixl` requires `nixl-cu13==1.3.0`, while the vendor image installs 1.3.1; the selected TP=1 single-stage diffusion route does not configure NIXL transfer. No new dependency conflicts were introduced.

Exact resource identities, private endpoints, registry coordinates and generated artifacts remain in access-controlled runtime storage. This page includes measurements and generic configuration only.

## Measured full-source augmentation

Eleven complete variants were generated from the same task-generated warehouse
robot video. The selected 30-second result changes the bright clean scene into
a dimmer warehouse with warm overhead illumination, rough dark damp concrete,
localized shallow puddles and restrained reflections. The original orange robot,
aisle geometry, camera travel and scene timing remain recognizable. The retained
comparison labels the **actual source on the left** and **augmentation on the
right**, correcting the earlier source/output labeling confusion.

The selected run used **35 steps, text guidance 5, control guidance 1, flow shift
10, high Canny thresholds 200/300, 297-frame windows, seed 480240 and 4096 tokens**.
BF16, TP=1, shared control/target temporal positions, audio off and guardrails off
remain as described in the [runtime contract](runtime-details.md). Its structured positive prompt specifies matte orange
paint, transparent shallow water over rough concrete, warm-white practical lamps,
restrained haze and source-matched travel. The structured negative prompt targets
glowing orange floor smears, trench-like lane markings, plastic materials and
motion/contact defects. Exact submitted and effective prompts are retained with
the private artifacts. Increasing steps to 50 or guidance did not consistently
improve realism; lower Canny thresholds did not improve wet-floor realism in
the same 297-frame configuration.

| Source interval, zero-based inclusive | Preparation (s) | Diffusion HTTP (s) | Peak allocator reserved (MiB) | Peak device used (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 0–296 | 24.84 | 167.09 | 38012 | 38918 |
| 292–588 | 6.85 | 165.86 | 38012 | 38918 |
| 584–719 | 6.51 | 53.08 | 38012 | 38920 |

Server wall time was **440.21 s**, including control preparation, generation,
validation and stitching. The client submission/download/validation interval was
**450.51 s**, excluding source S3 setup and output publication. The complete live
test case, including source setup, immutable publication/readback and recovery
verification, took **455.38 s**; JUnit reports **456.05 s** for the pytest suite. The
selected request overlapped two other augmentation requests on distinct existing
replicas for part of its execution; this is not an isolated throughput benchmark.
The memory sampling limits above apply. Input and output each fully decoded to
**720 frames, 24 fps, 832×480 and 30.000 s**; the synchronized comparison is
1664×480 with the same frame/timestamp contract.

The rubric was frozen before generation. Independent agent review produced the
following task-specific ordinal judgments; the hosted VLM results use a separate
0–1 scale and are retained without forcing agreement.

| Component | Agent (0–4) | Hosted VLM (0–1) |
| --- | ---: | ---: |
| Requested environmental change | 3 | 0.75 |
| Lighting realism | 3 | 0.50 |
| Materials and reflections | 3 | 0.50 |
| Robot identity | 3 | 0.75 |
| Source geometry, camera and timing | 4 | 0.75 |
| Motion and floor contact | 3 | 0.50–0.75 |
| Temporal continuity | 3 | 0.50 |

The motion rating initially was 2. A separately retained review of unmodified
native tire/contact crops supported qualitative source-following travel and
stable floor contact, yielding 3 with moderate confidence under the unchanged
anchors. The original review remains preserved. Exact tire angle, speed and
no-slip physics remain unverified: repetitive dark tread and hub covers provide
weak angular cues in both source and output. Automated CPU estimates did not
establish rolling fidelity. Complete browser playback decoded all 720 frames
without dropped frames; agent review also inspected dense sequences across both
joins. These observations are not a human rating or a standardized aggregate
quality score. The actual hosted VLM was MiniMax-M3; its 17 sampled-frame calls
sometimes inferred freezing from weak motion cues, reversed front/rear wheel
descriptions or used inconsistent verbal anchors, so its scores cannot certify
physical motion.

Remaining imperfections include weaker puddles and robot reflections late in the
clip, small hub/tire-detail changes and occasional rectangular floor sheen. The
result is a noticeable, plausible augmentation with these limits. It does not
establish an eight-way augmentation batch; the separate single-plus-eight test
above measures continuation.

To reuse the selected settings, provide a new immutable destination and the
desired structured positive, negative and system prompts through the environment:

```bash
npa workbench cosmos3 nano-video-augment \
  --input-path "$NPA_COSMOS3_AUGMENT_INPUT_URI" \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" \
  --prompt "$NPA_COSMOS3_AUGMENT_PROMPT" \
  --negative-prompt "$NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT" \
  --system-prompt "$NPA_COSMOS3_AUGMENT_SYSTEM_PROMPT" \
  --seed "${NPA_COSMOS3_AUGMENT_SEED:-480240}" \
  --num-inference-steps 35 --guidance-scale 5 --control-guidance 1 \
  --flow-shift 10 --edge-threshold high --chunk-frames 297 \
  --max-sequence-length 4096 --output-format json
```
