# Cosmos3 B200 checkpoint evaluation — 2026-08-14

Status: **complete**. The campaign generated and reviewed 72 still images on one
reserved-capacity NVIDIA B200 in Nebius: 40 images in the five
checkpoint primary matrix and 32 new images for the two-checkpoint consistency
phase. The primary-seed images were reused, not regenerated, in the three-seed
analysis.

## Decision

Invest further in **`Cosmos3-Super-Text2Image`** for still-image quality. It
scored 4.650/5 on the directly comparable primary seed, 0.700 points (17.7%)
above Nano, then ranked first across all three seeds at 4.717/5. Its seed-level
standard deviation was 0.051 and its worst seed averaged 4.650.

`Cosmos3-Super-Text2Image-4Step` is the throughput alternative. It won the
single primary seed at 4.750/5 and its cache-warm generation averaged 0.626
seconds/image, but its three-seed score fell to 4.608 with twice the seed-level
variation (0.101) and lower prompt adherence (3.958 versus 4.375). The
three-seed evidence therefore reverses the primary-only order rather than
overfitting the recommendation to seed 314159.

The quality gain beyond Nano is material for this Physical AI prompt set, but
it requires a much larger memory envelope: Super Text2Image peaked at 132,488
MiB versus Nano at 39,548 MiB and its cache-warm generation averaged 6.010
seconds/image versus Nano at 2.329 seconds/image.

## Evaluation contract

- Primary matrix: five checkpoints × eight prompts × seed `314159` = 40 images.
- Consistency: the primary image plus seeds `271828` and `161803` for each of
  the top two checkpoints × eight prompts = 48 reviewed images, of which 32
  were newly generated.
- Prompts cover block stacking, gear insertion, warehouse material handling,
  roadwork perception, quadruped contact, drone inspection, liquid pouring,
  and cable routing. The exact text and pinned revisions are in
  [`cosmos3-checkpoint-eval.json`](../../npa/workflows/workbench/configs/cosmos3-checkpoint-eval.json).
- All checkpoints used the pinned framework defaults, a 1:1 aspect ratio, and
  each checkpoint's supported native still-image resolution. The required
  4Step schedule was not replaced with an incompatible common step count.
- Blind review randomized checkpoint identity independently within each
  prompt. One Codex multimodal reviewer scored prompt adherence, visual
  quality, geometry, contact/physics plausibility, and visible artifacts on an
  integer 1–5 scale. Contact sheets normalized display size; original blind
  files were also inspected at native detail. For visible artifacts, 5 means
  artifact-free.

## Primary matrix

Wall time includes checkpoint download/load and all eight generations. “First”
and “warm” are the framework generation timer; warm is the mean after the first
sample. Peak memory is the maximum sampled `nvidia-smi` memory use on the single
B200. Byte totals cover the eight JPEG outputs.

| Checkpoint | Blind score | Adherence | Output | Wall (s) | First (s) | Warm (s/image) | Peak MiB | Bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Cosmos3-Edge | 1.925 | 1.250 | 640×640 | 166.297 | 19.726 | 1.838 | 12,344 | 911,834 |
| Cosmos3-Nano | 3.950 | 3.875 | 960×960 | 237.048 | 17.538 | 2.329 | 39,548 | 1,839,986 |
| Cosmos3-Super | 4.350 | 4.000 | 960×960 | 796.182 | 21.523 | 6.006 | 132,490 | 1,686,431 |
| Cosmos3-Super-Text2Image | 4.650 | 4.750 | 960×960 | 772.181 | 14.966 | 6.010 | 132,488 | 1,743,086 |
| Cosmos3-Super-Text2Image-4Step | 4.750 | 4.375 | 1024×1024 | 718.612 | 11.080 | 0.626 | 131,982 | 2,030,636 |

Every arm completed 8/8 samples, every JPEG was decodable and nonblank, and no
arm failed. Decodability/nonblank checks were health checks only; the blind
review supplied the quality judgment.

## Three-seed consistency

Scores aggregate 24 images per checkpoint: eight prompts at each of the three
fixed seeds. Seed σ is the population standard deviation of the three
eight-image seed means. Prompt σ is computed across the three seeds for each
prompt.

| Checkpoint | Overall | Adherence | Seed means (`161803`, `271828`, `314159`) | Seed σ | Worst seed | Mean prompt σ | Max prompt σ |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Cosmos3-Super-Text2Image | **4.717** | **4.375** | 4.725, 4.775, 4.650 | **0.051** | **4.650** | **0.196** | **0.377** |
| Cosmos3-Super-Text2Image-4Step | 4.608 | 3.958 | 4.550, 4.525, 4.750 | 0.101 | 4.525 | 0.267 | 0.471 |

The extra-seed generation arms also completed without sample failures:

| Checkpoint | Seed | Wall (s) | Warm (s/image) | Peak MiB | Output | Bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Super Text2Image | 271828 | 783.811 | 5.996 | 132,490 | 960×960 | 1,701,220 |
| Super Text2Image | 161803 | 92.740 | 6.012 | 132,488 | 960×960 | 1,805,329 |
| Super Text2Image 4Step | 271828 | 781.927 | 0.625 | 131,984 | 1024×1024 | 2,426,612 |
| Super Text2Image 4Step | 161803 | 44.881 | 0.620 | 131,982 | 1024×1024 | 2,190,123 |

The first extra seed for each checkpoint includes a fresh runtime checkpoint
download; the second uses the same arm-local cache. Checkpoint caches were
evicted at checkpoint boundaries, and no primary-seed work was duplicated.

## Nebius execution evidence

The campaign used one strict-reservation-backed NVIDIA B200 with 183,359 MiB of
memory and driver `580.95.05`. The primary workload succeeded in 45m48s with
5/5 arms and 40/40 images; the consistency workload succeeded in 29m07s with
4/4 arms and 32/32 newly generated images. Both reported zero recoveries.
Controller/setup-only failures produced no generation artifacts and are kept
separate from model-arm accounting. Temporary campaign compute was removed
after artifact verification.

Exact tenant, project, capacity, cluster, node, bucket, registry, network, job,
and artifact-location identifiers are retained only in access-controlled
external campaign evidence. They are intentionally excluded from Git and public
collaboration surfaces.

## Runtime and checkpoint provenance

The workload used one repaired, reusable runtime image for every checkpoint:

- Runtime image index: `sha256:7635e798c70ce8e53d0836170888dd77b1580bba7b46eba7eefd9d5a82f5cf15`
- Runtime amd64 manifest: `sha256:4973ebddfcaf223d1fabee4c60316df266d0a15c8d042748a85576b86740d0a6`
- Published parent index: `sha256:c65712832f6a50f3734d9b01a10699352d59b37949f551c38a33859ba0eedae8`
- Pinned Cosmos framework commit: `5e67049cd94acb667786f1e6dd0dab821cb90c97`

The additive campaign repair supplied command forwarding and `rsync` for the
SkyPilot Kubernetes runtime. A built-byte scan found no checkpoint or guardrail
weights among 75,342 image entries; a reserved-B200 smoke reported the expected
GPU and driver. All model and guardrail assets downloaded only at workload
runtime. The canonical Dockerfile now carries the same prerequisites, but its
standard image still requires a post-merge build, scan, and publication. The
campaign repair is evidence, not a claim that the canonical image was published.

| Asset | Provider revision verified before provisioning |
| --- | --- |
| `nvidia/Cosmos3-Edge` | `9eff1178c4e6dcbed1e27c8113109a6cd6d3706e` |
| `nvidia/Cosmos3-Nano` | `411f42a8fdfb8c5b2583cb8786e0938f49796eaa` |
| `nvidia/Cosmos3-Super` | `e0262be9d8f7586bc24c069a2aed2b665bdff266` |
| `nvidia/Cosmos3-Super-Text2Image` | `6d029507a7b5e0c35642aa48d24d043e379c6bcd` |
| `nvidia/Cosmos3-Super-Text2Image-4Step` | `0573a4b26b8e15d13d416e51f4680c8bc8b8c33d` |
| `nvidia/Cosmos-Guardrail1` | `d6d4bfa899a71454a700907664f3e88f503950cf` |
| `Wan-AI/Wan2.2-TI2V-5B` VAE | `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| `Qwen/Qwen3Guard-Gen-0.6B` | `fada3b2f655b89601929198343c94cd2f64d93cc` |

Authenticated provider metadata and artifact redirects were checked without
downloading weights during preflight. The phase manifests bind the normalized
runtime config at
`192f59f72961ac47e720e4cc8315a03d3c2bc6fcd24202ff3ae588d941e85554`;
the exact uploaded config bytes have SHA-256
`0691ea72935bfd7d34708ce5d39765696fd891780314722eebdb2f15e58e474f`.

## Artifacts

Bulky media, manifests, blind-review material, reconciliation records, teardown
records, and their exact object-store locations remain in access-controlled
external campaign evidence, not Git or public collaboration surfaces.

The primary review artifact SHA-256 is
`b03bf3638e758068851a8593080971452a2a91f2af7e9b28dde1f415d6a4bd36`.
The consistency unblinded review SHA-256 is
`139eee29bd3a0d3b1eda699f6a91813044b3f3beb44e2ddf2be52f32a41c6fdd`.
The completion evidence SHA-256 is
`9d3be61818308cbff68eef1579f331fd59ac73a1ea9f4a269b064c4696e267c2`;
it records 72 media objects, 111 pre-completion campaign objects, and zero
weight-file objects under the campaign prefix.

## License, consent, and guardrail posture

Each operator must independently accept and have access to the OpenMDW 1.1
Cosmos3 checkpoints, the NVIDIA Open Model License guardrail, and the
Apache-2.0 Wan VAE and Qwen guard model before runtime download. The completed
campaign's task-scoped acceptance record is retained in access-controlled
external evidence; it does not carry forward to another operator and never
authorizes redistribution or baking weights into an image.

Guardrails were requested and remained enabled. Prompt input used the upstream
blocklist plus Qwen3Guard and a blocked prompt would fail its arm. RetinaFace
postprocessing was active. At the pinned upstream commit, however, the
generated-media content-safety model list is intentionally empty because the
upstream filter was disabled after excessive false positives; every generation
therefore logged `No safety models found, returning safe`. These results have
**not** passed generated-image content-safety screening. The limitation was not
silently hidden or treated as a quality pass.

## Limitations

- A single multimodal reviewer produced all blind scores; this is a
  decision-quality internal comparison, not an inter-rater study.
- Native checkpoint resolutions differ (640, 960, or 1024 square). This
  preserves supported checkpoint behavior but means the review is not a strict
  pixel-count ablation; normalized contact sheets reduce display-size bias.
- Only eight Physical AI prompts and three seeds for the top two checkpoints
  were tested. The result does not establish performance on arbitrary domains.
- Latencies are from one reserved B200 and include different cache/download
  states as labeled. They are not fleet-wide throughput guarantees.
- Output-content safety was fail-open at the pinned upstream commit, as
  documented above.

## Reproduce

Validate and plan the reusable workflow before submitting it with a staged
campaign config and runtime credentials:

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/cosmos3-checkpoint-eval.yaml
npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/cosmos3-checkpoint-eval.yaml \
  --run-id <run-id>
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/cosmos3-checkpoint-eval.yaml \
  --image "${NPA_COSMOS3_DIGEST_REF}" \
  --var "source_sha=${NPA_SOURCE_SHA}" \
  --var "campaign_config_uri=${NPA_CAMPAIGN_CONFIG_URI}" \
  --var "bucket=${NPA_ARTIFACT_BUCKET}" \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

`NPA_COSMOS3_DIGEST_REF` must be a registry-qualified immutable digest reference,
not a tag, and `NPA_SOURCE_SHA` must be the exact source revision baked into that
image. Rendering fails before GPU submission when either proof is missing. The
campaign config's `runtime_image_digest` must match the selected canonical image;
stage that config and all exact infrastructure values outside the repository.

The workflow is
[`cosmos3-checkpoint-eval.yaml`](../../npa/workflows/workbench/npa-workflows/cosmos3-checkpoint-eval.yaml).
Use `eval_phase=primary` first. Select the top two checkpoints from the blind
primary review, then use `eval_phase=consistency` with both top-checkpoint
fields. The evaluator rejects a consistency run that repeats the primary seed.
