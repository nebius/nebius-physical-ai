---
name: flex-pi
description: Run, package, validate, or troubleshoot flex-pi inference and pinned public YAM training in NPA, including runtime-fetch assets, GPU workflows, and verified artifacts.
---

# flex-pi

Run the genuine upstream 6B world-action checkpoint. Do not replace policy
inference with an import check, random action fixture, or manifest-only smoke.

## Boundaries

- Source: `geyan21/flex-pi@20c1b2b71ea35a415d5d47c39b04443cfadad7a1`,
  MIT, baked without the vendored simulators.
- Checkpoint: `flex-pi/flexpi-robotwin@87d3833ea3bd89c4922945631db81b346e780785`,
  MIT-labelled, runtime fetch.
- Observation: `flex-pi/robotwin_3d@bee164afe94041d8c3d7dd1203725c41163fc3f4`,
  episode 0 frame 0. Its dataset card declares no license. Use only under the
  operator's authorized scope, fetch at runtime, verify each byte hash, and do
  not publish the media.
- DINOv3 uses its separate model license. The pinned converted Wan VAE/T5 and
  tokenizer repositories are Apache-2.0. They are operator runtime fetches;
  immutable revisions and heavyweight-file hashes are enforced before loading.
- The public image must contain none of those weights, media, credentials, or
  populated caches. Run `npa/scripts/scan_image_flex_pi_payload.py` against the
  built image before any registry write.

## Choose the regime and GPU

For real training use `workbench.flex_pi.train` and the maintained
`flex-pi-b200-public-training.yaml`. Its public `sort_utensils` dataset has
136/115,620 train and 16/12,390 validation episodes/anchors, separate CC-BY-4.0
rights, and the official YAM32 transforms. Require native four-GPU DDP,
BF16, microbatch one, accumulation 24, and effective batch 96. The final
36-sample update must preserve every anchor and use its actual denominator.
Training requires the full Wan VideoDiT and official derived ActionDiT
initialization; the checkpoint-only inference instructions below apply only to
the released RoboTwin inference checkpoint.

Freeze the bundled source/split manifests before profiling. Measure the
six excluded startup/trace updates and three eight-update steady windows
separately. Accept an execution-only optimization only after fixed-input
numerical and synchronized model-state parity. Full acceptance additionally
requires the complete epoch and validation split, finite gradients, non-regressed
validation, immutable checkpoint readback and matching fresh-process continuation.
Keep public fallback results explicitly non-comparable to any unverified private
reference, regardless of their numerical throughput. Record source-overlay
hashes separately from image qualification. See `docs/workbench/flex-pi.md`.

Use the released action-only regime for policy-serving validation: video and
DINO are observed, pointmap is absent, and all future-stream joint-denoising
flags are false. It produces the same 32-step, 14D bimanual action contract as
upstream evaluation without pretending an isolated observation proves simulator
success. The upstream model card reports equal average RoboTwin success for
action-only and full-joint inference, while action-only is substantially faster.

Route a reference workflow to one RTX PRO 6000 or one B200. The workload is
headless and does not need RT cores. These targets independently prove CUDA
`sm_120` and `sm_100`; neither substitutes for the other. Flex-pi inference
requires exactly one visible CUDA GPU. To exercise more capacity, fan out
independent one-GPU replicas with unique seeds and output prefixes. Never claim
that replica fan-out is model parallelism.

## Preflight and run

Before build, provisioning, or submission, prove Nebius and selected-project S3
credentials with `npa workbench health preflight --checks nebius,s3 --json`.
Probe the exact anonymous/public model and input objects before allocating a GPU.
Do not treat a generic Hugging Face token check as proof of those payloads.

Validate and plan the reference spec, then submit it with the selected project's
bucket and exact immutable image:

```bash
npa workbench workflow validate-spec workflows/testing/flex-pi-b200-inference.yaml
npa workbench workflow plan-spec workflows/testing/flex-pi-b200-inference.yaml
npa workbench workflow submit workflows/testing/flex-pi-b200-inference.yaml \
  --infra "$CONFIGURED_TARGET" --var "bucket=$OPERATOR_BUCKET" \
  --secret-env HF_TOKEN
```

Use `flex-pi-rtxpro-inference.yaml` for the independent RTX PRO 6000 path.
Both maintained workflow specs resolve through a toolRef that passes the
literal `--torch-compile` flag. This compiles only the upstream denoising step
after complete checkpoint load and performs its warmup before timed inference.
For a direct CLI or SDK call, opt in explicitly; do not infer compile mode from
the GPU name.

The cache may be run-owned and persistent across retries. Never place it in the
image build context or upload it as evidence. The checkpoint is public and the
token is not an access acceptance; forward an operator read token through this
secret-only channel to avoid anonymous rate limits during its multi-shard fetch.
The image defaults `MODELSCOPE_DOWNLOAD_PARALLELS=16`, the SDK's supported
maximum, because the converted UMT5 asset is one roughly 11 GB object. Retain
that default for normal cold starts; an operator may lower it for a constrained
network.

The released checkpoint contains the trained VideoDiT and ActionDiT experts.
Deployment must keep the maintained checkpoint-only patch enabled so upstream
constructs those experts with `skip_dit_load_from_pretrain=True` and immediately
loads the released checkpoint. Do not fetch or generate the training-only
ActionDiT initialization, and do not restore the redundant Wan VideoDiT shard
download. The strict-loader patch must reject every missing or unexpected MoT
key and require proprio, DINO, and pointmap state. Treat any checkpoint mismatch
as a failed run; `strict=False` in the upstream call is not permission to accept
partial model state.

## Artifacts and acceptance

Require all of the following from the exact image digest:

- terminal workflow/job success on the requested one-GPU B200 or RTX PRO 6000
  target;
- `actions.json` with schema `npa.flex_pi.actions.v1`, action-only regime, and
  exactly 32 rows of 14 finite numbers;
- positive inference latency and peak allocated GPU memory;
- CUDA device name and compute capability matching the request: B200 10.0 or
  RTX PRO 6000 12.0;
- pinned source/checkpoint/dataset identity plus checkpoint, stats, and input
  SHA-256 provenance;
- non-empty `result.json` and read-after-write verification in operator storage.
- exactly one `FLEX_PI_REAL_INFERENCE_PASSED` marker after strict state and
  artifact validation, with no traceback or checkpoint mismatch in the logs.

Keep raw logs and object locations access-controlled. Repository and PR evidence
may report only sanitized status, metrics, hashes, and generic GPU class.

The current `0.1.0-cu128-r2` release (`sha256:e27978b682056339fb332acfdd0df2369af1f180d93ba1e0fd86772d6649efe6`)
passed every publication gate and independent real compiled inference on one
B200 and one RTX PRO 6000 on 2026-09-17. Each exact-image worker passed native
architecture and baked-source checks, produced finite 32x14 actions, and ended
successfully with zero restarts and three read-back-verified JSON objects.
See `docs/workbench/flex-pi.md#r2-exact-image-acceptance` for measured values.

The following fan-out and paired benchmarks are historical evidence for
`0.1.0-cu128` (`sha256:88359258470d9622d9fb5274d8ad39627a57a5682cb8630c7ac85a3f303c7b91`), not r2 qualification.

The accepted 2026-09-16 RTX PRO 6000 run used the exact `0.1.0-cu128` digest,
torch 2.7.1+cu128, four Euler steps, 0.678 seconds of inference, and
25,268,430,336 bytes peak allocated GPU memory. It produced a finite 32×14
action chunk and three read-back-verified JSON objects.

The independent B200 validation used the same exact digest on the maximum live
capacity: 23 newly allocated reservation-backed devices plus one scheduler-free
device on a pre-existing shared node, for 24 requested/allocated/used GPUs.
Every one-GPU replica loaded the complete checkpoint, produced finite 32×14
actions, and published three durable objects. The run had 24 successes, zero
failures/restarts/tracebacks, 24 unique placement hashes, 24 unique action
hashes, and 72 unique object keys. It completed in 132.0 seconds at 0.1818
replicas/s, a 16.70× wall-throughput speedup and 69.56% efficiency versus the
91.823-second cached single-B200 baseline. Per-replica inference latency was
0.579–1.053 seconds (0.752-second median, 1.028-second p95), with
25,268,430,336 bytes peak allocated memory per replica. This was replica
fan-out, not model parallelism.

The 2026-09-17 runtime optimization retained the exact image, checkpoint,
public observation, seed, four Euler steps, and 32x14 output contract. With five
fixed-seed warm samples per mode, the maintained `--torch-compile` path reduced
median / p95 latency from 0.2701 / 0.2989 to 0.1021 / 0.1278 seconds on one RTX
PRO 6000 (2.645x), and from 0.1990 / 0.2075 to 0.08738 / 0.08759 seconds on one
B200 (2.278x). Peak allocated memory changed from 25,268,430,336 to
25,270,528,512 bytes. Require the measured correctness envelope as well as
finite actions: `atol=rtol=0.01`, relative L2 at most 0.5%, and action-L2 drift
at most 0.1%. The observed relative-L2 maxima were 0.2563% and 0.2927%,
respectively. Compile setup is not free: it added 30.94 seconds on RTX PRO 6000
and 24.16 seconds on B200, so keep eager mode available for one-shot callers.

Keep these historical observations tied to their exact digest. They are not latency SLAs or
closed-loop task-success claims; other GPU classes remain unmeasured for this
release.

Cancel the exact run before removing only run-owned infrastructure. Never tear
down a shared or pre-existing cluster.

## Troubleshoot

- ModelScope VAE/T5 fetch failures: this repository exposes only `master` to the
  SDK. Do not pass its raw commit as `revision`; require `master` to match the
  pinned Git commit, then enforce the two maintained SHA-256 file digests.
- Missing DINO weights: confirm access to the timm DINOv3 repository and its
  model terms before retrying.
- Missing ActionDiT initialization: confirm the checkpoint-only deployment patch
  is present. The released inference checkpoint replaces that training-only
  initialization; do not source an unrelated third-party copy.
- CUDA OOM: verify action-only flags and absence of competing workloads. Do not
  silently reduce cameras, horizon, action dimensions, or checkpoint fidelity.
- GPU mismatch: inspect scheduler labels and the artifact's device name. Do not
  infer RTX support from a B200 result, or B200 support from an RTX result.
- Invalid actions: retain the failed artifact privately and diagnose upstream;
  never coerce, clip, or fabricate values to pass validation.

## Verify changes

```bash
npa/.venv/bin/python /home/ubuntu/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/tools/flex-pi
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_flex_pi.py \
  npa/tests/docker/test_flex_pi_image_contract.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_three_tier_contract.py \
  npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py -q
```
