# flex-pi

NPA packages [flex-pi](https://github.com/geyan21/flex-pi) as a single-GPU,
runtime-fetch workbench for real robot-policy inference. The default path uses
the released RoboTwin checkpoint in its action-only regime: three RGB cameras,
a 14D robot state, and a language instruction produce a 32-step bimanual action
chunk. It does not run a simulator and does not claim task success from one
observation.

## Architecture and target

Flex-pi is a 6B multi-stream world-action model. A Wan2.2-TI2V-5B video expert
and roughly 1B-parameter ActionDiT share a 30-layer mixture-of-transformers
backbone; DINOv3 supplies frozen semantic features and an optional pointmap
stream supplies geometry. One checkpoint selects observed and generated streams
at inference time.

The reference configuration observes RGB and DINO, omits pointmap, and disables
future video/DINO/pointmap denoising. It keeps the trained action horizon and
four Euler steps. Upstream reports the same 94.6% average RoboTwin success for
action-only and full-joint modes, while action-only avoids unnecessary stream
generation. The maintained reference targets are one RTX PRO 6000 (`sm_120`) or
one B200 (`sm_100`); both supply ample memory above upstream's 16–26 GB
inference range. Flex-pi remains single-GPU inference. Capacity-wide B200 runs
use independent replicas, not tensor, pipeline, or other model parallelism.

## Packaging and terms

`npa-flex-pi:0.1.0-cu128-r2` contains pinned MIT flex-pi source, configs, the
RoboTwin policy adapter, and its CUDA/Python runtime. It contains no checkpoint,
Wan/T5/DINOv3 weights, observation media, credentials, actions, or populated
cache. A narrow maintained patch makes upstream inspect the lexical Hugging Face
snapshot path before resolving checkpoint symlinks into the blob store; this
ensures the released checkpoint's adjacent `config.yaml` defines the model
architecture. A second bounded deployment patch selects upstream's existing
checkpoint-override path: it constructs both DiT experts without the
training-only initialization weights and immediately loads the complete released
checkpoint. This removes a redundant multi-shard VideoDiT fetch and avoids
requiring the ActionDiT initialization file that upstream documents as a
training prerequisite but does not publish with this checkpoint. The upstream
follow-up is intentionally kept in this integration branch rather than a second
pull request. A third fail-closed patch rejects every missing or unexpected
mixture-of-transformers key and requires the released proprio, DINO, and
pointmap state. The runtime prints `FLEX_PI_REAL_INFERENCE_PASSED` only after
that complete-state check and artifact validation succeed.

The checkpoint repository is MIT-labelled. The selected public RoboTwin dataset
card declares no license; NPA does not infer one from the simulator's MIT source,
does not redistribute its bytes, and fetches only the five hash-pinned objects
needed for the authorized observation. The ModelScope converted Wan VAE/T5
repository is Apache-2.0; DINOv3 retains its separate model license. Ancillary
assets are fetched from immutable revisions and verified by SHA-256 before model
construction. ModelScope exposes only its `master` branch to the SDK for this
repository, so NPA first requires that branch to resolve to the pinned Git
commit and then verifies both selected files by SHA-256. See the image's
`REDISTRIBUTION.md` and `THIRD_PARTY_NOTICES.md` for the maintained boundary.
Cold workers use ModelScope's supported 16-way range downloader for the roughly
11 GB converted UMT5 object. Set `MODELSCOPE_DOWNLOAD_PARALLELS` explicitly to
reduce that concurrency when an operator-controlled network requires it.

## CLI and SDK

Inference emits exactly one JSON document on stdout, including real execution.
`--output-format json` is the default and the only supported format. Runtime
diagnostics and the success marker go to stderr; domain failures exit with code 1.

The runtime prepares and verifies the pinned DINO checkpoint before setting
`FLEX_PI_DINO_CHECKPOINT`. Direct encoder construction without that value fails
before model loading; it never falls back to an unpinned pretrained download.

Plan locally without downloading model assets:

```bash
npa workbench flex-pi infer \
  --input-path "s3://${OPERATOR_BUCKET}/inputs/flex-pi-observation.json" \
  --output-path "s3://${OPERATOR_BUCKET}/plans/flex-pi/" \
  --dry-run
```

Run inside the GPU image with an authorized S3 output prefix:

```bash
npa workbench flex-pi infer \
  --input-path /opt/flex-pi/npa/public_robotwin_sample.json \
  --output-path "s3://${OPERATOR_BUCKET}/runs/${RUN_ID}/flex-pi/" \
  --torch-compile \
  --expected-gpu RTXPRO6000
```

Use `--expected-gpu B200` for the B200 reference target. The maintained
`workbench.flex_pi.infer` toolRef passes `--torch-compile` literally for both
Blackwell workflows. Direct CLI and SDK callers must opt in explicitly. Compile
warmup is accounted as model setup; it does not change the checkpoint, public
observation, four Euler steps, seed, or 32×14 action contract.

The SDK exposes the same implementation as
`npa.sdk.workbench.flex_pi.infer(...)` and also supports local output directories.
For serving, set an owner-controlled
`NPA_FLEX_PI_TOKEN`, configure `NPA_FLEX_PI_OUTPUT_ROOT` to an authorized S3
prefix or local mode-0700 directory, and keep transport private or terminate
TLS. `/health`, `/status`, `/system-info`, `/list`, and `/run` all require bearer
authentication. HTTP callers cannot change the checkpoint or startup-snapshotted
input manifest.

## Workflow

Validate, plan, and submit the reference workflow:

```bash
npa workbench workflow validate-spec workflows/testing/flex-pi-b200-inference.yaml
npa workbench workflow plan-spec workflows/testing/flex-pi-b200-inference.yaml
npa workbench workflow submit workflows/testing/flex-pi-b200-inference.yaml \
  --infra "$CONFIGURED_TARGET" --var "bucket=$OPERATOR_BUCKET" \
  --secret-env HF_TOKEN
```

Use `flex-pi-rtxpro-inference.yaml` for the independent RTX PRO 6000 path. Each
spec requests exactly one GPU and routes `workbench.flex_pi.infer` to the
flex-pi image. It publishes only verified artifacts to operator-owned object
storage. Capacity-wide validation must discover live capacity first and fan out
independent one-GPU replicas with unique seeds and output prefixes; do not make
multiple devices visible to one flex-pi process or call that model parallelism.
The released checkpoint is public, so `HF_TOKEN` is not an access gate;
forwarding an operator read token through the secret channel avoids anonymous
multi-shard download throttling. Never put the token in the workflow YAML or an
artifact.

## Artifacts and acceptance

- `input.json`: immutable public-observation manifest, including object hashes;
- `actions.json`: 32×14 finite denormalized actions, real latency, peak allocated
  GPU memory, device/compute capability, and source/input/checkpoint provenance;
- `result.json`: request, image, model/data identity, metrics, and publication
  provenance under `npa.workbench.flex_pi.inference.v1`.

A passing run requires terminal job success, all three durable objects,
read-after-write verification, the requested device identity and compute
capability (RTX PRO 6000 12.0 or B200 10.0), and exact
source/checkpoint/data revisions. Raw logs and S3 locations stay outside Git;
reports and pull requests use only sanitized summaries.

### r2 exact-image acceptance

On 2026-09-17, `0.1.0-cu128-r2` was promoted from immutable source commit
`8904daf36d0cc9152193b0e87a687bce7e6fca46` at exact manifest digest
`sha256:e27978b682056339fb332acfdd0df2369af1f180d93ba1e0fd86772d6649efe6`. The replacement contains the
JSON stdout correction and fail-closed DINO checkpoint gate. Every publication
gate passed, including full-layer payload scans, the configured vulnerability
and secret policy, signed SBOM/provenance, and anonymous full-layer validation.

Independent real workflows ran the exact image without a source overlay on one
B200 and one RTX PRO 6000. Both verified native GPU architecture, baked source
hashes, strict checkpoint loading, the compiled four-step inference path,
terminal success, zero restarts, and three durable JSON objects read back twice.

| Target | GPU count | Finite action shape | Inference time | Peak allocated GPU memory | Verified objects |
| --- | --- | --- | --- | --- | --- |
| B200 | 1 | 32×14 | 0.094180 s | 25,270,528,512 bytes | 3 |
| RTX PRO 6000 | 1 | 32×14 | 0.151054 s | 25,270,528,512 bytes | 3 |

These are single-observation compiled inference results, not paired speedup or
fleet-scaling measurements. The historical results below remain bound to
`0.1.0-cu128`, digest `sha256:88359258470d9622d9fb5274d8ad39627a57a5682cb8630c7ac85a3f303c7b91`;
they do not qualify the replacement image. Neither release claims closed-loop
RoboTwin task success. Other GPU classes remain unmeasured for r2.

### Historical RTX PRO 6000 acceptance

On 2026-09-16, the exact public-development image later selected for
`0.1.0-cu128` completed a real action-only run on one NVIDIA RTX PRO 6000
Blackwell Server Edition (`sm_120`). The strict loader accepted the complete
released checkpoint, four Euler steps produced 32×14 finite actions in 0.678
seconds, and peak allocated GPU memory was 25,268,430,336 bytes. The durable
`actions.json`, `input.json`, and `result.json` objects were read back with
SHA-256 digests `b69156c1…`, `e2ff5ada…`, and `638e072d…`, respectively.

The run used torch 2.7.1+cu128 and ended with
`FLEX_PI_REAL_INFERENCE_PASSED`; it had no pod restarts or traceback. This is a
policy-inference and artifact-integrity result for one public observation, not a
closed-loop RoboTwin success measurement. L40S, Hopper, and B300 remain
supported by the measured wheel architectures but unmeasured for this release.

### Historical B200 acceptance

On 2026-09-16, the same exact accepted digest ran on the maximum B200 capacity
that fresh provider and scheduler evidence made available. The reserved pool
had 23 unallocated B200s; a pre-existing shared eight-GPU node had one additional
scheduler-free B200 while seven devices were already requested by workloads
that were left untouched. NPA therefore requested, allocated, and used 24
B200s: two reserved eight-GPU nodes, seven reserved one-GPU nodes, and one free
device on the shared node. Provider advice fell to zero available reservation
GPUs during the run and returned to the pre-run count after cleanup.

The exact digest first passed 24 independent `sm_100` probes with one visible
device and 24 distinct anonymized placements. A cached single-B200 baseline
completed in 91.823 seconds wall time with 0.578 seconds of model inference.
The measured capacity run then launched 24 independent action-only replicas,
not multi-GPU model parallelism. All 24 loaded the complete released checkpoint,
emitted `FLEX_PI_REAL_INFERENCE_PASSED`, and produced finite 32×14 actions with
zero failures, restarts, or tracebacks. The fan-out completed in 132.0 seconds
(0.1818 replicas/s): 16.70× throughput speedup and 69.56% wall-scaling
efficiency relative to the cached single-GPU baseline. Per-replica inference
latency was 0.579–1.053 seconds (0.752-second median, 1.028-second p95); peak
allocated GPU memory was 25,268,430,336 bytes per replica. Sampled GPU
utilization was nonzero on every device.

All 72 declared objects were read back from operator-owned storage under 72
unique keys. The run produced 24 unique action hashes and 24 unique result
hashes; the identical pinned input intentionally had one content hash. This is
observation-level policy inference and scaling evidence, not closed-loop
RoboTwin task success.

### Historical Blackwell compile optimization

On 2026-09-17, the unchanged `0.1.0-cu128` digest was measured in paired eager and
compiled runs on one RTX PRO 6000 (`sm_120`) and one B200 (`sm_100`). Each mode
used the released checkpoint, the same public observation, seed 42, four Euler
steps, two untimed warmups, and all five timed inferences. The optimized path is
upstream's `torch.compile` denoising-step specialization (`inductor`,
`reduce-overhead`); no image bytes or model semantics changed.

| Target | Requested / allocated / used | Eager median / p95 | Compiled median / p95 | Eager → compiled throughput | Median speedup | Eager → compiled peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 | 1 / 1 / 1 | 0.2701 / 0.2989 s | 0.1021 / 0.1278 s | 3.688 → 9.331 samples/s | 2.645× | 25,268,430,336 → 25,270,528,512 bytes |
| B200 | 1 / 1 / 1 | 0.1990 / 0.2075 s | 0.08738 / 0.08759 s | 4.992 → 11.450 samples/s | 2.278× | 25,268,430,336 → 25,270,528,512 bytes |

Cold asset preparation took 686.59 seconds on RTX PRO 6000 and 451.77 seconds
on B200. Eager model setup took 90.09 and 97.35 seconds, respectively; compiled
model setup including warmup took 121.03 and 121.51 seconds. The corresponding
end-to-end cold setup totals were 776.68→807.62 seconds on RTX PRO 6000 and
549.12→573.28 seconds on B200. Compile therefore trades a one-time 30.94- or
24.16-second setup increment for the repeated warm-inference gains above.

Every mode produced one stable fixed-seed action hash across all five repeats.
Compiled versus eager actions passed `atol=rtol=0.01`, a 0.5% relative-L2 cap,
and a 0.1% action-L2-drift cap. The observed maxima were 0.008001 absolute and
0.2563% relative L2 on RTX PRO 6000, and 0.008001 absolute and 0.2927% relative
L2 on B200. This bounded envelope accounts for BF16 kernel reordering without
clipping or changing outputs. Eager profiling identified `aten::addmm` as the
largest CUDA operation on both targets, supporting compilation of the repeated
denoising step rather than reducing solver work.

Both final Jobs terminated successfully with zero restarts, nonzero physical
GPU utilization, finite 32×14 actions, and five read-back-verified objects per
target. The standard action/result/placement identities were independently
different across the two GPUs: `889285410675…` / `10bd2ee9a53f…` /
`9bca9a4c6fd4…` on RTX PRO 6000 and `14355dc20f35…` / `c12ef1159c19…` /
`32bb6615e5a3…` on B200. Both used torch 2.7.1+cu128, CUDA runtime 12.8,
and driver 580.159.04. The prior 24-B200 fan-out remains the scaling record;
this one-B200 comparison proves only the shared runtime optimization.

## Build qualification

Build only from `npa/docker/workbench/flex-pi/Dockerfile` with the immutable
`dev-<full-git-sha>` tag. Before pushing, run dependency, vulnerability, secret,
license, and `scan_image_flex_pi_payload.py` gates over every layer. Resolve the
pushed digest anonymously and run the real workflow against that digest. Only
the exact digest that passed those gates may be promoted to the supported tag.

Cancel a live run before teardown and remove only resources created for that
run. Do not destroy shared clusters or buckets.
