# flex-pi

NPA packages [flex-pi](https://github.com/geyan21/flex-pi) for single-GPU
robot-policy inference and four-GPU public YAM training with runtime-fetched
inputs. The default inference path uses
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

The experimental [B300 workflow](../../workflows/testing/flex-pi-b300-inference.yaml)
requires the current NPA source overlay and `--expected-gpu B300`. The released
image's Triton assembler and LLVM cannot compile B300's `sm_103a` target. For
compiled B300 requests, NPA creates a private runtime with PyTorch 2.8.0+cu129,
Triton 3.4.0, matching torchvision/torchaudio/torchcodec, and CUDA libraries from
[hash-pinned official wheels](../../npa/src/npa/workbench/flex_pi/b300-runtime-requirements.txt).
It requires a successful native compiled reduction before downloading model
weights. NPA also runtime-fetches `nvidia-cuda-nvcc-cu12==12.9.86`, verifies
both the wheel and assembler SHA-256, and selects that assembler for the vendor
process. CUDA 12.9 adds this target according to the
[NVIDIA release notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html).
The compiler is governed by the NVIDIA CUDA Toolkit EULA; its included license
stays beside the binary in a private temporary run directory. NPA publishes only
compiler and vendor-runtime version/hash provenance in `result.json`; these are not baked
into the existing image or uploaded as an artifact. Eager requests and the B200
and RTX reference paths retain their existing compiler selection. This runtime
repair does not establish a B300 training speedup or qualify four single-GPU
hosts as equivalent to four GPUs on one host.

On 2026-09-23, this path completed real compiled inference on one preemptible
NVIDIA B300 (`sm_103`, compute capability 10.3). The native compiled-reduction
preflight and full policy both passed. Four Euler steps produced finite 32×14
actions in **0.151714 seconds**, with **25,270,528,512 bytes** peak allocated GPU
memory. NPA reported live-verified `SUCCEEDED`; the complete log contained one
success marker and no traceback, and all three JSON artifacts passed readback.
Their SHA-256 digests were `fe1394d7…` (actions), `0b155418…` (result), and
`e2ff5ada…` (input).

This measurement used the r2 base image plus NPA source fingerprint
`feec30a5483ae7ceb193eb65d37da2eae7a76cc8cf0b0747aa49d114b93b0978` and runtime
lock SHA-256 `e6de5c8518344b0cdeaf4655ce283f5a24c91f9baf8b6678a015cd294e2c577f`.
The separate PyTorch 2.8.0+cu129/Triton 3.4.0 environment retained the vendor's
NumPy 1.26.4 and PyAV 16.0.1. This is one warmed inference observation, with
installation, download and compilation excluded from its latency. It does not
qualify the unmodified r2 image, a repeated performance comparison, training,
multi-node scaling, or closed-loop robot success.

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
## Public YAM training

`npa workbench flex-pi train` runs the pinned upstream YAM utensil task on
exactly four GPUs. The same implementation is available through
`npa.sdk.workbench.flex_pi.train`, the authenticated `/train` service route, and
`workbench.flex_pi.train`. Use
`workflows/testing/flex-pi-b200-public-training.yaml` with the current NPA source
overlay, the standard `--runtime` submission mode, and the immutable r2 image.
For an operator-authorized run without a wait deadline, include
`--max-wait-seconds 0 --no-cancel-on-timeout` on workflow submission; the
runtime's default wait is shorter than a complete epoch of this workload.
Its single-node NCCL configuration retains NVLink peer transfers with socket
bootstrap, NVLS disabled, and cuMem allocation disabled. The r2 NCCL library
JIT-compiles Blackwell kernels on a cold cache; report that startup separately.
Three verified collectives run before any asset downloads or training.
Training hardware qualification is separate
from the existing one-GPU inference qualification below.

For four separate one-GPU B300 hosts, use
`workflows/testing/flex-pi-b300-multinode-public-training.yaml` with the same
verified original normalization object and SHA-256. Its `training_nodes: "4"`
sets the resource profile's `num_nodes`; each node requests `B300:1`, 16 CPUs
and 192 GiB of memory. This is one four-rank DDP training job. Inter-host NCCL
uses sockets, so throughput must be reported with this topology and must not be
presented as equivalent to four GPUs sharing one host's NVLink fabric.
The workflow sets `CUDA_CACHE_MAXSIZE=4294967296` (4 GiB) to retain more compiled
Blackwell kernels across fresh qualification and resume processes. The default
1 GiB cache filled during live startup and fresh phases repeated kernel
initialization. This setting addresses startup overhead; it does not establish
a training throughput improvement. See NVIDIA's
[CUDA cache controls](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

B300 diagnostic profiles use native CUPTI 13.0.85 with CUPTI Python 13.0.0.
The image's CUPTI 12.8 interface reported `CUPTI_ERROR_INVALID_DEVICE` and
produced a CPU-only trace; preloading a newer library did not repair the older
PyTorch profiler binding. NPA fetches a separate, hash-pinned profiler directory
for B300 profile phases without replacing the selected Torch, NumPy or CUDA
compute libraries. Update five records GPU kernels and memory activity; the
first six updates remain excluded from the steady windows. Zero kernels,
invalid timestamps or dropped records reject the profile. The result records
the profiler package-lock and native-library hashes separately from training
provenance. This repairs diagnostic visibility and does not establish a speedup.
See NVIDIA's [CUPTI Python interface](https://docs.nvidia.com/cupti-python/13.0.0/user-guide/topics/tutorial.html).
The private runtime retains the CUDA Toolkit, CUPTI Python and CUDA Python
license files plus cuda-pathfinder's Apache-2.0 license; these packages and
populated caches are never added to the public image.

Four profiles on four separate preemptible one-GPU B300 instances reached
live-verified terminal `SUCCEEDED` on 2026-09-23. Each measured 30 updates /
2,880 public training anchors and completed the initial and checkpoint
three-phase numerical qualifications, checkpoint readback, and exact fresh
continuation. All runs retained microbatch one, accumulation 24, global batch
96, eager BF16, the default optimizer, the pinned public dataset, and the
original normalization bytes.

| Four-instance B300 profile | Aggregate samples/s, all 30 updates | Median steady samples/s |
| --- | ---: | ---: |
| Original socket settings and cuDNN 9.7 | 2.565879 | 2.645909 |
| Four NCCL helper threads, four sockets each | 2.978842 | 3.109365 |
| Same sockets, private cuDNN 9.26 runtime | 4.473873 | 4.785060 |
| Same runtime, activation checkpointing off | 6.129717 | 6.684018 |

The steady measurement excludes the first six startup/profiler updates and
uses the median of three consecutive eight-update windows. The socket change
improved this median by **17.52%** while preserving all 31 step/sample/loss
records, complete model and training state, and all nine checkpoint file hashes.
It set `NCCL_SOCKET_NTHREADS=4` and `NCCL_NSOCKS_PERTHREAD=4`; the accepted
profiles retained automatic NCCL protocol/channel selection.

The cuDNN experiment replaced the complete loader/backend library set with
`nvidia-cudnn-cu12==9.26.0.51`, retaining Torch 2.7.1+cu128 and CUDA 12.8.
The private runtime verified the pinned wheel and all mapped library hashes;
package licenses remained with that runtime. The wheel SHA-256 is
`ce8603d4ea88d134be5a92c5e0833d513bea1ce0ee4bbc5d649fcb8718d51cbf`.
This runtime changes numerical results relative to the original cuDNN 9.7
stack: exact cross-runtime replacement qualification failed. The complete
epoch below evaluates this separate compute runtime against the predeclared
held-out quality and fresh-resume gates.

Disabling activation checkpointing within that same cuDNN 9.26 runtime
improved the steady median by **39.69%**, to **6.684018 samples/s**. Its three
windows were 6.673791, 6.684018, and 6.697200 samples/s. All 31 step/sample/loss
records, six qualification reports, complete initial/final and resumed training
state, and all nine checkpoint file hashes matched the activation-on profile
exactly. All eight phase receipts verified the three actual checkpointing flags
were off. Peak allocated GPU memory was 74,348,137,984 bytes. This establishes
an execution-only gain for the activation policy within the selected runtime.

The activation-off profile used source
`101cf4bb9c05cf1a3db2d958b86b9fc9460d53cc` with source-overlay fingerprint
`019281801cb99c92929befb364471be73e14fdf0a300b586282c9394452167de`.
All 19 recorded training source files matched that clean, pushed commit.
Its native GPU trace contained 501,837 kernels and 39,738 memory events with
zero dropped records; the published 131,354,010-byte trace matched worker
readback. The trace is a diagnostic for one instrumented update, not the
denominator for steady throughput.

The final profile median is **152.62% above the original four-instance B300
profile** and **5.04% numerically above** the public B200 steady median of
6.363063 samples/s. The B200 run used a different topology and compute runtime,
with shared-host contention recorded. These measurements do not isolate a
hardware-only improvement. The unavailable historical dataset remains
non-comparable, and `reference_benchmark_beaten` remains false.

The private runtime overlay is separate from the unchanged public image digest.
The maintained four-instance workflow still defaults to its original compute
stack and activation checkpointing on; the measured candidate must not be
described as qualification of the unmodified image or default workflow.

The same activation-off configuration subsequently completed the full epoch
and reached live-verified terminal `SUCCEEDED` on 2026-09-24.
It processed **115,620 anchors in 1,205 updates**, including the actual 36-anchor
tail, at **6.921861 aggregate samples/s** and **6.940044 median steady
samples/s**. The aggregate uses measured training-update durations; validation,
qualification, checkpoint I/O and the separate continuation update are excluded.
The steady median uses 149 complete eight-update windows after the first
six updates; the trailing partial window remains included in the aggregate.
It is **8.95% numerically above** the public B200 aggregate of
6.353262 samples/s, with the topology and compute-runtime differences described
above. This is a complete-epoch measurement, separate from the 30-update
profile comparisons.

Both complete held-out passes processed 12,390 anchors. Validation loss changed
from **3.069520630549** to **0.471179730231**. The final loss satisfies the
predeclared maximum of 0.47188222932935736; full-epoch quality and fresh-resume
acceptance passed.
All losses and update durations were finite. Both three-phase numerical
qualifications passed, and fresh step 1,206 matched the separately measured
uninterrupted continuation in sample count, loss, model/optimizer/scheduler
state and current CUDA RNG state. Checkpoint restoration also verified the
saved cursor, normalization and all four ranks' complete RNG state.
All eight phases verified the selected activation policy, loaded compute
runtime and the same 19 training-source file hashes. Published result artifacts
passed readback, and throughput was independently recomputed from the complete
measurement log.

After the terminal audit, NPA removed the owned controller and stopped its
owned local APIs, then destroyed this experiment's four GPU instances and CPU
helper. Provider inventory verified absence of the owned cluster, instances
and managed root disks. Artifact storage was retained. This cleanup does not
resolve the earlier B200 resources whose ownership and provider access remain
incomplete.

On 2026-09-25, matched profiles on one dedicated reserved B200 node measured
the larger-microbatch candidate. Each job exposed and used exactly four GPUs;
the provider's available reserved shape allocated eight GPUs on that node.
The two profiles used the same physical host, initial model, normalization,
public data, image, private cuDNN 9.26 runtime and 19 training-source file hashes.
Both retained eager BF16, the default AdamW optimizer, activation checkpointing
off, memory fill off and effective batch 96.

| Four-GPU single-node B200 profile | Aggregate samples/s, all 30 updates | Median steady samples/s |
| --- | ---: | ---: |
| Microbatch 1, accumulation 24 | 8.072713 | 8.990426 |
| Microbatch 3, accumulation 8 | 17.563931 | 20.268646 |

Microbatch three improved the steady median by **125.45%**, or **2.25447×**.
Its three complete windows measured 20.449261, 20.004106 and 20.268646 samples/s;
every window exceeded the baseline's maximum of 9.015391. The measurement used
the same exclusion of six initial updates and three consecutive eight-update
windows as the earlier profiles. Each profile processed 2,880 timed anchors.

Both profiles passed all six initial/checkpoint numerical qualification phases
and exact fresh continuation. For microbatch three, uninterrupted and fresh
step 31 each processed 96 anchors with loss 1.1332216262817383; model, optimizer,
scheduler and RNG checks also passed. All eight phases verified the selected
compute runtime, activation policy and training source. Published JSON artifacts
and the GPU trace passed independent readback. This establishes profile
throughput and within-configuration reproducibility.

The corresponding complete timed epoch recorded 115,620 anchors in 1,205 updates,
including the 36-anchor tail, over 5,453.162401 measured training seconds:
21.202376 aggregate samples/s and a 21.262443 steady median. Both held-out passes
evaluated 12,390 anchors. Loss fell from 3.069524735095065 to
0.47231008458883955, but exceeded the frozen quality ceiling of
0.47188222932935736. This configuration is therefore **rejected by the quality
gate**, despite its throughput gain. The ceiling is unchanged.

All six numerical qualification phases passed. Fresh step 1,206 matched the
uninterrupted continuation's 96 anchors, loss 0.44979333877563477, updated
model/optimizer/scheduler state and current CUDA RNG state. Checkpoint restoration
also verified every rank's saved cursor and complete RNG state. All six result
objects passed readback; all eight phases verified source, runtime and activation
policy. NPA reached live-verified terminal `SUCCEEDED`. These execution and
resume checks do not override the held-out quality rejection. Initialization,
qualification, validation, checkpoint I/O and separate continuation are excluded
from the training rate.

The candidate's instrumented rank-zero update recorded 216,066 GPU kernels,
versus 501,165 in the baseline. NCCL kernel time remained about 0.044 seconds.
These diagnostics support reduced per-microstep overhead; throughput comes
from the independent steady windows, not profiler timestamps. Recorded peak
allocated memory increased from 74,334,123,520 to 96,745,736,704 bytes.

The baseline training job reached `SUCCEEDED`, but its NPA monitor failed when
the native VM credential cache changed. A recovery bug briefly submitted a
duplicate after the timed measurement; that duplicate was cancelled. The audit
retains the failed monitor separately from the successful original workload and
its verified artifacts. NPA now preserves the supported native metadata cache
binding, restores an archived validation client configuration before retrying,
and reconciles `block_relaunch` jobs by their recorded identity. The candidate
used control source `e268b67cea39581cf2c1130765c7768528d054cb`, reached verified
NPA terminal `SUCCEEDED`, and completed controller/API cleanup before the
full-epoch attempt. The training-source bytes remained identical across profiles.

The renderer sets `NPA_FLEX_PI_NODE_COUNT` from the resolved resource profile
(one by default). The adapter cross-checks it against SkyPilot's node rank and
peer addresses, allowing only one node with four GPUs or four nodes with one
GPU each. The Kubernetes downward API supplies `NPA_FLEX_PI_HOST_ID` for a
hashed distinct-host placement receipt. Ports 29500 and 29501 must be reachable
between the task's peers for fresh torchrun groups and CPU coordination.
Node zero alone publishes results. Each node stages immutable inputs locally;
the checkpoint join collects every rank's saved RNG file on node zero, then
followers independently restore all hash-verified checkpoint bytes from S3.
Fresh processes repeat the existing qualification and continuation checks.
Peer heartbeats detect a disappeared instance without imposing a training
duration limit. Any worker failure stops the other live phase workers. This
path requires its own live four-node acceptance; CPU tests alone do not qualify
B300 training or establish a speedup.

The real public dataset is
[`flex-pi/sort_utensils`](https://huggingface.co/datasets/flex-pi/sort_utensils/tree/0780dd0a0b281df91abcef9434c4b3ac2757448c),
licensed CC-BY-4.0. Its 152 episodes contain 128,010 source frames. The pinned
upstream seed-42 split selects 136 episodes / 115,620 training anchors and
16 episodes / 12,390 validation anchors. Every anchor is visited once; the last
training update has 36 samples and uses its actual sample count for averaging.
The three cameras supply 640×360 RGB and depth. The official YAM transform uses
a 256×320 head view, 224×224 wrist views, a 384×320 composite, and 32-dimensional
state/actions. Thirty-three observations yield nine encoded video frames and
32 actions using the official frequency ratio and endpoint padding masks.

Training uses the upstream `yam_unified_flex_3cam_32d_rel_1e-4` model and loss,
Wan VideoDiT initialization plus the official derived ActionDiT backbone,
BF16 native DDP, AdamW with a peak learning rate of `1e-4`, and effective batch
96 (by default, microbatch one × 24 accumulation steps × four GPUs). Mixed-attention
gradient checkpointing is enabled by default. Model, tokenizer, VAE, DINOv3, and dataset
bytes are fetched only at runtime under their separate upstream terms. The
bundled manifests pin source revisions and every downloaded content hash.
Training uses strict deterministic PyTorch algorithms, deterministic cuDNN and
the pinned `:4096:8` cuBLAS workspace. Unsupported nondeterministic operators
fail before they can produce accepted results. This is necessary because the
default memory-efficient attention backward can use nondeterministic split-key
reductions. Compare baseline and candidate runs under this same execution
policy; earlier profiles without it are diagnostic only.
The adapter also fixes DDP bucket partitioning with unused-parameter discovery
enabled and static graphs disabled. Its guarded Accelerate 1.12.0 integration
runs before model preparation, preventing a fresh reducer from using different
first-iteration buckets than the uninterrupted run.

`--microbatch-per-rank 1|3` (`config.microbatch_per_rank`) selects one or three
samples per GPU. Accumulation automatically becomes 24 or 8 steps, preserving
global batch 96. Both widths divide the nine samples per rank in the real
36-sample tail; unsupported widths fail before launch. Qualification uses the
actual microbatch and native loader-end synchronization for both the full and
tail updates, and fresh resume checks the next ordered anchors. Validation
continues to use one sample per GPU with its original per-anchor seed.
Larger microbatches can change floating-point reductions and random draws, so
they require their own measured profile, exact within-configuration resume,
and full held-out quality check. The single-node B200 profile above establishes
a throughput gain for microbatch three, but its complete epoch failed the frozen
quality ceiling. It is not an accepted or execution-equivalent replacement for
microbatch one.

Training retains detached microbatch losses on the device until the optimizer
boundary. A collective finite-loss check covers every microbatch on every rank
before clipping or an optimizer update. Loss readback occurs once per update,
before the update timer stops, and preserves the original Python-float summation order. This
reduces scalar synchronization without changing finite-loss computation; its
performance remains subject to a measured GPU profile and full-epoch acceptance.

`--activation-checkpointing on|off` (`config.activation_checkpointing` in a
workflow) selects recomputation for both experts and mixed attention together.
The default `on` preserves the existing workload. `off` retains activations for
backward and needs more GPU memory; it is an experimental execution policy
until memory fit, exact numerical parity, checkpoint and fresh-resume checks
pass on the selected topology. Each phase records the actual model flags and
rejects a mismatch before training. This option does not disable durable model,
optimizer, scheduler or RNG checkpoints. Keep batch size, precision, optimizer,
data and eager execution fixed when comparing the two policies.

This public task is **non-comparable** to the private 5.66 samples/s reference:
dataset identity, split, source dimensions, task mix, initialization, action
representation and I/O layout differ. Its results always retain
`reference_benchmark_beaten=false`.

`--mode profile` performs 30 optimizer updates. The first six startup/profiler
updates are excluded; three subsequent eight-update windows report sustained
throughput. `--num-workers`, `--prefetch-factor`, and `--optimizer` expose
execution choices for controlled comparisons. CPU/CUDA and memory profiling
remain enabled, but tensor-shape recording is disabled because PyTorch 2.7.1
overflows while recording deterministic FlashAttention allocations
([upstream issue](https://github.com/pytorch/pytorch/issues/150601)). This changes
profiling metadata only; deterministic algorithms and attention math stay fixed.
Compare sample-order and model-state digests and numerical losses before
accepting any execution change; the same settings alone do not prove parity. Full training requires a complete
epoch, both complete initial/final validation passes, finite losses/gradients,
synchronized model states, a non-regressed validation loss, and an uploaded
checkpoint restored byte-for-byte into a fresh process. The continuation update
must match the uninterrupted update, including its model digest and scheduler.
The checkpoint also carries all four rank RNG files. Restored optimizer, RNG,
accumulation and cursor fingerprints must match before continuation; normalization
and workload hashes establish identity across separate processes and paths.
The checkpoint includes the original `dataset_stats.json`; fresh resume loads
those verified bytes instead of recomputing parallel statistics. For independent
baseline and candidate jobs, pass the same `--normalization-path` S3 object and
`--normalization-sha256`. The options must be supplied together. Both training
and validation load the verified phase-local file. Workload hashing canonicalizes
its location while retaining its exact content hash. Nondefault optimizer modes
require a separate fixed-input gradient, update and optimizer-state comparison
against default AdamW before selection.

`--memory-fill off` is an unqualified allocation candidate; the default is `on`.
It retains strict deterministic algorithms and requires the original verified
normalization input. Before execution, three isolated native DDP phases run the
real first 96 anchors and the actual final 36 anchors: fill-on capture, fill-on
replay, and fill-off candidate. Each phase preserves the full training schedule
and checks exact inputs, component losses, gradients before/after clipping,
updates, optimizer, scheduler, buffers and RNG state. Eight held-out anchors
also check evaluation-mode total/component losses. Any difference rejects the
candidate without widening tolerances. The same gates repeat from the verified
checkpoint before ordinary fresh-process resume, whose workload identity remains
strict. Diagnostic fixtures are excluded from epoch counts and throughput;
their elapsed time and hash-only evidence are reported separately. Rejections
retain a private receipt and attempt run-scoped publication with byte readback.

The candidate changes PyTorch's process-wide allocation flag around model
forward/backward and optimizer execution. Initialization, loader creation,
loader workers, and diagnostic hashing keep fills enabled. The concurrent
pin-memory thread shares the flag; its pinned PyTorch implementation fully
copies each allocated payload before use. This is not thread-local isolation.
No speedup or acceptance is claimed until target parity, repeated uncontended
measurements, a complete epoch, full validation and final resume all pass.

Cold preparation, initialization, checkpointing and validation are reported
separately from update throughput. No throughput or training acceptance is
claimed until the real target completes these gates.

Use `--mode profile-resume` before a complete epoch to exercise the same
30-update real-data profile plus full checkpoint upload, byte readback and
fresh-process continuation. Both continuations must use exactly the next 96
anchors from the frozen permutation, and their complete loaded state and next
update must match. The probe update is excluded from throughput. Its separate
`profile_checkpoint_resume_verified` result never claims a complete epoch or
full validation; `--mode train` retains the final epoch-boundary resume gate.

After checkpoint byte readback, matching fresh-process continuation, and final
result JSON readback succeed, the adapter removes the two local copies of the
published checkpoint state from that invocation's persistent working directory.
It verifies their paths and hashes before removal. Unexpected files, unpublished
upstream weight files, qualification/resume diagnostics, shared downloaded assets,
and failed runs remain available for operator-managed retention. A cleanup failure
warns on stderr and preserves the successful published result. Profile-only and
dry-run calls do not perform this cleanup.
