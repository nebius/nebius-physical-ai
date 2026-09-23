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
for B300 profile phases while retaining the original Torch, NumPy and CUDA
compute libraries. Update five records GPU kernels and memory activity; the
first six updates remain excluded from the steady windows. Zero kernels,
invalid timestamps or dropped records reject the profile. The result records
the profiler package-lock and native-library hashes separately from training
provenance. This repairs diagnostic visibility and does not establish a speedup.
See NVIDIA's [CUPTI Python interface](https://docs.nvidia.com/cupti-python/13.0.0/user-guide/topics/tutorial.html).
The private runtime retains the CUDA Toolkit, CUPTI Python and CUDA Python
license files plus cuda-pathfinder's Apache-2.0 license; these packages and
populated caches are never added to the public image.

Two profiles on four separate preemptible one-GPU B300 instances reached
live-verified terminal `SUCCEEDED` on 2026-09-23. Each measured 30 updates /
2,880 public training anchors and completed the initial and checkpoint
three-phase numerical qualifications, checkpoint readback, and exact fresh
continuation. The second profile set `NCCL_SOCKET_NTHREADS=4` and
`NCCL_NSOCKS_PERTHREAD=4` in each worker's pod environment:

| Four-host B300 profile | Aggregate samples/s, all 30 updates | Median steady samples/s |
| --- | ---: | ---: |
| Original socket settings | 2.565879 | 2.645909 |
| Four helper threads, four sockets each | 2.978842 | 3.109365 |

The steady windows exclude the first six updates and cover three consecutive
eight-update windows. The second profile measured 3.109365, 3.099759 and
3.120109 samples/s, a **17.52% median improvement** over the original four-host
profile. All 31 step/sample/loss records, including the separate continuation
probe, matched exactly. Both attempts also matched initial/final model state,
full checkpoint and resumed state, and all numerical fixture reports; all nine
checkpoint files had identical byte hashes. The second profile used source
`3b1b98f9936ebef122abe83ac047c0c04bcdc277` and retained the original training
compute libraries. Its native GPU trace contained 562,365 kernels and 38,490
memory events with zero dropped records; the published 147,038,296-byte trace
matched the worker readback.

This qualifies the profile and fresh-resume path. It does not establish a
complete B300 epoch or full held-out validation. Throughput remains below the
public B200 result, whose steady median was 6.363063 samples/s; the separate-host
socket topology and the recorded B200 shared-host contention remain distinct.
The unavailable historical dataset remains non-comparable, and
`reference_benchmark_beaten` remains false.

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
96 (microbatch one × 24 accumulation steps × four GPUs). Mixed-attention
gradient checkpointing is enabled. Model, tokenizer, VAE, DINOv3, and dataset
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
