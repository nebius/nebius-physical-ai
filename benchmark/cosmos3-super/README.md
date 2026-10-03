# Cosmos3-Super: request latency and node throughput

This benchmark compares four ways to serve NVIDIA Cosmos3-Super's text-to-video
Generator through vLLM-Omni on a complete eight-GPU Nebius node. More independent
replicas produced more technically valid video per node-hour, at the cost of
longer request latency, on both HGX B200 and HGX H200.

The reference table comes from **“Serving World Models: request latency and
node throughput tradeoffs” by Stewart Tong and Timothy Le**. Its public evidence
and methodology are maintained by Stewart Tong in
[`stewtong/cosmos3-super-serving`](https://github.com/stewtong/cosmos3-super-serving).
Source links below pin revision
[`951dc1162fcc6c47d694e29616a1f833f293b877`](https://github.com/stewtong/cosmos3-super-serving/tree/951dc1162fcc6c47d694e29616a1f833f293b877).
These are historical August 31, 2026 reference records, not measurements newly
produced by this folder.

## Published results

Each row uses all eight GPUs, one request in flight per replica, one excluded
warmup per replica, and 24 measured requests. A replica is an independent
vLLM-Omni service with its own endpoint and disjoint GPU group.

| Topology | Parallelism per replica | HGX B200 mean latency (s) | HGX B200 valid video-s/node-h | HGX H200 mean latency (s) | HGX H200 valid video-s/node-h |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 replica × 8 GPUs | CFG-2 / Ulysses-4 / HSDP-8, Ring-1 | 68.5 | 413.6 | 123.3 | 229.9 |
| 2 replicas × 4 GPUs | TP-4 | 121.5 | 466.3 | 230.0 | 245.6 |
| 4 replicas × 2 GPUs | TP-2 | 212.4 | 529.0 | 416.5 | 271.4 |
| 8 replicas × 1 GPU | TP-1 | 378.1 | 589.4 | 779.0 | 289.1 |

Sources: the pinned
[B200 record](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/results/b200-single-node-20260831.json)
and
[H200 record](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/results/h200-single-node-20260831.json).

Moving from 1×8 to 8×1 increased node throughput by 42.5% on B200 and 25.7% on
H200, while mean latency rose by 5.5× and 6.3×. The 4×2 topology retained 89.8%
of B200's 8×1 throughput and 93.9% of H200's, with lower latency. Replica count,
GPU allocation, and parallel method change together, so this comparison does
not isolate any one of them as the cause.

The full primary study contains **408 technically valid attempts out of 408**:
240 across ten B200 cells and 168 across seven H200 cells. The local
[`results.csv`](results.csv) contains all 17 cell aggregates, including median
and P95 latency, production-window seconds, concurrency, and source URLs. Values
retain the source record's precision. The earlier 147-attempt B200 study and
serving smoke tests are excluded from this count.

### Concurrency and repeat controls

Two requests in flight per replica added substantial mean latency for small
throughput gains in the measured cells:

| Platform | Topology | Node-throughput change, concurrency 1 → 2 | Mean-latency change, concurrency 1 → 2 |
| --- | --- | ---: | ---: |
| B200 | 1×8 | +3.24% | +47.14% |
| B200 | 2×4 | +2.10% | +47.72% |
| B200 | 4×2 | +1.23% | +48.77% |
| B200 | 8×1 | +0.41% | +32.92% |
| H200 | 2×4 | +1.43% | +48.78% |
| H200 | 4×2 | +0.48% | +49.62% |

Delayed repeats changed node throughput by −0.12% for B200 1×8 at concurrency
one, −0.14% for B200 1×8 at concurrency two, and −0.04% for H200 1×8 at
concurrency one. These are within-node observations, not fleet confidence
intervals. The B200 record combines two sessions on one node. See the upstream
[complete tables](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/BENCHMARKS.md).

## Measurement contract

| Control | Published reference value |
| --- | --- |
| Model | `nvidia/Cosmos3-Super`, Generator tower, BF16 |
| Model revision | `e0262be9d8f7586bc24c069a2aed2b665bdff266` |
| Container | `vllm/vllm-omni:cosmos3@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587` |
| Request | 1280×720, 189 frames, 24 fps; 7.875 seconds per valid clip |
| Sampling | 35 denoising steps, guidance 6.0, flow shift 10.0, maximum sequence length 4096 |
| Inputs | Fixed prompt and negative prompt from the pinned model snapshot; seeds 17/23/41 cycled within each replica |
| Other settings | Guardrails off; resolution/duration templates off; synchronous endpoint timeout 5400 seconds |
| Warmup and sample | One valid warmup per replica, excluded; 24 measured attempts per cell |
| Published v1 dispatch | Synchronized rounds across all replicas; next round waits for all requests in the current round |
| Published v1 latency | Client dispatch through receipt and persistence of the complete MP4; technical validation afterward |

Node throughput uses one shared production window:

```text
valid video-seconds per node-hour =
    technically valid clips × 7.875 × 3600 / production-window seconds
```

The published window includes request routing, generation, encoding, failed
attempts, replica skew, and the idle tail. It excludes service startup, model
loading, warmup, and post-response validation. Failed attempts stay in the
denominator and elapsed time but earn zero video-second credit.

The technical gate requires HTTP 200, non-empty MP4 bytes, complete decode,
1280×720, exactly 189 frames at 24 fps, duration 7.80–7.95 seconds, and spatial
detail and motion checks. It does not measure prompt adherence, visual quality,
physical plausibility, or downstream training utility. Exact prompt hashes,
runtime versions, host controls, and gate definitions are in the upstream
[method](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/reproduce/METHOD.md).

B200 and H200 match on the software and workload controls but differ in GPU
memory, VBIOS, host CPU, and host memory. Their absolute gap is a comparison of
complete systems. The hybrid parallel dimensions overlap on the same eight
GPUs; multiplying their labels does not give the GPU count.

## Workbench workflows

These four YAMLs are byte-identical copies of the maintained
`workflows/testing/cosmos3-super-*` specs, with a consistency check in the
documentation guardrails. The resource requirements and commands are unchanged.

| YAML | GPU allocation | Workbench coverage |
| --- | --- | --- |
| [`cosmos3-super-b200-benchmark.yaml`](cosmos3-super-b200-benchmark.yaml) | `B200:8` | Four primary cells; `suite=b200-full` selects all ten B200 cells and 240 measured attempts |
| [`cosmos3-super-h200-benchmark.yaml`](cosmos3-super-h200-benchmark.yaml) | `H200:8` | Four primary cells and 96 measured attempts |
| [`cosmos3-super-b200-single-gpu.yaml`](cosmos3-super-b200-single-gpu.yaml) | `B200:1` | One TP-1 service, one warmup, 24 measured requests |
| [`cosmos3-super-h200-single-gpu.yaml`](cosmos3-super-h200-single-gpu.yaml) | `H200:1` | One TP-1 service, one warmup, 24 measured requests |

Single-GPU checks report rates per GPU-hour and service-hour. They cannot stand
in for the eight-replica 8×1 node result.

**Protocol difference:** the current Workbench runner includes technical
validation in request and production-window time and uses per-replica worker
queues after an initial barrier. It does not implement the published v1
synchronized-round protocol. Its results must retain their own run identity and
timing definition. Matching the model and image pins alone does not make a run
an exact reproduction of the table above. The
[upstream review](upstream-review.md) records this and other coverage gaps.

### Validate and run

From the repository root with [NPA installed](../../docs/install.md), validate
all four entry points without allocating GPUs:

```bash
for spec in benchmark/cosmos3-super/*.yaml; do
  npa workbench workflow validate-spec "$spec"
done
npa workbench workflow plan-spec \
  benchmark/cosmos3-super/cosmos3-super-b200-benchmark.yaml \
  --run-id cosmos3-super-benchmark-plan
```

Expect one terminal `benchmark` stage invoking
`workbench.cosmos3.super_benchmark`. Validation and planning do not establish
model access, image pullability, or measured performance.

For execution on Nebius, follow the
[serving benchmark runbook](../../docs/workbench/cosmos3-super-serving.md#reproduce-the-primary-sweep-or-complete-b200-record)
and [workflow operations](../../workflows/README.md#commands). Supply an existing
GPU context, your bucket, and a validated immutable `runtime_image` for the
operator-built [SkyPilot wrapper](../../npa/docker/workbench/cosmos3-super-benchmark/REDISTRIBUTION.md).
The YAML's registry and bucket values are placeholders. The wrapper inherits
the pinned upstream image and adds worker bootstrap packages; it is separate
from the public `npa-cosmos3-serving` bootstrap. Keep the 32-GiB shared-memory
mount and the selected GPU count.

Check HF/S3 credentials and the `cosmos3-super-benchmark` access capability
before GPU submission. Supply the operator-reviewed runtime acceptance,
`HF_TOKEN`, and storage credentials through the existing secret mechanism.
Select `suite=b200-full` only on the B200 node workflow. Workbench does not yet
provide the full seven-cell H200 suite.

Inspect `benchmark.json` and each cell's `attempts.json`, `window.json`,
`derived.json`, `cell.json`, `complete.json`, and validated production MP4s.
The completion marker supports durable cell reuse only when the immutable
contract matches. Keep exact operational evidence in private storage, publish
sanitized metrics, and label new runs separately from these reference results.
Cancel the exact workflow run before removing its compute; follow the
[teardown guide](../../docs/teardown.md).

## Credits and provenance

- **Stewart Tong / [`stewtong/cosmos3-super-serving`](https://github.com/stewtong/cosmos3-super-serving):** source benchmark design, public measurement records, protocol clarification, and serving/reproduction reference. This folder summarizes factual aggregates and links to the source; it does not vendor the upstream harness. Its repository-authored material is [Apache-2.0 licensed](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/LICENSE), with separate [third-party provenance](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/PROVENANCE.md).
- **Stewart Tong and Timothy Le:** authors of the supplied “Serving World Models” article and its headline comparison.
- **NVIDIA / [`NVIDIA/cosmos`](https://github.com/NVIDIA/cosmos) and [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super):** the model and upstream guidance. The model has its own terms. Thanks to NVIDIA's Cosmos and performance teams for their input on benchmark controls and topology terminology, as acknowledged in the article.
- **[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni):** the serving engine and pinned container used by the study.
- **[`skypilot-org/skypilot`](https://github.com/skypilot-org/skypilot):** Workbench's workflow execution substrate on Nebius; this orchestration path is separate from Stewart's direct Docker launcher.

The CSV is a factual aggregate extract, not a copy of the raw evidence. Each row
identifies its source commit, record URL, record SHA-256, and cell key. The
upstream records retain the per-attempt inputs needed to rederive aggregates;
their verifier and checksums are linked in the [review](upstream-review.md).
