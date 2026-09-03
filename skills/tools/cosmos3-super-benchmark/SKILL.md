---
name: cosmos3-super-benchmark
description: Reproduce, operate, validate, or troubleshoot the fixed Cosmos3-Super eight-GPU benchmark or isolated single-H200 TP-1 validation through the immutable public vLLM-Omni image.
---

# Cosmos3-Super B200/H200 Benchmark

Use this skill for the production benchmark in
`npa/workflows/workbench/npa-workflows/cosmos3-super-b200-benchmark.yaml` or
`npa/workflows/workbench/npa-workflows/cosmos3-super-h200-benchmark.yaml`.
It is different from Cosmos Framework native Ray Serve (`cosmos3-ray-serve`):
this workload runs the public vLLM-Omni synchronous video endpoint. The default
`primary` suite measures four independent-service arrangements on one complete
B200 or H200 node. The `b200-full` suite reproduces the public machine-readable
ten-cell B200 record and its 240 measured attempts.

The separate `h200-single-gpu` suite is a functional/performance validation of
one TP-1 service on one H200. It fixes one warmup and 24 sequential measured
requests with the same model, image, prompt, seed, workload, timeout, and MP4
gates. Report its rates per GPU-hour and service-hour. Never label it eight-GPU
node throughput or compare it as the paper's `8x1` cell, which uses eight
independent one-GPU services on a complete node.

## Fixed contract

- Image: `vllm/vllm-omni:cosmos3` at the digest in the workflow.
- Model: `nvidia/Cosmos3-Super` at revision
  `e0262be9d8f7586bc24c069a2aed2b665bdff266`.
- Arrangements, in order: one 8-GPU hybrid service; two TP-4 services; four
  TP-2 services; eight TP-1 services. Every arrangement occupies all 8 GPUs.
- Workload: BF16 text-to-video, 1280x720, 189 frames, 24 fps, 35 steps,
  guidance 6.0, flow shift 10.0, max sequence length 4096, pinned model anchor
  and negative prompts, seed cycle 17/23/41, guardrails disabled, synchronous
  timeout 5400 seconds.
- One technically valid warmup per service is excluded. Every measured cell is
  24 attempts. The primary cells have exactly one request in flight per service.
- `b200-full` adds a concurrency-two cell for every topology and delayed repeats
  of the 1x8 concurrency-one and concurrency-two cells. Request concurrency is
  per service and is independent of replica count.
- `h200-single-gpu` fixes topology `1x1`, H200, 24 measured attempts, one request
  in flight, and reports an explicit non-paper validation scope.

Do not turn the reference frontier in the documentation into expected output or
a pass threshold. A new report is live evidence only when its per-attempt
records and window were emitted by the real command.

## Before submit

Load and follow `skills/atomic/health-preflight/SKILL.md`,
`skills/atomic/third-party-eula-preflight/SKILL.md`,
`skills/atomic/solution-licensing/SKILL.md`, and
`skills/atomic/protect-nebius-infra-details/SKILL.md`.

Run the exact access gates before provisioning:

```bash
npa/.venv/bin/npa workbench health preflight --checks hf,s3 --json
npa/.venv/bin/npa workbench health access --capability cosmos3-serving --json
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/cosmos3-super-<gpu>-benchmark.yaml
```

The operator must independently review the runtime terms and pass
`NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES` at submit time. Never commit,
persist, or bake acceptance, credentials, model weights, prompt text, generated
clips, or runtime caches.

## Run

Submit the shipped spec through `npa workbench workflow submit`, selecting the
operator's exact Kubernetes context and bucket. Pass the acceptance value,
`HF_TOKEN`, and S3 credentials through `--secret-env`; never render their values
into YAML or logs. Use the recipe for the intended hardware. The resource profile
must remain `B200:8` or `H200:8`, must keep the 32-GiB `/dev/shm`, and must retain
the exact external image digest. The H200 path sets PyTorch expandable segments
for the lower-memory one-GPU service cell; B200 behavior is unchanged.

The command starts services sequentially and refuses to open a cell's measured
window if any service warmup fails technical validation. It tears down one cell
before starting the next. Do not run two cells concurrently on the same node.

The shipped workflow defaults to `suite: primary`. Select the complete B200
record with `--var suite=b200-full`; that suite fails closed unless the GPU family
is B200, all four topologies are selected, and every cell has exactly 24 attempts.
After each cell, its records and validated clips are uploaded before an immutable
completion marker is created and read back. A resumed run reuses only a marker
whose complete cell record matches the current model/image/workload/prompt/run
contract; a partial cell is rerun and a conflicting completed cell fails closed.

## Evidence and interpretation

The durable root contains `benchmark.json`, per-cell `attempts.json`,
`window.json`, `derived.json`, `cell.json`, `complete.json`, and validated
production MP4s. A valid attempt
requires HTTP 200, non-empty bytes, full first-to-last decode, exact geometry,
frame count and rate, and passing blank/frozen checks. Failures remain in the
window and receive zero valid-video-second credit.

Use only the shared first-dispatch-to-final-completion window for node
throughput. It includes routing, generation, encoding, skew, failure time, and
tail idle time; it excludes startup, model load, and warmup. The final full-suite
report derives primary-frontier, concurrency-two, and delayed-repeat comparisons
from the live cell records. Report throughput as
*technically valid video-seconds per node-hour*, not accepted or useful output.

## Teardown

Cancel the exact workflow run before removing any cluster or controller. Follow
`skills/atomic/teardown-and-cost/SKILL.md`; shared controllers and operator
projects are not run-owned resources. Preserve exact infrastructure identifiers
only in access-controlled evidence.

## Verify

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_cosmos3_super_benchmark.py \
  npa/tests/workflows/test_cosmos3_super_b200_benchmark_workflow.py \
  npa/tests/workflows/test_cosmos3_super_h200_benchmark_workflow.py \
  npa/tests/cli/test_cosmos3_cli.py -q
```
