# Cosmos3-Nano on RTX PRO 6000 Blackwell Server Edition

This benchmark measures NVIDIA Cosmos3-Nano's Generator served by a
development build of vLLM-Omni (0.32.dev2, pinned by digest) on one node with
eight RTX PRO 6000 Blackwell Server Edition GPUs. These GPUs connect over PCIe
with no NVLink. The cells answer three questions for 720p text-to-video:
whether Nano serves on this part, how much video one node produces, and how
multi-GPU serving layouts behave over PCIe. Seven cells cover eight
independent one-GPU services, a relaunch with a pinned model revision,
guardrails, 480p, and three multi-GPU layouts; an NCCL all-reduce test
characterizes the node's PCIe fabric.

These are reference results from direct container runs on 2026-10-20,
2026-10-21 and 2026-10-25 UTC, not measurements produced by a Workbench
workflow. The per-request records are in [`records/`](records/), and the
latency and throughput columns of [`results.csv`](results.csv) are recomputed
from them.

## Cells

| Cell | Services × GPUs per service | Parallelism inside a service | Change from R8 |
| --- | --- | --- | --- |
| R8 | 8 × 1 | none | the reference cell |
| R8-H | 8 × 1 | none | relaunched four days later with the v1 harness's server command and a pinned model revision |
| R8-G2 | 8 × 1 | none | guardrails on (a second attempt; see below) |
| R8-480 | 8 × 1 | none | 832×480 instead of 1280×720 |
| R4-2 | 4 × 2 | tensor parallelism (TP), 2-way | |
| R2-4 | 2 × 4 | TP, 4-way | |
| R1-8U | 1 × 8 | 2-way classifier-free-guidance (CFG) parallelism × 4-way Ulysses sequence parallelism = 8 GPUs, with weights sharded across all 8 by HSDP (hybrid sharded data parallelism) | |

CFG parallelism runs the conditional and unconditional denoising passes on
separate GPU groups; Ulysses parallelism splits the token sequence across GPUs
and exchanges it with all-to-all collectives. The first guardrail attempt, R8-G,
ran with guardrails off because the image's stock deploy config disables them;
it is not published, and R8-G2 replaced it.

## Results

Every cell: 24 measured attempts, 24 technically valid; one request in flight
per service; text-to-video, 189 frames at 24 fps (7.875 s), 35 denoising
steps, BF16 weights.

| Cell | Shape | Mean latency (s) | P95 (s) | Valid clips per node-hour | Valid video-seconds per node-hour | Max memory used per GPU (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R8 | 1280×720 | 370.7 | 373.7 | 77.1 | 607.1 | 49,477 |
| R8-H | 1280×720 | 370.2 | 372.8 | 77.2 | 608.3 | 49,477 |
| R8-G2 | 1280×720 | 406.7 | 412.2 | 70.3 | 553.7 | 55,700 |
| R8-480 | 832×480 | 108.6 | 109.4 | 263.6 | 2,076.0 | 49,479 |
| R4-2 | 1280×720 | 242.7 | 245.3 | 59.0 | 464.7 | 35,965 |
| R2-4 | 1280×720 | 171.0 | 171.4 | 42.1 | 331.4 | 29,051 |
| R1-8U | 1280×720 | 71.3 | 71.9 | 50.5 | 397.5 | 26,221 |

P95 is the 23rd of 24 sorted latencies. Max memory used is the highest
per-GPU `nvidia-smi memory.used` sample, taken every 10 s during the measured
window, of 95.6 GiB usable per GPU. It is driver-level usage, including the
CUDA context and the allocator's cache, and it was flat for each whole window
(R2-4 and R1-8U varied by under 60 MiB). R8-480 reused R8's running services,
so its reading is R8's allocation. The samples are in each cell's
`gpu-telemetry.jsonl`.

What the cells show. These are within-node observations from one run of 24
attempts per cell, not fleet confidence intervals.
- **For 1280×720 text-to-video, eight one-GPU services gave the most
  output.** R8 produced 607.1 valid video-seconds per node-hour; the multi-GPU
  cells produced 76.5% of that (R4-2), 54.6% (R2-4) and 65.5% (R1-8U).
- **Splitting a request across GPUs cut its latency.** R1-8U returned a clip in
  71.3 s against R8's 370.7 s, at 50.5 clips per node-hour against 77.1. R1-8U
  had the node to itself; each R8 request shared it with seven others.
- **R8-H was indistinguishable from R8.** Its valid video-seconds per node-hour
  were 0.2% higher; R8's eight services alone ranged from 364.7 to 373.6 s mean
  latency. R8-H changed several launch settings at once (see the measurement
  contract), so it is a repeat of the whole system, not a one-variable control.
- **Guardrails cost 9.0% of valid video-seconds per node-hour** (R8-G2 against
  R8-H, run back to back with the same launcher), with 9.9% higher mean latency
  and 6,223 MiB more memory used per GPU. GPU utilization fell from 94.9% to
  87.4%.
- **480p yields 3.42× the valid video-seconds per node-hour of 720p** (R8-480
  against R8).
- **Board power and clocks stayed level.** In the first and last thirds of
  every 8 × 1 window, mean board power per busy GPU was 592 to 600 W and mean SM
  clock 2,219 to 2,236 MHz. R2-4's GPUs drew a mean of 460 W.

## PCIe fabric

`nvidia-smi topo -m` shows two groups of four GPUs, each behind one PCIe switch
attached to its own CPU socket: GPUs 0 to 3 and GPUs 4 to 7. `nvidia-smi topo
-p2p r` reports peer-to-peer reads as supported for all 56 ordered GPU pairs.
NCCL 2.29.7 used direct peer-to-peer transfers (`P2P/CUMEM`) inside each switch.
In the eight-GPU ring, the two links that cross sockets went through host
shared memory (`SHM/direct`; [`records/nccl/transports.txt`](records/nccl/transports.txt)).

| GPUs | Placement | All-reduce bus bandwidth at 16 MiB (GB/s) | At 1 GiB (GB/s) |
| --- | --- | ---: | ---: |
| 0, 1 | Same PCIe switch | 35.2 | 39.6 |
| 0 to 3 | Same PCIe switch | 39.8 | 41.6 |
| 4 to 7 | Same PCIe switch | 39.9 | 41.7 |
| 0 to 7 | Both switches, across CPU sockets | 26.0 | 27.8 |

Bus bandwidth follows the nccl-tests convention: algorithm bandwidth ×
2(n − 1)/n for an n-GPU all-reduce of BF16 data, in place. Each value is one
trial of 10 or 20 iterations timed on rank 0, with one group running at a
time on an otherwise idle node; smaller sizes were noisy (GPUs 4 to 7 measured
23.8 GB/s at 4 MiB against 37.3 for GPUs 0 to 3). This characterizes the
fabric's NCCL all-reduce. It does not measure the collectives inside the
serving cells: R1-8U's all-to-all and sharding traffic, whatever all-reduce
path vLLM-Omni chose for TP, or two groups running at once as in R2-4.

The multi-GPU cells sat on this topology as follows: R4-2 placed each service
on a pair inside one switch, R2-4 placed its two services on GPUs 0 to 3 and 4
to 7 at the same time, and R1-8U spans both switches.

## Relation to NVIDIA's published numbers

NVIDIA's
[Cosmos3-Nano Generator benchmarks](https://github.com/NVIDIA/cosmos/blob/e398d153/docs/reference/inference_benchmarks/cosmos3-nano-generator.md)
list vLLM-Omni text-to-video on one "RTX PRO 6000 Blackwell" at 371.53 s for
720p and 105.61 s for 480p, and note that the "vLLM-Omni numbers are for the
upcoming public release in the vLLM-Omni repo and are subject to change before
GA". Our mean client latencies with all eight services busy were 370.7 s and
108.6 s, 0.2% lower and 2.8% higher. NVIDIA's table does not say whether it
measured the Server Edition, the timing boundary or the runtime version, so
the differences are approximate.

An August 18 revision of NVIDIA's page
([`f9c42566`](https://github.com/NVIDIA/cosmos/blob/f9c42566/inference_benchmarks.md#cosmos3-nano-generator))
also listed multi-GPU vLLM-Omni rows for this GPU, defined as "total pipeline
time" and using tensor parallelism: 369.67 s on one GPU, 114.30 s on four and
68.66 s on eight. The current page lists only one-GPU rows.

| GPUs per request | NVIDIA, August | This benchmark | Ratio to one GPU, NVIDIA / here |
| --- | ---: | ---: | --- |
| 1 | 369.67 s | 370.7 s (R8, 8 services busy) | 1.00× / 1.00× |
| 2 | none | 242.7 s (R4-2, TP-2, 4 services busy) | none / 1.53× |
| 4 | 114.30 s | 171.0 s (R2-4, TP-4, 2 services busy) | 3.23× / 2.17× |
| 8 | 68.66 s (TP-8) | 71.3 s (R1-8U, a different layout, 1 service) | 5.38× / 5.20× |

Our latencies were measured with different numbers of services sharing the
node; NVIDIA's page does not state its load. R2-4 took 1.50× as long as
NVIDIA's four-GPU row. The cause was not isolated. Two TP-4 services ran at
once here, the vLLM-Omni builds differ, NVIDIA's August page names "RTX PRO
6000 Blackwell" in this table but lists "RTX PRO 6000 Blackwell Server
Edition" separately elsewhere, and its "total pipeline time" may not match our
client-side boundary. The NCCL results above do not separate these factors.
TP-8 was not measured here.

## Against the H200 Nano record

The same prompt files and request shape ran on one eight-GPU HGX H200 node in
the [Cosmos3-Nano H200 record](../cosmos3-nano/README.md):

| Node | Cell | Image | Mean latency (s) | Valid video-seconds per node-hour |
| --- | --- | --- | ---: | ---: |
| HGX H200 | N8 | `vllm/vllm-omni:cosmos3@sha256:6d2630c7…` | 213.3 | 1,051.9 |
| HGX H200 | N8-V30 | `vllm/vllm-omni:v0.30.0@sha256:fc77cfaa…` | 204.1 | 1,100.0 |
| RTX PRO 6000 | R8 | `vllm/vllm-omni@sha256:86fb713b…` | 370.7 | 607.1 |

R8 produced 58% of N8's and 55% of N8-V30's valid video-seconds per
node-hour, each counted under its own record's validity gate. Both nodes ran
NVIDIA driver 580.173.02. Everything else differs, so the ratio compares
complete systems:
- the image (three different builds);
- the benchmark client and its request strings (see the measurement contract);
- the latency boundary (the H200 timing also included writing the MP4 to disk);
- the validity gate (narrower here);
- warmup (one valid warmup per service there, one round here);
- memory sampling (every 5 s over the whole cell there, every 10 s over the
  window here).

The H200 record measured guardrails at 8.0% of valid video-seconds per
node-hour, against 9.0% here. Its guardrail cell ran more than three hours
after its baseline, after a service restart, using the harness's guardrails
flag on a different image, so the one-point gap is not attributable to the
GPU.

## Measurement contract

| Control | Value |
| --- | --- |
| Model | `nvidia/Cosmos3-Nano`, Generator, BF16, revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`. R8-H and R8-G2 pinned it with `--revision`; for the other cells, the node's environment manifest records this revision in the model cache, and the launches did not pin it |
| Container | `vllm/vllm-omni@sha256:86fb713bf73e118e537a55fa013dcdbdf2468ed61b98f4f2e7934a9482181dc1` (manifest list; vLLM 0.31.0 and vLLM-Omni 0.32.dev2+g4f62d6809, read from the server logs), the image behind the `cosmos3` tag on 2026-10-19. At startup it warns that its vLLM and vLLM-Omni versions do not match. Every service used `--omni --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml`, except R8-G2 (below) |
| Sampling | 35 steps, guidance 6.0, flow shift 10.0, max sequence length 4096 |
| Hardware | One node with eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (95.6 GiB usable each), PCIe, NVIDIA driver 580.173.02, two Intel Xeon 6776P sockets (192 vCPUs) |
| Guardrails | Off, by the stock deploy config, except R8-G2: a copy of that config with `guardrails: true` ([`records/deploy/`](records/deploy/cosmos3_omni_guardrails_on.yaml)) and access to the gated `nvidia/Cosmos-1.0-Guardrail` model |
| Inputs | The Nano snapshot's `assets/example_t2v_prompt.json` and `negative_prompt.json`, the same files as the H200 record. The client sends each file's JSON re-serialized without whitespace: 6,760 and 14,677 characters, SHA-256 `0ec8de42ababcdea182b5ea1a4b87a46fc9258f8cea93c8adf9f230a4d8c365b` and `bee47841d230ebdc16f9ba4bea9ece4a8bdd8cef9c48050db12a2fb90cffd721`. The H200 harness sent the file text (7,710 and 15,378 characters, `61c9c4b4…` and `007a1bdf…`). The JSON content is identical; the effect of the difference on timing was not measured. The `prompt_sha256` and `negative_sha256` fields in each `window.json` hash the raw files (`61c9c4b4…`, `7a60bd0c…`) |
| Warmup and sample | One warmup round per cell, excluded; then 24 measured attempts; seeds 17/23/41, one seed per round |
| Dispatch | Synchronized rounds: every service receives one request, and the next round starts when all have returned. A round therefore lasts as long as its slowest request, which put measured throughput 0.5% to 0.8% below what mean latency implies for the cells with four or eight services, and within 0.1% for R2-4 and R1-8U. The cells ran 3, 6, 12 and 24 rounds for 8, 4, 2 and 1 services |
| Latency | Client dispatch through receipt of the complete MP4 response |
| Throughput | Technically valid clips × clip seconds × 3600 / production-window seconds |

The technical gate requires HTTP 200 and, from `ffprobe`, the expected width,
height, decoded frame count and 24 fps. It is narrower than the H200 record's
gate (no duration, spatial detail or motion checks), it cannot detect a blank
or blocked output that still decodes, and it does not measure prompt
adherence, visual quality or physical plausibility.

R8-H differs from R8 in its launch, not its workload: the `vllm serve` entry
point instead of `vllm-omni serve`, `--revision`, `--init-timeout 1800`,
`--ipc=host`, `--shm-size 32g` instead of `8g`, a raised open-file limit, and
`VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400`.

[`records/cells/R8-G2/guardrail-gate.json`](records/cells/R8-G2/guardrail-gate.json)
records how guardrails were confirmed. Before the timed cell, eight
guardrails-on services were started; all eight logged "Cosmos3 guardrails
initialized.", and one seed-17 request produced an output whose size no
guardrails-off cell produced. Those services were then removed and the timed
services started fresh. In the timed cell, all eight services logged the same
line and none was launched with `--no-guardrails`, and memory used rose by
6,223 MiB per GPU against R8-H. The guardrails-off cells reproduced the same
two or three output sizes per seed across days, and none of R8-G2's 24 outputs
has one of those sizes, although all stay within about 1% of them.
Per-request guardrail verdicts were not recorded.

## Records and reproduction

[`records/`](records/) holds, for each cell:
- `requests.jsonl`: one row per measured attempt, with timings, seed, output
  size and gate result;
- `warmups.jsonl`: the warmup round. The client labels these rows
  `"kind": "production"`, as it does every request; the split comes from the
  window's warmup list;
- `window.json`: window boundaries, request shape, service ports and the
  client's own summary;
- `gpu-telemetry.jsonl`: the `nvidia-smi` samples (utilization, memory used,
  board power, SM clock per GPU, every 10 s).

The client wrote requests and warmups as one `records.jsonl`; this folder
splits it. The published `window.json` files omit four fields of the client's
output: the telemetry samples, now in `gpu-telemetry.jsonl`; three values the
client wrote identically for every cell (`topology: "8x1"`, `concurrency: 8`
and an unused comparison label), which are wrong for the multi-GPU cells; and
one internal planning figure in the summary. The layout is in `results.csv`
and `replica_ports`. `SHA256SUMS` covers every file as published here, not
the client's original output. [`records/nccl/`](records/nccl/) holds the
all-reduce results, topology maps and transports. The output videos are not
published.

`results.csv` uses the H200 record's columns, where `peak_device_memory_mib`
is the max memory used described above, and adds two: `mean_board_power_w`
(mean per-GPU board power over samples where that GPU's utilization exceeded
50%, during the window) and `launch` (how the services were started).

Every cell ran the client [`reproduce/r8-driver.py`](reproduce/r8-driver.py),
which calls each service's `/v1/videos/sync` directly. It needs Python 3 with
`ffprobe`, `nvidia-smi` and `sha256sum` on the path, reads
`$R8_BASE/prompts/example_t2v_prompt.json` and `negative_prompt.json`, and
writes to `$R8_BASE/$R8_RUN_DIR/`. Its settings come from `R8_CELL`,
`R8_RUN_DIR`, `R8_PORTS` (default 8100 to 8107), `R8_ROUNDS` (default 3),
`R8_WIDTH`, `R8_HEIGHT`, `R8_FRAMES` and `R8_GUIDANCE`. The published client
differs from the one that ran in two ways: its base directory comes from
`R8_BASE` instead of a fixed host path, and two lines that computed an
internal planning figure are removed. Its docstring still describes only R8,
and `import sys` is unused.

Prepare the prompts once:

```text
mkdir -p <run-dir>/prompts
cp <Cosmos3-Nano snapshot>/assets/example_t2v_prompt.json <Cosmos3-Nano snapshot>/assets/negative_prompt.json <run-dir>/prompts/
```

Every service ran under nerdctl with host networking, as root through `sudo`,
because this node's operating-system image provides nerdctl without Docker or
a CNI bridge plugin. R8-H and R8-G2 used
[`reproduce/launch-8x1.sh`](reproduce/launch-8x1.sh) (bash 4.4 or later), which
carries the [v1 harness](https://github.com/stewtong/cosmos3-super-serving/tree/951dc1162fcc6c47d694e29616a1f833f293b877)'s
server command and container options into nerdctl:

```text
BASE=<run-dir> HF_CACHE=<hf-cache> bash reproduce/launch-8x1.sh R8-H off
R8_BASE=<run-dir> R8_CELL=R8-H R8_RUN_DIR=r8-h python3 reproduce/r8-driver.py
```

For R8-G2, copy `records/deploy/cosmos3_omni_guardrails_on.yaml` into
`<hf-cache>`, export `HF_TOKEN` for an account that has accepted the
`nvidia/Cosmos-1.0-Guardrail` terms, pass `on` instead of `off`, and use
`R8_CELL=R8-G2 R8_RUN_DIR=r8-g2`. Wait for `/health` to return 200 on every
port before starting the client.

The other cells were launched by hand with the `vllm-omni serve` entry point
and the stock deploy config. R8 ran one container per GPU, `<i>` from 0 to 7:

```text
sudo nerdctl -n default run -d --network host --gpus device=<i> --shm-size 8g \
  -v <hf-cache>:/root/.cache/huggingface \
  vllm/vllm-omni@sha256:86fb713bf73e118e537a55fa013dcdbdf2468ed61b98f4f2e7934a9482181dc1 \
  vllm-omni serve nvidia/Cosmos3-Nano --omni \
  --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml \
  --tensor-parallel-size 1 --port 810<i> --no-guardrails
R8_BASE=<run-dir> R8_CELL=R8 R8_RUN_DIR=r8 python3 reproduce/r8-driver.py
```

The original launches named the image as `vllm/vllm-omni:cosmos3`, which
pointed to the digest above from 2026-10-19 through at least 2026-10-25. The
tag can move, so use the digest.
The other hand-launched cells changed only what each line lists:
- **R8-480:** R8's running services; client with `R8_CELL=R8-480
  R8_RUN_DIR=r8-480 R8_WIDTH=832 R8_HEIGHT=480`.
- **R4-2:** four containers, `<i>` from 0 to 3, each with `--gpus all -e
  CUDA_VISIBLE_DEVICES=<2i>,<2i+1>`, `--shm-size 16g`,
  `--tensor-parallel-size 2`, port 810`<i>`, and no `--no-guardrails`; client
  with `R8_CELL=R4-2 R8_RUN_DIR=r4-2 R8_PORTS=8100,8101,8102,8103 R8_ROUNDS=6`.
- **R2-4:** two containers with `--gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3`
  and `4,5,6,7`, `--shm-size 16g`, `--tensor-parallel-size 4`, ports 8100 and
  8101, and no `--no-guardrails`; client with `R8_CELL=R2-4 R8_RUN_DIR=r2-4
  R8_PORTS=8100,8101 R8_ROUNDS=12`.
- **R1-8U:** one container with `--gpus all`, `--shm-size 16g`,
  `--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8`,
  port 8100, and no `--no-guardrails`; client with `R8_CELL=R1-8U
  R8_RUN_DIR=r1-8u R8_PORTS=8100 R8_ROUNDS=24`.

The hand-launched services omitted `--ipc=host` and the sync timeout variable
that the launcher sets. The multi-GPU services also omitted
`--no-guardrails`; the stock deploy config keeps guardrails off, and their
logs show no guardrail initialization. They used `CUDA_VISIBLE_DEVICES`
because nerdctl on this node rejects `--gpus device=0,1`.

The all-reduce test ran
[`reproduce/allreduce_bench.py`](reproduce/allreduce_bench.py) in the same
image, once per GPU group, with no other process on the GPUs:

```text
sudo nerdctl -n default run --rm --network host --ipc=host --gpus all \
  -e CUDA_VISIBLE_DEVICES=<group> -e NCCL_DEBUG=INFO -v <dir>:/work --entrypoint torchrun \
  vllm/vllm-omni@sha256:86fb713bf73e118e537a55fa013dcdbdf2468ed61b98f4f2e7934a9482181dc1 \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29511 \
  --nproc_per_node=<n> /work/allreduce_bench.py
```

## Limits

- One node. Each cell ran once; R8-H is the only repeat of R8, and it changed
  launch settings as well.
- A development build of vLLM-Omni served every cell. Released images were not
  measured on this GPU, and NVIDIA's values are for an unreleased build.
- The comparison with the H200 record spans different images, benchmark
  clients, request strings, latency boundaries, validity gates and nodes.
- The technical gate is not a quality measure, and no output was reviewed for
  quality.
- The NCCL test measures all-reduce only, one group at a time, in one trial.
- The documented commands were not rerun after this record was assembled; they
  reproduce the server commands, ports and client settings recorded for each
  cell.
- Video-to-video, transfer, two requests per service, TP-8 and Cosmos3-Super
  were not measured on this GPU.

## Credits and provenance

- **[Stewart Tong (@stewtong)](https://github.com/stewtong) (Nebius):** ran these measurements on 2026-10-20, 2026-10-21 and 2026-10-25 UTC and assembled this folder.
- **NVIDIA / [`NVIDIA/cosmos`](https://github.com/NVIDIA/cosmos) and [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano):** the model, its example prompts and the published single-GPU benchmarks. The model and assets have their own terms; neither is redistributed here.
- **[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni):** the serving engine, the pinned image, and the deploy config adapted in `records/deploy/` (Apache-2.0; the file states its change).
- **[`stewtong/cosmos3-super-serving`](https://github.com/stewtong/cosmos3-super-serving):** the v1 harness whose server command R8-H and R8-G2 use, and the method this folder follows.
