# Cosmos3-Nano on H200: node throughput against Cosmos3-Super

This benchmark measures NVIDIA Cosmos3-Nano's Generator served by vLLM-Omni
on one eight-GPU HGX H200 node, with eight independent one-GPU services. Its
anchor cell uses the request, seeds, layout and image of the
[Cosmos3-Super](../cosmos3-super/README.md) 8×1 H200 reference cell and the same
v1 benchmark protocol, so the two models' node throughput can be compared.
Nine further cells measure other workloads, a repeat, a second request per
service, guardrails and a newer runtime.

These are reference results from a direct Docker run on 2026-10-05 and
2026-10-06 UTC, not measurements produced by a Workbench workflow. Workbench
runs Nano in several workflows, but its Cosmos3 serving benchmark runner pins
Super, so no Workbench workflow measures Nano node throughput and this folder
has no workflow YAML. The per-request records are in [`records/`](records/),
and the latency and throughput columns of [`results.csv`](results.csv) are
recomputed from them.

## Nano against Super on one H200 node

| Model | Cell | Mean latency (s) | Valid clips per node-hour | Valid video-seconds per node-hour |
| --- | --- | ---: | ---: | ---: |
| Cosmos3-Super | H4_8x1 (2026-08-31) | 779.0 | 36.7 | 289.1 |
| Cosmos3-Nano | N8 (2026-10-05) | 213.3 | 133.6 | 1,051.9 |

Nano produced about 3.6× as many technically valid video-seconds per
node-hour as Super. Both cells ran eight TP-1 services with one request in
flight each, one excluded warmup per service and 24 measured attempts, all 24
valid. Both used image `vllm/vllm-omni:cosmos3@sha256:6d2630c7…`, the fixed
prompt and negative prompt (same request-text SHA-256 in both records), seeds
17/23/41, and 1280×720, 189 frames at 24 fps, 35 steps, BF16, guardrails off.

The two cells ran on different H200 nodes five weeks apart. The Super record
lists NVIDIA driver 580.173.02, the driver on the Nano node. Host CPU, host
memory and VBIOS were not matched, and variance between nodes was not
measured, so the ratio compares two complete systems. The Super values come
from the pinned
[H200 record](https://github.com/stewtong/cosmos3-super-serving/blob/951dc1162fcc6c47d694e29616a1f833f293b877/results/h200-single-node-20260831.json)
(SHA-256 `c7746bbc…`), as summarized in the Super folder's `results.csv`; 36.7
clips per node-hour is derived from that row as 24 × 3600 / 2,353.69 s.

## All Nano cells

Every cell: eight services × one H200, 24 measured attempts, 24 technically
valid.

| Cell | Workload | Shape | Steps | Mean latency (s) | P95 (s) | Valid clips per node-hour | Valid video-seconds per node-hour |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| N8 | Text-to-video | 1280×720, 189 frames, 24 fps | 35 | 213.3 | 215.6 | 133.6 | 1,051.9 |
| N8C2 | N8 with up to two requests in flight per service | same | 35 | 282.5 | 422.6 | 134.8 | 1,061.9 |
| N8R | N8 repeated about 93 minutes after N8 began | same | 35 | 213.6 | 216.0 | 133.3 | 1,049.9 |
| N8-480 | Text-to-video at 480p | 832×480, 189 frames, 24 fps | 35 | 61.8 | 62.4 | 461.1 | 3,631.4 |
| N8-V2V | Video-to-video; an N8 output clip as `input_reference`, first two latent frames conditioned | 1280×720, 189 frames, 24 fps | 35 | 216.2 | 218.3 | 131.9 | 1,038.8 |
| N8-TE2 | Edge transfer, NVIDIA cookbook asset, guidance 3, control guidance 1.5 | 1280×720, 121 frames, 30 fps | 50 | 504.0 | 511.2 | 56.3 | 227.2 |
| N8-TD2 | Depth transfer, same settings | 1280×720, 121 frames, 30 fps | 50 | 509.3 | 515.7 | 55.9 | 225.5 |
| N8-TL | Edge transfer of a LeRobot episode, medium Canny derived server-side, guidance 3, control guidance 1.5 | 832×480, 192 frames, 24 fps | 35 | 376.8 | 383.0 | 75.5 | 603.9 |
| N8-G | N8 with guardrails on | 1280×720, 189 frames, 24 fps | 35 | 232.8 | 234.2 | 122.9 | 968.2 |
| N8-V30 | N8 on `vllm/vllm-omni:v0.30.0@sha256:fc77cfaa…` | same | 35 | 204.1 | 206.4 | 139.7 | 1,100.0 |

Video-seconds per node-hour weight each clip by its length: 7.875 s for the
189-frame cells, 4.033 s for the cookbook transfer cells and 8.0 s for N8-TL.
Compare clip counts only between cells of the same shape. The cells ran in the
order N8, N8C2, N8-480, N8-V2V, N8-TL, N8R, N8-TE2, N8-TD2, N8-G and N8-V30;
N8-G and N8-V30 ran last, more than three hours after N8, each after a service
restart.

`results.csv` also gives peak device memory per cell, from `nvidia-smi`
sampled every 5 s on all eight GPUs over the whole cell, including warmup. The
telemetry itself is not published. N8 peaked at 49,807 MiB and N8-G at
56,121 MiB, of 143,771 MiB.

What the cells show. These are within-node observations from one run of 24
attempts per cell, not fleet confidence intervals; the one repeat moved
throughput by 0.19%.
- **A second request per service did not add throughput.** Each service ran
  its two requests one after the other: per service, the first finished after
  about 214 s and the second after about 422 s. N8C2's valid video-seconds per
  node-hour were 0.95% above N8's, with 32% higher mean latency. N8C2 needed
  two dispatch rounds instead of three, which can account for the small gain.
- **480p yields 3.45× the valid video-seconds per node-hour of 720p** (N8-480
  against N8).
- **Video-to-video ran at text-to-video speed.** N8-V2V's valid video-seconds
  per node-hour were 1.2% below N8's.
- **Transfer was the slowest of the measured workloads.** N8-TE2 and N8-TD2
  yield 4.6× fewer valid video-seconds per node-hour than N8; per frame and
  denoising step, their mean latency is about 2.6× N8's. N8-TL yields 6.0×
  fewer than N8-480, and this larger gap was not isolated.
- **Guardrails cost 8.0% of valid video-seconds per node-hour** on this prompt
  (N8-G against N8), with 9.1% higher mean latency and 6,314 MiB more peak
  device memory.
- **The v0.30.0 image produced 4.6% more valid video-seconds per node-hour**
  than the `cosmos3` image (N8-V30 against N8), with mean latency 4.3% lower.
  The workload controls were unchanged; the image swap changes vLLM, vLLM-Omni
  and their dependencies together.

## Relation to NVIDIA's published numbers

NVIDIA's
[Cosmos3-Nano Generator benchmarks](https://github.com/NVIDIA/cosmos/blob/c3e4354c96cc8c9c311babc93b51edc6fea43fb9/inference_benchmarks/cosmos3-nano-generator.md)
report single-GPU latency per GPU and engine. For vLLM-Omni on one H200 they
list 208.36 s for 720p text-to-video and 58.14 s for 480p, and note that the
vLLM-Omni values are pre-release. Our mean client latencies, with all eight
services busy, were 213.3 s and 61.8 s, which are 2.4% and 6.4% higher. The
page does not state the timing boundary, step budget or runtime version behind
the vLLM-Omni text-to-video values, so these differences are approximate. It
also lists single-GPU transfer latencies without their frame count or step
budget, so this folder does not compare against them, and it lists no
vLLM-Omni video-to-video values.

This folder adds node-level throughput with every GPU busy, under the Super
reference's controls, plus the concurrency, repeat, guardrail and
runtime-version cells above.

## Measurement contract

| Control | Value |
| --- | --- |
| Model | `nvidia/Cosmos3-Nano`, Generator, BF16, revision `e59a53c25979a090fa8706c9acc0c254a6e89b92` |
| Container | `vllm/vllm-omni:cosmos3@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587` (vLLM 0.25.0, vLLM-Omni 0.25.0rc2.dev62); N8-V30 only: `vllm/vllm-omni:v0.30.0@sha256:fc77cfaac7b1b43c30cbf32eaea214864434c6e0f303e6e9eb9f64104241a994` (vLLM 0.30.0, vLLM-Omni 0.30.0; other component versions not recorded) |
| Hardware | One node with eight H200 SXM 141 GB GPUs, NVIDIA driver 580.173.02 |
| Layout | Eight vLLM-Omni services, one GPU each (TP-1), on loopback `/v1/videos/sync` |
| Inputs | Text cells: the Nano snapshot's `assets/example_t2v_prompt.json` and `negative_prompt.json`, whose request text hashes to the Super record's prompt values. Transfer cells: NVIDIA cookbook prompts, negative prompt and control videos. N8-TL: the prompt in [`records/prompts/`](records/prompts/lerobot-cups-open.txt) and the cookbook negative prompt |
| Warmup and sample | One valid warmup per service, excluded; 24 measured attempts per cell; seeds 17/23/41 cycled |
| Dispatch | Synchronized rounds across the eight services, as in the Super reference |
| Latency | Client dispatch through receipt and persistence of the complete MP4 |
| Throughput | Technically valid clips × clip seconds × 3600 / production-window seconds |

The technical gate requires HTTP 200, a complete decode, the expected size,
frame count, frame rate and duration, and spatial detail and motion checks. It
does not measure prompt adherence, visual quality or physical plausibility.
For N8-TE2, N8-TD2 and N8-TL, the expected size, frame count and frame rate
were taken from the first warmup output rather than fixed in advance. In each
case they equal the requested values, recorded in the cell's `window.json`.

N8-TL uses Workbench's Nano augmentation defaults for size, frame rate, steps,
guidance, control guidance and Canny threshold. It differs from the Workbench
augmentation path in other ways: one 297-frame chunk setting rather than
121-frame chunks with a five-frame overlap, the cookbook negative prompt and no
system prompt rather than an empty negative prompt and Workbench's default
system prompt, and edge extraction on the server rather than a precomputed
control video.

## Records and reproduction

[`records/`](records/) holds, for each cell, `requests.jsonl` (one row per
measured attempt, with timings, seed, request shape, output SHA-256 and gate
result), `warmups.jsonl` and `window.json` (the window boundaries, rounds and,
for the driver cells, the request spec). `SHA256SUMS` covers every file under
`records/`. Host paths are rewritten to the portable names `inputs/…` and
`prompts/…`. The output videos and GPU telemetry are not published.

Every cell needs the v1 harness at
[`stewtong/cosmos3-super-serving@951dc116`](https://github.com/stewtong/cosmos3-super-serving/tree/951dc1162fcc6c47d694e29616a1f833f293b877);
the Nano run's `reproduce/benchmark.py`, `derive-results.py` and
`validate-video.py` were byte-identical to that revision. Start the eight
services with the harness launcher and a profile from
[`records/profiles/`](records/profiles/), which is the harness profile with
the model changed to Nano:

```text
bin/cosmos3-super --config profiles-nano.json serve --platform h200 --topology 8x1 --cache-dir <hf-cache>
```

Add `--guardrails` for N8-G, and use `profiles-nano-v0.30.0.json` for N8-V30.
The text cells N8, N8C2, N8R, N8-G and N8-V30 then ran:

```text
reproduce/benchmark.py --topology 8x1 --ports 8100,8101,8102,8103,8104,8105,8106,8107 \
  --prompt example_t2v_prompt.json --negative-prompt negative_prompt.json \
  --attempts 24 --concurrency 1 --warmups-per-replica 1 --output <cell-dir>
```

with `--concurrency 2` for N8C2 and `--guardrails` for N8-G.

The other five cells ran [`reproduce/p2-bench.py`](reproduce/p2-bench.py), a
driver around the same harness:

```text
python3 p2-bench.py --harness <harness-checkout> --spec <spec.json> --output <cell-dir>
```

It keeps the harness's dispatch, window, record schema and gate, and adds
per-cell request shapes, file uploads and expected outputs from
[`records/specs/`](records/specs/). It patches harness internals, so it works
only with the 951dc116 harness. N8-TE2 and N8-TD2 used `N8-TE.json` and
`N8-TD.json`; their transfer specs name a `control_path` that the server reads,
so the control video must be mounted in the service containers at that path.
N8-V2V needs an N8 output clip as its input: a rerun uses its own clip, so its
input bytes will differ from the original.

The published `p2-bench.py` is the final revision, used for N8-TE2, N8-TD2 and
N8-TL. N8-480 and N8-V2V ran an earlier revision without lines 102 to 113, which
was not kept. Those lines re-space the gate's frame samples over the expected
frame count and give the same indices for 189 frames. They were added after
the first N8-TE and N8-TD attempts stopped at warmup: the harness samples
frames 0, 47, 94, 141 and 188, which do not all exist in a 121-frame clip.
Those failed first attempts are not included here.

Input files and their SHA-256:

| Input | Source | SHA-256 |
| --- | --- | --- |
| `inputs/transfer/edge/control_edge.mp4`, `edge/prompt.json` | [NVIDIA cookbook transfer assets](https://github.com/NVIDIA/cosmos/tree/c3e4354c96cc8c9c311babc93b51edc6fea43fb9/cookbooks/cosmos3/generator/transfer/assets) | `9d0fb1d1…`, `2b3f6354…` |
| `inputs/transfer/depth/control_depth.mp4`, `depth/prompt.json` | same | `0552c261…`, `2caf2ec1…` |
| `inputs/transfer/negative_prompt.json` | same | `86b667ce…` |
| `inputs/n8-clip.mp4` | N8 output `production-r0-a000` | `0a52387f…` |
| `inputs/lerobot/episode0-832x480-24fps.mp4` | [`lerobot/aloha_static_cups_open`](https://huggingface.co/datasets/lerobot/aloha_static_cups_open) revision `d793c969`, `observation.images.cam_high/chunk-000/file-000.mp4` (`681a1081…`), first 8 s | `f8b73374…` |

The LeRobot input was made with FFmpeg 6.1.1:

```text
ffmpeg -ss 0 -t 8 -i file-000.mp4 -vf "fps=24,scale=640:480,pad=832:480:96:0:black" \
  -frames:v 192 -c:v libx264 -crf 12 -pix_fmt yuv420p -an episode0-832x480-24fps.mp4
```

FFmpeg 8.1.1 produces the same 832×480, 24 fps, 192-frame shape with
different bytes.

## Limits

- One node, one session. Each cell ran once; N8R is the only repeat.
- The Super comparison spans two nodes and five weeks, with the host matched
  only on driver and image.
- The technical gate is not a quality measure, and no output was reviewed for
  quality here.
- Cost to serve, Ray Serve batching, multi-camera episodes and Super transfer
  were not measured.
- B200 and other GPUs were not measured for Nano in this folder.

## Credits and provenance

- **Stewart Tong (Nebius):** ran these measurements on 2026-10-05 and 2026-10-06 UTC and assembled this folder.
- **NVIDIA / [`NVIDIA/cosmos`](https://github.com/NVIDIA/cosmos) and [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano):** the model, the transfer cookbook assets and the published single-GPU benchmarks. The model and assets have their own terms; neither is redistributed here.
- **[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni):** the serving engine and both pinned images.
- **[`stewtong/cosmos3-super-serving`](https://github.com/stewtong/cosmos3-super-serving):** the v1 harness, the Super reference record and the method this folder reuses.
- **[LeRobot](https://github.com/huggingface/lerobot) and the MIT-licensed `aloha_static_cups_open` dataset:** the source episode for N8-TL, used as input only and not redistributed.
