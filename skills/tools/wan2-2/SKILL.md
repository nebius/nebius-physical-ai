---
name: wan2-2
description: Use when packaging, running, reviewing, or extending the Alibaba Wan 2.2 TI2V-5B BYOF solution, its official video artifacts, or its verified Rerun evidence.
---

# Wan 2.2 Workbench support

Use this skill for the public Wan 2.2 registry candidate and its verified video
evidence. Read these files before changing behavior:

- `npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml`
- `npa/workflows/workbench/npa-workflows/byof-wan2.2-multigpu.yaml`
- `npa/src/npa/workflows/wan_rerun.py`
- `docs/workbench/wan2.2.md`

Also load `byof-onboard`, `oss-solution-registry-onboard`,
`author-npa-workflow`, `real-components`, `solution-licensing`, `gpu-selection`,
`nebius-infra`, `testing-conventions`, `npa-agent`, and
`agent-visual-feedback` when their surfaces are involved.

## Ground truth

- Official source: `https://github.com/Wan-Video/Wan2.2.git`, pinned to
  `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.
- Official model: `Wan-AI/Wan2.2-TI2V-5B`, pinned to
  `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`.
- TI2V-5B is a stock generative-video model supporting text and image inputs.
- A historical operator-only validation record accepted the real single-GPU
  text-to-video path on RTX PRO 6000 Blackwell (`sm_120`) from immutable image
  digest `sha256:1baa4e2e89999ea26df81891ac786fa99c7498cbf173e5c5abad54c6f1dd1d13`,
  including exact MP4/RRD byte identity.
- A historical operator-only validation record accepted one shared official
  generation from that same observed image digest on four B200s (`sm_100`) with
  world size 4, NCCL, T5 and DiT FULL_SHARD FSDP, Ulysses size 4, and exact
  MP4/RRD byte identity.
- Those records used Torch 2.7.1/CUDA 12.8 and NCCL 2.27.7. The current
  acceptance gate is Torch 2.13.0/CUDA 13.0 and NCCL 2.29.7; it requires fresh
  operator-accepted single- and four-GPU evidence before publication.
- I2V, A14B, speech-to-video, Animate, and training are separate capabilities.
- Stock Wan does not predict robot actions. Never claim that it is
  action-conditioned.

For changing facts, use only the official Wan repository, official Wan-AI model
cards, and primary framework documentation.

## Packaging contract

Use `workbench.byof.repo`; do not add a fake Wan toolRef. Keep the repo and all
model inputs immutable. The image may contain pinned source and dependencies but
no checkpoint weights, credentials, private code, or user data. The runtime
must remain non-root, with `/opt/byof` and its venv readable and executable.

The single-GPU baseline requests one RTX PRO 6000 Blackwell (`sm_120`), uses the
security-fixed PyTorch 2.13.0 CUDA 13.0 wheel line, and binds pinned Wan
attention to native PyTorch SDPA instead of FlashAttention. Record the device,
compute capability, driver, CUDA, torch version, compiled arch list, and finite
SDPA probe.

The distributed spec uses `byof-solution-smoke-wan22-b200-4gpu.yaml` and exactly
four ranks. Invoke the official path with:

```text
/opt/byof/.venv/bin/python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc_per_node=4 wan22_distributed_wrapper.py
# The generated wrapper instruments all ranks, then executes pinned official
# /opt/byof/generate.py as __main__ with --dit_fsdp --t5_fsdp --ulysses_size 4.
```

Fail closed unless every rank proves NCCL initialization/all-reduce, a unique
local B200, T5 and DiT FULL_SHARD wrappers, live Ulysses distributed-attention
and all-to-all calls, the upstream final barrier, observer terminal
synchronization, and compute capability 10.0 with `sm_100` support.

The single-GPU smoke writes:

- `wan2_2_ti2v_5b.mp4`
- `wan2_2_ti2v_5b_text_to_video.json`
- `wan2_2_runtime_inventory.json`

The distributed smoke additionally writes:

- `wan2_2_ti2v_5b_multigpu.mp4`
- `wan2_2_ti2v_5b_multigpu.json`
- `wan2_2_multigpu_topology.json`
- `wan2_2_multigpu_runtime_inventory.json`
- `wan2_2_multigpu_rank_0.json` through `wan2_2_multigpu_rank_3.json`

Decode every frame and fail on invalid dimensions/count/FPS, a corrupt or empty
container, an implausibly small file, or uniform content. Keep
`capabilities_exercised` exact and `deferred` empty for hard-gated runs.

## Rerun evidence contract

Every successful named Wan solution smoke is postprocessed by
`npa.workflows.wan_rerun`. The postprocessor runs after the existing BYOF S3
upload and must fail the parent command if source validation, RRD generation,
local parsing, `rerun rrd verify`, upload, S3 byte verification, remote parsing,
or manifest verification fails.

Use Rerun SDK `0.31.4`, matching the agent-compatible `npa[viz]` extra. Embed
the exact MP4 at `/wan2_2/video/asset`, log one timestamped
`VideoFrameReference` per decoded frame at `/wan2_2/video/frame`, and use the
`video_time` duration timeline. Static JSON facts belong in the summary,
validation, runtime, distributed, rank, and metric entities; do not invent a
time series.

The distributed filenames are:

- `wan2_2_ti2v_5b_multigpu.rrd`
- `wan2_2_ti2v_5b_multigpu_rrd_manifest.json`

The manifest must contain source object URIs plus ETags, byte sizes, and
SHA-256 values; RRD URI/hash/size/version/entities; embedded-video identity;
and local plus remote verification. Only a successfully uploaded and remotely
verified manifest may name `wan2.2_verified_rerun_recording`.

## Capability status

| Capability | Status |
| --- | --- |
| `wan2.2_ti2v_5b_text_to_video` | accepted current evidence; exact public-dev digest ran the Torch 2.13.0/CUDA 13.0 closure on RTX PRO 6000 |
| `wan2.2_decoded_mp4_validation` | accepted current evidence; 17 decoded 1280×704 frames at 24 fps |
| `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | accepted historical evidence; current runtime needs a fresh 4×B200 official run |
| `wan2.2_distributed_rank_topology_validation` | accepted historical evidence; four unique ranks/devices and collective/barrier evidence |
| `wan2.2_verified_rerun_recording` | accepted current single-GPU evidence; exact MP4 identity and uploaded RRD were independently re-verified |
| `wan2.2_ti2v_5b_image_to_video` | deferred |
| A14B / S2V / Animate | deferred |
| official TI2V fine-tuning | deferred; no pinned-source entrypoint |
| stock Wan action prediction | rejected as an upstream capability |

## Licensing

Track official source, baked dependencies, runtime-fetched CUDA software,
run-time model/tokenizer, and data separately. Source/model declarations do not
classify a built image. The promoted first-class `npa-wan2-2` contract is public
eligible only when a pushed digest proves all `nvidia-*`, CUDA/cuDNN/NCCL,
checkpoint, credential, and cache bytes absent from every layer and history via
`npa/scripts/scan_image_wan_payload.py`. CUDA/PyTorch installation and use remain
governed by the upstream package terms; NPA adds no per-run consent variable.
Model/tokenizer acquisition remains runtime-only. Do not treat access as permission
beyond the applicable licenses and never publish merely because the Dockerfile looks
clean. `HF_TOKEN` is optional for public assets and remains a submission secret when
supplied.

## Validation

Use the repository venv, never bare Python:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml --run-id wan22-plan
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2-multigpu.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2-multigpu.yaml \
  --run-id wan22-multigpu-plan
npa/.venv/bin/python -m pytest npa/tests/workflows/test_wan_rerun.py -q
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q
```

The gated live tests are `npa/tests/e2e/test_byof_wan22_live_e2e.py` and
`npa/tests/e2e/test_byof_wan22_multigpu_live_e2e.py`. Future compatibility
changes require fresh live evidence rather than inference from an older run.
