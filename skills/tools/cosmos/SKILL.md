---
name: cosmos
description: Use when working on Cosmos world model serving, inference, serverless training smoke validation, backend selection, or rendering limitations.
---

# Cosmos

Cosmos is the world model tool for synthetic data generation and video generation.

It requires a GPU. RT cores are not required for standard serving, inference,
or the serverless training smoke path, unlike Isaac Lab. Cosmos
visual-generation/rendering paths have the same container EGL/DRI gap as
Genesis.

## Interfaces

Cosmos3-specific guidance lives as agent skills, not CLI commands:

- `skills/atomic/cosmos3-setup/SKILL.md`
- `skills/atomic/cosmos3-codebase-nav/SKILL.md`
- `skills/atomic/cosmos3-env-troubleshoot/SKILL.md`
- `skills/workflows/cosmos3-inference/SKILL.md`
- `skills/workflows/cosmos3-post-training/SKILL.md`

API:

- `POST /serve`
- `POST /infer`
- `POST /train` for serverless Jobs smoke validation
- `GET /status`
- `GET /system-info`
- `GET /list`

CLI:

```bash
npa workbench cosmos deploy
npa workbench cosmos serve
npa workbench cosmos infer
npa workbench cosmos train --runtime serverless --smoke
npa workbench cosmos finetune
npa workbench cosmos optimize
npa workbench cosmos status
npa workbench cosmos system-info
npa workbench cosmos list
```

## Backend Selection

Use `--backend` to select one of:

- `basic`
- `nim`
- `triton`

Only `basic` is implemented today. `nim` and `triton` are exposed as enum
choices but intentionally exit as not implemented. For multiple models, use
named workbenches or the deploy/serve model swap pattern.

## E2E Status

Cosmos is validated end-to-end on Nebius through the public CLI serverless
training smoke path:

```bash
npa workbench cosmos train --runtime serverless --smoke
```

W13 run `w13-cosmos-e2e-20260521T233523Z` completed on `gpu-h100-sxm` and
uploaded `checkpoint.json` to S3. This closes the named Workbench tool matrix
gap for an artifact-bearing Cosmos workflow.

Known constraints:

- `finetune` and `optimize` are placeholders.
- Basic serverless endpoint inference validates endpoint/job completion, but
  generated endpoint outputs do not yet have a public CLI serverless-side S3
  export contract.
- EGL/DRI-dependent visual-generation/rendering paths remain deferred.

## Predict2 CUDA Wheel Contract

The `npa-cosmos` Predict2 1.0.9 image uses NVIDIA's complete v1.2.0
`cu128_torch27` wheel set: torch 2.7.0, torchvision 0.22.0, flash-attn 2.7.3,
NATTEN 0.21.0, and Transformer Engine 1.13.0. Keep these as one ABI-locked
unit. Do not bump torch alone, and do not replace either custom-kernel wheel
with a source build during an image refresh.

Predict2 1.0.9's package metadata still pins triton 3.2.0 for its former torch
2.6 stack, while torch 2.7 requires triton 3.3.0. Install Predict2 itself with
`--no-deps`, exclude torch/torchvision/triton and the three NVIDIA kernel
packages from its derived dependency closure, and constrain every subsequent
resolver pass to torch 2.7.0, torchvision 0.22.0, and triton 3.3.0. Otherwise a
later broad dependency can silently replace the selected cu128 stack.

An architecture import check is insufficient. A release validation must read
`torch._C._cuda_getArchFlags()` and find `sm_100`, then execute both custom
kernels on B200: a real flash-attn forward and the exact pinned
Predict2 `NeighborhoodAttention` module with one of the model's shipped NATTEN
configurations. Run checkpoint-backed Video2World with `--natten` whenever the
operator has access to NVIDIA's gated checkpoint. If access is denied, record
that generation as unverified with the HTTP evidence; the model-module kernel
smoke is valid kernel-compatibility evidence, but it is not a generated-video
result.

Predict2 1.0.9 rejects B300 capability 10.3 in its own `[90, 100]` allowlist;
forward-compatible `sm_100` wheel SASS does not bypass that check. Route this
pin to B200 or H100 and require a real-forward negative test when rechecking
B300.

## Cosmos Transfer B300 Contract

The published Cosmos Transfer 2.5 cu128 image is validated for B200, not B300.
On physical B300 it reaches real `Control2WorldInference` model construction,
then `torch.nn.init.trunc_normal_` JIT-compiles an `erfinv` kernel and CUDA 12.8
NVRTC rejects capability 10.3 with `invalid value for --gpu-architecture`.
This demonstrates that wheel SASS coverage alone cannot establish compatibility
for workloads that generate kernels at runtime. A B300 port must move the whole
locked environment to CUDA 13/cu130 and pass the full depth-conditioned
Video2Video smoke; a CUDA probe or import is not sufficient.

## Sim2Real VLM (self-hosted Reason2 + Reason3)

Sim2Real stage 8 evaluates rollouts with **two** workbench-hosted Cosmos Reason
models in parallel sibling GPU jobs — not Token Factory:

- `nvidia/Cosmos-Reason2-8B` (`vlm_eval_reason2`)
- `nvidia/Cosmos-Reason2-2B` (`vlm_eval_reason3`, self-hosted default second checkpoint)

`nvidia/Cosmos3-Super-Reasoner` is a **Token Factory hosted** model id only
(no Hugging Face repo). Use `npa workbench token-factory reason` for that path;
do not set it as `VLM_REASON3_MODEL` on self-hosted sim2real runs.

Implementation lives in `npa.workbench.cosmos.reason`. The `npa-cosmos3-reason`
image runs `component-vlm-eval`; dual eval merges judgments via
`merge_dual_reason_evaluations`. Pool sizing divides `k8s_max_parallel_gpus` by
two jobs per rollout (`NPA_SIM2REAL_VLM_DUAL_REASON=1`, default). With
`k8s_max_parallel_gpus=16` and `ROLLOUT_COUNT=8`, all 16 GPUs can run VLM eval.

**Hugging Face setup (required once per account):** accept each gated repo at
https://huggingface.co while signed in, then put `HF_TOKEN` in
`~/.npa/credentials.yaml` and mirror it into the cluster `hf-ngc-tokens` secret.
See [sim2real-workflow.md](../../../docs/workbench/guides/sim2real-workflow.md#hugging-face-model-access-self-hosted-workbench).

Env knobs: `VLM_REASON2_MODEL`, `VLM_REASON3_MODEL`, `VLM_REASON2_IMAGE`,
`VLM_REASON3_IMAGE`, `NPA_COSMOS_REASON2_CACHE`, `NPA_COSMOS_REASON3_CACHE`.

## Operational Safety

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

BYOVM deploys record `endpoint_strategy: public` or `endpoint_strategy:
ssh_fallback` in `~/.npa/config.yaml`. Live `status`, `serve`, and `infer`
commands honor that strategy and self-heal blocked public endpoints through a
transient SSH-local route.
