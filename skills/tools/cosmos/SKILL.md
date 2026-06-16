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
