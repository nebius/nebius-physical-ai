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

- `.agents/skills/cosmos3-setup/SKILL.md`
- `.agents/skills/codebase-nav/SKILL.md`
- `.agents/skills/env-troubleshoot/SKILL.md`
- `.agents/skills/inference/SKILL.md`
- `.agents/skills/cosmos3-post-training/SKILL.md`

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

## Operational Safety

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

BYOVM deploys record `endpoint_strategy: public` or `endpoint_strategy:
ssh_fallback` in `~/.npa/config.yaml`. Live `status`, `serve`, and `infer`
commands honor that strategy and self-heal blocked public endpoints through a
transient SSH-local route.
