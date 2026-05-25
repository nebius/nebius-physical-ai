---
name: groot
description: Use when working on NVIDIA GR00T deployment, status checks, validation, routing, or CUDA 13 alignment.
---

# GR00T

GR00T is NVIDIA's robot foundation model for imitation learning.

## Interfaces

API surfaces cover deploy, status, and system-info.

CLI:

```bash
npa workbench groot deploy
npa workbench groot status
npa workbench groot system-info
```

## Routing And Validation

- Routes to H100 or H200.
- Does not require RT cores.
- Validated on B300 as Tier 1.

Known issue: silent output truncation at `--steps > 32`. Promotion criteria must be evidence-based; do not rely on subjective evaluation.

CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not Nebius-blocked.

## Operational Safety

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

BYOVM deploys record `endpoint_strategy: public` or `endpoint_strategy:
ssh_fallback` in `~/.npa/config.yaml`. Live `status`, `serve`, and `infer`
commands honor that strategy and self-heal blocked public endpoints through a
transient SSH-local route.
