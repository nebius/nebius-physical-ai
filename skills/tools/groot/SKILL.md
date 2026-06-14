---
name: groot
description: Use when working on NVIDIA GR00T deployment, model download, finetuning, evaluation, serving, inference, conversion, status checks, validation, routing, or CUDA alignment.
---

# GR00T

## When To Use

Use this skill for NVIDIA GR00T robot foundation model workbench changes,
especially NGC/Hugging Face model handling, embodiment tags, checkpoint
conversion, serving, inference, and validation.

## Procedure

1. Start with the current command surface:

   ```bash
   npa workbench groot --help
   ```

2. Use `download` to stage model artifacts and credentials, `finetune` for
   training, `eval` for offline scoring, `serve` and `infer` for runtime calls,
   and `convert` when transforming checkpoints for downstream use.
3. Use `status`, `system-info`, and `list` for operational checks. Keep
   `ensure-ingress`, `register-byovm`, `reload-env`, and `cleanup-partial`
   scoped to setup and recovery flows.
4. Preserve credential redaction for NGC, Hugging Face, S3, and SSH values.

## Three-Tier Contract

- CLI: `list`, `deploy`, `download`, `finetune`, `eval`, `serve`, `infer`,
  `convert`, `status`, and `system-info` are the main user commands.
- SDK/API: keep model, checkpoint, and storage path normalization in shared
  helpers so service and CLI routes do not diverge.
- YAML: workflow YAML should pass model refs, checkpoint S3 URIs, output S3
  prefixes, and GPU selection as env vars or explicit task inputs.

## Routing And Validation

- GR00T does not require RT cores for the standard model paths.
- Route throughput-heavy training/eval to H100/H200 unless a command or image
  specifically requires another target.
- CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not a
  Nebius infrastructure blocker.

## Gotchas

- NGC credentials are required for gated NGC model refs. Do not print token
  values in diagnostics.
- Managed VM `deploy` defaults to in-place updates for existing aliases.
  Terraform plans that would destroy or replace critical infrastructure are
  blocked unless the operator passes `--replace` and confirms with `--yes`.
- BYOVM deploys record `endpoint_strategy: public` or
  `endpoint_strategy: ssh_fallback` in `~/.npa/config.yaml`; live commands honor
  that strategy and can self-heal blocked public endpoints through a transient
  SSH-local route.
- Known issue: output truncation at high step counts must be validated with
  artifacts, not subjective evaluation.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The skill smoke invokes current GR00T `download` and `convert` help, which
protects against stale deploy/status-only documentation.
