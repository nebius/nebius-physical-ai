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
   For declarative N1.7 training, use
   `npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml`; its
   `gpu_count` is a positive single-node world size that controls both the
   scheduler allocation and the real trainer.
3. Use `status`, `system-info`, and `list` for operational checks. Keep
   `ensure-ingress`, `register-byovm`, `reload-env`, and `cleanup-partial`
   scoped to setup and recovery flows.
4. Preserve credential redaction for NGC, Hugging Face, S3, and SSH values.

## Three-Tier Contract

- CLI: `list`, `deploy`, `download`, `finetune`, `eval`, `serve`, `infer`,
  `convert`, `status`, and `system-info` are the main user commands.
- SDK/API: keep model, checkpoint, and storage path normalization in shared
  helpers so service and CLI routes do not diverge.
- YAML: `groot-1-7-finetune.yaml` calls `workbench.groot.finetune` in the
  stage's own image. Keep model/source pins, checkpoint S3 URIs, run ID, and
  GPU count explicit; counts above one must use the upstream `torchrun` path.

## Routing And Validation

- GR00T does not require RT cores for the standard model paths.
- Route throughput-heavy training/eval to H100/H200 unless a command or image
  specifically requires another target.
- CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not a
  Nebius infrastructure blocker.

## Runtime Isaac bootstrap (the container ships no Isaac Sim)

The `npa-groot` image contains **no NVIDIA Isaac Sim or Isaac Lab code**. It used to bake
Omniverse Kit, which made it non-redistributable; Isaac is now downloaded on first use of
`/isaac-sim/python.sh` from `https://pypi.nvidia.com`, into a cache volume, under the
**operator's own EULA acceptance**. Full rationale:
`docs/workbench/container-packaging.md` and `skills/atomic/solution-licensing/SKILL.md`.

What this changes in practice:

- **Set both variables, or Isaac will not start.** Missing either
  `OMNI_KIT_ACCEPT_EULA=YES` or `ISAACSIM_ACCEPT_EULA=YES` makes the container exit **78**
  with a message naming them. That refusal is deliberate and load-bearing — do not "fix"
  it by baking acceptance into the image; a guard fails the build if anyone does.
- **Reach Isaac through `/isaac-sim/python.sh`** (the value of `ISAAC_LAB_PYTHON`). That is
  the bootstrap shim, and it is what every SkyPilot template, the sim2real engine and the
  workbench CLI already use. A bare `python3` is the *system* interpreter and will not
  find Isaac.
- **Never invoke the shim from a Dockerfile `RUN`.** It would download and bake ~4.5 GB of
  Isaac into a layer. Build-time work uses the image's own venv python.
- **Budget the first start.** Measured on RTX PRO 6000: 111 s cold, 32 ms warm, 10.04 GiB
  of cache. Pre-warm a shared volume once per node/PVC with
  `npa/docker/workbench/common/warm-isaac-cache.yaml`, then run workload pods with
  `NPA_ISAAC_CACHE_READONLY=1`. Otherwise every pod pays it, and 8 GPU pods on a node
  download ~36 GB.
- `isaac-bootstrap status` reports what is cached without needing acceptance or network;
  `isaac-bootstrap verify` additionally launches Isaac Sim headless (needs a GPU).
- No NGC credentials are needed to build or run this image.

GR00T inference and fine-tuning run in `GROOT_VENV`
(`python -m npa.smoke.test_groot_functional` covers inference) and need no Isaac
or EULA acceptance, so they pay no first-run download. Only the Isaac Lab
simulation paths do.

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
- Multi-GPU fine-tuning defaults to NCCL's native transport selection. When a
  clean two-rank collective proves that both P2P and SHM are unsafe on a
  single-node host, use the workflow's `nccl_transport=socket` compatibility
  fallback. It disables both transports, so do not use it speculatively on
  healthy high-bandwidth hosts.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/test_groot_finetune_workflow.py -q
```

The skill smoke invokes current GR00T training help and parses the reference
N1.7 workflow in addition to the download/convert command checks.
