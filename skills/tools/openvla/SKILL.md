---
name: openvla
description: Use when working on OpenVLA fine-tuning/serving/evaluation workbench stages, the OpenVLA-OFT LoRA recipe, or the npa workbench openvla CLI.
---

# OpenVLA

OpenVLA is the vision-language-action tool for fine-tuning, serving, and
evaluating OpenVLA policies (OpenVLA-OFT recipe) in npa.

The three core stages (`train`, `serve`, `eval`) are currently stubs: they
expose the intended argument signatures but exit with a clear "not yet
implemented" message until the OpenVLA-OFT fine-tuning pipeline lands
(tracking issue nebius/nebius-physical-ai#500). The base model (default
`openvla/openvla-7b`) is resolved at runtime through the HF Hub cache via
`npa.workbench.model_access` — weights are never bundled in the repo or
image.

## Interfaces

- CLI: `npa workbench openvla <train|serve|eval> --help`
- Python SDK: `npa.sdk.workbench.openvla`
  (`train`, `serve`, `eval`)
- Workflow module: `npa.workflows.byof.openvla_pipeline`
  (argument validation, upstream argv planning, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.openvla` (multi-command package form for
  new tools); the SDK surface lives at `npa.sdk.workbench.openvla`.
- The `train` stage is the headline `openvla/train` three-tier contract
  (CLI <-> SDK <-> `workflows/testing/openvla-train.yaml`); `serve` and
  `eval` remain public-reusable sibling toolRefs.
