---
name: molmoact
description: Use when working on MolmoAct VLA fine-tuning/serving/evaluation workbench stages, MolmoAct policy configs, or the npa workbench molmoact CLI.
---

# MolmoAct

MolmoAct is the vision-language-action tool for fine-tuning, serving, and
evaluating MolmoAct VLA policies in npa.

The three core stages (`finetune`, `serve`, `eval`) are currently stubs:
they validate their arguments and return a plan-only manifest — actual
trainer/server/evaluator execution is not wired up yet (tracking issue
nebius/nebius-physical-ai#502). The base model (default
`allenai/MolmoAct-7B-O-0812`) is resolved at runtime through the HF Hub
cache via `npa.workbench.model_access` — weights are never bundled in the
repo or image.

## Interfaces

- CLI: `npa workbench molmoact <finetune|serve|eval> --help`
- Python SDK: `npa.sdk.workbench.molmoact`
  (`finetune`, `serve`, `eval`)
- Workflow module: `npa.workflows.byof.molmoact_pipeline`
  (argument validation, stub-manifest plumbing, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.molmoact` (multi-command package form
  for new tools); the SDK surface lives at `npa.sdk.workbench.molmoact`.
- The `finetune` stage is the headline `molmoact/finetune` three-tier
  contract (CLI <-> SDK <-> `workflows/testing/molmoact-finetune.yaml`);
  `serve` and `eval` remain public-reusable sibling toolRefs.
