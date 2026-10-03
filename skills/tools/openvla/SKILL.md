---
name: openvla
description: Use when working on OpenVLA fine-tuning/serving/evaluation workbench stages, the OpenVLA-OFT LoRA recipe, or the npa workbench openvla CLI.
---

# OpenVLA-OFT

OpenVLA-OFT is the vision-language-action runtime for native OFT LIBERO
fine-tuning and evaluation in NPA. It uses upstream's continuous action head
and 8-D proprioception projector; do not pass an OFT adapter to the stock
OpenVLA decoder or treat a decoder-only checkpoint as compatible.

The native path is `prepare → train → rollout → evaluate → visualize`:
RLDS/normalization inspection, upstream OFT fine-tuning, closed-loop LIBERO
rollout with MP4s, numerical success verification, and factual SVG/CSV
comparison output. Every stage hands off a hash-bound artifact bundle. Source,
base model, OFT adapters, LIBERO data, and populated caches are runtime-only.
Training materializes the exact Hub revision into that cache and calls the
upstream script through single-node eight-process `torchrun`.

The current pinned OFT package declares an unversioned `moojink/dlimp_openvla`
direct dependency whose repository has no declared license. The runtime fails
closed before fetching it; do not claim an OFT train/rollout acceptance until
the upstream owners publish an authoritative license or permission for a pinned
revision. This is not an NPA acceptance control.

## Interfaces

- CLI: `npa workbench openvla <prepare|train|rollout|evaluate|visualize> --help`
- Python SDK: `npa.sdk.workbench.openvla`
  (`prepare`, `train`, `rollout`, `evaluate`, `visualize`)
- Workflow module: `npa.workflows.byof.openvla_pipeline`
  (native execution, component validation, upstream argv, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.openvla` (multi-command package form for
  new tools); the SDK surface lives at `npa.sdk.workbench.openvla`.
- The `train` stage is the headline `openvla/train` three-tier contract
  (CLI <-> SDK <-> `workflows/testing/openvla-oft-libero.yaml`).
- Official suite checkpoint IDs, immutable revisions, and required LoRA/action-
  head/proprio-projector members are in `OFFICIAL_SUITE_CHECKPOINTS`. Preserve
  that exact mapping for sibling work; a locally trained bundle is not one of
  those official adapters.
- Read `docs/workbench/openvla-oft.md` before packaging or publishing. The OFT
  source is MIT, while the current OpenVLA README separately identifies the
  Llama Community License for its Llama-2-derived pretrained model. Runtime
  accessibility does not grant redistribution rights.
