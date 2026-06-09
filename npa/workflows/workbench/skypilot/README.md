# SkyPilot Workflow YAMLs

Runnable SkyPilot YAMLs for Workbench reference pipelines. Each YAML is a
pipeline definition; the narrative walkthrough, prerequisites, and verification
steps live in the published docs under `docs/`, and the thin submission wrappers
live in `npa/scripts/`.

## Why the guides live in `docs/` and not here

The guide-to-YAML relationship is many-to-many, so the docs are not colocated
1:1 with each YAML:

- One guide often drives several YAMLs (the SONIC locomotion cookbook uses
  `sonic-locomotion-finetuning.yaml`, `retargeting.yaml`, and `mjlab-eval.yaml`).
- One YAML is often referenced by several guides (`bdd100k-pipeline.yaml` is used
  by its cookbook, the demo writeup, and `docs/workbench-yaml-guide.md`).
- Guides are customer-facing product documentation that cross-link to the
  quickstart, getting-started, architecture, and CLI references. They belong in
  the navigable `docs/` tree rooted at `docs/README.md`.

To keep the YAML and its guide easy to traverse in both directions, each YAML
carries a header comment pointing to its guide and runner, and this index maps
them out. Update both ends when you add or rename a workflow.

## Reference pipelines

| Workflow YAML | Guide / cookbook | Submission wrapper |
| --- | --- | --- |
| `bdd100k-pipeline.yaml` | [cookbooks/bdd100k-pipeline.md](../../../../docs/workbench/cookbooks/bdd100k-pipeline.md), [demos/bdd100k-lancedb-demo.md](../../../../docs/demos/bdd100k-lancedb-demo.md) | `npa/scripts/run_bdd100k_pipeline.py` |
| `sim-to-real-pipeline.yaml` | [cookbooks/sim-to-real-pipeline.md](../../../../docs/workbench/cookbooks/sim-to-real-pipeline.md), [sim-to-real-quickstart.md](../../../../docs/workbench/sim-to-real-quickstart.md) | `npa/scripts/run_sim_to_real_pipeline.py`, `npa/scripts/run_sim_to_real_quickstart.py` |
| `sim-to-real-loop.yaml` | [cookbooks/vlm-eval-loop-runbook.md](../../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) | `npa/scripts/run_sim_to_real_pipeline.py` |
| `isaac-lab-rl-train.yaml` | [cookbooks/byof-isaac-lab/README.md](../../../../docs/workbench/cookbooks/byof-isaac-lab/README.md), [workbench-yaml-guide.md](../../../../docs/workbench-yaml-guide.md) | `npa/scripts/run_isaac_lab_rl.py` |
| `isaac-lab-rl-sweep.yaml` | [workbench-yaml-guide.md](../../../../docs/workbench-yaml-guide.md) | `npa/scripts/run_isaac_lab_rl.py` |
| `sonic-train-standalone.yaml` | [cookbooks/sonic-train-runbook.md](../../../../docs/workbench/cookbooks/sonic-train-runbook.md), [sonic-image-catalog.md](../../../../docs/workbench/sonic-image-catalog.md) | `npa workflow` / `npa workbench sonic` |
| `sonic-locomotion-finetuning.yaml` | [cookbooks/sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md), [cookbooks/sonic-mvp-g1-mujoco.md](../../../../docs/workbench/cookbooks/sonic-mvp-g1-mujoco.md) | `npa workflow` / `npa workbench sonic` |
| `sonic-export.yaml`, `sonic-export-eval.yaml` | [cookbooks/sonic-eval-runbook.md](../../../../docs/workbench/cookbooks/sonic-eval-runbook.md), [cookbooks/sonic-whole-body-control.md](../../../../docs/workbench/cookbooks/sonic-whole-body-control.md) | `npa workbench sonic` |
| `mjlab-eval.yaml` | [cookbooks/sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) | `npa workbench mjlab` |
| `retargeting.yaml` | [cookbooks/sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) | `npa workbench retargeting` |
| `vlm-eval.yaml`, `vlm-eval-benchmark.yaml` | [cookbooks/vlm-eval-loop-runbook.md](../../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) | `npa workbench vlm-eval` |
| `cosmos2-transfer.yaml`, `cosmos3-*.yaml` | `.agents/skills/inference/SKILL.md`, `.agents/skills/workbench/cosmos/SKILL.md` | `npa workbench cosmos` |

The `sim2real-actions.yaml` and `sim2real-envgen-split.yaml` step YAMLs are
components of the self-contained Sim2Real runbook at
[`../sim2real/README.md`](../sim2real/README.md), which is the one workflow
that keeps its guide and YAML colocated because it is a single CLI/SDK-driven
chain rather than a docs-site cookbook.

## Conventions

- Shared parameter, artifact, naming, GPU, and safety rules:
  [`../schemas/workflow-conventions.md`](../schemas/workflow-conventions.md).
- General YAML structure (label maps, env vars, endpoints, S3 paths):
  [`../../../../docs/workbench-yaml-guide.md`](../../../../docs/workbench-yaml-guide.md).
- Submission, SkyPilot runtime, and cleanup pattern: [`../README.md`](../README.md).
