# Main workflows

[Workflow catalog](../README.md) · [Testing and reference workflows](../testing/README.md)

Start with a guide, then use the YAML in the same row. Follow its setup,
configuration, launch and output-verification steps from the repository root.

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [`paidf-cosmos3.yaml`](paidf-cosmos3.yaml) | [Setup and run](../guides/paidf-cosmos3.md) · [Twelve variants and preserved padding](../../docs/workbench/guides/paidf-appearance-12.md#apply-the-recipe) · [Recipe YAML](../../docs/workbench/examples/paidf-appearance-12.yaml) | Video or LeRobot episode → Cosmos3 appearance variants → evaluation, relabeling, curation and Rerun. The twelve-profile recipe is opt-in; padding detection and preservation are automatic. |
| [`nurec-reconstruct.yaml`](nurec-reconstruct.yaml) | [NuRec guide and measured evidence](../../docs/workbench/guides/neural-reconstruction.md#promotion-evidence) | NCore V4 capture → 3DGUT training on an RT-core GPU → USDZ → rig-offset novel views → Rerun. See the [readiness record](nurec-reconstruct.readiness.json). |
| [`sim2real.yaml`](sim2real.yaml) | [Sim2Real guide](../../docs/workbench/guides/sim2real-workflow.md) | Canonical fourteen-stage robot-learning pipeline through the standard SkyPilot runtime. |

Detailed guides remain in their existing locations; this table is the index
beside the executable specs. The PAIDF recipe is a configuration overlay, not a
standalone workflow: its guide shows how to apply it to `paidf-cosmos3.yaml`.

For shared CLI steps, see [validate, plan and submit](../README.md#commands).
For component checks and other examples, use the [testing index](../testing/README.md).
