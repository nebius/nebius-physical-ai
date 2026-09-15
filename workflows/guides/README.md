# Workflow runbooks

[Workflow catalog](../README.md) · [Workbench guides](../../docs/workbench/guides/README.md)

Use a runbook to prepare inputs, provision the right compute, submit a workflow,
and inspect its outputs. YAML files alone contain placeholder storage and may
require images or model access that you must prepare first.

| Result | Runbook | Input and output |
| --- | --- | --- |
| Source-conditioned video augmentation | [PAIDF + Cosmos 3](paidf-cosmos3.md) | Public starter, local MP4, or one LeRobot episode/camera → generated variants, evaluation, curation, and Rerun |
| A labeled dataset | [Physical AI Data Factory](../../docs/workbench/guides/physical-ai-data-factory-deploy.md) | Source video → Cosmos Transfer augmentation, labels, curation, and recording |
| 3D reconstruction | [NuRec](../../docs/workbench/guides/neural-reconstruction.md) | Sensor capture → scene USDZ, novel views, and Rerun |
| Robot-learning pipeline | [Sim2Real](../../docs/workbench/guides/sim2real-workflow.md) | Task-aligned trigger and prepared images → training/evaluation and review artifacts |

For PAIDF + Cosmos 3, jump directly to
[local MP4 input](paidf-cosmos3.md#r3b-run-the-full-pipeline-from-a-local-mp4),
[LeRobot input](paidf-cosmos3.md#r3a-augment-one-lerobot-episode-and-camera), or
[generation and acceptance settings](paidf-cosmos3.md#r5-find-and-change-generation-and-evaluation-settings)
after completing its shared setup.

To create a new workflow, use the
[authoring guide](../../docs/workbench/npa-workflow-guide.md) and
[tool catalog](../../docs/workbench/npa-workflow-tool-catalog.md). Validate and
plan locally before attempting cloud execution. After a run, preserve its ID
and inspect the declared outputs before [cleanup](../../docs/teardown.md).
