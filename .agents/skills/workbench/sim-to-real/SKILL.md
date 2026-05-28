# Sim-To-Real Workflow Tools

Use this skill when designing, reviewing, or operating generic robotics
sim-to-real workflows in the workbench. The workflow shape is intended for
robotics teams that need to move data and artifacts through iterative simulation,
policy training, synthetic data generation, and evaluation loops without baking
project-specific names or infrastructure into the implementation.

## Scope

- Treat object storage URIs, project aliases, bucket names, registry paths,
  endpoint names, and thresholds as configuration.
- Keep customer data import, synthetic data generation, training, evaluation, and
  loop-control steps composable through CLI commands and workflow YAML.
- Prefer dry-run plans for data movement, autoscaling, and external service calls
  when validating workflow shape.
- Keep artifacts partitioned by pipeline run ID so repeated experiments do not
  overwrite each other.

## Workflow Shape

1. Import source robot, scene, or task data from a configured object-storage
   prefix into a run-scoped pipeline prefix.
2. Generate or augment simulation data using configured model-serving or batch
   inference resources.
3. Train or fine-tune the policy against the run-scoped dataset and write
   checkpoints back to configured storage.
4. Evaluate the policy with deterministic metrics or a configured VLM backend.
5. Decide whether to stop, continue iterating, or route artifacts for human
   review based on configurable thresholds.

## Implementation Guidance

- Do not hardcode customer names, event names, personal names, fixed tenant IDs,
  registry IDs, bucket names, or private endpoints.
- Keep CLI defaults generic, for example `sim-to-real`, `robot-data`, or
  `pipeline-run`, and let users override them with flags or environment values.
- Use existing workbench tools for S3 data sync, Cosmos autoscaling, model
  inference, training, and VLM evaluation instead of adding one-off scripts.
- Unit tests should mock storage, model-serving, and network clients at the call
  site and should use generic fixture paths.
