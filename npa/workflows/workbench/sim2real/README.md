# Sim2Real workflow

Use the single canonical spec:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sim2real.yaml \
  --runtime --run-id <run-id> \
  --var bucket=<bucket> \
  --var controller_image=<immutable-ref> \
  --var transfer_image=<immutable-ref> \
  --var envgen_image=<immutable-ref> \
  --var reason_image=<immutable-ref> \
  --var isaac_image=<immutable-ref> \
  --var viewer_image=<immutable-ref> \
  --var isaac_cache_pvc=<pvc>
```

The YAML exposes all 14 stages and runs through the standard workflow runtime.
Each real solution has its own image/resource state, S3 inputs and outputs, and
ComponentRecord. Parallel Stage 4/8 leaves publish attributable lane records,
and their barrier consumers own the canonical aggregate record. Stage 11 early
exit is explicit (`allow_early_exit`), Stage 13/14 use the completed loop
iteration, shard cardinality is validated before submission, and visualization
downloads only its declared artifact set into cleaned ephemeral storage.
Runtime values are operator inputs; this directory contains no
tenant, project, registry, bucket, cluster, credential, or run identifier.

`controller_image` must be the small CPU-only image built from
`docker/workbench/sim2real-control/Dockerfile`. It contains the exact source and
pinned S3 dependencies but no Genesis, Isaac, CUDA, trainer, or injected source
bootstrap. GPU solution images remain attached only to their corresponding
workflow states.

The legacy `npa.workflows.sim2real` controller modules have a finite compatibility
window for archived callers and artifacts. They are lazy, are not called by the
canonical workflow, and cannot materialize or submit its retired controller.

See `docs/workbench/guides/sim2real-workflow.md` for preflight, reduced proof,
resume, monitoring, and evidence-audit commands.
