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
ComponentRecord. Runtime values are operator inputs; this directory contains no
tenant, project, registry, bucket, cluster, credential, or run identifier.

`controller_image` must be the small CPU-only image built from
`docker/workbench/sim2real-control/Dockerfile`. It contains the exact source and
pinned S3 dependencies but no Genesis, Isaac, CUDA, trainer, or injected source
bootstrap. GPU solution images remain attached only to their corresponding
workflow states.

The legacy `npa.workflows.sim2real` controller modules remain only to read and
replay archived runs. They are not called by the canonical workflow.

See `docs/workbench/guides/sim2real-workflow.md` for preflight, reduced proof,
resume, monitoring, and evidence-audit commands.
