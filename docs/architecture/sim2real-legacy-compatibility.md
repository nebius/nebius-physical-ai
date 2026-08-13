# Sim2Real legacy compatibility boundary

The only canonical operator surface is
`npa/workflows/workbench/npa-workflows/sim2real.yaml`, executed by the ordinary
`npa workbench workflow ... --runtime` planner, renderer, and SkyPilot runtime.

`npa.workflows.sim2real.engine`, its bounded `legacy_*` modules,
`stage_execution.py`, and the older scheduler/`sim2real.dag.yaml` pair remain
lazy compatibility code for pre-standard-runtime Python callers and archived
artifact replay. They must not be imported by the canonical spec,
`workflow_stage.py`, or standard runtime. No direct-controller YAML,
materializer implementation, or `k8s_submit` surface remains.

This compatibility window targets removal in NPA 0.5.0, no earlier than
2027-02-01. During the window, changes are limited to security/correctness fixes
and tests needed to keep archived inputs readable; new stages, orchestration,
and product behavior belong only in the compositional workflow.
