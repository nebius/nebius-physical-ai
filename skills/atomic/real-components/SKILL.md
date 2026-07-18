---
name: real-components
description: Use when authoring or reviewing an NPA workbench pipeline/blueprint that advertises specific components (Cosmos Transfer, FiftyOne, VLM eval, etc.) — ensure every advertised stage invokes the REAL component, not an echo/manifest stub masquerading as real work.
---

# Real Components As Advertised

A pipeline that advertises a component (e.g. "Cosmos Transfer 2.5", "FiftyOne
curation") MUST actually run it. A stub that only `echo`s or writes a
`contract_ready` manifest while the spec advertises real output is a correctness
bug: it looks green in validate/plan/smoke but produces no real artifacts, and
downstream stages that consume them fail (or silently pass on fake data).

## Rule

- Every advertised stage invokes the real component and writes real artifacts.
- A stage may be a stub ONLY if the spec description says so explicitly; never
  advertise a stub as the real capability.
- Verify on a live run that each stage's output artifact is real (a real
  video/frames/report), not a manifest or an echoed string.

## Audit method

For each `toolRef`, inspect its argv in
`npa/src/npa/orchestration/npa_workflow/catalog.py`:

- `argv[0] == "echo"` → **stub** (e.g. `workbench.fiftyone.launch_app`,
  `workbench.sim2real.finalize`).
- a Python one-liner that writes `"status": "contract_ready"` or a fixed
  `write_decision` → **stub / demo** (e.g. `workbench.cosmos2.transfer`
  manifest mode, `workbench.sim2real.write_decision`).
- invokes a real CLI (`npa workbench <tool> ...` with real flags) or a real
  module function → **real**.

Known stub toolRefs (do NOT advertise as real output):

| Stub | Real replacement |
| --- | --- |
| `workbench.cosmos2.transfer` (manifest only) | `workbench.cosmos2.transfer_execute` (`--execute`; real GPU model + uploads video/frames to S3) |
| `workbench.fiftyone.launch_app` (echo) | real `npa workbench fiftyone load-dataset`, or a real `run.shell` curation function |
| `workbench.sim2real.finalize` / `write_decision` (echo/demo) | real `run.shell` module fns (e.g. `npa.workflows.data_factory_stages.finalize` / `grade_gate`) |

`run.shell` stages count as real when they invoke a real `npa workbench ...`
command or import a real, tested module (e.g.
`npa.workflows.data_factory_stages`, `npa.workflows.data_factory_viz`). Put the
logic in a tested module, not inline.

## Reference implementation

`physical-ai-data-factory.yaml` uses only real components: Token Factory VLM
caption, `cosmos2.transfer_execute` (real Cosmos Transfer 2.5 on GPU), `vlm_eval`
attribute verification, and real `run.shell` module functions for config-gen,
grade gate, curation, and finalize. `build_run_rrd` writes a real Rerun `.rrd`.

## Enforced by

`npa/tests/orchestration/npa_workflow/test_real_components.py` fails if the
blueprint uses a known-stub toolRef, if a `run.shell` stage isn't a real
command/module call, or if the augment stage isn't `cosmos2.transfer_execute`.
Live-infra verification is a priority (`skills/atomic/testing-conventions`).
