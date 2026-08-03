---
name: real-components
description: Use when authoring or reviewing an NPA workbench pipeline/blueprint that advertises specific components (Cosmos Transfer, Cosmos Evaluator, Cosmos Curator, FiftyOne, VLM eval, etc.) — ensure every advertised stage invokes the REAL component, not an echo/manifest stub masquerading as real work.
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
  `write_decision` → **stub / demo** (e.g.
  `workbench.sim2real.write_decision`).
- invokes a real CLI (`npa workbench <tool> ...` with real flags) or a real
  module function → **real**.

Known stub toolRefs (do NOT advertise as real output):

| Stub | Real replacement |
| --- | --- |
| `workbench.fiftyone.launch_app` (echo) | real `npa workbench fiftyone load-dataset`, or a real `run.shell` curation function |
| `workbench.sim2real.finalize` / `write_decision` (echo/demo) | real `run.shell` module fns (e.g. `npa.workflows.data_factory_stages.finalize` / `grade_gate`) |

`run.shell` stages count as real when they invoke a real `npa workbench ...`
command or import a real, tested module (e.g.
`npa.workflows.data_factory_stages`, `npa.workflows.data_factory_viz`). Put the
logic in a tested module, not inline.

`workbench.cosmos2.transfer_execute` is the real Config-Gen-aware execution
toolRef used by the Data Factory and the standalone procedural-input smoke.
`workbench.cosmos2.transfer_conditioned_execute` is the real input-conditioned
execution toolRef for workflows that do not carry a Config-Gen manifest. Both
include `--execute` and `--condition-on-input`, so a missing runtime or input
video fails closed. The direct `workbench.cosmos2.transfer` / `npa workbench
cosmos2 transfer` surface retains reference/local augmentation behavior and must
not be used by an advertised real Cosmos stage.

## Name the real project, or run it

Advertising a *named third-party component* is a stronger claim than advertising
"real work": it says that project's code produced the artifact. Two ways to
honour it, both acceptable, and the difference must be visible in the artifact:

1. **Run upstream's code.** Preferred. Import upstream from a checkout baked into
   the tool's image and call its own classes (e.g. the `cosmos-curate` stage
   drives upstream's `CuratorStage` objects directly), or invoke upstream's
   documented CLI in upstream's container.
2. **Implement upstream's published algorithm or protocol**, cite it, and tag the
   artifact with which engine produced the number (e.g. the evaluator's
   hallucination check records `engine: cosmos-evaluator-upstream` vs
   `cosmos-evaluator-npa-port`, and the two agree to ~1e-3 on the same input).

What is never acceptable: a stage named after a project that neither runs it nor
implements it. When upstream cannot run in an environment, record
`engine: unavailable` **with the reason** (see
`npa workbench cosmos-curate engine`) rather than emitting a plausible-looking
report. Attribution belongs in a NOTICE file — see
`skills/NOTICE-NVIDIA-COSMOS-OSS` for the level of specificity expected: which
upstream modules run, which are reimplemented, and where NPA substitutes its own
endpoint.

## Reference implementation

`physical-ai-data-factory.yaml` uses only real components: Token Factory VLM
caption, `cosmos2.transfer_execute` (real Cosmos Transfer 2.5 on GPU),
`cosmos_evaluator.evaluate` (real NVIDIA Cosmos Evaluator hallucination +
attribute-verification checks), `cosmos_curate.curate` (real NVIDIA Cosmos Curator
stages), and real `run.shell` module functions for config-gen, grade gate,
FiftyOne review, and finalize. `build_run_rrd` writes a real Rerun `.rrd`.

## Enforced by

`npa/tests/orchestration/npa_workflow/test_real_components.py` fails if the
blueprint uses a known-stub toolRef, if a `run.shell` stage isn't a real
command/module call, if the augment stage isn't `cosmos2.transfer_execute`, if the
grade loop stops grading with Cosmos Evaluator, or if curation stops running
Cosmos Curator before FiftyOne review. Live-infra verification is a priority
(`skills/atomic/testing-conventions`).
