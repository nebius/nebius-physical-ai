# NPA Workbench workflow catalog

This directory holds the **supported, customer-facing** Workbench workflow
catalog for robotics, simulation, perception, eval, and synthetic-data
workloads. The supported specs are declarative `npa.workflow` YAMLs; SkyPilot
remains the underlying execution engine, but the raw SkyPilot task templates are
no longer part of the shown catalog (see [Layout](#layout)).

## ➡️ Start here: the workflow catalog

**[`npa-workflows/README.md`](npa-workflows/README.md)** is the catalog of every
supported workflow spec (`apiVersion: npa.workflow/v0.0.1`). Author and submit
these with:

```bash
npa workbench workflow validate-spec <spec.yaml>
npa workbench workflow plan-spec <spec.yaml> --run-id demo
npa workbench workflow submit <spec.yaml> --run-id demo
```

Authoring skills: `skills/workflows/author-npa-workflow/SKILL.md` (edit) and
`skills/workflows/generate-npa-workflow/SKILL.md` (design new pipelines).

## Layout

- `npa-workflows/`: the supported declarative `npa.workflow` specs plus the
  [workflow catalog](npa-workflows/README.md). This is the only workflow YAML
  set we show and support.
- `sim2real/`: the self-contained Sim2Real runbook (guide + YAML colocated); the
  14-stage engine detects the runbook and routes to direct K8s.
- `schemas/`: conventions for parameters, artifacts, naming, and runtime
  constraints.
- `steps/` and `templates/`: legacy placeholders kept for compatibility with
  older examples.

### SkyPilot task templates (internal)

The raw SkyPilot task YAMLs that the `npa.workflow` engine and the
`npa/scripts/run_*.py` wrappers render and launch live under
`npa/src/npa/workflows/skypilot/` as internal, package-owned runtime resources.
They are not the shown catalog and should not be authored by customers; the
supported entry point is always an `npa.workflow` spec above. Their per-file
reference notes (S3 I/O, GPU targets, HF/NGC rights, raw `sky launch` caveats)
are documented in
[`npa/src/npa/workflows/skypilot/README.md`](../../src/npa/workflows/skypilot/README.md).

## Sim-To-Real

The H100 quickstart submits:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

It renders and runs the internal template
`npa/src/npa/workflows/skypilot/sim-to-real-pipeline.yaml`. The deeper reference
path is documented in `docs/workbench/cookbooks/sim-to-real-pipeline.md`.

## Submission Pattern

Use the thin Python wrappers under `npa/scripts/` when a workflow needs runtime
substitution, S3 paths, secret-env injection, GPU validation, or cleanup:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py --help
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py --help
npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py --help
```

Invoke SkyPilot through `NPA_SKYPILOT_BIN`, normally resolved by:

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
```

## Cleanup

Wrappers that create live GPU resources must use explicit SkyPilot cleanup and
must poll for absence when they own the user-facing lifecycle. Do not rely on a
detached terminal or manual cleanup as the only teardown path.
