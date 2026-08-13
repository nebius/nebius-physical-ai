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
- `sim2real/`: operator notes and the finite legacy compatibility DAG. The only
  supported Sim2Real submission spec is `npa-workflows/sim2real.yaml`; it uses
  the ordinary standard runtime and never routes to direct Kubernetes.
- `schemas/`: conventions for parameters, artifacts, naming, and runtime
  constraints.
- `steps/` and `templates/`: legacy placeholders kept for compatibility with
  older examples.

### Raw SkyPilot YAML

The retired raw SkyPilot workflow catalog is gone. The `npa.workflow` engine
still renders specs to SkyPilot at submit time, and `npa workbench workflow
submit` still accepts customer-owned raw SkyPilot YAML, but shipped raw YAMLs
must live only in guarded, tool-specific homes such as burst examples, BYOF
resource profiles, or the NuRec single-pod example.

## Sim-To-Real

Submit the staged VLM-to-RL loop:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sim2real.yaml \
  --run-id <run-id> \
  --var NPA_SIM2REAL_BUCKET=<your-bucket> \
  --var NPA_SIM2REAL_TRIGGER_DATASET_URI=s3://<your-bucket>/<trigger-prefix>/
```

The legacy `sim_to_real` H100 quickstart and its template are retired: that path ran
`npa.workflows.sim_to_real real-loop`, which raises a `DeprecationWarning` pointing at the
compositional standard runtime. The single canonical YAML is
[`npa-workflows/sim2real.yaml`](npa-workflows/sim2real.yaml); the deeper reference is
[`docs/workbench/guides/sim2real-workflow.md`](../../../docs/workbench/guides/sim2real-workflow.md).

## Submission Pattern

Use the thin Python wrappers under `npa/scripts/` when a workflow needs runtime
substitution, S3 paths, secret-env injection, GPU validation, or cleanup:

```bash
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
