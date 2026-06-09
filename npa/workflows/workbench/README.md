# NPA SkyPilot Workflow Templates

This directory contains Workbench reference workflows for robotics,
simulation, perception, eval, and synthetic-data workloads. SkyPilot is the
workflow orchestrator for these templates.

## Layout

- `skypilot/`: runnable SkyPilot YAMLs for reference pipelines. See
  `skypilot/README.md` for the index that maps each YAML to its guide in `docs/`
  and its submission wrapper in `npa/scripts/`.
- `schemas/`: conventions for parameters, artifacts, naming, and runtime
  constraints.
- `steps/` and `templates/`: legacy placeholders kept for compatibility with
  older examples; new workflow work should use `skypilot/`.

## Sim-To-Real

The H100 quickstart submits:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

It renders and runs:

```text
npa/workflows/workbench/skypilot/sim-to-real-pipeline.yaml
```

The deeper reference path is documented in
`docs/workbench/cookbooks/sim-to-real-pipeline.md`.

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
