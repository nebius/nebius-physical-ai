---
name: gemini-robotics
description: Use when working on Gemini Robotics 2 toolRefs — ER embodied-reasoning planning or rubric evaluation via the hosted Gemini API.
---

# Gemini Robotics

Gemini Robotics is the closed-weight VLA tool for ER (embodied reasoning)
planning and rubric evaluation, all served through the
hosted Gemini API (``GOOGLE_API_KEY`` required).

The toolRefs are a provisional override-required API adapter: every stage
requires an explicit ``--model`` (no default) and the model id is pinned to
the provisional ``gemini-robotics-er-2-preview`` constant until the production
model id is confirmed.

## Interfaces

- CLI: `npa workbench gemini-robotics <plan|eval> --help`
- Python SDK (workbench-first): `npa.workbench.gemini_robotics`
  (`plan`, `eval` — aliases of `run_er_planning_stage`,
  `run_eval_stage`)
- Workflow module: `npa.workflows.byof.gemini_robotics_pipeline`
  (stage functions, receipt writing, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.gemini_robotics` (multi-command package
  form for new tools); the SDK surface lives at `npa.workbench.gemini_robotics`
  and does not go through `make_cli_wrapper`.
- Stage receipts are written to the requested output path before any API call
  returns, so partial progress is always inspectable.
