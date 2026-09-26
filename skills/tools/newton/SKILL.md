---
name: newton
description: Use when working on Newton physics engine workbench stages (teacher training, demo generation, evaluation), the Newton XPBD simulation pipeline, or the npa workbench newton CLI.
---

# Newton

Newton is the physics engine tool for teacher policy training, demonstration
generation, and policy evaluation in simulation.

The three core stages (`train-teacher`, `generate-demos`, `eval`) are
currently stubs: they expose the intended argument signatures but exit with
a clear "not yet implemented" message until the Newton simulation pipeline
lands (tracking issue nebius/nebius-physical-ai#499). Real physics validation
(a double-pendulum XPBD simulation) runs on CPU via the installed Newton
package — no GPU required for the validation path.

## Interfaces

- CLI: `npa workbench newton <train-teacher|generate-demos|eval> --help`
- Python SDK (workbench-first): `npa.workbench.newton`
  (`train_teacher`, `generate_demos`, `evaluate`)
- Workflow module: `npa.workflows.byof.newton_pipeline`
  (argument validation, stub-manifest plumbing, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.newton` (multi-command package form for
  new tools); the SDK surface lives at `npa.workbench.newton` and does not go
  through `make_cli_wrapper`.
- Stub stages write a `not_implemented` manifest to the requested
  `output_uri` before raising, so partial progress is always inspectable.
