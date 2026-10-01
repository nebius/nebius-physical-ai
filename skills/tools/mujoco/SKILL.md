---
name: mujoco
description: Use when running contact-rich manipulation simulation (peg insertion, screw driving, reach), rolling out scripted policies in MuJoCo, or using the npa workbench mujoco CLI.
---

# mujoco

`mujoco` runs scripted policies on contact-rich manipulation tasks in real
MuJoCo simulation: `peg_insertion` (position-controlled gripper inserts a peg
into a hole; success is tip depth plus radial tolerance), `screw_driving`
(the screw hangs from a compliant two-hinge holder so contact torques tilt it
for real; success is drive depth with tilt under threshold), and `reach` (a
contact-free smoke task).

Scenes are MJCF built programmatically as XML strings — no downloads, no
asset servers — and the simulation runs headless by default.

## Interfaces

- CLI: `npa workbench mujoco run --help`
- Python SDK (workbench-first): `npa.workbench.mujoco_manip`
  (`run`, `make_env`, `get_policy`, `TASKS`, `build_scene`)
- Workflow module: `npa.workflows.byof.mujoco_pipeline`
  (real stage implementation, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.mujoco` (multi-command package form for
  new tools); the SDK surface lives at `npa.workbench.mujoco_manip` and does
  not go through `make_cli_wrapper`.
- The `mujoco` package is required at run time but imports stay lazy, so the
  CLI and SDK import cleanly without it.
- Environments expose `reset(seed)`, `step(action)`, `action_space` /
  `observation_space`, and `info["success"]` — the same interface the
  `eval-harness` tool consumes for standardized A/B policy comparison.
- `run` writes a trajectories + metrics JSON report to `--output-uri`
  (file:// or local path); trajectories are stored downsampled (every 20th
  control step) to keep artifacts small.
