---
name: eval-harness
description: Use when standardizing policy evaluation, A/B comparing two policies with paired seeds, computing success-rate confidence intervals, or using the npa workbench eval-harness CLI.
---

# eval-harness

`eval-harness` is the standardized policy evaluation tool: run one policy on
a registered manipulation task for N episodes (`run`), or A/B compare two
policies with paired episode seeds (`compare`). Every episode is scored by a
success judge and every report carries success rates with Wilson confidence
intervals; comparisons add a paired bootstrap confidence interval on the
success-rate difference.

Built-in tasks: `mujoco_manip.peg_insertion`, `mujoco_manip.screw_driving`,
`mujoco_manip.reach` (real MuJoCo contact simulation).

## Interfaces

- CLI: `npa workbench eval-harness <run|compare> --help`
- Python SDK (workbench-first): `npa.workbench.eval_harness`
  (`run`, `compare`, plus `run_policy`, `compare_policies`, `metrics`,
  `HeuristicJudge`, `VLMJudge`)
- Workflow module: `npa.workflows.byof.eval_harness_pipeline`
  (real stage implementations, argparse entrypoint)

## Conventions

- The CLI lives at `npa.cli.workbench.eval_harness` (multi-command package
  form for new tools); the SDK surface lives at `npa.workbench.eval_harness`
  and does not go through `make_cli_wrapper`.
- Policy specs: `scripted:<expert|noisy|random>` for built-ins, or
  `python:<module>:<callable>` for user callables with signature
  `policy(obs) -> action`.
- Judges: `heuristic` trusts the environment's `info["success"]` signal;
  `vlm` is a real client over `npa.workbench.vlm_eval` that requires
  `--vlm-endpoint-url` (and rendered frames) and raises an informative error
  when unconfigured — it never invents a score.
- Public CLI and SDK writes use required S3 `--output-path` destinations;
  `--output-uri` remains a compatibility alias. Reports are create-only:
  `report.json` + `report.md` (run) or `compare.json` + `compare.md` (compare)
  appear below the requested prefix. An interrupted retry resumes only if a
  pre-existing object has exactly the same bytes; a different object fails
  closed. Local paths are limited to hermetic pipeline tests, never cross-tool
  handoffs.
