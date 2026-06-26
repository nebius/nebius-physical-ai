# Nebius Physical AI

Nebius Physical AI provides containerized workbench tools and SkyPilot workflows for robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. Codex should use this file as a lightweight index, scan `skills/index.yaml`, and load the relevant root `skills/` entry before changing behavior or operations.

## Key Conventions

- Use `npa/.venv/bin/python`; never use bare `python` for repo validation.
- Use `RELAXED_DIRTY_TREE_MODE`: dirty files outside the run's target paths are not blockers.
- Do not add time, cost, or job-count limits unless the operator explicitly asks for them.

## Codex Skills

The source of truth is `skills/index.yaml`. The tree is organized as:

- `skills/workflows/`: workflow-level procedures such as sim-to-real, policy training, Cosmos3 inference, and reference SkyPilot workflows.
- `skills/atomic/`: reusable actions and review conventions such as GPU selection, workflow submission, testing conventions, image build/push, and Cosmos3 setup/troubleshooting.
- `skills/tools/`: concrete workbench and platform tools such as LeRobot, FiftyOne, Genesis, Isaac Lab, Cosmos, LanceDB, GR00T, SONIC, MJLab, Retargeting, SkyPilot, and Nebius infra.
- `skills/workflows/sim2real-operate/SKILL.md`: operate the staged Sim2Real pipeline on a K8s GPU cluster — runbook, direct-K8s submit, preflight health checks, storage secret sync, and job monitoring.
- `skills/workflows/author-npa-workflow/SKILL.md`: author and validate declarative `npa.workflow/v0.0.1` specs (`validate-spec`, `plan-spec`, toolRef catalog).
- `skills/workflows/generate-npa-workflow/SKILL.md`: design new creative npa.workflow pipelines from the catalog (loops, gates, golden YAML).
- `skills/workbench/sim2real-engine/SKILL.md`: canonical 14-stage Sim2Real engine map (`run_preamble` / `run_inner_loop` / `run_single_outer_iteration` / `run_finalize`) and K8s sibling job glue.

Compatibility symlinks exist at `.agents/skills` and `.claude/skills`; do not add new skills there directly.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.

## Cursor Cloud specific instructions

`npa` is a CLI/SDK (Typer), not a long-running server; there is no dev server to
start. The dev loop is lint + unit tests + invoking the `npa` CLI. The update
script already provisions the venv at `npa/.venv` and installs `npa[dev]`.

- Always use the venv interpreter: `npa/.venv/bin/python` and `npa/.venv/bin/npa`
  (per the repo `Key Conventions`).
- Lint / test / build commands are defined in the root `Makefile` and
  `CONTRIBUTING.md`. Pass the venv explicitly, e.g.
  `make test PYTHON=/workspace/npa/.venv/bin/python` (also `make lint`,
  `make test-smoke`). `make` targets run `pytest`/`ruff` from inside `npa/`.
- The full `make test` suite (~2240 unit tests) is hermetic (no cloud/GPU/network)
  but takes ~3.5 min; run it in the background and poll rather than blocking.
- `make lint` currently reports pre-existing `ruff` failures on a clean tree
  (mostly unused-import F401 in tests); they are not caused by your changes. Use
  `make format` to autofix only what you touched, and do not mass-rewrite
  unrelated files.
- No-cloud smoke / "hello world": the offline stub VLM-eval benchmark from the
  README runs with zero credentials and writes a ranked report
  (`best_config.metrics.accuracy == 1.0`):
  `npa/.venv/bin/npa workbench vlm-eval benchmark --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json --output /tmp/vlm-eval-benchmark.json --backend stub --thresholds 0.5,0.8,0.9 --rubrics default,strict --models Qwen/Qwen2-VL-7B-Instruct --format json`
- Anything beyond the stub path (real workbench tools, `npa configure`, e2e
  markers, SkyPilot, GPU) needs Nebius credentials in `~/.npa/credentials.yaml`
  and external infra; those are intentionally out of scope for local dev.
- System dependency note: creating the venv requires the `python3.12-venv` apt
  package (already present in the snapshot); it is not installed by `pip`.
