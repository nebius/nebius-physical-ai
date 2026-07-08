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
- `skills/workflows/agent-fresh-operate/SKILL.md`: npa-driven agent teardown, fresh-setup, tiered verify gates, and deploy failure recovery on the operator/dev VM.
- `skills/workflows/author-npa-workflow/SKILL.md`: author and validate declarative `npa.workflow/v0.0.1` specs (`validate-spec`, `plan-spec`, toolRef catalog).
- `skills/workflows/byof-onboard/SKILL.md`: BYOF OSS repo onboarding (Ubuntu/Isaac base, container-verify, agent `onboard_solution`).
- `skills/workflows/generate-npa-workflow/SKILL.md`: design new creative npa.workflow pipelines from the catalog (loops, gates, golden YAML).
- `skills/workflows/diagram-to-npa-workflow/SKILL.md`: turn an architecture diagram + step write-up into a working npa.workflow/v0.0.1 YAML (boxes/arrows/diamonds/back-edges → states, loops, gates, catalog toolRefs); generalizes across sim2real, AV, RL, and Cosmos pipelines.
- `skills/workbench/sim2real-engine/SKILL.md`: canonical 14-stage Sim2Real engine map (`run_preamble` / `run_inner_loop` / `run_single_outer_iteration` / `run_finalize`) and K8s sibling job glue.

Compatibility symlinks exist at `.agents/skills` and `.claude/skills`; do not add new skills there directly.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.
