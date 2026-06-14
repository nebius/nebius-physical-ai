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

Compatibility symlinks exist at `.agents/skills` and `.claude/skills`; do not add new skills there directly.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.
