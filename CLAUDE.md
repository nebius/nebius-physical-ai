# Nebius Physical AI

Nebius Physical AI is the workbench and workflow layer for running robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. The repo centers on the `npa` CLI/SDK, containerized workbench tools, and SkyPilot workflows that compose those tools through S3 artifacts. The current product shape is a tool marketplace that customers can adapt without hardcoding project-specific infrastructure.

Claude Code should treat this file as an index. Load the relevant skill before making architecture, review, or domain judgments.

## Claude Skills

- `.claude/skills/platform/quickstart/SKILL.md`: load for first-time setup, the zero-credential first run, install, Nebius auth, and the contributor dev/test loop.
- `.claude/skills/workbench/cookbooks/SKILL.md`: load for running working end-to-end cookbooks (BDD100K, sim-to-real, VLM-eval loop, LeRobot benchmarks, Isaac Lab BYOF) mapped to their validated entrypoints.
- `.claude/skills/platform/architecture/SKILL.md`: load for platform architecture, tool-layer scope, orchestrator decisions, partner model, and validation state.
- `.claude/skills/platform/review-checklist/SKILL.md`: load for code reviews and risk classification.
- `.claude/skills/workbench/physical-ai-context/SKILL.md`: load for robotics, sim-to-real, GPU routing, Genesis, Isaac Lab, LeRobot, SONIC, GR00T, Cosmos, or BDD100K context.
- `.claude/skills/workbench/mjlab/SKILL.md`: load for MJLab locomotion evaluation and SONIC checkpoint scoring.
- `.claude/skills/workbench/retargeting/SKILL.md`: load for motion retargeting in SONIC locomotion workflows.
- `.agents/skills/workbench/sim-to-real/SKILL.md`: load for generic sim-to-real data import, Cosmos autoscale, VLM evaluation, and controller-loop workflow planning.
- `.agents/skills/platform/context-efficiency/SKILL.md`: apply on every turn to minimize context ingestion, keep chat memory lean, avoid full-workspace scans, and route work to the right model tier.

### Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.

## Project Instructions

- Do not hardcode project IDs, tenant IDs, registry IDs, bucket names, or secrets. Credentials live in `~/.npa/credentials.yaml`; machine-managed config lives in `~/.npa/config.yaml`.
- Unit tests must not touch real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site.
- Do not import GPU-heavy packages such as `torch`, `genesis`, or `lerobot` at module level in unit tests; use targeted imports or `pytest.importorskip()`.
- CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.
- The repo's current operational context is the workbench architecture, not the older LeRobot-only VM research-script flow.
