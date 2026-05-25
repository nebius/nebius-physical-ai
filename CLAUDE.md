# Nebius Physical AI

Nebius Physical AI is the workbench and workflow layer for running robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. The repo centers on the `npa` CLI/SDK, containerized workbench tools, and SkyPilot workflows that compose those tools through S3 artifacts. The current product shape is a tool marketplace that customers can adapt without hardcoding project-specific infrastructure.

Claude Code should treat this file as an index. Load the relevant skill before making architecture, review, or domain judgments.

## Claude Skills

- `.claude/skills/platform/architecture/SKILL.md`: load for platform architecture, tool-layer scope, orchestrator decisions, partner model, and validation state.
- `.claude/skills/platform/review-checklist/SKILL.md`: load for code reviews and risk classification.
- `.claude/skills/workbench/physical-ai-context/SKILL.md`: load for robotics, sim-to-real, GPU routing, Genesis, Isaac Lab, LeRobot, SONIC, GR00T, Cosmos, or BDD100K context.

## Project Instructions

- Do not hardcode project IDs, tenant IDs, registry IDs, bucket names, or secrets. Credentials live in `~/.npa/credentials.yaml`; machine-managed config lives in `~/.npa/config.yaml`.
- Unit tests must not touch real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site.
- Do not import GPU-heavy packages such as `torch`, `genesis`, or `lerobot` at module level in unit tests; use targeted imports or `pytest.importorskip()`.
- CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.
- The repo's current operational context is the workbench architecture, not the older LeRobot-only VM research-script flow.
