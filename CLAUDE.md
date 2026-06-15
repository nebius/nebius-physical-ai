# Nebius Physical AI

Nebius Physical AI is the workbench and workflow layer for running robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. The repo centers on the `npa` CLI/SDK, containerized workbench tools, and SkyPilot workflows that compose those tools through S3 artifacts. The current product shape is a tool marketplace that customers can adapt without hardcoding project-specific infrastructure.

Claude Code should treat this file as a lightweight index. Scan
`skills/index.yaml` first, then load the relevant root `skills/` entry before
making architecture, review, or domain judgments.

## Skill Index

- `skills/index.yaml`: root manifest with name, when-to-use, path, and CI smoke
  expectations.
- `skills/atomic/architecture/SKILL.md`: platform architecture and validation
  state.
- `skills/atomic/review-checklist/SKILL.md`: review risk classification.
- `skills/atomic/physical-ai-context/SKILL.md`: robotics, sim-to-real,
  GPU-routing, Genesis, Isaac Lab, LeRobot, SONIC, GR00T, Cosmos, or BDD100K
  context.
- `skills/tools/mjlab/SKILL.md`: MJLab locomotion evaluation and SONIC checkpoint
  scoring.
- `skills/tools/retargeting/SKILL.md`: motion retargeting in SONIC locomotion
  workflows.
- `skills/workflows/sim-to-real/SKILL.md`: generic sim-to-real workflow
  planning.

Compatibility symlinks exist at `.claude/skills` and `.agents/skills`; do not
create a new split skill tree.

### Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.

## Project Instructions

- Do not hardcode project IDs, tenant IDs, registry IDs, bucket names, or secrets. Credentials live in `~/.npa/credentials.yaml`; machine-managed config lives in `~/.npa/config.yaml`.
- Unit tests must not touch real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site.
- Do not import GPU-heavy packages such as `torch`, `genesis`, or `lerobot` at module level in unit tests; use targeted imports or `pytest.importorskip()`.
- CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.
- The repo's current operational context is the workbench architecture, not the older LeRobot-only VM research-script flow.
