# Nebius Physical AI

Nebius Physical AI provides containerized workbench tools and SkyPilot workflows for robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. Codex should use this file as a lightweight index and load the relevant `.agents/skills/` file before changing behavior or operations.

## Key Conventions

- Use `npa/.venv/bin/python`; never use bare `python` for repo validation.
- Use `RELAXED_DIRTY_TREE_MODE`: dirty files outside the run's target paths are not blockers.
- Do not add time, cost, or job-count limits unless the operator explicitly asks for them.

## Codex Skills

- `.agents/skills/platform/skill-authoring/SKILL.md`: frontmatter contract, required sections, and mirroring rules for all SKILL.md files.
- `.agents/skills/platform/skill-curation/SKILL.md`: triage/promote/drop/escalate loop, drift checklist, and curation-log ledger.
- `.agents/skills/workbench/workbench-tool/SKILL.md`: workbench API/CLI/SDK/container pattern and S3 data flow.
- `.agents/skills/platform/skypilot-workflows/SKILL.md`: SkyPilot workflow authoring, runner scripts, limitations, and cleanup.
- `.agents/skills/platform/nebius-infra/SKILL.md`: cluster, storage, registry, credential, GPU routing, and namespace facts.
- `.agents/skills/platform/testing-conventions/SKILL.md`: pytest, ruff, gates, expected baseline, and known failures.
- `.agents/skills/platform/super-prompt-patterns/SKILL.md`: repo super-prompt phase, dirty-tree, NOVEL_ISSUE template, skill-deltas log, and commit-lock conventions.
- `.agents/skills/workbench/lerobot/SKILL.md`: LeRobot policy training, serving, inference, datasets, and validation.
- `.agents/skills/workbench/fiftyone/SKILL.md`: FiftyOne curation, visualization, public access, and app behavior.
- `.agents/skills/workbench/genesis/SKILL.md`: Genesis simulation, RL teacher training, and EGL/DRI rendering limits.
- `.agents/skills/workbench/isaac-lab/SKILL.md`: Isaac Lab RT-core routing, headless training, workflows, and custom forks.
- `.agents/skills/workbench/cosmos/SKILL.md`: Cosmos world-model serving, backend selection, downloads, and rendering limits.
- `.agents/skills/workbench/lancedb/SKILL.md`: LanceDB vector store, BDD100K UDFs, materialized views, and CLIP embeddings.
- `.agents/skills/workbench/groot/SKILL.md`: GR00T deployment, status, routing, validation, and CUDA 13 alignment.
- `.agents/skills/workbench/sonic/SKILL.md`: SONIC training, H100 routing, validation, and known job ID issue.
- `.agents/skills/workbench/workflows/SKILL.md`: reference SkyPilot YAMLs, runners, S3 outputs, and cookbooks.

## Self-Improvement Loop

See [docs/agents/loop.md](docs/agents/loop.md) for the full capture → triage → promote/drop/escalate flow, roles, file map, and cadence. The short version: builders log to `/tmp/<run-id>/` during a run, persist to `.agents/runs/<run-id>/` at end of Phase L, and a different reviewer triages those logs into the affected skills per `skill-curation`.

