# Nebius Physical AI

Nebius Physical AI provides containerized workbench tools and SkyPilot workflows for robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. Codex should use this file as a lightweight index and load the relevant `.agents/skills/` file before changing behavior or operations.

## Key Conventions

- Use `npa/.venv/bin/python`; never use bare `python` for repo validation.
- Use `RELAXED_DIRTY_TREE_MODE`: dirty files outside the run's target paths are not blockers.
- Do not add time, cost, or job-count limits unless the operator explicitly asks for them.

## Codex Skills

- `.agents/skills/platform/quickstart/SKILL.md`: first-time setup, zero-credential first run, install, Nebius auth, and the contributor dev/test loop.
- `.agents/skills/workbench/cookbooks/SKILL.md`: working end-to-end cookbooks mapped to their validated entrypoints (BDD100K, sim-to-real, VLM-eval loop, LeRobot benchmarks, Isaac Lab BYOF).
- `.agents/skills/workbench/workbench-tool/SKILL.md`: workbench API/CLI/SDK/container pattern and S3 data flow.
- `.agents/skills/platform/skypilot-workflows/SKILL.md`: SkyPilot workflow authoring, runner scripts, limitations, and cleanup.
- `.agents/skills/platform/nebius-infra/SKILL.md`: cluster, storage, registry, credential, GPU routing, and namespace facts.
- `.agents/skills/platform/testing-conventions/SKILL.md`: pytest, ruff, gates, expected baseline, and known failures.
- `.agents/skills/platform/super-prompt-patterns/SKILL.md`: repo super-prompt phase, dirty-tree, NOVEL_ISSUE, and commit-lock conventions.
- `.agents/skills/workbench/lerobot/SKILL.md`: LeRobot policy training, serving, inference, datasets, and validation.
- `.agents/skills/workbench/fiftyone/SKILL.md`: FiftyOne curation, visualization, public access, and app behavior.
- `.agents/skills/workbench/vlm-eval/SKILL.md`: VLM-eval scoring, stub/self-hosted/api backends, benchmark sweeps, the sim-to-real loop, and the zero-credential first run.
- `.agents/skills/workbench/token-factory/SKILL.md`: Nebius Token Factory native workflows — the OpenAI-compatible hosted-inference client, the token-factory tool (caption/generate), the vlm-eval api backend, and the zero-GPU Token Factory SkyPilot workflows.
- `.agents/skills/workbench/genesis/SKILL.md`: Genesis simulation, RL teacher training, and EGL/DRI rendering limits.
- `.agents/skills/workbench/isaac-lab/SKILL.md`: Isaac Lab RT-core routing, headless training, workflows, and custom forks.
- `.agents/skills/workbench/cosmos/SKILL.md`: Cosmos world-model serving, backend selection, downloads, and rendering limits.
- `.agents/skills/workbench/lancedb/SKILL.md`: LanceDB vector store, BDD100K UDFs, materialized views, and CLIP embeddings.
- `.agents/skills/workbench/groot/SKILL.md`: GR00T deployment, status, routing, validation, and CUDA 13 alignment.
- `.agents/skills/workbench/sonic/SKILL.md`: SONIC training, H100 routing, validation, and known job ID issue.
- `.agents/skills/workbench/workflows/SKILL.md`: reference SkyPilot YAMLs, runners, S3 outputs, and cookbooks.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (NuRec, CAD-to-SimReady, USD tooling, defect-image SDG, video data augmentation, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`. Those are not yet implemented in the workbench; add each as a real skill only when its solution lands on Nebius + SkyPilot, with tests.
