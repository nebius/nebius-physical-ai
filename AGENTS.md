# Nebius Physical AI

Nebius Physical AI provides containerized workbench tools and SkyPilot workflows for robotics, simulation, perception, and synthetic-data workloads on Nebius infrastructure. Codex should use this file as a lightweight index, scan `skills/index.yaml`, and load the relevant root `skills/` entry before changing behavior or operations.

## Key Conventions

- Use `npa/.venv/bin/python`; never use bare `python` for repo validation.
- Use `RELAXED_DIRTY_TREE_MODE`: dirty files outside the run's target paths are not blockers.
- Do not add time, cost, or job-count limits unless the operator explicitly asks for them.

## Codex Skills

The source of truth is `skills/index.yaml`. The tree is organized as:

- `skills/workflows/`: workflow-level procedures such as sim-to-real, policy training, Cosmos3 inference, and reference SkyPilot workflows.
- `skills/atomic/`: reusable actions and review conventions such as GPU selection, workflow submission, testing conventions, image build/push, Cosmos3 setup/troubleshooting, and agent visual feedback (Describe this).
- `skills/atomic/agent-visual-feedback/SKILL.md`: Describe-this multimodal feedback for Rerun / video / image / data viewers.
- `skills/tools/`: concrete workbench and platform tools such as LeRobot, FiftyOne, Genesis, Isaac Lab, Cosmos, LanceDB, GR00T, SONIC, MJLab, Retargeting, SkyPilot, Scenario Gen, Dataset-of-record, and Nebius infra.
- `skills/tools/scenario-gen/SKILL.md`: adversarial scenario generation — an RL adversary that maximizes failures of a policy-under-test, scenario ranking, and the adversarial-scenario-hardening workflow.
- `skills/tools/dataset/SKILL.md`: dataset-of-record — ingest, validate, curate, and query production sensor data as a versioned, lineage-tracked dataset (FiftyOne curation + LanceDB query index).
- `skills/tools/foxglove/SKILL.md`: Foxglove embedded viewer — the `@foxglove/embed` TypeScript SDK in the agent UI, MCAP recordings (convert/inspect/publish), and the `npa-foxglove-embed` container.
- `skills/tools/insights/SKILL.md`: lineage graph + common metrics store over workflow-run artifacts — non-invasive ingest-run, query, compare, lineage traversal, and dashboard (CPU-only, append-only S3 JSONL, LanceDB-optional).
- `skills/workflows/sim2real-operate/SKILL.md`: operate the staged Sim2Real pipeline on a K8s GPU cluster — runbook, direct-K8s submit, preflight health checks, storage secret sync, and job monitoring.
- `skills/workflows/agent-fresh-operate/SKILL.md`: npa-driven agent teardown, fresh-setup, tiered verify gates, and deploy failure recovery on the operator/dev VM.
- `skills/workflows/author-npa-workflow/SKILL.md`: author and validate declarative `npa.workflow/v0.0.1` specs (`validate-spec`, `plan-spec`, toolRef catalog).
- `skills/workflows/byof-onboard/SKILL.md`: BYOF OSS repo onboarding (Ubuntu/Isaac base, container-verify, agent `onboard_solution`).
- `skills/workflows/contribute-workbench-image/SKILL.md`: external fork PR through licensing review, trusted image build, registry-byte validation, and incremental GHCR publication.
- `skills/workflows/onboard-world-model/SKILL.md`: generic playbook for onboarding and containerizing a world model (learned action-conditioned simulator) as a multi-GPU BYOF registry candidate — containerize, stage a real dataset, encode the train→tokenize→dynamics→dream→visualize loop as capability smokes, validate on real GPUs (Open Dreamer is the reference example).
- `skills/workflows/generate-npa-workflow/SKILL.md`: design new creative npa.workflow pipelines from the catalog (loops, gates, reference YAML).
- `skills/workflows/diagram-to-npa-workflow/SKILL.md`: turn an architecture diagram + step write-up into a working npa.workflow/v0.0.1 YAML (boxes/arrows/diamonds/back-edges → states, loops, gates, catalog toolRefs); generalizes across sim2real, AV, RL, and Cosmos pipelines.
- `skills/workflows/physical-ai-data-factory/SKILL.md`: author/run/view the NVIDIA Physical AI Data Factory blueprint on Nebius + SkyPilot (no OSMO): annotate → Cosmos Transfer augment → Cosmos Evaluator gate → re-label → Cosmos Curator + FiftyOne curate → Rerun visualize. The evaluator and curator are the real Apache-2.0 NVIDIA projects (`npa workbench cosmos-evaluator` / `cosmos-curate`); see `skills/NOTICE-NVIDIA-COSMOS-OSS`.
- `skills/workflows/neural-reconstruction/SKILL.md`: NuRec/NRE neural reconstruction on Nebius — NCore V4 capture (incl. deriving the `rig → world` pose edge NRE requires) → 3DGUT Gaussian training → renderable USDZ → rig-offset novel views → `reports/sim2real.rrd`. RT-core GPU only (L40S / RTX PRO 6000, never H100/H200).
- `skills/atomic/real-components/SKILL.md`: ensure every advertised workbench pipeline stage invokes the real component (Cosmos Transfer, FiftyOne, VLM), not an echo/manifest stub.
- `skills/atomic/solution-licensing/SKILL.md`: when adding a solution, tool, image, model, or dataset — classify what the artifact actually bakes, decide whether it may be redistributed (`public` vs `restricted`), and record it in the packaging contract where the guards enforce it. Verify the claim against the BUILT image with `npa/scripts/scan_image_omniverse_payload.py`, not by reading the Dockerfile.
- `skills/workbench/sim2real-engine/SKILL.md`: canonical 14-stage Sim2Real engine map (`run_preamble` / `run_inner_loop` / `run_single_outer_iteration` / `run_finalize`) and K8s sibling job glue.

Compatibility symlinks exist at `.agents/skills` and `.claude/skills`; do not add new skills there directly.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (CAD-to-SimReady, USD tooling, defect-image SDG, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`; those are not yet implemented in the workbench. **NuRec/NRE has landed** (`skills/workflows/neural-reconstruction/SKILL.md`), as has video data augmentation (`skills/workflows/physical-ai-data-factory/SKILL.md`). Add each remaining capability as a real skill only when its solution lands on Nebius + SkyPilot, with tests.
