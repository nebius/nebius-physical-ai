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
- `skills/atomic/agent-development/SKILL.md`: build, enhance, or test the NPA
 chat agent backend (grounded-first routing, cost-aware Token Factory model
 selection, embedded-backend mechanism, cheap-token test tiers).
- `skills/atomic/agent-visual-feedback/SKILL.md`: Describe-this viewer feedback
  for Rerun / video / image / data panes (multimodal vision tier).
- `skills/atomic/review-checklist/SKILL.md`: review risk classification.
- `skills/atomic/physical-ai-context/SKILL.md`: robotics, sim-to-real,
  GPU-routing, Genesis, Isaac Lab, LeRobot, SONIC, GR00T, Cosmos, or BDD100K
  context.
- `skills/tools/scenario-gen/SKILL.md`: adversarial scenario generation — an RL
 adversary that maximizes failures of a policy-under-test, scenario ranking, and
 the adversarial-scenario-hardening workflow.
- `skills/tools/dataset/SKILL.md`: dataset-of-record — ingest, validate, curate,
 and query production sensor data as a versioned, lineage-tracked dataset
 (FiftyOne curation + LanceDB query index).
- `skills/tools/foxglove/SKILL.md`: Foxglove embedded viewer — the
 `@foxglove/embed` TypeScript SDK in the agent UI, MCAP recordings
 (convert/inspect/publish), and the `npa-foxglove-embed` container.
- `skills/tools/insights/SKILL.md`: lineage graph + common metrics store over
 workflow-run artifacts — non-invasive ingest-run, query, compare, lineage
 traversal, and dashboard (CPU-only, append-only S3 JSONL, LanceDB-optional).
- `skills/tools/fleet/SKILL.md`: deploy a fleet of Nebius Managed Kubernetes
 (k8s-training) clusters across one or many projects in a tenant from an
 `npa.fleet/v0.0.1` spec — identical and/or custom clusters, create-on-demand
 projects, and a k8s-training source that can consume the latest upstream recipe.
- `skills/tools/mjlab/SKILL.md`: MJLab locomotion evaluation and SONIC checkpoint
 scoring.
- `skills/tools/retargeting/SKILL.md`: motion retargeting in SONIC locomotion
  workflows.
- `skills/workflows/sim-to-real/SKILL.md`: generic sim-to-real workflow
  planning.
- `skills/workflows/sim2real-operate/SKILL.md`: run, monitor, and debug the
  compositional Sim2Real `npa.workflow` through the standard SkyPilot runtime
  (validate/plan/submit, durable S3 resume, health checks, job monitoring).
- `skills/workflows/agent-fresh-operate/SKILL.md`: npa-driven agent teardown,
  fresh-setup, tiered verify gates, and deploy failure recovery on the
  operator/dev VM.
- `skills/workflows/author-npa-workflow/SKILL.md`: author and validate
  declarative `npa.workflow/v0.0.1` specs (toolRef catalog, validate/plan/run CLI).
- `skills/workflows/generate-npa-workflow/SKILL.md`: design new creative
 npa.workflow pipelines from the workbench tool catalog.
- `skills/workflows/diagram-to-npa-workflow/SKILL.md`: turn an architecture
 diagram + step write-up into a working npa.workflow/v0.0.1 YAML (boxes, arrows,
 decision diamonds, and loop back-edges → states, loops, gates, catalog
 toolRefs); generalizes across sim2real, AV, RL, and Cosmos pipelines.
- `skills/workflows/physical-ai-data-factory/SKILL.md`: author, run, submit, or
 view the NVIDIA Physical AI Data Factory blueprint on Nebius + SkyPilot (no
 OSMO): annotate → Cosmos Transfer augment → Cosmos Evaluator gate → re-label →
 Cosmos Curator + FiftyOne curate → Rerun visualize. The evaluator and curator
 are the real Apache-2.0 NVIDIA projects, wrapped as
 `npa workbench cosmos-evaluator` and `npa workbench cosmos-curate`; see
 `skills/NOTICE-NVIDIA-COSMOS-OSS` for which upstream code runs where.
- `skills/workflows/neural-reconstruction/SKILL.md`: NuRec/NRE neural
 reconstruction on Nebius — NCore V4 capture (including deriving the
 `rig → world` pose edge NRE requires) → 3DGUT Gaussian training → renderable
 USDZ → rig-offset novel views → `reports/sim2real.rrd`. RT-core GPU only
 (L40S / RTX PRO 6000, never H100/H200).
- `skills/workflows/onboard-world-model/SKILL.md`: generic playbook for
 onboarding and containerizing a world model (learned action-conditioned
 simulator) as a multi-GPU BYOF registry candidate — containerize, stage a real
 dataset, encode the train→tokenize→dynamics→dream→visualize loop as capability
 smokes, and validate on real GPUs (Open Dreamer is the reference example).
- `skills/atomic/real-components/SKILL.md`: ensure every advertised workbench
 pipeline stage invokes the real component (Cosmos Transfer, FiftyOne, VLM),
 not an echo/manifest stub masquerading as real work.
- `skills/atomic/solution-licensing/SKILL.md`: when adding a solution, tool,
 image, model, or dataset — classify what the artifact actually bakes, decide
 whether it may be redistributed (`public` vs `restricted`), and record it in
 the packaging contract where the guards enforce it. Verify the claim against the
 BUILT image with `npa/scripts/scan_image_omniverse_payload.py`, not by reading
 the Dockerfile: the Isaac images were cleared that way, and two of the three
 problems it found were invisible in the Dockerfile.
- `skills/workbench/sim2real-engine/SKILL.md`: canonical 14-stage Sim2Real graph,
  stateless stage adapters, parallel lane joins, ComponentRecords, and durable
  standard-runtime resume. The preamble/inner/outer/finalize controller is
  finite legacy compatibility, never the canonical execution path.

Compatibility symlinks exist at `.claude/skills` and `.agents/skills`; do not
create a new split skill tree.

### Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (CAD-to-SimReady, USD tooling, defect-image SDG, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`; those are not yet implemented in the workbench. **NuRec/NRE has landed** (`skills/workflows/neural-reconstruction/SKILL.md`), as has video data augmentation (`skills/workflows/physical-ai-data-factory/SKILL.md`). Add each remaining capability as a real skill only when its solution lands on Nebius + SkyPilot, with tests.

## Project Instructions

- Do not hardcode project IDs, tenant IDs, registry IDs, bucket names, or secrets. Credentials live in `~/.npa/credentials.yaml`; machine-managed config lives in `~/.npa/config.yaml`.
- Unit tests must not touch real infrastructure. Mock SSH, S3, Nebius APIs, GPUs, and network calls at the call site.
- Do not import GPU-heavy packages such as `torch`, `genesis`, or `lerobot` at module level in unit tests; use targeted imports or `pytest.importorskip()`.
- CLI tests use `typer.testing.CliRunner` against `npa.cli.main:app`.
- The repo's current operational context is the workbench architecture, not the older LeRobot-only VM research-script flow.
