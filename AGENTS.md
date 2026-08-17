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
- `skills/tools/`: concrete workbench and platform tools such as LeRobot, FiftyOne, Genesis, Isaac Lab, Cosmos, LanceDB, GR00T, SONIC, MJLab, Retargeting, SkyPilot, Scenario Gen, Dataset-of-record, Fleet, and Nebius infra.
- `skills/workflows/first-run-setup/SKILL.md`: zero to a first verified result on a fresh machine or new project — an ordered, gated path (install → configure → credential preflight → cheapest-proof workload → validate spec → provision → image pullability → submit) with a stop condition at every step.
- `skills/atomic/health-preflight/SKILL.md`: there is no `npa doctor` — prove HF/NGC/S3/Token Factory credentials and gated-model access with `npa workbench health preflight` / `access` before spending GPU time.
- `skills/atomic/debug-failed-run/SKILL.md`: triage a run that failed, hung, or produced no artifacts — status and pod-level reason, stage logs, S3 evidence, image pullability, scheduling, and the resume-vs-cancel decision.
- `skills/atomic/teardown-and-cost/SKILL.md`: stop spend safely — cancel-before-destroy ordering, cloud versus local state, and the orphan audit for leaked clusters, agent VMs, controllers, buckets, and cross-project fleets.
- `skills/tools/token-factory/SKILL.md`: zero-GPU hosted inference (captioning, batch generation, Cosmos physical-AI reasoning) — the cheapest tier that produces a real artifact, with no cluster and no provisioning.
- `skills/tools/vlm-eval/SKILL.md`: score rollouts with a VLM and turn the score into a gate — `run` vs `loop`, rubric/threshold benchmark sweeps, backend selection, and judging against a plan an earlier stage wrote.
- `skills/tools/golden-eval/SKILL.md`: prove a container image actually works — per-container hello-world manifest, dry-run/local/serverless tiers, batch runs, and the offline manifest validation that gates CI.
- `skills/tools/burst/SKILL.md`: one gang-scheduled multi-node GPU job with torchrun rendezvous, deliberately not a workflow surface.
- `skills/tools/gpu-cluster-provisioning/SKILL.md`: managed-image vs GPU-Operator driver strategy (operator mode is unsafe on NVSwitch), the post-apply health gates (fabric, CUDA vectorAdd, stability window), accelerator-name discovery, and triage for nodes whose GPUs do not work.
- `skills/tools/detection-training/SKILL.md`: Faster R-CNN detectors trained from LanceDB materialized views (BDD100K failure-mode slices).
- `skills/tools/artifact-viz-share/SKILL.md`: sim demos → LeRobotDataset → `.rrd`/MP4, and time-boxed presigned Rerun share links.
- `skills/tools/fleet/SKILL.md`: deploy a fleet of Nebius Managed Kubernetes (k8s-training) clusters across one or many projects in a tenant from an `npa.fleet/v0.0.1` spec — identical and/or custom clusters, create-on-demand projects, and a k8s-training source that can consume the latest upstream recipe.
- `skills/tools/scenario-gen/SKILL.md`: adversarial scenario generation — an RL adversary that maximizes failures of a policy-under-test, scenario ranking, and the adversarial-scenario-hardening workflow.
- `skills/tools/dataset/SKILL.md`: dataset-of-record — ingest, validate, curate, and query production sensor data as a versioned, lineage-tracked dataset (FiftyOne curation + LanceDB query index).
- `skills/tools/foxglove/SKILL.md`: Foxglove embedded viewer — the `@foxglove/embed` TypeScript SDK in the agent UI, MCAP recordings (convert/inspect/publish), and the `npa-foxglove-embed` container.
- `skills/tools/insights/SKILL.md`: lineage graph + common metrics store over workflow-run artifacts — non-invasive ingest-run, query, compare, lineage traversal, and dashboard (CPU-only, append-only S3 JSONL, LanceDB-optional).
- `skills/workflows/sim2real-operate/SKILL.md`: operate the compositional Sim2Real `npa.workflow` through the standard SkyPilot runtime — validate/plan/submit, durable S3 resume, preflight health checks, storage secret sync, and job monitoring.
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
- `skills/atomic/protect-nebius-infra-details/SKILL.md`: sanitize commits, docs, reports, examples, tests, PR/issue text, and live-validation handoffs so concrete live Nebius infrastructure details remain only in access-controlled external evidence.
- `skills/workbench/sim2real-engine/SKILL.md`: canonical 14-stage Sim2Real graph, stateless adapters, parallel lane joins, ComponentRecords, and durable standard-runtime resume. Retired controller entrypoints remain finite legacy compatibility only.

Compatibility symlinks exist at `.agents/skills` and `.claude/skills`; do not add new skills there directly.

## Partner Capability Roadmap

Onboarding NVIDIA Physical AI / Omniverse capabilities (CAD-to-SimReady, USD tooling, defect-image SDG, SDG infrastructure) is tracked in `docs/architecture/partner-skills-roadmap.md`; those are not yet implemented in the workbench. **NuRec/NRE has landed** (`skills/workflows/neural-reconstruction/SKILL.md`), as has video data augmentation (`skills/workflows/physical-ai-data-factory/SKILL.md`). Add each remaining capability as a real skill only when its solution lands on Nebius + SkyPilot, with tests.
