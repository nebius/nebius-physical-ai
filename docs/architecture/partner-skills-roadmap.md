# Partner Skills Roadmap (NVIDIA Physical AI / Omniverse)

Status: **roadmap / design**. None of the capabilities below are implemented in
the NPA workbench yet, and none of these skills have been validated end-to-end.
This document captures the onboarding analysis so it is not lost; each skill
should be added to `.agents/skills/` and `.claude/skills/` only **when its
workbench solution lands**, and only **with tests** (see "Gating" below).

## Why This Is A Doc, Not Live Skills

NPA agent skills are description-matched and can auto-load. A skill that
describes a capability NPA does not have can lead an agent to route a user toward
a non-existent `npa workbench` flow or imply NPA supports, e.g., NuRec. The
repo's established pattern (the `cosmos3-*` skills) is to land a skill **alongside
its implementation and tests**, never ahead of it:

- Implementation: `npa/src/npa/workbench/cosmos/cosmos3.py`, checked-in workflow
  YAMLs, and CLI commands.
- Validation: `test_cosmos3_agent_skills_are_discoverable_and_well_formed`
  asserts frontmatter, attribution, and that the workflow renders the right
  commands.

Until a partner capability has that footing, it stays here as a blueprint.

## Architecture Constraints (must hold for every onboarded skill)

- **SkyPilot is the sole orchestrator.** Every multi-stage job is a SkyPilot YAML
  under `npa/workflows/workbench/skypilot/`, submitted via
  `npa workbench workflow submit`. Do not add a second orchestrator.
- **Nebius substrate.** Managed Kubernetes (`npa-workbench-eu-north1`,
  `eu-north1`), S3 on `storage.eu-north1.nebius.cloud`, vLLM/serverless serving,
  GPU routing per `nebius-infra`.
- **No hardcoded infra.** Buckets, endpoints, registry IDs, and model names are
  configuration. Secrets via env / `~/.npa/credentials.yaml`.
- **Partner model.** Partner workloads accessed through the platform run on
  Nebius infrastructure.

## Attribution

Adapted analysis from NVIDIA agent skills at https://github.com/NVIDIA/skills.
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Upstream licenses are
Apache-2.0, except defect-image-generation and video-data-augmentation which are
CC-BY-4.0 AND Apache-2.0. Trademarks (NVIDIA, Omniverse, NuRec, NRE, Isaac Sim,
Cosmos) belong to NVIDIA. When a skill is onboarded, ship a `NOTICE-NVIDIA-SKILLS`
attribution file alongside it (mirror `NOTICE-NVIDIA-COSMOS3`).

## Onboarding Tiers

### Tier A — good fit, new capability, low conflict (do first)

These are upstream **routers**: orchestrator-agnostic (Docker + GPU + NGC/HF), so
they run on Nebius GPUs directly. They match NPA's existing "adapted-from-NVIDIA
router to upstream" precedent.

| Skill | Capability | Upstream | License | Lands when |
| --- | --- | --- | --- | --- |
| `neural-reconstruction` | NuRec/NRE: sensor recordings → renderable USDZ; NCore conversion; 3DGS; gRPC sensor sim; PhysicalAI HF datasets | `physical-ai-neural-reconstruction` → https://github.com/NVIDIA/nurec-skills | Apache-2.0 | A NuRec/NRE workbench tool or a checked-in SkyPilot YAML that runs NRE containers on Nebius exists and is validated |
| `cad-to-simready` | CAD/source-asset → SimReady USD via Omniverse Content Agents (convert, material/physics assignment, conformance, validation, packaging) | `omniverse-cad-to-simready` → https://github.com/nvidia-omniverse/content-agents | Apache-2.0 | Content Agents can be deployed on Nebius and a validated conversion path exists |

### Tier B — useful but peripheral USD tooling (defer)

NPA's default visualization is Rerun; these are Omniverse/USD-app-centric.

| Skill | Capability | Upstream | License | Lands when |
| --- | --- | --- | --- | --- |
| `usd-performance-tuning` | USD scene perf diagnosis + Scene Optimizer optimization | `omniverse-usd-performance-tuning` | Apache-2.0 | USD becomes a first-class NPA artifact and a Kit/Scene-Optimizer runtime is available on Nebius |
| `realtime-viewer` | ovrtx/ovstream USD viewer apps (browser/local/native) | `omniverse-realtime-viewer` | Apache-2.0 | An interactive USD viewer is a product requirement beyond Rerun |

### Tier C — valuable capability, requires a native build (do not copy upstream)

Upstream ships these as a different orchestrator/cloud stack. Onboarding requires
**building** the NPA-native pipeline described below, not porting upstream
plumbing.

| Skill | Capability | Upstream | License | Lands when |
| --- | --- | --- | --- | --- |
| `defect-image-generation` | AOI defect SDG (usd2roi, image-edit, AnomalyGen; PCBA/metal/glass; Day 0/Day 1) | `physical-ai-defect-image-generation` | CC-BY-4.0 AND Apache-2.0 | A validated SkyPilot defect-SDG pipeline + image-edit model serving on Nebius exists |
| `video-data-augmentation` | Cosmos-Transfer augmentation + VLM auto-labeling | `physical-ai-video-data-augmentation` | CC-BY-4.0 AND Apache-2.0 | A validated SkyPilot VDA pipeline driving NPA Cosmos + vLLM serving exists |
| `infrastructure-resilient-scaling` | SDG infra setup/scaling/recovery | `physical-ai-infrastructure-setup-and-resilient-scaling` | Apache-2.0 | Captured as Nebius-K8s + SkyPilot provisioning/runbooks; overlaps `nebius-infra` + `skypilot-workflows` |

## NPA-Native Target Architecture (Tier C build notes)

When building the Tier C solutions, implement each concern directly on NPA's
stack:

| Concern | NPA implementation |
| --- | --- |
| Multi-stage orchestration | SkyPilot YAML under `npa/workflows/workbench/skypilot/`, submitted via `npa workbench workflow submit`; status/logs via `npa workbench workflow status/logs` |
| GPU scheduling | SkyPilot `--gpus` + Nebius GPU routing (`nebius-infra`); RT-core paths (Isaac-render pose defects) on L40S / RTX PRO 6000 |
| Inference endpoints | NPA vLLM/serverless serving on Nebius (OpenAI-compatible); `vlm-eval` for VLM scoring/labeling |
| Video augmentation worker | NPA `cosmos` tool (`npa workbench cosmos`) |
| Artifact storage / handoff | `s3://$NPA_S3_BUCKET/...` on `storage.eu-north1.nebius.cloud`; `--input-path`/`--output-path` between stages; `npa workbench data sync` for retrieval |
| Cluster + namespaces | Nebius Managed Kubernetes; `workbench` (services), `default` (SkyPilot task pods) |

## Gating: Definition Of Done For Onboarding A Partner Skill

Before a skill in this roadmap moves into `.agents/skills/` and `.claude/skills/`:

1. The underlying capability runs on Nebius + SkyPilot (a checked-in workflow
   YAML, runner, or workbench tool), or — for pure routers — a validated upstream
   fetch + run path on a Nebius GPU.
2. A test mirrors `test_cosmos3_agent_skills_are_discoverable_and_well_formed`:
   asserts frontmatter (`name`, `description`), a "Source And Attribution"
   section, and the `NOTICE-NVIDIA-SKILLS` reference.
3. A `NOTICE-NVIDIA-SKILLS` attribution file ships alongside the skill.
4. The skill body points only to real entrypoints; no invented `npa workbench`
   commands, and no second orchestrator. Use the NPA-native table above.
5. Indexed in `AGENTS.md` (Codex) and `CLAUDE.md` (Claude).
