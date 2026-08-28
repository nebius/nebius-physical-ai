---
name: sim2real-engine
description: Use when navigating, reviewing, or changing the compositional Sim2Real workflow, stateless stage adapters, durable standard-runtime resume, ComponentRecords, and S3 lineage.
---

# Compositional Sim2Real engine

The canonical engine surface is the ordinary workflow
`npa/workflows/workbench/npa-workflows/sim2real.yaml`. Its leaf states call
`npa.workflows.sim2real.workflow_stage`; S3/content-addressed evidence primitives
live in `workflow_io`. The shared `npa_workflow` interpreter/runtime owns loops,
parallel waves, SkyPilot jobs, ledger persistence, and resume.

The small `engine.py` facade and bounded `legacy_*` modules remain compatibility
surfaces for archived runs. New canonical states must never call
`run_preamble`, `run_inner_loop`, `run_single_outer_iteration`, `run_finalize`,
or spawn sibling Jobs.

## Stage map

| Stage | Canonical record | Adapter boundary |
| ---: | --- | --- |
| 1 | `stage_01_trigger` | task-aligned trigger/seed validation |
| 2 | `stage_02_assets` | task/assets/camera/strict-success contract |
| 3 | `stage_03_augment` | real Cosmos Transfer 2.5 |
| 4 | `stage_04_envs_raw` | parallel raw EnvGen shards |
| 5 | `stage_05_envs_train` | sealed train/validation/gold split |
| 6 | `stage_06_tokens` | explicit S3 token/scenario manifest |
| 7 | `stage_07_actions_train` | real Isaac multi-camera rollout |
| 8 | `stage_08_vlm_eval_train` | single hosted Token Factory Cosmos3 CPU evaluator |
| 9 | `stage_09_training_signal` | temporal merge, real PPO, validation selection |
| 10 | `stage_10_eval_heldout` | exact-checkpoint untouched-gold Isaac eval |
| 11 | `stage_11_outer_loop` | strict metric + standard decision artifact |
| 12 | `stage_12_external_validation` | designed external `SEAM` |
| 13 | `stage_13_retrigger` | retrigger/loop record |
| 14 | `stage_14_rerun_viz` | final report, RRD, and MCAP |

## Invariants

- Use named `{{loop.*}}` tokens for iteration-scoped S3 paths.
- Every real stage publishes `WORKS`; Stage 12 alone publishes `SEAM`.
  Canonical pointers have immutable content-addressed history.
- GPU images are immutable and source-attested. Isaac rollout/train/eval use
  `NPA_SIM2REAL_INLINE_TASK=1` inside the workflow-owned GPU task.
- The Cosmos3 evaluator uses `nvidia/Cosmos3-Super-Reasoner` through Token Factory,
  receives `NEBIUS_TOKEN_FACTORY_KEY` by secret name, requests no GPU, and
  publishes evaluator tokens/latency/retries/request IDs/cost separately from
  model-agent accounting. Stage 9 must reject missing, duplicate, or extra
  evaluations by comparing that single result with the authoritative Stage 7
  rollout set before PPO or checkpoint selection.
- Train, validation, and gold digests are disjoint. Checkpoint ranking reads
  validation only; Stage 10 reads gold only and preserves exact render lineage.
- Temporal rewards remain bounded and simulator-grounded. Strict success remains
  stable placement within 5 cm. Pipeline success and policy quality are distinct.
- Final RRD/MCAP use configured capture FPS and contain non-empty multi-camera,
  progress, policy, and evaluation evidence.

## Verify

Validate and plan the canonical YAML, run the focused compositional workflow and
Isaac payload tests, then run repository-required lint/type/docs/guardrails. A
live proof must use the same standard `workflow submit --runtime` path.
