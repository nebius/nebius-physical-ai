---
name: sim2real-engine
description: Use when navigating, reviewing, or changing the compositional Sim2Real workflow, stateless stage adapters, durable standard-runtime resume, ComponentRecords, and S3 lineage.
---

# Compositional Sim2Real engine

The canonical engine surface is the ordinary workflow
`workflows/main/sim2real.yaml`. Its leaf states call
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
| 8 | `stage_08_vlm_eval_train` | single hosted Token Factory CPU evaluator |
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
- The hosted evaluator uses `MiniMaxAI/MiniMax-M3` through Token Factory,
  receives `NEBIUS_TOKEN_FACTORY_KEY` by secret name, requests no GPU, and
  publishes evaluator tokens/latency/retries/request IDs/cost separately from
  model-agent accounting. Stage 9 must reject missing, duplicate, or extra
  evaluations or a mismatched configured model/family by comparing that single result with the authoritative Stage 7
  rollout set before PPO or checkpoint selection.
- The canonical Stage 8/9 interface uses `token_factory_model`,
  `token-factory.json`, and `npa.sim2real.token_factory_evaluator.v1`; it has no
  canonical Cosmos3 reasoning selector. Only the archived legacy loop retains
  its source-supported `vlm_cosmos3_model` and `vlm_eval_cosmos3` compatibility
  interface. Changing the canonical model or evaluator contract requires a new run.
- Train, validation, and gold digests are disjoint. After the full PPO pass,
  checkpoint ranking evaluates every saved 100-update checkpoint and the final
  checkpoint on the same validation split. Preserve actual checkpoint update
  indices alongside full pass telemetry. Retry requires the complete, exact
  committed candidate pool. Ranking reads validation only; Stage 10 reads gold only and preserves exact render lineage.
- Native stock-Franka scenarios share the attested task-space control profile:
  upstream arm gains and gravity compensation, 200 Hz physics and 50 Hz policy
  actions. Training demonstrations initialize the actor; validation and Gold
  execute that learned actor alone. Physics refinement preserves episode duration
  and PPO update counts, with the strict placement predicate retained.
- Stock scenario policies share `stock-franka-cartesian-gru-v2`: native absolute
  TCP-pose actions, measured geometry, initialized observation normalization,
  a GRU, and the real reset clock saturated at five seconds. Fresh training uses
  `franka-cartesian-gru-distillation-v2`, retains at least three 32-epoch demonstration
  fits, and repeats corrective fitting until a strict actor-only handoff across
  the complete training split passes before PPO. Archive every assessment.
  A weak initial mixed collection triggers another guarded teacher-only fit and
  a new mixed collection. Both accepted warmup rounds retain the half-split
  placement guard; archive every failed collection and recovery fit.
  Freeze trained observation statistics after the successful actor-only handoff.
  Stock demonstration and resumed transport/convergence passes default to fixed
  PPO updates at `0.00001`, zero entropy, and action noise `0.005`, while explicit
  operator overrides remain effective. Every checkpoint binds the actual
  normalization count and update bound; exact loads restore that bound.
  Expert phases only weight training loss. Every periodic and final checkpoint
  binds the policy semantics; all native inference paths reset
  recurrent state on actual `dones`. Validation retains the first native
  checkpoint, including index zero with one completed update, as well as the
  configured periodic and final checkpoints. Keep the full PPO count and held-out gate.
- Temporal rewards remain bounded and simulator-grounded. Strict success remains
  stable placement within 5 cm. Pipeline success and policy quality are distinct.
- Native rollout actions preserve every reset between sparse samples using
  `episode_boundary`; cameras separately identify the physical simulator episode.
  Hosted v5 evaluations reject missing or contradictory boundaries. Reset-crossing
  intervals remain archived but are excluded from rewards, advantages, fallback,
  action credit, and trainer statistics. Never relabel an archived v4 run as v5.
- Final RRD/MCAP use configured capture FPS and contain non-empty multi-camera,
  progress, policy, and evaluation evidence.

## Verify

Validate and plan the canonical YAML, run the focused compositional workflow and
Isaac payload tests, then run repository-required lint/type/docs/guardrails. A
live proof must use the same standard `workflow submit --runtime` path.
