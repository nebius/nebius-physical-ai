---
name: sim2real-engine
description: Use when navigating, reviewing, or changing the Sim2Real staged pipeline engine — stage map, preamble/inner/outer/finalize entrypoints, K8s sibling jobs, and S3 artifact contracts.
---

# Sim2Real Engine

## When To Use

Load this skill when you need the **canonical 14-stage map** for the VLM-to-RL
Sim2Real loop, not the older generic sim-to-real workflow YAMLs. Use it for
engine edits, monitor/status alignment, walkthrough docs, and debugging sibling
K8s jobs (`s2r-*`).

## Source Files

| File | Role |
| --- | --- |
| `npa/src/npa/workflows/sim2real/engine.py` | Stage glue, K8s siblings, inner/outer/finalize |
| `npa/src/npa/workflows/sim2real_stages.py` | Mandatory preamble stages 3–7 helpers |
| `npa/src/npa/workflows/sim2real/monitor.py` | `_STAGE_SPECS` S3 marker rules for live status |
| `npa/src/npa/workflows/sim2real/runner.py` | `Sim2RealWorkflow` orchestration CLI entry |
| `npa/src/npa/workflows/sim2real_assets.py` | Stage 2 assets consumption |

## Stage Map

| Stage | Monitor name | Entrypoint | Primary artifacts |
| --- | --- | --- | --- |
| 1 | `stage_01_trigger` | `run_preamble` | `stage_01_trigger/trigger.json` |
| 2 | `stage_02_assets` | `run_preamble` → `run_assets_stage` | `stage_02_assets/consumed_*_spec.json` |
| 3 | `stage_03_augment` | `run_preamble` → `run_augment_stage` | `augment/manifest.json` |
| 4 | `stage_04_envs_raw` | `run_envgen_split_stage` | `envs/raw/` |
| 5 | `stage_05_envs_train` | `run_envgen_split_stage` | `envs/train/envs.jsonl` |
| 6 | `stage_06_tokens` | `run_envgen_split_stage` | `tokens/manifest.json` |
| 7 | `stage_07_actions_train` | `run_inner_loop` → `run_policy_rollouts` | `actions/train/` |
| 8 | `stage_08_vlm_eval_train` | `run_inner_loop` → `evaluate_rollout_with_vlm` | `vlm_eval/train/` |
| 9 | `stage_09_training_signal` | `run_inner_loop` (signal + trainer) | `training_signal/train/` |
| 10 | `stage_10_eval_heldout` | `run_single_outer_iteration` → `run_heldout_eval` | `eval/heldout/report.json` |
| 11 | `stage_11_outer_loop` | `run_single_outer_iteration` → `threshold_decision` | `outer_loop/decision.json` |
| 12 | `stage_12_external_validation_stub` | `run_finalize` | `stage_12_external_validation/external_stub.json` |
| 13 | `stage_13_retrigger` | `run_finalize` | `stage_13_retrigger/retrigger.json` |
| 14 | `stage_14_rerun_viz` | `run_finalize` → `_run_sim2real_viz_stage` | `reports/sim2real.rrd` (+ `reports/sim2real.mcap`) |

## Phase Boundaries

- **Preamble (1–6):** `run_preamble` — trigger, assets, augment, envgen split, tokens.
- **Outer iteration (7–11):** `run_single_outer_iteration` — one inner loop (7–9),
  held-out eval (10), threshold decision (11). Repeats for `outer_iterations`.
- **Finalize (12–14 + report):** `run_finalize` — external-validation stub,
  loop-of-loops retrigger record, local Rerun `.rrd` plus a Lichtblick/Foxglove
  `.mcap` of the same rollout data (`NPA_SIM2REAL_MCAP`, default on when rerun is
  on), `sim2real-report.json`, optional S3 upload.

Stages 4–6 share one component name in monitor: `stage_04_06_env_gen_split_tokens`.

## Gotchas

- Monitor infers stage 01 from later artifacts (`infer_from_later=True`); early
  stages may show `PENDING` while later stages succeed — trust `current_stage`.
- K8s sibling job names embed only the first 22 characters of `run_id`.
- Registry-qualified images gate K8s execution; placeholders fall back to local
  reference paths (SEAM tier in component records).
- Do not hardcode bucket names, tenant IDs, or registry paths in engine code.

## Verify

```bash
npa/.venv/bin/ruff check npa/src/npa/workflows/sim2real/engine.py
npa/.venv/bin/python -m pytest npa/tests/workflows/sim2real/ -q --tb=no
```
