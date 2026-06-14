---
name: compose-cloud-tokenfactory
description: Use when composing NPA workbench pipelines that pair Nebius AI Cloud compute (serverless GPU, Kubernetes, or VM) with hosted Nebius Token Factory inference, or when a user asks how to get the AI Cloud + Token Factory tokens and chain a GPU stage into a hosted text/vision/reasoning stage.
---

# Composing Nebius cloud + Token Factory pipelines

Combo pipelines pair **real Nebius compute** (GPU/sim work) with **hosted Token
Factory inference** (zero-GPU). Full guide for users:
`docs/workbench/composing-cloud-and-token-factory.md`. This skill is the agent
fast-path.

## The one contract

```
[ Nebius compute stage ] --writes--> [ S3 ] --reads--> [ Token Factory stage ]
```

Compute stage (GPU/VM/k8s) uploads artifacts to an `s3://` `--output-path`; the
Token Factory stage (`caption`/`generate`/`reason`/`vlm-eval --backend api`)
reads that URI and calls the hosted API. Glue = S3 URIs + two credentials.

## Two tokens (they are different keys)

- **AI Cloud** (compute + storage): `nebius profile create` →
  `nebius iam get-access-token`; S3 keys under `storage:` in
  `~/.npa/credentials.yaml`; project ID passed as `--project-id` (never
  hardcoded). Verify: `nebius iam get-access-token >/dev/null`.
- **Token Factory** (hosted inference): mint at <https://tokenfactory.nebius.com/>,
  set `NEBIUS_API_KEY` (via `npa configure`, env, or `tokens:` in credentials).
  Verify: `npa workbench token-factory verify`. Registration detail:
  `docs/workbench/token-factory.md`.

## Two composition styles

1. **Python runner** (fan-out / branching / local glue): shell out to `npa` for
   the GPU stage, call Token Factory tool functions in-process. Keep pure logic
   in `npa/src/npa/workflows/token_factory_combos.py` (unit-tested) and I/O in
   the runner. Always add `--render-only` (no infra) and a cheap no-GPU mode.
   Examples: `run_tokenfactory_train_triage.py`, `run_tokenfactory_sim_sweep.py`.
2. **SkyPilot serial YAML** (clean hand-off): `execution: serial`, one doc per
   stage; GPU stage requests `accelerators`, hosted stage omits them. Pass `s3://`
   URIs via `envs`; fail fast if `NEBIUS_API_KEY` is unset; launch with
   `--secret NEBIUS_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY`.
   Examples: `tokenfactory-rollout-judge.yaml`,
   `tokenfactory-scene-to-rollout-judge.yaml`.

## Reusable pure helpers (token_factory_combos.py)

`summarize_run_artifacts` (bounded, binary-skipping digest), prompt builders
(`build_triage_prompt`, `build_ranking_prompt`, `build_sweep_design_prompt`),
`join_uri` / `sweep_variant_output_uri`, ID/job-name derivation (`utc_stamp`,
`triage_job_name`, `sweep_variants`). Add new pure logic here, not in the runner.

## Rules

- No hardcoded project/registry/bucket IDs — flags, `--var`, or `--secret` only.
- Token Factory stages are CPU-only: no `accelerators`, no vLLM serving.
- Unit tests stay infra-free; mock S3/Nebius/GPU and never call the live API.
  Live checks go in `npa/tests/e2e/` behind a key, or via the runner's
  `--render-only` / no-GPU modes.
