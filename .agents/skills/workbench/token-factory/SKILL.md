---
name: token-factory
description: Use when working on Nebius Token Factory native workflows in NPA — the OpenAI-compatible hosted-inference client, the token-factory workbench tool (caption/generate), the vlm-eval api backend, or the zero-GPU Token Factory SkyPilot workflows.
---

# Nebius Token Factory

Nebius Token Factory is an OpenAI-compatible hosted-inference API for open text
and vision models. Because inference is hosted, Token Factory workflows are
**zero-GPU**: they run on CPU and call the API. Base URL defaults to
`https://api.tokenfactory.nebius.com/v1/`; auth is the `NEBIUS_API_KEY`
environment variable.

## Single Source Of Truth

All endpoint, auth, and request logic lives in
`npa/src/npa/clients/token_factory.py` (`TokenFactoryClient`, `resolve_config`).
Do not re-derive the base URL or read `NEBIUS_API_KEY` elsewhere; call the
client. Base URL overrides: `NEBIUS_TOKEN_FACTORY_BASE_URL` or
`NEBIUS_BASE_URL`.

`NEBIUS_API_KEY` is a first-class credential: `npa configure` prompts for it,
or put it under `tokens:` in `~/.npa/credentials.yaml`. It is injected into every
workbench/workflow run via `shared_credential_env`. `npa workbench token-factory
verify` does a live authenticated `list_models` call (exits non-zero on
auth/connectivity failure); `status` reports the resolved key/base URL with no
network call. User-facing setup guide: `docs/workbench/token-factory.md`.

## token-factory Workbench Tool

`npa workbench token-factory ...`, behavior in
`npa/src/npa/workbench/token_factory/__init__.py`. Like every tool it uses
`--input-path` / `--output-path` S3 URIs.

- `caption`: caption images / rollout frames with a hosted vision model →
  `captions.json`.
- `generate`: batch text generation from a JSONL/text prompt file →
  `generations.jsonl` (synthetic task instructions, Cosmos scene prompts, sim
  variation).
- `reason`: physical-AI scene reasoning with `nvidia/Cosmos3-Super-Reasoner`
  (default) → `scene_reasoning.json`. Scene images + a task in, scene
  understanding + plan of action out. Backs the physical-common-sense challenge.
- `models`: list models available to the key. `verify`: live authenticated
  models call (exits non-zero on failure). `status` / `list`: observability.

Mocked tests inject a `TokenFactoryClient` built on `httpx.MockTransport`; never
hit the live API. CLI tests monkeypatch `token_factory._default_client`.

## Live Testing (first-class)

Live tests live in `npa/tests/e2e/test_token_factory_e2e.py`, marked
`token_factory_e2e`, and self-skip without a key. The marker is in
`_LIVE_MARKERS` so the conftest credential scrub does not strip `NEBIUS_API_KEY`
for them. Run with `NEBIUS_API_KEY=... pytest npa/tests/e2e/test_token_factory_e2e.py`.
Cosmos model availability is key-dependent; confirm with
`npa workbench token-factory models`.

## vlm-eval Over Token Factory

The vlm-eval `api` backend defaults its base URL to Token Factory and accepts
`NEBIUS_API_KEY` (falling back to `OPENAI_API_KEY`). This gives zero-GPU rollout
scoring with no vLLM serving stage.

## Workflows

Zero-GPU workflows in `npa/workflows/workbench/skypilot/` (CPU-only,
`cloud: kubernetes`, no `accelerators`):

- `token-factory-caption.yaml`
- `token-factory-generate.yaml`
- `token-factory-cosmos-reason.yaml`
- `vlm-eval-token-factory.yaml`

Pass the key as a SkyPilot secret at launch:
`sky jobs launch --secret NEBIUS_API_KEY --secret AWS_ACCESS_KEY_ID --secret AWS_SECRET_ACCESS_KEY <yaml>`.
Each `run` block fails fast if `NEBIUS_API_KEY` is unset.

## Compute Combos (Nebius GPU + Token Factory)

Two workflows deliberately pair **real Nebius GPU compute** with the hosted
Token Factory stage, so the pipeline exercises both. Pure logic lives in
`npa/src/npa/workflows/token_factory_combos.py` (infra-free, unit-tested);
network/storage calls live in the runner and existing tool modules.

- `npa/scripts/run_tokenfactory_train_triage.py` (**serverless**): a LeRobot
  serverless GPU Job writes run artifacts to S3, then `token-factory generate`
  has a text model write a triage report. Needs `--project-id` + `--output-path`
  unless the workbench config supplies a project and
  `storage.checkpoint_bucket`. `--render-only` previews with no infra;
  `--from-output-path` triages an existing prefix without launching a GPU Job.
- `npa/scripts/run_tokenfactory_sim_sweep.py` (**serverless fan-out**): a text
  model designs the sweep, a deterministic `--steps` grid launches one LeRobot
  serverless GPU train per variant, then a text model ranks the completed runs.
  `--render-only` previews; `--rank-existing a,b` ranks existing prefixes with no
  GPU spend. `lerobot train` has no `--seed`, so the grid varies `--steps`.
- `npa/workflows/workbench/skypilot/tokenfactory-rollout-judge.yaml`
  (**kubernetes**): stage 1 renders a `lerobot-eval` rollout on a k8s GPU and
  uploads videos to S3; stage 2 is CPU `vlm-eval --backend api` judging via a
  hosted VLM. Two serial docs, two images (lerobot GPU, then token-factory CPU).
- `npa/workflows/workbench/skypilot/tokenfactory-scene-to-rollout-judge.yaml`
  (**kubernetes**): three serial stages — `token-factory reason` over scene
  images (CPU) → `lerobot-eval` rollout (k8s GPU) → `vlm-eval --backend api`
  judging the rollout against the reasoner's plan (CPU). The hackathon
  physical-common-sense loop end to end.

Guide: `docs/workbench/cookbooks/tokenfactory-compute-combos.md`. To compose new
combos (the contract, both tokens, the two styles), see
`docs/workbench/composing-cloud-and-token-factory.md` and the
`compose-cloud-tokenfactory` skill. All combos are smoke-sized to stay cheap.
The runners export `~/.npa/credentials.yaml` into the environment themselves
because they are launched as plain scripts, not via the CLI.
