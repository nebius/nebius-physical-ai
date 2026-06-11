# Token Factory + Nebius compute combos

Most Token Factory workflows are **zero-GPU** — they only call the hosted API.
These two **combo** workflows are different: each pairs *real Nebius cloud
compute* (a GPU job) with *hosted Token Factory inference*. The GPU stage
produces artifacts; the hosted, zero-GPU stage reasons over them. They were
built for the hackathon to show Token Factory working alongside Nebius compute,
not just on its own.

| Workflow | Nebius compute | Token Factory stage | Entry point |
| --- | --- | --- | --- |
| **train-triage** | Serverless GPU Job (LeRobot train) | Text model writes a triage report from the run's artifacts | [`run_tokenfactory_train_triage.py`](../../../npa/scripts/run_tokenfactory_train_triage.py) |
| **rollout-judge** | Managed Kubernetes GPU (LeRobot eval rollout) | Hosted VLM scores the rollout (`vlm-eval --backend api`) | [`tokenfactory-rollout-judge.yaml`](../../../npa/workflows/workbench/skypilot/tokenfactory-rollout-judge.yaml) |

Both are intentionally **smoke-sized** so they are cheap to run end-to-end.

## Prerequisites

- A Token Factory API key in `NEBIUS_API_KEY` (see
  [token-factory.md](../token-factory.md) to register and mint one).
- Nebius cloud credentials for compute + storage. The serverless path needs a
  Nebius **project ID** and an S3 **bucket** you can write to; the Kubernetes
  path needs SkyPilot bootstrapped (`npa skypilot bootstrap`).
- Object-storage credentials in `~/.npa/credentials.yaml` (the runner exports
  them for you; SkyPilot reads them as `--secret`s).

Never hardcode project, registry, or bucket IDs in committed files — pass them
at launch via flags, `--var`, or SkyPilot secrets.

## 1. train-triage (serverless GPU → Token Factory report)

A LeRobot **serverless GPU Job** trains a policy (smoke settings by default) and
uploads its run artifacts (configs, logs, metrics) to S3. A Token Factory text
model then reads those artifacts and writes a triage + next-steps report next to
the run.

```bash
# No-infrastructure preview of exactly what will run.
python npa/scripts/run_tokenfactory_train_triage.py --render-only

# Full live run: serverless GPU smoke train, then Token Factory triage.
# --project-id and --output-path are required unless your workbench config
# already provides a project and storage.checkpoint_bucket.
NEBIUS_API_KEY=... python npa/scripts/run_tokenfactory_train_triage.py \
  --project-id project-xxxxxxxx \
  --output-path s3://your-bucket/tf-triage/<run-id>/ \
  --gpu-type h200

# Cheap iteration: skip the GPU stage and only triage an existing run prefix.
NEBIUS_API_KEY=... python npa/scripts/run_tokenfactory_train_triage.py \
  --from-output-path s3://your-bucket/lerobot-serverless-test/<ts>/
```

Output: a `generations.jsonl` triage report under `<artifacts>/triage/` (or
`--triage-root`). Choose the triage model with `--model` (default
`meta-llama/Llama-3.3-70B-Instruct`; `nvidia/Cosmos3-Super-Reasoner` also works).

## 2. rollout-judge (Kubernetes GPU → Token Factory VLM judge)

A two-stage serial SkyPilot pipeline. Stage 1 runs `lerobot-eval` on a Nebius
**Managed Kubernetes GPU**, renders rollout videos, and uploads them to S3.
Stage 2 is zero-GPU: `vlm-eval --backend api` scores the rollout with a hosted
Token Factory VLM — no local vLLM serving stage.

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"

npa workbench workflow submit \
  npa/workflows/workbench/skypilot/tokenfactory-rollout-judge.yaml \
  --run-id rollout-judge \
  --var NPA_LEROBOT_IMAGE=cr.eu-north1.nebius.cloud/<registry>/npa-lerobot:0.5.1 \
  --var NPA_TOKEN_FACTORY_IMAGE=cr.eu-north1.nebius.cloud/<registry>/npa-cosmos:1.0.9 \
  --var ROLLOUTS_URI=s3://your-bucket/tokenfactory/<run-id>/rollouts/ \
  --var JUDGE_URI=s3://your-bucket/tokenfactory/<run-id>/vlm-judge/
```

Pass the key and storage creds as secrets when launching the YAML directly:

```bash
sky jobs launch --secret NEBIUS_API_KEY --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/skypilot/tokenfactory-rollout-judge.yaml
```

Output: a `vlm-eval` task-success report under `JUDGE_URI` with per-rollout
`{passed, score, rationale}`. The default rollout uses the public
`lerobot/diffusion_pusht` checkpoint on the `pusht` environment, so it runs out
of the box; swap `CHECKPOINT`/`ENV_TYPE` for your own policy.

## Why these are "combo" workflows

The other Token Factory workflows (`token-factory-caption`,
`token-factory-generate`, `token-factory-cosmos-reason`,
`vlm-eval-token-factory`) are CPU-only and only call the hosted API. These two
deliberately put a real Nebius GPU stage in front of the hosted stage so the
pipeline exercises **both** Nebius cloud compute and Token Factory in one run.
```
