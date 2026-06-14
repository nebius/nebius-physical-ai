# Token Factory + Nebius compute combos

Most Token Factory workflows are **zero-GPU** — they only call the hosted API.
These two **combo** workflows are different: each pairs *real Nebius cloud
compute* (a GPU job) with *hosted Token Factory inference*. The GPU stage
produces artifacts; the hosted, zero-GPU stage reasons over them. They show
Token Factory working alongside Nebius compute, not just on its own.

| Workflow | Nebius compute | Token Factory stage | Entry point |
| --- | --- | --- | --- |
| **train-triage** | Serverless GPU Job (LeRobot train) | Text model writes a triage report from the run's artifacts | [`run_tokenfactory_train_triage.py`](../../../npa/scripts/run_tokenfactory_train_triage.py) |
| **sim-sweep** | N serverless GPU Jobs (LeRobot train fan-out) | Text model designs the sweep, then ranks the runs | [`run_tokenfactory_sim_sweep.py`](../../../npa/scripts/run_tokenfactory_sim_sweep.py) |
| **rollout-judge** | Managed Kubernetes GPU (LeRobot eval rollout) | Hosted VLM scores the rollout (`vlm-eval --backend api`) | [`tokenfactory-rollout-judge.yaml`](../../../npa/workflows/workbench/skypilot/tokenfactory-rollout-judge.yaml) |
| **scene-to-rollout-judge** | Managed Kubernetes GPU (LeRobot eval rollout) | Reasoner extracts a plan, then a VLM judges the rollout against it | [`tokenfactory-scene-to-rollout-judge.yaml`](../../../npa/workflows/workbench/skypilot/tokenfactory-scene-to-rollout-judge.yaml) |

All are intentionally **smoke-sized** so they are cheap to run end-to-end. New to
composing these? Read
[composing-cloud-and-token-factory.md](../composing-cloud-and-token-factory.md)
first — it explains the contract, both tokens, and the two composition styles.

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

## 3. sim-sweep (Token Factory design → N serverless GPUs → Token Factory rank)

A fan-out sweep that uses Token Factory **twice** around a batch of Nebius GPU
jobs. A hosted text model writes a per-variant hypothesis; a deterministic grid
launches one LeRobot serverless GPU smoke train per variant (varying `--steps`,
the real comparable knob — `lerobot train` has no `--seed`); a hosted text model
then ranks the completed runs from their real artifacts and names a winner.

```bash
# No-infrastructure preview of design prompt + grid + per-variant commands.
python npa/scripts/run_tokenfactory_sim_sweep.py --render-only --num-variants 2

# Full live run: design -> N serverless GPU trains -> ranking.
NEBIUS_API_KEY=... python npa/scripts/run_tokenfactory_sim_sweep.py \
  --project-id project-xxxxxxxx \
  --bucket s3://your-bucket/tf-sim-sweep \
  --num-variants 2

# Cheap iteration: rank existing run prefixes (skips design + GPU stages).
NEBIUS_API_KEY=... python npa/scripts/run_tokenfactory_sim_sweep.py \
  --rank-existing s3://your-bucket/runA/,s3://your-bucket/runB/
```

Output: a `generations.jsonl` ranking report under `<sweep-root>/ranking/`, plus
the design notes under `<sweep-root>/design/` and per-variant artifacts under
`<sweep-root>/variants/<id>/`.

## 4. scene-to-rollout-judge (Token Factory reason → k8s GPU → Token Factory VLM)

The physical-common-sense loop as one serial SkyPilot pipeline. Stage
1 (zero-GPU) runs `token-factory reason` over scene images with
`nvidia/Cosmos3-Super-Reasoner` and writes a plan of action. Stage 2 rolls out a
policy on a Nebius **Managed Kubernetes GPU**. Stage 3 (zero-GPU) folds the
Stage 1 plan into the `vlm-eval` task and has a hosted VLM judge whether the
rollout accomplished the plan.

```bash
npa skypilot bootstrap

npa workbench workflow submit \
  npa/workflows/workbench/skypilot/tokenfactory-scene-to-rollout-judge.yaml \
  --run-id scene-judge \
  --var NPA_LEROBOT_IMAGE=cr.eu-north1.nebius.cloud/<registry>/npa-lerobot:0.5.1 \
  --var NPA_TOKEN_FACTORY_IMAGE=cr.eu-north1.nebius.cloud/<registry>/npa-cosmos:1.0.9 \
  --var SCENE_URI=s3://your-bucket/tokenfactory/<run-id>/scene/ \
  --var PLAN_URI=s3://your-bucket/tokenfactory/<run-id>/plan/ \
  --var ROLLOUTS_URI=s3://your-bucket/tokenfactory/<run-id>/rollouts/ \
  --var JUDGE_URI=s3://your-bucket/tokenfactory/<run-id>/vlm-judge/
```

Put a few scene images (jpg/png) under `SCENE_URI` first. Output: a
`scene_reasoning.json` plan under `PLAN_URI` and a per-rollout
`{passed, score, rationale}` report under `JUDGE_URI`.

## Why these are "combo" workflows

The other Token Factory workflows (`token-factory-caption`,
`token-factory-generate`, `token-factory-cosmos-reason`,
`vlm-eval-token-factory`) are CPU-only and only call the hosted API. These four
deliberately put a real Nebius GPU stage alongside the hosted stage(s) so the
pipeline exercises **both** Nebius cloud compute and Token Factory in one run.
See [composing-cloud-and-token-factory.md](../composing-cloud-and-token-factory.md)
to build your own.
```
