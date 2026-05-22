# Nebius Physical AI Workbench — YAML Pipeline Guide

> Living document. Updated as new pipeline patterns are introduced.
> Last updated: 2026-05-20

## Overview

A Workbench pipeline is a SkyPilot multi-document YAML file. Each document defines
one task: the compute it needs, the environment it runs in, and the command it
executes. Tasks call workbench tool HTTP endpoints via `curl`. SkyPilot
orchestrates dependencies and schedules tasks on the Nebius MK8s cluster.

The reference pipeline is `npa/workflows/skypilot/bdd100k-pipeline.yaml`.

## Pipeline Structure

The BDD100K reference file starts with a workflow document:

```yaml
name: bdd100k-pipeline
execution: serial
```

Each subsequent `---` document is one SkyPilot task. The fields used by the
current pipeline are:

- `name`: unique task name, for example `bdd100k-ingest`.
- `resources`: Kubernetes scheduling and container settings.
- `envs`: environment variables injected into the task shell.
- `setup`: package bootstrap, currently used to ensure `curl` and `jq` exist.
- `run`: the shell command for the task.

Minimal single-task shape:

```yaml
---
name: example-workbench-task
resources:
  cloud: kubernetes
  cpus: 4
  memory: 16
  image_id: "docker:<registry>/<image>:<tag>"
envs:
  LANCEDB_ENDPOINT: http://npa-lancedb.workbench.svc.cluster.local:8686
run: |
  set -euo pipefail
  curl -fsS "${LANCEDB_ENDPOINT}/health"
```

SkyPilot 0.12.2 does not expand same-block environment variables inside
`image_id`. Keep committed YAMLs on explicit placeholders such as:

```yaml
# Replace <your-registry-id> with your Nebius container registry ID.
# Replace <image-tag> with the current tool image tag.
image_id: "docker:cr.eu-north1.nebius.cloud/<your-registry-id>/<image>:<image-tag>"
```

Use `npa workbench lancedb system-info` or the corresponding workbench tool's
`system-info` command to find the current image tag for live submissions.

The BDD100K pipeline currently runs serially because the checked-in comment notes
that SkyPilot `0.12.2` supports serial pipelines and all-parallel job groups, but
not the mixed dependency graph needed to train and evaluate the three views in
parallel after shared upstream stages.

## Resources

All committed BDD100K tasks use:

```yaml
resources:
  cloud: kubernetes
```

CPU-only stages:

- Stage 1, `bdd100k-ingest`: imports source or synthetic rows into LanceDB.
- Stage 2, `bdd100k-backfill-cpu`: computes `has_person`, `has_rider`,
  `person_bbox_area_pct`, `dhash`, and `is_duplicate`.
- Stage 4, `bdd100k-create-mvs`: creates the three materialized views.

GPU stages:

- Stage 3, `bdd100k-backfill-clip`: uses `accelerators: H100:1` for CLIP image
  embeddings.
- Stage 5, `bdd100k-train-*`: uses `accelerators: H100:1` for Faster R-CNN
  training.
- Stage 6, `bdd100k-eval-*`: uses `accelerators: H100:1` for checkpoint
  evaluation.

The detection-training deploy path uses the H100 Kubernetes node selector value
`gpu-h100-sxm`. The pipeline YAML itself requests GPUs with SkyPilot's
`accelerators: H100:1` field.

## Environment Variables (`envs`)

The `envs` block is the task contract between the rendered pipeline and the
shell in `run`. Values are referenced as shell variables with `${VAR}` and are
passed into JSON request bodies with `jq`.

Common run-scoped variables in the BDD100K pipeline:

- `NPA_PIPELINE_RUN_ID`: logical run ID.
- `S3_BUCKET`: artifact bucket.
- `S3_PREFIX`: per-run prefix under the bucket.
- `PIPELINE_ROOT_URI`: full `s3://` root for this run.
- `LANCE_URI`: per-run LanceDB URI.
- `LANCEDB_ENDPOINT`: LanceDB service URL.
- `DETECTION_TRAINING_ENDPOINT`: detection-training service URL.

Example JSON payload pattern:

```bash
payload=$(jq -n \
  --arg table "${LANCE_TABLE}" \
  --arg lance_uri "${LANCE_URI}" \
  --argjson batch_size "${CPU_BACKFILL_BATCH_SIZE}" \
  '{table: $table, lance_uri: $lance_uri, batch_size: $batch_size}')
```

Use `--arg` for strings and `--argjson` for values that should remain JSON
numbers, objects, arrays, booleans, or null.

### Label Map Injection (BDD100K Pattern)

Workbench tools that operate on labeled data can accept a `label_map` parameter
to translate string category names to integer IDs. The map is injected as a JSON
environment variable in the pipeline YAML and passed in the `curl` POST body.

Why this pattern:

- Training tools are dataset-agnostic; they do not hardcode any category schema.
- Dataset-specific configuration belongs in the pipeline YAML, not in the tool.
- Any dataset can be supported by injecting its own label map.

The committed BDD100K pipeline keeps the synthetic map active because the
checked-in mock and synthetic workflows emit those names:

```yaml
envs:
  # Synthetic BDD100K data - category names match the synthetic data generator.
  BDD100K_LABEL_MAP: '{"person":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motor":6,"bike":7,"traffic light":8,"traffic sign":9}'
  # Real BDD100K data - uncomment the line below and comment the line above.
  # BDD100K_LABEL_MAP: '{"pedestrian":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motorcycle":6,"bicycle":7,"traffic light":8,"traffic sign":9}'
```

Use the synthetic map for runs that import generated rows with
`BDD100K_SYNTHETIC_ROWS` or the runner's `--synthetic` flag. Use the real map for
runs that import BDD100K label files from `BDD100K_SOURCE_URI`.

The category IDs stay stable between the two maps, but three category names
differ:

| ID | Synthetic category | Real BDD100K category |
|---:|---|---|
| 0 | `person` | `pedestrian` |
| 6 | `motor` | `motorcycle` |
| 7 | `bike` | `bicycle` |

To switch a production run to real BDD100K labels, update each training task's
`envs` block by commenting the synthetic `BDD100K_LABEL_MAP` line and
uncommenting the real BDD100K line. SkyPilot `0.12.2` does not support
self-referencing interpolation inside `envs`, so the pipeline uses explicit
comment blocks instead of deriving one map variable from another.

In the `run` block:

```bash
payload=$(jq -n \
  --arg view "${VIEW_NAME}" \
  --arg lance_uri "${LANCE_URI}" \
  --arg output_uri "${TRAIN_OUTPUT_URI}" \
  --argjson label_map "${BDD100K_LABEL_MAP}" \
  --argjson epochs "${TRAIN_EPOCHS}" \
  --argjson batch_size "${TRAIN_BATCH_SIZE}" \
  --argjson learning_rate "${TRAIN_LEARNING_RATE}" \
  '{view: $view, lance_uri: $lance_uri, output_uri: $output_uri, label_map: $label_map, epochs: $epochs, batch_size: $batch_size, learning_rate: $learning_rate}')
```

`num_classes` is auto-inferred from `len(label_map) + 1`; the extra class is the
background class. Do not pass `num_classes` manually unless overriding the
inferred value.

Extending to other datasets: replace `BDD100K_LABEL_MAP` with that dataset's
category-to-integer mapping. The detection-training tool accepts any
`label_map`; it is not BDD100K-specific.

## Service Endpoints

The pipeline uses cluster-internal Kubernetes DNS:

```text
http://<service-name>.workbench.svc.cluster.local:<port>
```

Services used by the BDD100K reference pipeline:

| Tool | Service | Port | Endpoints Used |
|---|---|---:|---|
| LanceDB | `npa-lancedb` | `8686` | `GET /health`, `POST /import-bdd100k`, `POST /backfill`, `POST /create-mv` |
| Detection training | `npa-detection-training` | `8790` | `GET /health`, `POST /train`, `GET /status`, `GET /runs`, `POST /eval` |

The `/train` request schema accepts:

- `view`: Lance materialized view name.
- `lance_uri`: LanceDB URI.
- `output_uri`: checkpoint and metrics output URI.
- `label_map`: optional string-label-to-integer mapping.
- `num_classes`: optional manual class count override.
- `epochs`, `batch_size`, `learning_rate`: training hyperparameters.
- `validation_filter_sql`: optional validation filter, currently not used by the
  committed BDD100K pipeline.

## S3 Artifact Paths

The runner renders per-run paths before submission. The convention is:

```text
s3://<bucket>/bdd100k-pipeline/<run-id>/
```

With `NPA_S3_BUCKET=your-bucket-name` and a run ID of `example-run`:

```text
s3://${NPA_S3_BUCKET}/bdd100k-pipeline/example-run/
```

Derived paths:

- LanceDB: `${PIPELINE_ROOT_URI}/lancedb/`
- Training: `${PIPELINE_ROOT_URI}/training/${VIEW_SLUG}`
- Evaluation: `${PIPELINE_ROOT_URI}/eval/${VIEW_SLUG}`

`npa/scripts/run_bdd100k_pipeline.py` renders these values into each task's
`envs` block. Cleanup is controlled by the runner's `--cleanup` flag, which calls
the SkyPilot cleanup path for the run after terminal workflow status.

## Standard Pipeline Stages (BDD100K Reference)

The BDD100K pipeline is the canonical reference implementation:

1. `bdd100k-ingest`: imports BDD100K source data or synthetic rows into LanceDB.
2. `bdd100k-backfill-cpu`: computes CPU UDF columns needed by later filters.
3. `bdd100k-backfill-clip`: computes CLIP embeddings with the GPU UDF path.
4. `bdd100k-create-mvs`: creates `bdd100k_rider_train`,
   `bdd100k_nighttime_person_train`, and `bdd100k_distant_person_train`.
5. `bdd100k-train-rider`, `bdd100k-train-nighttime`, `bdd100k-train-distant`:
   train Faster R-CNN models from the three views.
6. `bdd100k-eval-rider`, `bdd100k-eval-nighttime`, `bdd100k-eval-distant`:
   evaluate the latest completed training run for each view.

Related docs:

- `docs/getting-started.md`
- `docs/demos/bdd100k-lancedb-demo.md`
- `docs/cookbooks/bdd100k-pipeline.md`
- `npa/workflows/skypilot/bdd100k-pipeline.yaml`

## Isaac Lab RL Training

Isaac Lab RL jobs are batch training workloads, not persistent service calls.
Use the committed SkyPilot consumers:

- `npa/workflows/skypilot/isaac-lab-rl-train.yaml` for one RSL-RL training job.
- `npa/workflows/skypilot/isaac-lab-rl-sweep.yaml` for an all-parallel sweep.
- `npa/scripts/run_isaac_lab_rl.py` to render per-run values and submit.

Single run:

```bash
export NPA_S3_BUCKET=your-bucket-name
python npa/scripts/run_isaac_lab_rl.py \
  --yaml npa/workflows/skypilot/isaac-lab-rl-train.yaml \
  --task Isaac-Cartpole-v0 \
  --iterations 10 \
  --run-id isaac-cartpole-smoke
```

The training command uses the Isaac Lab RSL-RL entry point:

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${ISAAC_LAB_TASK}" \
  --num_envs "${ISAAC_LAB_NUM_ENVS}" \
  --max_iterations "${ISAAC_LAB_ITERATIONS}" \
  --headless \
  --experiment_name "${ISAAC_LAB_EXPERIMENT_NAME}" \
  --run_name "${ISAAC_LAB_RUN_NAME}" \
  agent.save_interval=1
```

Parameter sweep:

```bash
python npa/scripts/run_isaac_lab_rl.py \
  --yaml npa/workflows/skypilot/isaac-lab-rl-sweep.yaml \
  --task Isaac-Cartpole-v0 \
  --iterations 10 \
  --run-id isaac-cartpole-sweep
```

The sweep YAML uses `execution: parallel`, which is the SkyPilot 0.12.2 pattern
for independent parallel tasks. It avoids a mixed dependency graph and writes
each variant under:

```text
s3://<bucket>/isaac-lab-rl/<run-id>/<variant>/
```

Isaac Lab requires RT-core GPUs for simulation. The YAMLs request:

```yaml
resources:
  cloud: kubernetes
  accelerators: L40S:1
```

Use L40S first. RTX Pro 6000 is the fallback when exposed in the Kubernetes GPU
catalog. Do not run Isaac Lab on H100 or H200 for these jobs; those accelerators
do not provide the RT cores required by Isaac Sim rendering/simulation paths.

Custom Isaac Lab forks can be layered by overriding the image in the YAML:

```yaml
resources:
  image_id: "docker:cr.eu-north1.nebius.cloud/<registry>/flexion-isaac-lab:<tag>"
```

The replacement image must keep the Isaac Lab source tree at
`/workspace/isaaclab` or provide the same
`scripts/reinforcement_learning/rsl_rl/train.py` entry point. The runner also
accepts `--image cr.../custom-isaac-lab:<tag>` to rewrite `image_id` in the
rendered workflow.

## Adding a New Pipeline

Use the Isaac Lab and BDD100K YAMLs as the current reference patterns.

Current minimum pattern:

- Start from a multi-document SkyPilot YAML file.
- Add a workflow document with `name` and `execution`.
- Add one task document per stage.
- Use `resources.cloud: kubernetes`.
- Put per-run paths and service URLs in `envs`.
- Build HTTP request bodies with `jq` in `run`.
- Validate tool health with `/health` before making state-changing requests.
- Add a mock-endpoint or render-only validation path before live submission.

New workbench tool endpoints should be documented here only after the endpoint is
present in committed source.

## Changelog

| Date | Change | Run |
|---|---|---|
| 2026-05-20 | Added Isaac Lab RSL-RL single-job and parallel sweep SkyPilot YAML patterns. | W9-isaac-lab-e2e-fix |
| 2026-05-16 | Initial guide. Label map injection pattern (BDD100K). | W9-label-schema-fix |
