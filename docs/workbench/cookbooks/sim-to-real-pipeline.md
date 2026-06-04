# Sim-To-Real Pipeline Runbook

This is the customer runbook for the generic sim-to-real workflow. It extends
the existing cookbook path instead of creating a second overlapping guide.

The example uses the pinned public LeRobot dataset `lerobot/pusht` at revision
`7628202a2180972f291ba1bc6723834921e72c19`. The dataset is MIT licensed and has
vision, state, action, episode, frame, and timestamp fields. The default staged
copy is:

```text
s3://$NPA_S3_BUCKET/datasets/lerobot-pusht/
```

## Prerequisites

```bash
export NPA_SKYPILOT_BIN=/home/ubuntu/.npa/skypilot-venv/bin/sky
export S3_BUCKET=npa-sim2real-d87cf691
export NPA_S3_BUCKET="$S3_BUCKET"
export S3_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export AWS_ENDPOINT_URL="$S3_ENDPOINT_URL"
export NEBIUS_S3_ENDPOINT="$S3_ENDPOINT_URL"
export POLICY_IMAGE=npa-lerobot-policy:0.1.0
export NPA_GPU_TYPE=H100:1
export NPA_GPU_FAILOVER=H200:1,A100:1,L40S:1,RTX6000:1
export EVAL_BACKEND=state-success
export FEEDBACK_SOURCE=vlm
export FEEDBACK_TYPE=critique
export AWS_ACCESS_KEY_ID=<s3-access-key>
export AWS_SECRET_ACCESS_KEY=<s3-secret-key>
```

SkyPilot is installed outside the NPA venv. Use `$NPA_SKYPILOT_BIN`; do not rely
on `sky` being on `PATH`.

## Raw SkyPilot Path

The checked-in YAML is runnable without the NPA SDK or CLI. Launch it directly
with raw `sky` and override the image, bucket, endpoint, and run prefix at the
SkyPilot env layer:

```bash
RUN_ID=sim-to-real-example
"$NPA_SKYPILOT_BIN" launch \
  --yes \
  --cluster "s2r-${RUN_ID}" \
  --workdir . \
  --infra nebius/eu-north1 \
  --gpus "${NPA_GPU_TYPE}" \
  --env "NPA_SIM_TO_REAL_RUN_ID=${RUN_ID}" \
  --env "S3_ENDPOINT_URL=${S3_ENDPOINT_URL}" \
  --env "NEBIUS_S3_ENDPOINT=${S3_ENDPOINT_URL}" \
  --env "AWS_ENDPOINT_URL=${S3_ENDPOINT_URL}" \
  --env "S3_BUCKET=${S3_BUCKET}" \
  --env "NPA_S3_BUCKET=${S3_BUCKET}" \
  --env "S3_PREFIX=sim-to-real/${RUN_ID}" \
  --env "PIPELINE_ROOT_URI=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/" \
  --env "INPUT_DATA_URI=s3://${S3_BUCKET}/datasets/lerobot-pusht/" \
  --env "LEROBOT_DATASET_URI=s3://${S3_BUCKET}/datasets/lerobot-pusht/" \
  --env "RAW_ENVS_URI=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/raw-envs/" \
  --env "TRAIN_ENVS_URI=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/splits/train/" \
  --env "HELDOUT_ENVS_URI=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/splits/heldout/" \
  --env "POLICY_IMAGE=${POLICY_IMAGE}" \
  --env "CHECKPOINT_URI=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/checkpoints/policy/" \
  --env "RERUN_RRD_PATH=s3://${S3_BUCKET}/sim-to-real/${RUN_ID}/viz/${RUN_ID}.rrd" \
  --env "NPA_GPU_TYPE=${NPA_GPU_TYPE}" \
  --env "NPA_GPU_FAILOVER=${NPA_GPU_FAILOVER}" \
  --env "EVAL_BACKEND=${EVAL_BACKEND}" \
  --env "FEEDBACK_SOURCE=${FEEDBACK_SOURCE}" \
  --env "FEEDBACK_TYPE=${FEEDBACK_TYPE}" \
  --env "VLM_EVAL_BACKEND=stub" \
  --env "VLM_EVAL_SCORE=0.82" \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/skypilot/sim-to-real-pipeline.yaml
```

Tear down explicitly after the run:

```bash
"$NPA_SKYPILOT_BIN" down --yes "s2r-${RUN_ID}"
until ! "$NPA_SKYPILOT_BIN" status --refresh | grep -q "s2r-${RUN_ID}"; do sleep 30; done
```

The checked-in YAML defaults to ordered SkyPilot accelerator failover:
`H100:1`, `H200:1`, `A100:1`, `L40S:1`, `RTX6000:1`. For raw SkyPilot launches,
`--gpus` can override the primary accelerator and the `NPA_GPU_TYPE` /
`NPA_GPU_FAILOVER` envs keep the runtime report aligned with the resource choice.
The SkyPilot Nebius catalog uses `RTX6000` for NVIDIA RTX PRO 6000.

## CLI Wrapper Path

The CLI wrapper renders the same YAML, fills the same envs, submits it through
the NPA SkyPilot helper, and then polls the managed job:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py \
  --run-id sim-to-real-example \
  --bucket "$S3_BUCKET" \
  --s3-endpoint "$S3_ENDPOINT_URL" \
  --input-data-uri "s3://$S3_BUCKET/datasets/lerobot-pusht/" \
  --dataset-repo-id lerobot/pusht \
  --dataset-revision 7628202a2180972f291ba1bc6723834921e72c19 \
  --policy-image "$POLICY_IMAGE" \
  --eval-backend state-success \
  --feedback-source vlm \
  --feedback-type critique \
  --vlm-eval-backend stub \
  --vlm-eval-score 0.82 \
  --gpu H100:1 \
  --gpu-failover H200:1,A100:1,L40S:1,RTX6000:1 \
  --task-cloud nebius \
  --controller-backend nebius \
  --cleanup
```

## SDK Path

The Python SDK path runs the same structural spine directly. It is useful for
local smoke and for applications that want typed return objects instead of a
subprocess wrapper:

```python
from npa.sdk.workbench import sim_to_real

report = sim_to_real.local_smoke(
    run_id="sim-to-real-sdk-example",
    s3_bucket="npa-sim2real-d87cf691",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_prefix="sim-to-real/sim-to-real-sdk-example",
    input_data_uri="s3://npa-sim2real-d87cf691/datasets/lerobot-pusht/",
    policy_image="npa-lerobot-policy:0.1.0",
    gpu="H100:1",
    gpu_failover="H200:1,A100:1,L40S:1,RTX6000:1",
    eval_backend="state-success",
    feedback_source="vlm",
    feedback_type="critique",
    vlm_eval_backend="stub",
    vlm_eval_score=0.82,
    attempt_s3_roundtrip=True,
)
print(report.status)
```

Expected artifacts in the JSON output:

- Seeded real-episode split: `train=165`, `heldout=41`, `seed=42`.
- Feedback object with `{success, score, rationale}`.
- Checkpoint marker or checkpoint URI under `s3://$NPA_S3_BUCKET/sim-to-real/<run-id>/checkpoints/policy/`.
- Rerun recording under `s3://$NPA_S3_BUCKET/sim-to-real/<run-id>/viz/<run-id>.rrd`.
- Per-component tiers. Treat `SEAM` and `BLOCKED` literally.

## Pluggable Eval And Feedback

Eval backends are selected consistently through CLI `--eval-backend`, SDK
`eval_backend`, and YAML env `EVAL_BACKEND`:

- `state-success`: pose/state predicate backend. The real `lerobot-eval` /
  `pc_success` path should be wired here at merge.
- `vlm-frames`: frame subset rendered to a VLM/VLA scorer.
- `heldout-metrics`: heldout imitation metrics.

Feedback sources are selected through CLI `--feedback-source`, SDK
`feedback_source`, and YAML env `FEEDBACK_SOURCE`:

- `none`: pure imitation, no feedback loop.
- `sim-env`: feedback derived from the selected eval/env metric.
- `vlm`: VLM critique or score.
- `byo-container`: neutral HTTP or CLI BYO feedback container contract.

Feedback type is selected through CLI `--feedback-type`, SDK `feedback_type`,
and YAML env `FEEDBACK_TYPE`. Supported types are `scalar`, `dense-per-step`,
`pass-fail`, `critique`, and `preference`; each has an adapter to the standard
training signal schema. `byo-container` declares which type it emits and can run
in `provided-rollout` or `self-rollout` mode via `BYO_FEEDBACK_MODE`.

View a downloaded Rerun artifact with:

```bash
rerun /tmp/npa-sim-to-real-<run-id>/<run-id>.rrd
```

The recording uses logical paths for input demonstrations, the policy rollout,
and per-episode feedback. The report includes verified entity counts for paths
such as `input_dataset/episodes/.../state/dim_00`,
`policy_rollout/episodes/.../actions/dim_00`, and
`eval/episodes/.../score`.

## Secure Inputs

Pass S3 credentials through environment variables or SkyPilot secret injection.
Do not write credentials into source files, workflow YAML, logs, image tags, S3
keys, or `.rrd` recordings.

## Bring Your Own Dataset

Point `--input-data-uri` at a LeRobotDataset directory in S3:

```bash
--input-data-uri "s3://$NPA_S3_BUCKET/datasets/my-lerobot-dataset/"
```

Keep `--dataset-repo-id` and `--dataset-revision` aligned with the source when
they are known. The adapter visualizes the same camera, state, action, timestamp,
rollout, and feedback paths.

## Bring Your Own Policy Image

`POLICY_IMAGE` defaults to the platform BYO-compatible LeRobot policy container.
Override it with any image that keeps the same contract:

- `POST /infer` for observation-to-action inference.
- `POST /rollout` for batched rollout actions.
- `POST /feedback/train-step` for `{success, score, rationale}` feedback batches.
- S3 inputs from `INPUT_DATA_URI`; checkpoint outputs to `CHECKPOINT_URI`.

The feedback hook runs real update steps and writes an adapter checkpoint. It is
still a `SEAM`: calibration, convergence, and full closed-loop policy improvement
are separate milestones.

Build and push the policy image from the repo root:

```bash
docker build \
  -f npa/docker/workbench/lerobot-policy/Dockerfile \
  -t npa-lerobot-policy:0.1.0 \
  npa

docker tag npa-lerobot-policy:0.1.0 \
  "cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/npa-lerobot-policy:0.1.0"
docker push "cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/npa-lerobot-policy:0.1.0"
export POLICY_IMAGE="cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/npa-lerobot-policy:0.1.0"
```

## Teardown

The runner uses SkyPilot cleanup and does not use unsupported `--down` flags.
After the command exits:

```bash
"$NPA_SKYPILOT_BIN" status
"$NPA_SKYPILOT_BIN" jobs queue
```

Both should show no in-progress clusters or managed jobs for the run.

## Swap Matrix

| Setting | Example default | BYO override |
| --- | --- | --- |
| `S3_BUCKET` / `NPA_S3_BUCKET` / `--bucket` | `npa-sim2real-d87cf691` | BYO bucket |
| `S3_ENDPOINT_URL` / `NEBIUS_S3_ENDPOINT` / `AWS_ENDPOINT_URL` / `--s3-endpoint` | `https://storage.eu-north1.nebius.cloud` | BYO S3-compatible endpoint |
| `--input-data-uri` / `LEROBOT_DATASET_URI` | `s3://$S3_BUCKET/datasets/lerobot-pusht/` | Any LeRobotDataset S3 URI |
| `--dataset-repo-id` | `lerobot/pusht` | Dataset repo ID |
| `--dataset-revision` | `7628202a2180972f291ba1bc6723834921e72c19` | Dataset revision |
| `POLICY_IMAGE` / `--policy-image` | `npa-lerobot-policy:0.1.0` | Custom LeRobot policy image or registry-qualified tag |
| `--eval-backend` / `EVAL_BACKEND` / `eval_backend` | `state-success` | `vlm-frames` or `heldout-metrics` |
| `--feedback-source` / `FEEDBACK_SOURCE` / `feedback_source` | `vlm` | `none`, `sim-env`, or `byo-container` |
| `--feedback-type` / `FEEDBACK_TYPE` / `feedback_type` | `critique` | `scalar`, `dense-per-step`, `pass-fail`, or `preference` |
| `--gpu` / `NPA_GPU_TYPE` / `gpu` | `H100:1` | Primary SkyPilot accelerator; examples: `H100:1`, `H200:1`, `A100:1`, `L40S:1`, `RTX6000:1` |
| `--gpu-failover` / `NPA_GPU_FAILOVER` / `gpu_failover` | `H200:1,A100:1,L40S:1,RTX6000:1` | Ordered fallback accelerator list |
| `--vlm-eval-backend` | `stub` | Live VLM backend |
| `--task-cloud` | `nebius` | Task backend for acceptance runs when Kubernetes GPU capacity is occupied |
| `--controller-backend` | `nebius` | Managed-jobs controller fallback for clusters that cannot validate the Kubernetes controller pod |
| `--rerun-max-frames-per-episode` | `32` | Lower for smoke, higher for inspection |

## Tier Semantics

- `WORKS`: live or local evidence exists for that component.
- `PARTIAL`: structural validation passed, but live backend evidence is missing.
- `SEAM`: typed extension point exists, but the full backend/calibration is not
  complete.
- `BLOCKED`: exact missing credential, tool, or backend failure was observed.
