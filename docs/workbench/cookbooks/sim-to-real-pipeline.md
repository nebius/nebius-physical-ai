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
export NPA_S3_BUCKET=npa-sim2real-d87cf691
export NPA_REGISTRY_ID=<registry-id>
export POLICY_IMAGE="cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/npa-lerobot-policy:0.1.0"
export AWS_ACCESS_KEY_ID=<s3-access-key>
export AWS_SECRET_ACCESS_KEY=<s3-secret-key>
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export NEBIUS_S3_ENDPOINT="$AWS_ENDPOINT_URL"
```

SkyPilot is installed outside the NPA venv. Use `$NPA_SKYPILOT_BIN`; do not rely
on `sky` being on `PATH`.

## One Command

Run this exact command from the repo root:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py \
  --run-id sim-to-real-example \
  --bucket "$NPA_S3_BUCKET" \
  --input-data-uri "s3://$NPA_S3_BUCKET/datasets/lerobot-pusht/" \
  --dataset-repo-id lerobot/pusht \
  --dataset-revision 7628202a2180972f291ba1bc6723834921e72c19 \
  --policy-image "$POLICY_IMAGE" \
  --vlm-eval-backend stub \
  --vlm-eval-score 0.82 \
  --gpu H100:1 \
  --cleanup
```

Expected artifacts in the JSON output:

- Seeded real-episode split: `train=165`, `heldout=41`, `seed=42`.
- Feedback object with `{success, score, rationale}`.
- Checkpoint marker or checkpoint URI under `s3://$NPA_S3_BUCKET/sim-to-real/<run-id>/checkpoints/policy/`.
- Rerun recording under `s3://$NPA_S3_BUCKET/sim-to-real/<run-id>/viz/<run-id>.rrd`.
- Per-component tiers. Treat `SEAM` and `BLOCKED` literally.

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
| `NPA_S3_BUCKET` | `npa-sim2real-d87cf691` | BYO bucket |
| `--input-data-uri` / `LEROBOT_DATASET_URI` | `s3://$NPA_S3_BUCKET/datasets/lerobot-pusht/` | Any LeRobotDataset S3 URI |
| `--dataset-repo-id` | `lerobot/pusht` | Dataset repo ID |
| `--dataset-revision` | `7628202a2180972f291ba1bc6723834921e72c19` | Dataset revision |
| `POLICY_IMAGE` / `--policy-image` | `cr.eu-north1.nebius.cloud/$NPA_REGISTRY_ID/npa-lerobot-policy:0.1.0` | Custom LeRobot policy image |
| `--vlm-eval-backend` | `stub` | Live VLM backend |
| `--feedback-source` | `vlm` | `vla` when configured |
| `--gpu` | `H100:1` | `H100:1,H200:1,A100:1,L40S:1,RTX6000:1` failover string |
| `--rerun-max-frames-per-episode` | `32` | Lower for smoke, higher for inspection |

## Tier Semantics

- `WORKS`: live or local evidence exists for that component.
- `PARTIAL`: structural validation passed, but live backend evidence is missing.
- `SEAM`: typed extension point exists, but the full backend/calibration is not
  complete.
- `BLOCKED`: exact missing credential, tool, or backend failure was observed.
