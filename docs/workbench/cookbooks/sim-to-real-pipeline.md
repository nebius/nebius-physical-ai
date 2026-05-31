# Sim-To-Real Pipeline Runbook

This runbook describes the generic sim-to-real training and evaluation pipeline.
It is parameterized for Nebius S3-compatible storage, a custom LeRobot policy
container, simulator backends, and VLM/VLA feedback.

## One-Command Launch

Render and submit the SkyPilot workflow with the runner:

```bash
export NPA_SKYPILOT_BIN=/home/ubuntu/.npa/skypilot-venv/bin/sky
export NPA_S3_BUCKET=your-bucket-name
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export NPA_REGISTRY_ID=your-registry-id

npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py \
  --run-id sim-to-real-example \
  --bucket "$NPA_S3_BUCKET" \
  --input-data-uri "s3://$NPA_S3_BUCKET/sim-to-real/input/" \
  --policy-image "cr.eu-north1.nebius.cloud/$NPA_REGISTRY_ID/npa-lerobot:0.5.1" \
  --vlm-eval-backend stub \
  --cleanup
```

For shape validation without launching infrastructure:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py \
  --run-id sim-to-real-render \
  --render-only
```

## Credentials

S3 credentials are supplied at runtime through environment or SkyPilot secret
injection. Do not write access keys into workflow YAML, source files, logs, or
Rerun recordings.

Required storage settings:

```bash
export NEBIUS_S3_ENDPOINT=https://storage.eu-north1.nebius.cloud
export AWS_ENDPOINT_URL="$NEBIUS_S3_ENDPOINT"
export NPA_S3_BUCKET=your-bucket-name
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

The workflow writes artifacts under:

```text
s3://$NPA_S3_BUCKET/sim-to-real/<run-id>/
```

## Policy Image

`POLICY_IMAGE` defaults to the platform LeRobot workbench image resolved by
`npa.deploy.images.container_image_for_tool("lerobot")`. Override it with any
custom LeRobot-compatible policy container that preserves the expected input and
output contract.

The policy container consumes:

- LeRobot-format data or simulator rollout references from `INPUT_DATA_URI`.
- Vision and language observations: workspace camera, wrist camera, robot state,
  and instruction text.
- VLM/VLA feedback as a bounded training signal.

The policy container produces:

- A checkpoint under `CHECKPOINT_URI`.
- Optional training logs under the run-scoped S3 prefix.

## Inputs And Outputs

Inputs:

- `INPUT_DATA_URI`: source LeRobot data, raw rollout data, or task assets.
- `POLICY_IMAGE`: custom or platform LeRobot policy container.
- `SIM_BACKEND`: `genesis`, `lightwheel`, or `isaac`.
- `FEEDBACK_SOURCE`: `vlm` or `vla`.
- `SUCCESS_THRESHOLD`: outer-loop promotion threshold.

Outputs:

- Raw environment specs: `raw-envs/`.
- Seeded train and held-out split manifests: `splits/train/`, `splits/heldout/`.
- Feedback JSON from `vlm-eval`: `feedback/`.
- Training signal JSON: `training-signal.json`.
- Checkpoint marker or promoted checkpoint: `checkpoints/policy/`.
- Tiered report: `reports/sim-to-real-report.json`.
- Rerun recording: `viz/<run-id>.rrd`.

View a local Rerun artifact with:

```bash
rerun /tmp/npa-sim-to-real-<run-id>/<run-id>.rrd
```

## Backend Swaps

Use the same workflow with different backend flags:

```bash
--sim-backend genesis
--sim-backend lightwheel
--sim-backend isaac
--feedback-source vlm
--feedback-source vla
--vlm-eval-backend self-hosted
--vlm-eval-backend api
--vlm-eval-backend stub
```

`vlm` feedback uses the existing `npa workbench vlm-eval` implementation. `vla`,
Lightwheel, Isaac Lab, Cosmos augmentation, and LanceDB cache hooks are reported
as seams until a configured backend produces live evidence.

## Tier Semantics

Every report labels components:

- `WORKS`: live evidence exists.
- `PARTIAL`: local or structural validation passed, but live backend evidence is
  unavailable.
- `SEAM`: typed extension point exists, but the backend is not implemented or not
  configured.
- `BLOCKED`: exact missing credential, tool, or backend failure was observed.

Do not treat a partial report as a live result. A non-draft release requires a
live Nebius/GPU/S3/VLM run with evidence in the tier table.
