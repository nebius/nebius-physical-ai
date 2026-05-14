# LeRobot Benchmark on Workbench Serverless

This runbook shows how to run LeRobot training benchmarks on Nebius Physical AI
Workbench with Serverless Jobs.

Use it when you want to submit a benchmark job, collect `wallclock_results.json`,
and store the profile artifacts in S3. The runbook is intentionally
instructional: it describes how to run the workflow and inspect the output, not
what a particular benchmark run measured.

## Prerequisites

You need:

- A Nebius project with Serverless Jobs access
- A Workbench project alias configured for `npa`
- S3-compatible object storage credentials in `~/.npa/credentials.yaml`
- A Hugging Face token in `~/.npa/credentials.yaml` if the dataset download
  path requires one
- The LeRobot profile script in this repository

Set these values before running commands:

```bash
export NPA_REPO="$HOME/repos/nebius-physical-ai"
export PROJECT_ALIAS="<PROJECT_ALIAS>"
export NEBIUS_PROJECT_ID="<NEBIUS_PROJECT_ID>"
export NEBIUS_S3_BUCKET="<YOUR_BUCKET>"
export NEBIUS_S3_ENDPOINT="https://storage.<REGION>.nebius.cloud"
export WORKBENCH_NAME="lerobot-profile"
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export SCRIPT_PATH="$NPA_REPO/research/lerobot-deploy/training/profile_train.py"
export LOCAL_RESULTS_DIR="$NPA_REPO/.npa-results/lerobot-profile-$RUN_TAG"
export WAIT_TIMEOUT_SECONDS=5400

mkdir -p "$LOCAL_RESULTS_DIR"
```

Check local tooling:

```bash
cd "$NPA_REPO"
source npa/.venv/bin/activate

npa --help >/dev/null
nebius profile list
aws --version
jq --version
test -f "$SCRIPT_PATH"
aws s3 ls "s3://$NEBIUS_S3_BUCKET" --endpoint-url "$NEBIUS_S3_ENDPOINT" >/dev/null
```

`~/.npa/credentials.yaml` should be mode `600`:

```bash
stat -f "%Sp %N" ~/.npa/credentials.yaml 2>/dev/null || stat -c "%A %n" ~/.npa/credentials.yaml
```

## GPU Selectors

Use these `--gpu-type` values with `profile-train --runtime serverless`:

| GPU | `--gpu-type` |
| --- | --- |
| H200 | `h200` |
| L40S | `l40s` |
| B300 | `b300` |
| RTX PRO 6000 | `gpu-rtx-pro-6000` |

If a region exposes a lower-level platform name, you can pass that platform
name directly as `--gpu-type`.

## Policy Inputs

Use these policy and dataset pairs for the standard benchmark matrix:

| Policy | Dataset |
| --- | --- |
| VQ-BeT | `lerobot/pusht` |
| ACT | `lerobot/pusht` |
| Diffusion Policy | `lerobot/pusht` |
| SmolVLA | `lerobot/aloha_sim_insertion_human` |

The examples below use:

- `--mode wallclock`
- `--steps 100`
- `--batch-size 8`
- `--warmup-steps 10`
- `--num-workers 0`

In the Workbench CLI, `--num-workers 0` means "use the available CPU count" for
serverless profile jobs.

## Run One Benchmark Job

This command runs Diffusion Policy on H200 and uploads the profile artifacts to
S3:

```bash
npa workbench lerobot \
  -p "$PROJECT_ALIAS" \
  -n "$WORKBENCH_NAME" \
  profile-train \
  --runtime serverless \
  --project-id "$NEBIUS_PROJECT_ID" \
  --script "$SCRIPT_PATH" \
  --mode wallclock \
  --policy-type diffusion \
  --dataset-repo-id lerobot/pusht \
  --steps 100 \
  --batch-size 8 \
  --num-workers 0 \
  --warmup-steps 10 \
  --gpu-type h200 \
  --gpu-count 1 \
  --job-name "h200-diffusion-$RUN_TAG" \
  --output-path "s3://$NEBIUS_S3_BUCKET/lerobot-benchmarks/$RUN_TAG/h200-diffusion/" \
  --wait-timeout "$WAIT_TIMEOUT_SECONDS" \
  --poll-interval 30
```

To submit without waiting for completion, add:

```bash
--submit-only
```

## Run a Matrix

Create a small helper so every job uses the same command shape:

```bash
cat > "$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

gpu_type="$1"
policy_type="$2"
dataset_repo_id="$3"
label="$4"

run_dir="$LOCAL_RESULTS_DIR/$label"
mkdir -p "$run_dir"

job_name="${label}-${RUN_TAG}"
output_path="s3://${NEBIUS_S3_BUCKET}/lerobot-benchmarks/${RUN_TAG}/${label}/"

npa workbench lerobot \
  -p "$PROJECT_ALIAS" \
  -n "$WORKBENCH_NAME" \
  profile-train \
  --runtime serverless \
  --project-id "$NEBIUS_PROJECT_ID" \
  --script "$SCRIPT_PATH" \
  --mode wallclock \
  --policy-type "$policy_type" \
  --dataset-repo-id "$dataset_repo_id" \
  --steps 100 \
  --batch-size 8 \
  --num-workers 0 \
  --warmup-steps 10 \
  --gpu-type "$gpu_type" \
  --gpu-count 1 \
  --job-name "$job_name" \
  --output-path "$output_path" \
  --wait-timeout "$WAIT_TIMEOUT_SECONDS" \
  --poll-interval 30 \
  2>&1 | tee "$run_dir/npa-profile-train.log"

printf '%s\n' "$output_path" > "$run_dir/output-path.txt"
SH

chmod +x "$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh"
```

Run H200:

```bash
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" h200 diffusion lerobot/pusht h200-diffusion
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" h200 act lerobot/pusht h200-act
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" h200 smolvla lerobot/aloha_sim_insertion_human h200-smolvla
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" h200 vqbet lerobot/pusht h200-vqbet
```

Run L40S:

```bash
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" l40s diffusion lerobot/pusht l40s-diffusion
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" l40s act lerobot/pusht l40s-act
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" l40s smolvla lerobot/aloha_sim_insertion_human l40s-smolvla
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" l40s vqbet lerobot/pusht l40s-vqbet
```

Run B300:

```bash
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" b300 diffusion lerobot/pusht b300-diffusion
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" b300 act lerobot/pusht b300-act
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" b300 smolvla lerobot/aloha_sim_insertion_human b300-smolvla
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" b300 vqbet lerobot/pusht b300-vqbet
```

Run RTX PRO 6000:

```bash
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" gpu-rtx-pro-6000 diffusion lerobot/pusht rtx6000-diffusion
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" gpu-rtx-pro-6000 act lerobot/pusht rtx6000-act
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" gpu-rtx-pro-6000 smolvla lerobot/aloha_sim_insertion_human rtx6000-smolvla
"$LOCAL_RESULTS_DIR/run_lerobot_profile_job.sh" gpu-rtx-pro-6000 vqbet lerobot/pusht rtx6000-vqbet
```

## Output Layout

Each job writes artifacts under the S3 prefix you pass with `--output-path`:

```text
s3://$NEBIUS_S3_BUCKET/lerobot-benchmarks/$RUN_TAG/<gpu-policy>/
```

For wallclock mode, the important files are:

```text
wallclock_results.json
wallclock_summary.txt
```

List the output for one run:

```bash
aws s3 ls \
  "s3://$NEBIUS_S3_BUCKET/lerobot-benchmarks/$RUN_TAG/h200-diffusion/" \
  --recursive \
  --endpoint-url "$NEBIUS_S3_ENDPOINT"
```

Fetch and inspect the JSON:

```bash
mkdir -p "$LOCAL_RESULTS_DIR/h200-diffusion"

aws s3 cp \
  "s3://$NEBIUS_S3_BUCKET/lerobot-benchmarks/$RUN_TAG/h200-diffusion/wallclock_results.json" \
  "$LOCAL_RESULTS_DIR/h200-diffusion/wallclock_results.json" \
  --endpoint-url "$NEBIUS_S3_ENDPOINT"

jq . "$LOCAL_RESULTS_DIR/h200-diffusion/wallclock_results.json"
```

`wallclock_results.json` includes fields such as:

- `mode`
- `policy_type`
- `dataset_repo_id`
- `batch_size`
- `warmup_steps`
- `measured_steps`
- `throughput_steps_per_sec`
- `seconds_per_step`

## Build a Local Summary

After fetching any completed result files, create a compact summary:

```bash
python3 - "$LOCAL_RESULTS_DIR" <<'PY' \
  | tee "$LOCAL_RESULTS_DIR/lerobot-profile-summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/wallclock_results.json")):
    data = json.loads(path.read_text())
    rows.append({
        "label": path.parent.name,
        "policy_type": data.get("policy_type"),
        "dataset_repo_id": data.get("dataset_repo_id"),
        "batch_size": data.get("batch_size"),
        "warmup_steps": data.get("warmup_steps"),
        "measured_steps": data.get("measured_steps"),
        "throughput_steps_per_sec": data.get("throughput_steps_per_sec"),
        "seconds_per_step": data.get("seconds_per_step"),
    })
json.dump(rows, sys.stdout, indent=2)
print()
PY
```

## Troubleshooting

**Job stays in `PROVISIONING`**

The requested GPU type may be temporarily unavailable in that project or
region. Check platform availability and try another project with the same GPU
quota if you have one:

```bash
nebius compute platform list --parent-id "$NEBIUS_PROJECT_ID"
```

**B300 Diffusion startup or throughput looks unusual**

B300 Diffusion can be sensitive to the PyTorch/CUDA image because Blackwell
native kernel coverage depends on the image build. If you are investigating
B300 specifically, keep the image tag, CUDA version, PyTorch version, and
`torch.compile` setting fixed across runs.

**RTX PRO 6000 selector is not accepted**

Some regions expose the platform as `gpu-rtx6000`. Use the platform name shown
by `nebius compute platform list` as the `--gpu-type` value.

**S3 upload fails**

Check that the `storage` block in `~/.npa/credentials.yaml` points at the
endpoint for the bucket you are writing to, and verify that the bucket can be
listed with the same endpoint:

```bash
aws s3 ls "s3://$NEBIUS_S3_BUCKET" --endpoint-url "$NEBIUS_S3_ENDPOINT"
```

**Dataset download fails**

Confirm that `HF_TOKEN` is present in `~/.npa/credentials.yaml`, then rerun the
same job. The profile script uses the LeRobot dataset loaders inside the
container.

## Cleanup

Serverless Jobs remain visible as job history. No VM teardown is required.

Remove this run's S3 artifacts only after you no longer need them:

```bash
aws s3 rm \
  "s3://$NEBIUS_S3_BUCKET/lerobot-benchmarks/$RUN_TAG/" \
  --recursive \
  --endpoint-url "$NEBIUS_S3_ENDPOINT"
```

Cancel a still-running job by ID if needed:

```bash
nebius ai job cancel --id <JOB_ID>
```

## Related Docs

- [LeRobot GPU Benchmarks Cookbook](lerobot-gpu-benchmarks.md)
- [npa quickstart](../quickstart.md)
- [npa LeRobot CLI reference](../cli/lerobot.md)
