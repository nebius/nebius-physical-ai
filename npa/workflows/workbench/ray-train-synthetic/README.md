# Train on two GPUs with native Ray Train

[Ray guide](../../../../docs/workbench/ray.md) · [Reference examples](../README.md)

Train synthetic linear regression with PyTorch DDP on **two B200 hosts**, one
CUDA rank per host. Native Ray Train saves recoverable checkpoints to S3 and
exports the trained model, metrics, and a Rerun recording. Ray Jobs handles
source delivery and job control; SkyPilot owns the hosts and service task.

This example verifies distributed optimization and checkpoint recovery. It is
not a robotics-policy benchmark. Use the [`npa.workflow` catalog](../../../../workflows/README.md)
for production pipelines composed of multiple Workbench capabilities.

## 1. Prepare clients, compute, and storage

Complete the [private SkyPilot platform setup](../ray-clip-development/platform/README.md).
Retain its private `KUBECONFIG`, verified context, API endpoint, and
`NPA_SKYPILOT_BIN`. Select two compatible B200 GPU nodes and verify their
accelerator spelling with `"$NPA_SKYPILOT_BIN" gpus list`. Each hosting pod requests
12 CPUs and at least 48 GiB memory in addition to its GPU.

From the repository root, use the [contributor environment](../../../../npa/README.md#developing-and-testing-npa)
and install the separate application Jobs client:

```bash
export NPA_REPO="$PWD"
export TRAIN_EXAMPLE="$NPA_REPO/npa/workflows/workbench/ray-train-synthetic"
export RAY_BIN="$NPA_REPO/npa/.venv/bin/ray"
npa/.venv/bin/python -m pip install 'ray[default,train]==2.58.0'
export TRAIN_RESULTS="$HOME/ray-train-results"
```

Select a workload bucket owned by the authorized project and a fresh run prefix.
Verify ownership and write/read/delete permission at that exact prefix. Export
these variables privately before launch:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL_S3     # verified HTTPS endpoint for this bucket
AWS_DEFAULT_REGION     # bucket signing region
TRAIN_STORAGE_URI      # s3://<your-bucket>/<fresh-training-prefix>
```

Only trusted code may share this runtime's storage identity. Launch forwards
credential variable names; keep their values outside source files, Jobs runtime
environment files, and explicit `pyarrow.fs.S3FileSystem` arguments.

## 2. Start and connect to the Ray service

Launch **from the example directory** so its relative `./cluster` mount resolves:

```bash
cd "$TRAIN_EXAMPLE"
export TRAIN_CLUSTER='<unique-development-cluster-name>'
export KUBE_CONTEXT='<verified-kubernetes-context>'
"$NPA_SKYPILOT_BIN" launch --yes --detach-run -c "$TRAIN_CLUSTER" \
  --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY \
  --env AWS_ENDPOINT_URL_S3 --env AWS_DEFAULT_REGION cluster.yaml
"$NPA_SKYPILOT_BIN" queue "$TRAIN_CLUSTER"
```

Record the exact service-task ID. Preparation installs the pinned application
environment in a fresh private `/opt/npa-ray-train` on each host. It refuses an
existing runtime directory; preserve failed-attempt evidence and use fresh pods.
See [runtime versions and checks](runtime-validation.md#application-environment).

Use the generated SSH alias for an authenticated loopback tunnel:

```bash
export TRAIN_SOCKET="$HOME/.ssh/${TRAIN_CLUSTER}-jobs.sock"
ssh -M -S "$TRAIN_SOCKET" -fNT -o ExitOnForwardFailure=yes \
  -L 18265:127.0.0.1:8265 "$TRAIN_CLUSTER"
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:18265
"$RAY_BIN" job list --address "$RAY_API"
```

Require Jobs readiness and both hosts joined to application GCS port 6381, with
two GPUs total. The separate SkyPilot management runtime must remain intact.
Keep the two ambient Ray address variables unset: the Jobs client can prioritize
them over `--address`. Do not expose Jobs publicly or use a broad `ray stop`.

## 3. Train and inspect completion

Still in `$TRAIN_EXAMPLE`:

```bash
"$RAY_BIN" job submit --address "$RAY_API" --submission-id train-baseline \
  --working-dir . -- /opt/npa-ray-train/env/bin/python train.py \
  --storage-path "$TRAIN_STORAGE_URI" --run-name baseline \
  --output-dir /opt/npa-ray-train/exports/train-baseline
"$RAY_BIN" job status --address "$RAY_API" train-baseline
"$RAY_BIN" job logs --address "$RAY_API" train-baseline
```

Require `SUCCEEDED`. The default 32 optimizer steps use deterministic rank shards
and SGD momentum. Every fourth step and the final step save the journal,
model, and optimizer. Export verifies held-out loss improvement and matching
final parameters across CUDA ranks.

Each new experiment needs a fresh `--run-name`. Do not run concurrent drivers
with the same storage path and run name.

## 4. Download and view the durable result

Exports appear under `$TRAIN_STORAGE_URI/baseline/exports/`. Each upload is
read back and verified; conflicting existing bytes are refused.

| File | What it proves |
| --- | --- |
| `state.pt` | Final model and SGD momentum checkpoint |
| `metrics.json` | Every optimizer step, loss, gradient, parameter update, and CUDA rank |
| `metrics.rrd` | Derived visualization of the recorded measurements |
| `result.json` | Recipe, validation, runtime, and artifact bindings |
| `SHA256SUMS` | Exact exported file hashes |

Use a fresh destination directory:

```bash
"$NPA_REPO/npa/.venv/bin/python" "$TRAIN_EXAMPLE/inspect_results.py" \
  "$TRAIN_RESULTS/baseline" --download "$TRAIN_STORAGE_URI/baseline/exports"
"$NPA_REPO/npa/.venv/bin/rerun" rrd verify "$TRAIN_RESULTS/baseline/metrics.rrd"
"$NPA_REPO/npa/.venv/bin/rerun" "$TRAIN_RESULTS/baseline/metrics.rrd"
```

The inspector requires four verified artifacts, 32 decoded optimizer steps,
five metric entities, and eight checkpoint events for the default recipe.
The RRD values must match the full journal. The local inspector uses NPA's
Rerun dependencies; the deeper live suite additionally installs matching Torch.
Download again into another fresh directory after compute teardown to verify
that the S3 result outlives its hosts.

## Optional: prove checkpoint recovery

Use a separate native run name:

```bash
"$RAY_BIN" job submit --address "$RAY_API" --submission-id train-recovery \
  --working-dir . -- /opt/npa-ray-train/env/bin/python train.py \
  --storage-path "$TRAIN_STORAGE_URI" --run-name recovery \
  --output-dir /opt/npa-ray-train/exports/train-recovery --fail-after-step 8
```

Rank one raises only after checkpoint 8 is committed. Train restarts both ranks,
restores model and optimizer, and continues at step 9. The final journal must
contain every step exactly once and prove restoration on both ranks. This uses
Train V2; do not substitute V1 `TorchTrainer.restore()` or `resume_from_checkpoint`.
For driver retries, retain the same native `RunConfig(name, storage_path)` pair
and exact saved recipe. `ray.train.Result.from_path()` reopens checkpoint state
through the same S3 filesystem but does not restore Jobs terminal status.

If export delivery fails after training, preserve the host's export and retry
its exact bytes without retraining:

```bash
"$RAY_BIN" job submit --address "$RAY_API" --submission-id train-export-retry \
  --working-dir . -- /opt/npa-ray-train/env/bin/python inspect_results.py \
  /opt/npa-ray-train/exports/train-baseline --publish "$TRAIN_STORAGE_URI/baseline/exports"
```

If those host files were lost, recover the checkpoint into a fresh export
destination. A regenerated RRD carries new event timestamps and is a new artifact.
The [live validation procedure](runtime-validation.md#committed-live-suite) tests
baseline training, recovery, independent checkpoint decoding, and active cancellation.

## 5. Stop the application and its hosts

Stop each remaining exact Ray submission ID and confirm a terminal state before
removing its hosting task. A stop request alone does not prove cancellation:

```bash
"$RAY_BIN" job stop --address "$RAY_API" '<exact-submission-id>'
"$RAY_BIN" job status --address "$RAY_API" '<exact-submission-id>'
# After all owned application jobs stop and outputs are saved:
"$NPA_SKYPILOT_BIN" cancel "$TRAIN_CLUSTER" '<service-task-id>' --yes
"$NPA_SKYPILOT_BIN" down "$TRAIN_CLUSTER" --yes
ssh -S "$TRAIN_SOCKET" -O exit "$TRAIN_CLUSTER"
```

Verify the owned pods are absent. Retain the shared platform and Kubernetes
cluster unless you own their lifecycle. Keep S3 results and private ownership
receipts. [Compatibility and licenses](runtime-validation.md#compatibility-and-licenses)
describe the pinned upstream runtime and its NVIDIA CUDA terms.
