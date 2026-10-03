# Native Ray Tune: search, checkpoint, and recover

[Ray guide](../../../../docs/workbench/ray.md) · [Reference examples](../README.md)

Run a three-trial CPU search on generated numeric inputs. Ray Tune selects the
best step size and retries one deliberately interrupted trial from its checkpoint.
The result is a set of JSON reports that an independent inspector verifies.
Start locally; the optional cloud path runs the same source through Ray Jobs.

| You provide | You get |
| --- | --- |
| Python environment with Ray 2.58.0; no model or dataset | Three completed trials, best step size `0.2`, and zero best loss |
| Fresh storage and export directories | Native Tune checkpoints plus checksum-verified result reports |
| Optional cloud platform | A reusable CPU service with native Jobs submit, logs, status, and stop |

This is a native Ray application example. Use the
[`npa.workflow` catalog](../../../../workflows/README.md) for pipelines that
compose several Workbench capabilities.

## Run locally first

From the repository root, use the
[contributor environment](../../../../npa/README.md#developing-and-testing-npa)
and install the application's additional dependency:

```bash
export NPA_REPO="$PWD"
export TUNE_EXAMPLE="$NPA_REPO/npa/workflows/workbench/ray-tune-synthetic"
export RAY_BIN="$NPA_REPO/npa/.venv/bin/ray"
npa/.venv/bin/python -m pip install 'ray[default,tune]==2.58.0'

unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export TUNE_RESULTS="$(mktemp -d "${TMPDIR:-/tmp}/npa-tune.XXXXXX")"
npa/.venv/bin/python "$TUNE_EXAMPLE/search.py" \
  --local \
  --storage-path "$TUNE_RESULTS/local-storage" \
  --run-name local-search \
  --output-dir "$TUNE_RESULTS/local-result" \
  --fail-step-size 0.3
npa/.venv/bin/python "$TUNE_EXAMPLE/inspect_results.py" "$TUNE_RESULTS/local-result"
```

Expected inspector output:

```json
{"artifacts_verified": 3, "resumed_trials": 1, "trials_verified": 3}
```

The `0.3` trial deliberately fails after its first checkpoint, then resumes at
iteration two. An error from that injected failure is expected; the final search
and inspection must still succeed. This exact local path was verified on macOS
with Ray 2.58.0. It establishes CPU search and checkpoint recovery.

Use fresh paths for another run: existing export directories are refused.
Copy results somewhere durable before deleting temporary files. The application
shuts down its locally started Ray runtime when it finishes.

## What to inspect

| File | Contents |
| --- | --- |
| `result.json` | Completion, selected optimum, and experiment path |
| `trials.json` | Every trial's loss, iterations, checkpoint, and retry state |
| `runtime.json` | Python/Ray versions and exact source hash |
| `SHA256SUMS` | Hashes of the three JSON files |

`inspect_results.py` checks the complete file set, source and artifact hashes,
trial completion, checkpoint evidence, and the known optimum. The export is a
reviewable summary. Tune's experiment directory holds the state needed for
restoration; preserve both when recovery matters.

## Optional: prepare the cloud platform

Complete the [private SkyPilot platform setup](../ray-clip-development/platform/README.md)
first. It supplies the pinned SkyPilot 0.12.2 executable, authenticated access,
and an explicitly selected Kubernetes context. The example requests one CPU
pod with 6 CPUs and 12 GiB memory; leave room for platform/system pods.

Keep the local variables above, then select the exact context supplied by setup:

```bash
export TUNE_CLUSTER='<unique-development-cluster-name>'
export KUBE_CONTEXT='<verified-kubernetes-context>'
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa workbench health preflight --checks nebius --json
```

For S3 checkpoints, also verify the selected bucket's ownership and
write/read/delete access to your run prefix. Export `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`, and `AWS_DEFAULT_REGION` privately.
Append name-only `--env` forwards for those four variables to the launch below.
Keep credential values out of YAML and Jobs runtime-environment files.
The default example below stores experiment state on the service host.

## Start the reusable Ray service

Launch **from the example directory** so `cluster.yaml` and its relative
`./cluster` file mount resolve correctly:

```bash
cd "$TUNE_EXAMPLE"
"$NPA_SKYPILOT_BIN" launch --yes --detach-run -c "$TUNE_CLUSTER" \
  --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" \
  cluster.yaml
"$NPA_SKYPILOT_BIN" queue "$TUNE_CLUSTER"
```

Record the exact service-task ID. Preparation creates a fresh private
`~/.npa-ray-tune` on the **remote host** using the pinned Ray image and Ray 2.58.0
application environment. If preparation finds an existing runtime, preserve
that attempt and use fresh hosting pods.

Connect through the SkyPilot-generated SSH alias. Read the remote runtime path
instead of expanding your local `$HOME` into a remote command:

```bash
export TUNE_RUNTIME="$(ssh "$TUNE_CLUSTER" 'printf "%s/.npa-ray-tune\n" "$HOME"')"
export TUNE_SOCKET="$HOME/.ssh/${TUNE_CLUSTER}-jobs.sock"
ssh -M -S "$TUNE_SOCKET" -fNT -o ExitOnForwardFailure=yes \
  -L 18265:127.0.0.1:8265 "$TUNE_CLUSTER"
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:18265
"$RAY_BIN" job list --address "$RAY_API"
```

Wait until Jobs responds. The dashboard and GCS remain private; application
ports are separate from SkyPilot's management Ray. Submit only trusted code and
use the explicit Jobs address throughout.

## Submit and inspect through Ray Jobs

Still in `$TUNE_EXAMPLE`, choose a fresh submission ID and output paths:

```bash
export TUNE_JOB=tune-reference
"$RAY_BIN" job submit --address "$RAY_API" --submission-id "$TUNE_JOB" \
  --working-dir . -- \
  "$TUNE_RUNTIME/env/bin/python" search.py \
  --storage-path "$TUNE_RUNTIME/experiments" \
  --run-name reference \
  --output-dir "$TUNE_RUNTIME/exports/$TUNE_JOB" \
  --fail-step-size 0.3
"$RAY_BIN" job status --address "$RAY_API" "$TUNE_JOB"
"$RAY_BIN" job logs --address "$RAY_API" "$TUNE_JOB"
```

Require `SUCCEEDED`, then download the export and local experiment state:

```bash
mkdir -p "$TUNE_RESULTS/cloud"
rsync -az "$TUNE_CLUSTER:$TUNE_RUNTIME/exports/$TUNE_JOB/" "$TUNE_RESULTS/cloud/export/"
rsync -az "$TUNE_CLUSTER:$TUNE_RUNTIME/experiments/reference/" "$TUNE_RESULTS/cloud/experiment/"
"$NPA_REPO/npa/.venv/bin/python" "$TUNE_EXAMPLE/inspect_results.py" "$TUNE_RESULTS/cloud/export"
```

Expect the same three-artifact, three-trial, one-resumed-trial receipt. For
S3-backed experiments, use the verified run-scoped S3 URI as `--storage-path`
instead; preserve the export separately. Host-local checkpoints disappear when
the hosting pod is deleted.

## Failure and recovery boundaries

The injected exception exercises native Tune checkpointed retry with
`FailureConfig(max_failures=1)`. It does not simulate node or head loss. A second
failure exceeds the recipe's retry allowance and cannot produce a successful
export.

After a driver interruption, keep the same source, `--storage-path`, and
`--run-name`, add `--restore`, and select a fresh output directory. The example
uses `Tuner.can_restore` and `Tuner.restore(..., resume_errored=True)`. Restore
cannot repair an incompatible search space or deterministic application bug.
Only restore experiment state written by trusted code.

To stop an active job:

```bash
"$RAY_BIN" job stop --address "$RAY_API" "$TUNE_JOB"
"$RAY_BIN" job status --address "$RAY_API" "$TUNE_JOB"
```

Wait for `STOPPED`. Preserve partial evidence; an interrupted export is not a
completed result.

## Cleanup

After all owned Ray Jobs are terminal and needed outputs are saved, cancel the
recorded SkyPilot service task and remove its development cluster:

```bash
"$NPA_SKYPILOT_BIN" cancel "$TUNE_CLUSTER" '<service-task-id>' --yes
"$NPA_SKYPILOT_BIN" down "$TUNE_CLUSTER" --yes
ssh -S "$TUNE_SOCKET" -O exit "$TUNE_CLUSTER"
```

Verify the owned pod and tunnel are gone. Keep shared Kubernetes and the
SkyPilot API running; a separately owned platform has its own cleanup procedure.
Do not use `ray stop`, which can affect other Ray processes on the host.

## Agent guidance and compatibility

Keep the cluster name, service-task ID, Jobs ID, tunnel, storage prefix, and
result destination in private run notes. A successful CPU search establishes
no GPU or multi-node behavior.

The implementation follows the Ray 2.58 primary sources for
[`Tuner`](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/tune/tuner.py),
[`ResultGrid`](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/tune/result_grid.py),
and [Tune fault tolerance](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/tune/tutorials/tune-fault-tolerance.rst).
Ray is Apache-2.0. The example downloads no model or dataset, builds no image,
and adds nothing to NPA's public image catalog.
