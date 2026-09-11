# Tune a synthetic objective with native Ray Tune

This guarded reference runs a real three-trial Ray Tune search on public,
generated numeric inputs. It demonstrates the supported boundary without adding
an NPA Tune wrapper: SkyPilot owns one reusable CPU host and its Ray service
task; native Ray Jobs owns source delivery, submission, logs, status, and stop;
the application owns its Tune recipe and exported result contract.

This example is outside the `npa.workflow` catalog. Use `npa.workflow` for a
production graph that composes several Workbench capabilities. This reference
adds no CLI group, service, container, controller, or GPU claim.

## What it proves

The search evaluates step sizes `0.1`, `0.2`, and `0.3` against a deterministic
target of `0.2`. Every trial reports three iterations and a native Tune
checkpoint. The optional failure probe raises once after the `0.3` trial's first
checkpoint; `FailureConfig(max_failures=1)` reschedules that trial, which resumes
at iteration two. The final result must select `0.2` with zero loss.

`search.py` writes a fresh owner-only export containing:

- `result.json`: completion, Ray version, experiment path, and selected optimum;
- `trials.json`: every search value, final loss, completion, checkpoint, and
  observed retry state;
- `runtime.json`: Python and Ray versions;
- `SHA256SUMS`: complete-byte hashes for the three JSON artifacts.

`inspect_results.py` checks the complete file set, every digest, all trials,
checkpoint availability, and the known optimum. The export is independent
review evidence; Tune's experiment directory remains the recovery source.

## Run locally first

Use the repository interpreter after installing the application dependency:

```bash
cd <npa-checkout>
export TUNE_EXAMPLE=npa/workflows/workbench/ray-tune-synthetic
npa/.venv/bin/pip install 'ray[default,tune]==2.58.0'
npa/.venv/bin/python "$TUNE_EXAMPLE/search.py" \
  --local \
  --storage-path "$PWD/local-storage" \
  --run-name local-search \
  --output-dir "$PWD/local-result" \
  --fail-step-size 0.3
npa/.venv/bin/python "$TUNE_EXAMPLE/inspect_results.py" "$PWD/local-result"
```

Use fresh paths. Existing output directories are refused so a retry cannot
overwrite earlier evidence. This CPU run proves Tune scheduling, checkpointed
trial retry, result selection, and artifact validation; it proves no GPU or
multi-node behavior.

## Start the reusable Ray service

Use the private SkyPilot platform setup documented beside the native Ray CLIP
reference. Bootstrap SkyPilot 0.12.2 through NPA and invoke only the returned
`NPA_SKYPILOT_BIN`; do not use an ambient `sky` executable. Select the exact
Kubernetes context and a unique cluster name:

```bash
export TUNE_CLUSTER=tune-synthetic
export KUBE_CONTEXT="$(kubectl config current-context)"
"$NPA_SKYPILOT_BIN" launch --yes --detach-run -c "$TUNE_CLUSTER" \
  --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" \
  cluster.yaml
"$NPA_SKYPILOT_BIN" queue "$TUNE_CLUSTER"
```

The digest-pinned upstream Ray image and the separately prepared environment
both use Ray 2.58.0. Preparation fails if its private runtime already exists;
retain that attempt and use a fresh host rather than adopting unknown bytes.
The dashboard binds to loopback. Reach it through an authenticated SSH tunnel,
keep GCS and Jobs private, and allow only trusted application code on the
service.

```bash
export TUNE_SOCKET="$HOME/.ssh/${TUNE_CLUSTER}-jobs.sock"
ssh -M -S "$TUNE_SOCKET" -fNT -o ExitOnForwardFailure=yes \
  -L 18265:127.0.0.1:8265 "$TUNE_CLUSTER"
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:18265
ray job list --address "$RAY_API"
```

Wait for the native Jobs endpoint before submitting. Application Ray uses
ports separate from SkyPilot's management Ray; never use `ray stop` or ambient
Ray discovery.

## Submit and inspect through Ray Jobs

From this example directory, choose one exact submission ID and fresh host
paths. For a durable experiment, use a run-scoped S3 `--storage-path` whose
bucket ownership and exact prefix were verified privately before launch. Pass
the matching S3 credential variables to the SkyPilot launch environment; never
put values in source or Jobs runtime-environment JSON.

For S3, add these name-only forwards to the earlier `launch` command after the
exact bucket and prefix pass ownership and write/read/delete checks:

```bash
--env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY \
--env AWS_ENDPOINT_URL_S3 --env AWS_DEFAULT_REGION
```

```bash
export TUNE_JOB=tune-reference
export TUNE_RUNTIME="$HOME/.npa-ray-tune"
ray job submit --address "$RAY_API" --submission-id "$TUNE_JOB" \
  --working-dir . -- \
  "$TUNE_RUNTIME/env/bin/python" search.py \
  --storage-path "$TUNE_RUNTIME/experiments" \
  --run-name reference \
  --output-dir "$TUNE_RUNTIME/exports/$TUNE_JOB" \
  --fail-step-size 0.3
ray job status --address "$RAY_API" "$TUNE_JOB"
ray job logs --address "$RAY_API" "$TUNE_JOB"
```

Require `SUCCEEDED`; a submission banner is not completion evidence. Preserve
both the export and, for local storage, the Tune experiment directory before
teardown:

```bash
mkdir -p "$RESULTS"
rsync -az "$TUNE_CLUSTER:$TUNE_RUNTIME/exports/$TUNE_JOB/" "$RESULTS/export/"
rsync -az "$TUNE_CLUSTER:$TUNE_RUNTIME/experiments/reference/" "$RESULTS/experiment/"
# After returning to the repository root:
npa/.venv/bin/python "$TUNE_EXAMPLE/inspect_results.py" "$RESULTS/export"
(cd "$RESULTS/export" && sha256sum -c SHA256SUMS)
```

An S3 experiment path makes Tune's checkpoints and experiment state survive
host teardown, but the checksum-bound summary still must be copied to durable
storage. A local experiment path is not durable after `sky down`.

## Failure and recovery boundaries

The injected trial failure is recovered automatically from its last checkpoint.
That validates one application exception, not node loss, head loss, or exactly-once
external side effects. A second failure exceeds the declared budget and the
application refuses to export a successful result.

If the Jobs driver, head, or service is interrupted after Tune has persisted
experiment state, rerun the exact recipe with `--restore`, the same
`--storage-path` and `--run-name`, and a fresh output directory. The application
uses `Tuner.can_restore` and `Tuner.restore(..., resume_errored=True)`. Restore
does not accept a changed search space and cannot fix deterministic application
bugs. Treat Tune experiment state as trusted executable state; never restore a
path writable by an untrusted party.

Stopping the Ray Job is separate from Tune restoration:

```bash
ray job stop --address "$RAY_API" "$TUNE_JOB"
ray job status --address "$RAY_API" "$TUNE_JOB"
```

Poll the exact ID to `STOPPED`; a stop request is not terminal evidence. An
interrupted export is not a completed result. Preserve any partial evidence and
use a fresh output directory after restoration.

## Cleanup

Cancel exact Ray Jobs before their hosting service. Then cancel only the
recorded SkyPilot service task and remove only the named development cluster:

```bash
"$NPA_SKYPILOT_BIN" cancel "$TUNE_CLUSTER" <service-task-id> --yes
"$NPA_SKYPILOT_BIN" down "$TUNE_CLUSTER" --yes
ssh -S "$TUNE_SOCKET" -O exit "$TUNE_CLUSTER"
```

Verify the owned pod and tunnel are absent. Retain shared Kubernetes and the
SkyPilot API unless their separate owner explicitly authorized teardown.

## Agent guidance and compatibility

When operating this reference, keep the native ownership split visible. Start
with the local CPU run; before cloud work, run `npa workbench health preflight
--checks nebius --json` and verify the exact target context. If S3 is selected,
verify bucket ownership plus write/read/delete access to the exact run prefix.
Track the submission ID, service-task ID, cluster name, tunnel socket, storage
path, and result destination privately. Never infer GPU, multi-node, or workload
quality evidence from this numeric CPU search.

The implementation follows the Ray 2.58 primary sources for
[`Tuner`](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/tune/tuner.py),
[`ResultGrid`](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/tune/result_grid.py),
and [Tune fault tolerance](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/tune/tutorials/tune-fault-tolerance.rst).
Ray is Apache-2.0. The example downloads no model or dataset, builds no image,
and adds nothing to NPA's public image catalog.
