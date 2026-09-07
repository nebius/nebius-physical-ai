# Train on two GPUs with native Ray Train

This reference learns a synthetic linear regression with PyTorch DDP on two
B200 hosts, one CUDA rank per host. It saves native Ray Train checkpoints to
S3 and exports the final model, optimizer momentum, rank evidence, metrics and
a factual Rerun recording. Ray Jobs handles source delivery and application
control; SkyPilot owns the hosts and the Ray service task.

This is a tool-specific development example, outside the `npa.workflow`
catalog. It introduces no NPA CLI, service, container or job controller.
Production pipelines composed of several capabilities belong in `npa.workflow`.

## Prepare the platform and storage

Use the existing [private SkyPilot platform setup](../ray-clip-development/platform/README.md).
It supplies an isolated Kubernetes namespace, private kubeconfig, SkyPilot
0.12.2 API and `NPA_SKYPILOT_BIN`. Keep its authenticated SSH and NetworkPolicy
boundary. The Ray Jobs dashboard binds to loopback; never expose it publicly.

The selected Nebius Kubernetes cluster must have two compatible B200 GPU nodes.
Use `npa workbench health preflight --checks nebius --json` before provisioning,
then the [GPU provisioning procedure](../../../../skills/tools/gpu-cluster-provisioning/SKILL.md).
Confirm the actual accelerator spelling with `"$NPA_SKYPILOT_BIN" gpus list`.
Change `resources.accelerators` in a private copy of `cluster.yaml` if the
verified cluster uses another spelling. Two ranks on one host prove distributed
training, but do not prove the two-host claim made by this profile.

Select a workload bucket owned by the authorized project and a fresh run prefix.
Verify provider ownership, then write/read/delete a probe in that exact prefix
using the executing credentials and endpoint. A generic S3 list or authentication
check does not prove write permission. Keep storage credentials outside this
source directory. Provide the following variables in the private operator shell:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL_S3     # verified HTTPS endpoint for this bucket
AWS_DEFAULT_REGION     # bucket signing region
TRAIN_STORAGE_URI      # s3://<your-bucket>/<fresh-training-prefix>
```

The SkyPilot launch forwards the named variables without putting their values
on the command line. Both Ray hosts inherit them before service startup.
Do not put keys in a Jobs runtime-env file or pass them explicitly to
`pyarrow.fs.S3FileSystem`, which serializes explicit keys with the filesystem.
Only trusted code may share this Ray cluster and its storage identity.

## Start the application Ray service

From this directory, select a unique development-cluster name:

```bash
export TRAIN_CLUSTER=synthetic-train
export KUBE_CONTEXT="$(kubectl config current-context)"
"$NPA_SKYPILOT_BIN" launch --yes --detach-run -c "$TRAIN_CLUSTER" \
  --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY \
  --env AWS_ENDPOINT_URL_S3 --env AWS_DEFAULT_REGION cluster.yaml
"$NPA_SKYPILOT_BIN" queue "$TRAIN_CLUSTER"
```

Record the exact service-task ID from the queue. Preparation installs application
Ray **2.58.0** with Train V2, Arrow **23.0.1**, NumPy **2.4.6** and Rerun
**0.31.4**, retaining Torch **2.12.1+cu130** from the immutable upstream image.
The image digest and the separately recorded dependency freeze identify different
parts of the runtime; this is not a fully hash-locked transitive installation.
There is no model download or external training dataset.
The Jobs driver and every new or restarted training worker also check the exact
Torch version before training, so environment drift after preparation fails closed.

Use the SkyPilot-generated SSH alias to reach Jobs through one owned tunnel:

```bash
export TRAIN_SOCKET="$HOME/.ssh/${TRAIN_CLUSTER}-jobs.sock"
ssh -M -S "$TRAIN_SOCKET" -fNT -o ExitOnForwardFailure=yes \
  -L 18265:127.0.0.1:8265 "$TRAIN_CLUSTER"
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:18265
# Install ray[default,train]==2.58.0 in your own client environment.
ray job list --address "$RAY_API"
```

Wait for Jobs readiness; SkyPilot launch completion alone does not establish it.
Ray's native Jobs client gives those two environment variables precedence over
an explicit address. Keep them unset in the operator shell; the live suite
rejects them before client contact. Each submitted driver receives its separate
application GCS address from the native Jobs service.
Check both hosts joined the application GCS on port 6381, with two total GPUs.
Application ports are separate from SkyPilot's management Ray; never use
ambient discovery or a broad `ray stop`.
The driver propagates its selected GCS address through the native runtime
environment to Train's actors. This also keeps the head-pinned placement cleanup
actor's State API on the application service when management Ray is present.

## Train, inspect, recover

```bash
ray job submit --address "$RAY_API" --submission-id train-baseline \
  --working-dir . -- /tmp/ray-train-env/bin/python train.py \
  --storage-path "$TRAIN_STORAGE_URI" --run-name baseline \
  --output-dir /tmp/train-baseline
ray job status --address "$RAY_API" train-baseline
ray job logs --address "$RAY_API" train-baseline
```

The default 32 steps train distinct deterministic shards with SGD momentum.
Every step records globally reduced loss, gradient norm, parameter change,
applied learning rate, synchronized samples/second and actual CUDA rank evidence.
An exact-zero gradient is valid when SGD momentum still produces a measured
parameter update; every recorded step must retain that update evidence.
Every fourth step, plus the last, saves the full journal and model/optimizer.
Export reloads the checkpoint, checks held-out improvement and matches its
parameters against the final CUDA-rank hashes. These measurements demonstrate
execution and optimizer progress, not a robotics-policy benchmark.

Each new training run needs a fresh `--run-name`. A retry of the same run uses
the same native `RunConfig(name, storage_path)` pair. The saved recipe must
match exactly. Run the worker-failure exercise under a separate name:

```bash
ray job submit --address "$RAY_API" --submission-id train-recovery \
  --working-dir . -- /tmp/ray-train-env/bin/python train.py \
  --storage-path "$TRAIN_STORAGE_URI" --run-name recovery \
  --output-dir /tmp/train-recovery --fail-after-step 8
```

Rank one raises after native Train confirms checkpoint 8 is committed. Native
Train restarts the worker group; both ranks load its model and optimizer and
continue at step 9. The final journal must contain each step exactly once and
prove both restored ranks. Do not use V1 `TorchTrainer.restore()` or
`resume_from_checkpoint`: Ray 2.58 defaults to Train V2.

## Preserve and view the outputs

Exports live under `<storage-prefix>/<run-name>/exports/`. The application uploads
`state.pt`, `metrics.json`, `metrics.rrd`, `result.json`, then `SHA256SUMS`, checking
each object with read-after-write. Conflicting existing exports are refused.
Do not launch two drivers concurrently with the same native run name.

If S3 export delivery fails after training, preserve the host's export directory.
Retry those exact bytes without retraining or regenerating the RRD:

```bash
ray job submit --address "$RAY_API" --submission-id train-export-retry \
  --working-dir . -- /tmp/ray-train-env/bin/python inspect_results.py \
  /tmp/train-baseline --publish "$TRAIN_STORAGE_URI/baseline/exports"
```

An interrupted upload is safe to retry with the same export; differing bytes
are refused. If the host and its export were lost before publication finished,
recover the native checkpoint and write a fresh export destination, preserving
the partial earlier attempt. Rerun recordings carry event timestamps, so a
regenerated recording is a new artifact even when its metric values agree.

From an installed NPA checkout, use its interpreter and a fresh durable directory:

```bash
npa/.venv/bin/python npa/workflows/workbench/ray-train-synthetic/inspect_results.py \
  "$HOME/ray-train-results/baseline" \
  --download "$TRAIN_STORAGE_URI/baseline/exports"
npa/.venv/bin/rerun rrd verify "$HOME/ray-train-results/baseline/metrics.rrd"
npa/.venv/bin/rerun rrd print -vv "$HOME/ray-train-results/baseline/metrics.rrd"
npa/.venv/bin/rerun "$HOME/ray-train-results/baseline/metrics.rrd"
```

The inspector compares every decoded optimizer step and scalar value with the
journal, including CUDA rank counts and checkpoint events. The recording uses
the application run name as its recording ID and logs no infrastructure IDs.
Repeat the download into another fresh directory after compute teardown to
prove the artifacts outlive their hosts. Native checkpoint recovery can also
reopen `ray.train.Result.from_path` with the same custom S3 filesystem; its
`checkpoint.as_directory()` downloads through that filesystem. Preserve Jobs
terminal status separately: `Result.from_path()` does not restore failure status.

The committed live suite runs baseline training, checkpoint recovery and active
cancellation against this prepared runtime. In an owner-only file outside the
checkout, set `address` to the loopback Jobs URL, `storage_uri` to the verified
fresh S3 prefix, and `evidence_dir` to an owner-only local directory. Keep the
same verified AWS variables in the test process. From the repository root:

```bash
export NPA_RAY_TRAIN_LIVE_CONFIG="$HOME/.config/ray-train/live.json"
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_ray_train_synthetic_live.py -v
```

Install the pinned Ray client and matching Torch in this checkout's own venv
first; they are optional reference dependencies. The suite independently loads
the downloaded model and SGD momentum, recomputes held-out loss, verifies two
distinct B200 hosts, decodes every RRD step and checks that recovered parameters
match uninterrupted training. It records intent before each native submission
and attempts cleanup for every owned ID even after a transport or assertion
failure. The hosting task, platform and provider teardown remain operator-owned.
Cancellation also verifies that the captured placement groups are removed and
the detached Train cleanup actor exits; terminal Jobs status alone is insufficient.

## Stop only this application

For an active run, `ray job stop --address "$RAY_API" <exact-submission-id>`
must reach `STOPPED`; confirm using native status and logs. Download useful
evidence first when possible. The committed live suite exercises a real active
training cancellation after checkpoint progress, in addition to success and
worker recovery.

```bash
# Stop any remaining exact Ray submission IDs before the hosting task.
"$NPA_SKYPILOT_BIN" cancel "$TRAIN_CLUSTER" <service-task-id> --yes
"$NPA_SKYPILOT_BIN" down "$TRAIN_CLUSTER" --yes
ssh -S "$TRAIN_SOCKET" -O exit "$TRAIN_CLUSTER"
```

Verify the owned pods are absent independently. Retain the shared Kubernetes
cluster/platform unless you also own their lifecycle. Destroy separately owned
validation infrastructure only after workloads stop; retain verified S3 outputs.

## Compatibility and licenses

The [Ray 2.58 Train storage contract](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/train/user-guides/persistent-storage.rst),
[V2 report API](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/train_fn_utils.py)
and [Torch 2.12.1 serialization](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/serialization.py)
govern this implementation. The committed live test is
`npa/tests/e2e/test_ray_train_synthetic_live.py`; it requires an explicitly
selected private runtime configuration and performs real GPU work.

This repository ships application source only. Ray and Arrow are Apache-2.0,
PyTorch uses its [BSD-style license](https://github.com/pytorch/pytorch/blob/v2.12.1/LICENSE),
and Rerun is Apache-2.0/MIT. The upstream image supplies CUDA/cuDNN under their
respective [NVIDIA terms](https://docs.nvidia.com/cuda/eula/index.html).
Runtime use requires the operator's acceptance; no new image is built or
redistributed here, and the NPA public image catalog is unchanged. Inputs are
generated tensors and outputs are this run's trained parameters and measurements.
