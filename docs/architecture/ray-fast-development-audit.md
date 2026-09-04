# Ray and faster source iteration

Audit date: 2026-09-04. This document describes existing NPA behavior and
development options; it introduces no scheduler, container, or public CLI.

## Recommendation

Keep SkyPilot as NPA's workflow execution engine. For a Python change with
unchanged dependencies, first run the changed source in the existing compatible
container using a read-only source mount. For a representative remote workflow,
use NPA's existing content-addressed source staging and opt-in source overlay.
Both paths can execute changed code without building an image.

Use Ray where the workload benefits from distributed Python tasks, model actors,
or persistent batched inference. NPA already has that last path for Cosmos3-Nano.
A general Ray migration would add operational work without being necessary to
remove the container build from the development loop.

The [reproduction and measured workload](../testing/fast-source-iteration.md)
exercise dataset ingestion, validation, curation, and querying on synthetic
records. This is a CPU data-processing workload; it does not establish GPU ABI
compatibility, model throughput, or remote scheduling performance. Changes to
GPU execution still need the intended model or simulator on its supported
Nebius GPU, using the existing access, image, placement, and cleanup gates.

## What already exists

| Mechanism | Repository evidence | What it provides |
| --- | --- | --- |
| Editable local package | [CONTRIBUTING.md](../../CONTRIBUTING.md), [pyproject.toml](../../npa/pyproject.toml) | `npa/.venv` with `npa[dev]`; ordinary Python edits are visible without reinstalling the package. |
| Source staging | [src_staging.py](../../npa/src/npa/orchestration/npa_workflow/src_staging.py), `ensure_npa_source` | Fingerprinted source prefixes, manifest written last, reuse of a matching manifest, and inclusion of dirty tracked and nonignored untracked files. |
| Baked-image source overlay | [skypilot_render.py](../../npa/src/npa/orchestration/npa_workflow/skypilot_render.py), `default_npa_setup` and `render_task_run_script` | `NPA_SRC_OVERLAY=1`, editable installation, explicit source precedence on `PYTHONPATH`, and interpreter recording across setup/run shells. |
| Workflow lifecycle | [runtime.py](../../npa/src/npa/orchestration/npa_workflow/runtime.py), [run lifecycle](../run-lifecycle.md) | Durable workflow/wave identity, output checks, retries, and cancellation through the existing SkyPilot runtime. |
| Persistent Cosmos inference | [ray_server.py](../../npa/src/npa/workbench/cosmos/ray_server.py), [service guide](../workbench/cosmos3-ray-serve.md) | NVIDIA `OmniModelDeployment`, native Ray Serve batching, resident model weights, and authenticated NPA HTTP ingress. |
| Durable Ray service client | [ray_serve.py](../../npa/src/npa/workbench/cosmos/ray_serve.py), [cosmos3-ray-batch.yaml](../../npa/workflows/workbench/npa-workflows/cosmos3-ray-batch.yaml) | A CPU client persists request, response, media hashes, and provenance through S3; the service must already be running. |

The Cosmos service calls `ray.init()` and binds upstream deployment handles; it
does not expose a general NPA Ray Jobs developer API. Its
[Dockerfile](../../npa/docker/workbench/cosmos3-ray-serve/Dockerfile) pins Ray
2.46.0 and the Cosmos framework revision. KubeRay Terraform modules exist in the
vendored infrastructure recipe, but
[mk8s_render.py](../../npa/src/npa/cluster_backends/mk8s_render.py) explicitly
disables both `enable_kuberay_cluster` and `enable_kuberay_service`. Their presence
is not evidence of an enabled NPA cluster service.

The older [May workflow-engine recommendation](workflow-engine-recommendation-20260514T233740Z.md)
is historical. Its Argo recommendation predates the current SkyPilot-only
execution contract.

## The immediate development loop

Create the repository environment once, following the declared setup:

```bash
set -euo pipefail
uv venv --python 3.12 npa/.venv
uv pip install --python npa/.venv/bin/python -e 'npa[dev,adapter]'
npa/.venv/bin/python -m pytest npa/tests/orchestration/npa_workflow/test_src_staging.py -q
```

For an existing container, mount only the source needed by the change. Select
the exact image digest and its dependency-complete interpreter; vendor runtimes
can have multiple Python installations. This example verifies which source the
container imports before running the full workload in the linked reproduction:

```bash
set -euo pipefail
# NPA_DEV_IMAGE is an existing compatible image reference pinned by digest.
# NPA_DEV_PYTHON is its absolute Python interpreter path.
docker run --rm --network none \
  --mount "type=bind,source=$PWD/npa/src,target=/npa-dev/src,readonly" \
  --env PYTHONPATH=/npa-dev/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint "$NPA_DEV_PYTHON" "$NPA_DEV_IMAGE" \
  -c 'import hashlib,pathlib,npa.workbench.dataset.validation as m; p=pathlib.Path(m.__file__); print(p); print(hashlib.sha256(p.read_bytes()).hexdigest())'
```

The image retains its installed dependencies and the host source is read-only
inside the container. Docker bind mounts refer to the daemon host, so this
command assumes a local daemon; a remote daemon cannot mount the client's local
checkout. A source mount does not update installed distribution metadata,
console entrypoints, native extensions, or files copied elsewhere during an
image build. Use the interpreter's `-m npa` entrypoint for changed NPA Python
code, and mount a separate output directory when running an artifact-producing
workload. [Docker bind-mount documentation](https://docs.docker.com/engine/storage/bind-mounts/)

A fresh process is part of this loop. Editing a mounted file does not replace a
module already imported into a long-lived service or an existing model actor.
Record source hashes and behavior, rather than using a successful container
start as proof that the changed code ran.

## Remote execution with existing source staging

`workflow stage-src` is the supported staging entrypoint. Prefer it to
`scripts/stage-npa-src.sh`: the older shell helper uploads to a mutable prefix
and lacks the current manifest and sensitive-file filters.

For a configured project and a compatible single-stage development workflow:

```bash
set -euo pipefail
# Resolve these variables from private operator configuration.
umask 077
npa/.venv/bin/npa workbench health preflight --checks s3 --json
npa/.venv/bin/npa workbench workflow stage-src \
  --bucket "$NPA_DEV_BUCKET" \
  --prefix "$NPA_DEV_PREFIX" \
  --project "$NPA_PROJECT" \
  --run-id "$NPA_DEV_RUN_ID"

# stage-src persists the exact source URI in this project's local configuration.
# Clear older explicit overrides so submit selects that freshly staged source.
unset NPA_SRC_S3_URI NPA_E2E_NPA_SRC_S3_URI
npa/.venv/bin/npa workbench workflow validate-spec "$NPA_DEV_SPEC"
NPA_SRC_OVERLAY=1 npa/.venv/bin/npa workbench workflow submit "$NPA_DEV_SPEC" \
  --project "$NPA_PROJECT" --run-id "$NPA_DEV_RUN_ID" \
  --image "$NPA_DEV_IMAGE" --plan-only
NPA_SRC_OVERLAY=1 npa/.venv/bin/npa workbench workflow submit "$NPA_DEV_SPEC" \
  --project "$NPA_PROJECT" --run-id "$NPA_DEV_RUN_ID" \
  --image "$NPA_DEV_IMAGE"
```

Keep the staging and submission output private: it includes exact storage and
run locations. The example assumes the project's infrastructure and workload
inputs are already configured; model and GPU workflows also require their
tool-specific health/access and GPU placement checks. For a workflow with
different tool images, use repeatable `--image-override TOOL_REF=IMAGE` rather
than one global image. Preserve artifacts and cancel the run's managed jobs
before destroying only resources created for its development run.

There are three separate controls in
[the submission CLI](../../npa/src/npa/cli/workbench/workflow/__init__.py):

- Automatic staging handles tasks that lack an image. A stale saved source
  fingerprint can trigger restaging; an explicit environment URI wins over the
  saved setting and is treated as a deliberate selection.
- `submit --stage-src` forces staging. It does **not** set `NPA_SRC_OVERLAY`.
  Staging source alone does not replace the NPA package in an existing image.
  Global `--image` and `--stage-src` are rejected together; standalone staging
  followed by the submission above avoids that conflict.
- `NPA_SRC_OVERLAY=1` opts compatible baked images into overlay installation.
  `config.require_baked_npa=true` intentionally bypasses this mechanism and
  requires a digest-pinned image with an exact source SHA. Content Agents also
  uses a separate narrow baked runtime that bypasses generic overlay setup.

Restage after each code revision and use a new run identity/output prefix for a
new comparison. A new source fingerprint changes the source identity; it is not
a compatible resume of a completed run with the old source. This path removes
the image build, but retains source transfer, image pulling, scheduling, setup,
and any model initialization.

## Optional warm development cluster

For an operator-owned, isolated development cluster already launched through
SkyPilot, its pinned `exec` command can sync `workdir` and run changed code while
skipping provisioning, `setup`, and `file_mounts` synchronization. This behavior
is present in [SkyPilot 0.12.2's execution implementation](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/execution.py#L772).
Use NPA's isolated executable, not an arbitrary `sky` on `PATH`:

```bash
set -euo pipefail
npa/.venv/bin/npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
"$NPA_SKYPILOT_BIN" exec "$NPA_DEV_CLUSTER" "$NPA_DEV_TASK_YAML"
```

The customer-owned task YAML should set `workdir` to a reviewed source-only
directory and `run` to the real workload command using the existing worker
interpreter. Do not turn this into a parallel repository workflow-authoring
surface: NPA pipelines remain `npa.workflow/v0.0.1` specifications.

SkyPilot copies workdir files to `~/sky_workdir`. Its documented default filters
include `.gitignore` and `.git/info/exclude`; adding `.skyignore` replaces those
filters. Review the actual payload and keep operator configuration, datasets,
credentials, caches, and generated artifacts outside it. Changes to setup or
dependencies require `launch`; changed file mounts require synchronization via
`launch`, optionally `--no-setup` when setup truly is unchanged.
[SkyPilot source and artifact syncing](https://docs.skypilot.ai/en/latest/examples/syncing-code-artifacts.html)

This is an upstream development option, not a new NPA-managed service or a
validated performance result in this audit. The operator owns cluster lifetime,
job reconciliation, artifacts, and cleanup. Reusing a machine does not by itself
keep a model resident when each command starts a new process.

## Where Ray can help

| Candidate | Useful fit | Boundary |
| --- | --- | --- |
| Existing Cosmos3-Nano Ray Serve | Repeated generation requests that can share loaded weights and native request batching | Use the shipped service/client contract. Do not confuse it with Cosmos3-Super's vLLM-Omni runtime. |
| Ray Data with model actors | A measured backlog of image/video inference or preprocessing that benefits from partitioning and actor reuse | A future integration needs representative throughput/memory evidence, checkpoint/output semantics, GPU placement, and lifecycle handling. |
| Ray Jobs with `working_dir` / `py_modules` | Python application revisions submitted to a compatible existing Ray cluster | Code packaging is useful, but NPA has no general product surface for it today. Keep the existing workflow controller. |

Ray Data's callable-class actor pattern loads model state once per actor and
reuses it for batches. That gives a concrete reason to consider it for repeated
inference. Fine-grained remote tasks can instead add enough scheduling overhead
to slow the job; benchmark the actual work units before choosing Ray.
[Ray batch inference](https://docs.ray.io/en/latest/data/batch_inference.html),
[fine-grained task overhead](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html)

Ray Jobs accepts an entrypoint and runtime environment, survives client
disconnection, and remains bound to the Ray cluster's lifetime. The submitter
must handle resubmission/retries. It therefore cannot simply replace NPA's
durable workflow, output verification, and cross-tool cancellation contracts.
[Ray Jobs overview](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html),
[Jobs quickstart](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html)

Ray `working_dir` distributes code and `py_modules` distributes Python modules.
A Jobs runtime environment covers the driver and child work; calling
`ray.init(runtime_env=...)` does not alter the already-running driver. Local
working-directory uploads have a documented 500 MiB limit, and dependencies/code
are cached per node. Keep large stable dependencies in the image. These runtime
environments do not make an incompatible native dependency stack compatible.
[Ray dependency handling](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)

Ray Serve updates also have restart boundaries: rerunning `serve run`
redeploys deployments, and changing an application's runtime environment or
import path restarts them. Supported `user_config`/`reconfigure` updates are a
different operation. A source refresh must not promise that already-loaded model
weights stay in GPU memory.
[Serve development workflow](https://docs.ray.io/en/latest/serve/advanced-guides/dev-workflow.html),
[Serve application updates](https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html)

The current upstream Ray pages describe versions newer than the image's pinned
2.46.0. Any proposed Ray Jobs or runtime-environment feature needs verification
against the exact selected version before it becomes a supported NPA command.

## Rebuild, reproducibility, and isolation boundaries

Rebuild and validate the image when changing Python ABI, Torch/CUDA libraries,
compiled extensions, OS packages, entrypoints, dependencies or their lockfile,
vendor source baked outside the overlaid package, or installed package metadata.
Image rebuilds remain the release and immutable validation boundary even when a
source overlay proves a development change.

The current overlay puts source first on `PYTHONPATH` before editable
installation. If a non-root image's installed launcher is immutable and the
`--no-deps` install fails, it creates `/tmp/npa-overlay-venv` with
`--system-site-packages`, installs the overlay there without dependencies, and
records the selected interpreter. This fallback has already landed; a new
container-development feature need not duplicate it. The virtualenv deliberately
sees the image's packages and is not a security boundary.

The overlay still falls back to dependency resolution if `npa.cli.main` cannot
import. It is therefore not a guarantee that
the runtime's dependency set stays frozen. Record the interpreter and relevant
package versions for the measured run, and investigate dependency fallback
rather than labeling that run equivalent to the original image.

Source staging fingerprints paths, executable bits, and file contents, and its
manifest is a useful upload completion marker. However, current verification
checks the manifest, not every uploaded file. Worker downloads do not recompute
the fingerprint or restore executable bits; the download loop also lacks exact
prefix-delimiter and traversal checks. Fixed worker scratch paths can retain
stale files if manually reused. Use restricted source-prefix writers and fresh
task containers; do not describe the current manifest as cryptographic proof of
the complete executed tree.

For a reproducible experiment, retain the image digest, source fingerprint,
dirty patch or source snapshot, actually imported module path/hash, interpreter
and dependency versions, input hash, workload parameters, command exit codes,
output hashes, and timing boundaries. A Git SHA alone omits uncommitted edits;
an image label still describes the baked source after an overlay. Keep concrete
infrastructure locations and credentials in access-controlled evidence.

Runtime environments and Ray namespaces are not tenant isolation boundaries.
Ray requires trusted code and external network/access controls, with separate
clusters where workloads require isolation. NPA's Cosmos HTTP bearer token
protects its ingress routes; it does not authenticate the Ray Dashboard, Jobs,
or Client endpoints. Do not expose those control ports to untrusted clients or
assume current Ray authentication features exist in the pinned older image.
[Ray security guidance](https://docs.ray.io/en/latest/ray-security/index.html)

Docker source mounts likewise grant code access to the process's existing
capabilities. Mount only reviewed source, use a separate output directory,
provide only workload-required credentials, and avoid Docker socket or broad
operator-home mounts. A read-only mount prevents writes through that mount; it
does not sandbox untrusted Python code.

## Follow-up priorities

1. Make the existing staging/overlay distinction discoverable in CLI help and
   operator docs; it is currently easy to stage code without executing it.
2. Add worker verification of an immutable per-file manifest, safe relative
   paths, executable modes, and an emitted source receipt. Preserve explicit
   dependency boundaries instead of silently broadening resolution.
3. Consider an isolated warm-cluster development helper only after measurements
   show scheduling/setup dominates the existing loop. Use pinned SkyPilot
   behavior and explicit resource ownership.
4. Add a Ray-specific lane only for a workload with measured actor/batching or
   distributed-data benefit. Avoid deploying KubeRay merely to transfer source.

These are follow-up opportunities, not implementation claims in this change.

## Measured evidence

Three real container executions each ingested and validated 100,000 synthetic
sensor metadata records, then curated and queried their manifest. A temporary
quality-comparison change selected 10,000 records instead of 12,500; restoring
the source restored all 12,500 IDs and the original module hash. Assertions
checked the complete selected-ID sequence, validation reports, lineage, and the
actually imported file hash. All three container exit codes were zero.

The changed-code iteration took **6.893 seconds** including container startup
(**5.918 seconds** inside the workload). Baseline and restored wall times were
5.653 and 6.905 seconds. The image and isolated dependency directory stayed the
same, with **zero image builds**. The image was cached; dependency preparation
was excluded. This is evidence of a working development loop, not a measured
speedup over rebuilding or a Ray benchmark.

See the [complete reproduction](../testing/fast-source-iteration.md) for the
executed workload, source-change proof, artifacts, commands, and measured limits.

## Repository validation

After reconciliation with current main, repository Ruff, 13,479-test
collection, 2,564 combined guardrail/source-staging/SkyPilot-renderer/dataset
tests, CLI documentation drift, and the reproduction's shell syntax/link checks
passed. The documentation embeds the executed workload and
runner; staged confidentiality and credential scans passed.

The full `make test` run reported 12,929 passed, 36 skipped, 12 deselected,
1 xpassed, and three failures. All three reproduced with the working tree
restored exactly to base `918196ed8c96c354a95f297eb1c188257bbecd2a`, verified by
an empty `git diff origin/main` before rerunning them:

- Two Rerun visualization tests selected a broken host executable. Both passed
  after prepending this checkout's `npa/.venv/bin` to `PATH`.
- `test_run_cosmos_transfer_names_gated_access_denial_without_leaking_prompt`
  omits the guardrail-preparation mock used by adjacent tests. It reaches real
  preparation before its mocked vendor subprocess and fails because the test
  environment lacks `huggingface_hub`. Installing that optional runtime package
  would introduce a real fetch instead of repairing test isolation.

The same Cosmos failure is present in the base's
[Python 3.12 CI job](https://github.com/nebius/nebius-physical-ai/actions/runs/33911450997/job/101148584343).
Its test, implementation, conftest, and CI workflow remain unchanged in the
subsequent main revision used by this audit.

These are qualified base/environment failures, not a claim that the complete
suite passed. Product code and tests remain unchanged. Detailed reproduction
and file-identity evidence is retained outside Git.
