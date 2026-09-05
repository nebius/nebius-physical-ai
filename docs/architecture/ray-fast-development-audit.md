# Ray + SkyPilot for application development

Updated: 2026-09-05. The original CPU source-mount audit is retained below as
historical evidence. The expanded integration adds an explicit image runtime and
a reproducible CLIP application recipe. **Medium and complex GPU checks passed,
including durable workflow completion: one GPU on one node, then two GPUs
across two Kubernetes worker nodes.** Earlier startup and lifecycle failures are recorded below.

## Recommendation

Use SkyPilot to allocate Nebius GPU resources and own the NPA workflow lifecycle;
use an independent application Ray cluster for customer Jobs, `runtime_env`
source delivery, and GPU actors. This gives customers upstream Ray APIs while
retaining `npa.workflow/v0.0.1` for durable workflow identity, credentials,
artifacts, cancellation and infrastructure ownership. SkyPilot itself documents
this separation; application Ray must not connect to its management cluster.
[SkyPilot distributed Ray example, v0.12.2](https://github.com/skypilot-org/skypilot/blob/v0.12.2/examples/distributed_ray_train/ray_train.yaml)

For a direct Python script, native SkyPilot `workdir` and `exec` already remove
the image build from compatible source iterations. Ray is useful when the
application also needs distributed tasks, actor model reuse, Jobs status, and a
familiar source-submission contract. A new infrastructure autoscaler or KubeRay
migration is unnecessary for this path. Native `exec` performance has not been
measured here; the comparison is architectural.
[SkyPilot execution, v0.12.2](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/execution.py#L772)

## Existing mechanisms and the scoped addition

| Mechanism | Repository surface | Support boundary |
| --- | --- | --- |
| Editable NPA installation | [Contributing](../../CONTRIBUTING.md), [dependencies](../../npa/pyproject.toml) | Fast local Python development with `npa/.venv`; GPU changes still require real compatible GPU work. |
| Existing NPA source staging/overlay | [Source staging](../../npa/src/npa/orchestration/npa_workflow/src_staging.py), [renderer](../../npa/src/npa/orchestration/npa_workflow/skypilot_render.py) | NPA-owned source distribution for NPA development; staging and activating an image overlay are distinct operations. |
| Existing Cosmos3-Nano Ray Serve | [Service](../../npa/src/npa/workbench/cosmos/ray_server.py), [guide](../workbench/cosmos3-ray-serve.md) | Persistent NVIDIA framework inference, resident weights and authenticated inference ingress; not a general Jobs endpoint or Cosmos3-Super runtime. |
| Image-owned application runtime | [Spec](../../npa/src/npa/orchestration/npa_workflow/spec.py), [renderer](../../npa/src/npa/orchestration/npa_workflow/skypilot_render.py) | New explicit `config.runtime_setup: image` for raw `run.argv`/`run.shell` stages using a registry-qualified immutable image digest. |
| Ray application recipe | [Session workflow](../../npa/workflows/workbench/npa-workflows/ray-clip-development-session.yaml), [Jobs application](../../npa/workflows/workbench/ray-clip-development/README.md) | Standard Ray Jobs carries reviewed application source; the NPA session owns preparation, S3 receipts and scoped shutdown within operator-configured network isolation. Medium and two-node complex GPU integration passed, including durable completion. |

Image mode skips NPA installation, source-URI injection, `PYTHONPATH` replacement
and interpreter shims. It retains the existing workflow, resource, secret and
model-cache handling. It rejects native `toolRef` stages, mutable images, and a
conflicting `require_baked_npa` setting. It does not invent a baked NPA source
attestation: the image owns the application environment. SkyPilot worker
bootstrap prerequisites and application dependency/model checks still apply.
[Image runtime contract](../workbench/npa-workflow-guide.md#application-runtimes-in-immutable-images)

The associated SkyPilot CLI fix handles pip's long-interpreter-path launcher
header when an isolated runtime is atomically moved from staging into place.
Real generated launchers execute successfully after relocation, including paths
with spaces; the existing ordinary-shebang behavior remains covered. This fixes
a preparation blocker, not a Ray application feature.

The [older May workflow-engine recommendation](workflow-engine-recommendation-20260514T233740Z.md)
is historical: its Argo recommendation predates the current SkyPilot contract.
Vendored KubeRay infrastructure definitions are also not an enabled NPA service;
[the managed Kubernetes renderer](../../npa/src/npa/cluster_backends/mk8s_render.py)
disables their cluster/service flags.

## Medium and complex checks

The [reproduction guide](../testing/fast-source-iteration.md) uses the repository's
real Workbench `udf_clip_embedding`, a pinned public CLIP snapshot, and synthetic
rendered RGB inputs. It reuses an official PyTorch 2.12.1/CUDA 13 runtime image
and prepares pinned application dependencies once per session. The UDF is a
fourth reviewed source file submitted through Ray, alongside the application,
worker and validation modules; it is held fixed across crop-policy revisions.
The workload performs inference, materializes Parquet and Lance outputs, and
validates retrieval. Both live qualification checks completed on the measured
stack, with final results below.

The first GPU session attempt exposed a prerequisite the earlier CPU audit did
not test: the published LanceDB image runs as `ubuntu` without `sudo`, so
SkyPilot's SSH bootstrap exited before application Ray or GPU inference started.
An anonymously pullable image is not sufficient evidence that SkyPilot can
bootstrap it. The revised recipe uses the existing root PyTorch image instead
of rebuilding LanceDB; its application environment is prepared at session start.
Root inside the isolated workload pod is not a privileged pod or host-root
access, and must still be permitted by the target cluster's security policy.
[SkyPilot 0.12.2 Kubernetes bootstrap](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/templates/kubernetes-ray.yml.j2)

| Check | Intended scope | Required evidence | Live result |
| --- | --- | --- | --- |
| Medium | One GPU node, one actor, 4,096 records per source revision | Baseline → changed image crop → restored source; every worker's imported source hash, changed vectors, restoration and output completeness | Passed: three successful Jobs; source/output/recovery/cancellation, artifact persistence, process cleanup and durable terminal status verified |
| Complex | Two GPU actors on two Kubernetes worker nodes, one GPU each, 16,384 records per revision | CPU partitions → concurrent GPU inference → aggregation/retrieval; source propagation, owned-actor failure/replacement, committed-shard replay and exact-job cancellation | Passed: source/output/recovery/cancellation, head driver placement, durable artifacts, both node shutdowns and terminal workflow/stage status verified |

### Measured medium result

The final medium session passed both live SkyPilot status and the durable NPA
stage receipt: **SUCCEEDED**, with its end timestamp present. One Kubernetes worker node
with one **RTX PRO 6000 Blackwell Server Edition** GPU completed baseline,
changed crop, restored source and a separate cancellation job in **104.720
seconds** of client wall time. Each source job processed 4,096 records in 64
shards, persisted all 4,096 Lance rows and passed five retrieval checks. The
image was reused with **zero image builds**.

| Revision | SDK package/upload/submit (s) | Total iteration (s) |
| --- | ---: | ---: |
| Baseline | 0.577 | 30.290 |
| Changed crop policy | 0.049 | 29.894 |
| Restored source | 0.030 | 30.053 |

The restored submission reused the baseline source package; it still started a
new job and loaded new actors/models. SDK timing combines source hashing,
packaging, upload and submission. These observations are not a controlled
comparison of cold and warm caches or of Ray versus native SkyPilot execution.

Session preparation took **37.795 seconds**, including **7.856 seconds** to
fetch and verify the model snapshot. This starts at the application's
preparation command and stops before its Ray CLI starts; it excludes image
pulling and SkyPilot bootstrap. Preparation happens once, outside every source
iteration.

For the changed-source job, cluster connection and initial actor readiness took
**5.902 seconds**; initial model loading was **2.437 seconds**. The
preprocessing/inference interval was **19.027 seconds**, including the controlled
actor replacement and its **2.358-second** model reload. Synchronized inference
totaled **8.368 actor-seconds** within that interval; aggregation took **0.870
seconds**. The application measured **27.775 seconds**, while the client's
**29.894-second** total also includes Jobs startup and polling. These intervals
overlap and should not be added as independent costs.

All 4,096 changed vectors differed by L2 distance above 0.01; the mean embedding
changed by **0.229948**. Restoration produced **zero maximum absolute error**
and identical persisted Parquet and vector-byte hashes. Worker SHA-256 prefixes
were `7c9b617426935a28` for baseline/restoration and `9c4629dd941110b4` for the
edit. Every driver/actor import matched the submitted four-file manifest; model,
UDF and runtime fingerprints stayed fixed. Full hashes remain in private evidence.

Each source job killed its exact first actor after a committed shard, loaded a
replacement, and replayed that shard with **zero additional inference calls**.
The replacement then handled 63 batches with its resident model. Source
redeployment reloaded weights. The separate cancellation job performed GPU work
and took **10.021 seconds** from submission to `STOPPED`; this is not the
latency of the stop API alone. Independent inspection found no live workload
actors or remaining GPU compute processes.

Artifact publication verified **416 files totaling 95,567,696 bytes**, plus the
manifest object. Independent downloads reopened Parquet and Lance data and
rechecked completeness, source provenance, full vectors and retrieval. The
finish client completed in **15.572 seconds**. Scoped Ray shutdown took **2.435
seconds** and ended with zero surviving children, signal errors or fallback
TERM/KILL signals. Both scheduler status and the durable stage receipt then
reported success.

Actual worker versions were Python **3.12.3**, Ray **2.46.0**, Torch
**2.12.1+cu130**, CUDA **13.0**, Transformers **5.10.2**, PyArrow **23.0.1**,
LanceDB **0.30.2** and NumPy **2.4.6**. The official PyTorch image digest stayed
fixed across all source iterations.

### Measured complex result

The final complex session used **two Kubernetes worker nodes with one RTX PRO 6000
Blackwell Server Edition GPU each**. Its three source jobs each processed
**16,384 records in 256 shards**, materialized 16,384 Lance rows and passed five
retrieval checks. Baseline, crop-policy change, restoration and the separate
cancellation job completed in **156.965 seconds** of client wall time, with
**zero image builds**.

| Revision | SDK package/upload/submit (s) | Total iteration (s) |
| --- | ---: | ---: |
| Baseline | 0.707 | 44.439 |
| Changed crop policy | 0.052 | 49.224 |
| Restored source | 0.038 | 48.955 |

CPU partitions fed two GPU actors, followed by aggregation, Parquet persistence,
LanceDB materialization and retrieval validation. Every CPU shard and GPU actor,
including the replacement actor, matched the submitted source manifest.
Independent state checks verified two Kubernetes worker nodes and all Jobs drivers on
the head, where the recipe keeps its durable-output staging and checkpoints.
The coordinator observed overlapping inference calls on both actors; its start
and post-CUDA-synchronization finish events include RPC edges. This establishes
concurrent application execution, not a precise cross-node CUDA-kernel timeline.

Application preparation took **35.873 seconds** on the head and **36.995 seconds**
on the worker, including model fetch/verification of **8.000** and **8.183
seconds**. These per-node intervals overlap and exclude image pull, SkyPilot
bootstrap and Ray startup. Both nodes reused their prepared environment across
source submissions.

For the changed-source job, initial actor readiness took **6.062 seconds**.
The two initial model loads were **2.502** and **2.472 seconds**; the controlled
replacement reloaded in **2.564 seconds**. Preprocessing/inference took **31.878
seconds** wall time. Its CPU tasks totaled **92.048 task-seconds** and GPU
inference **30.774 actor-seconds** across the workers; these are sums of parallel
work and are not additional wall-clock costs. Aggregation took **3.453 seconds**,
application execution **47.131 seconds**, and the client iteration **49.224
seconds**.

All 16,384 edited vectors changed by L2 distance above 0.01; the mean embedding
changed by **0.230283**. Restored vectors matched baseline exactly, with zero
maximum absolute error and identical output hashes. Each revision killed one
owned actor after a committed shard, replayed that shard with zero additional
inference calls, and completed all outputs. The final actors served **127 and
128 batches** with resident weights; actor replacement and each new source job
loaded weights again. The cancellation job performed GPU work and reached
`STOPPED` **9.936 seconds** after submission. Independent state and GPU queries
found no remaining workload actors or compute processes.

Publication verified **1,568 files totaling 375,856,595 bytes**, plus the manifest,
and independent downloads revalidated full vectors, every CPU shard's source,
Parquet, Lance rows and retrieval. The finish client took **33.347 seconds**.
Head/worker application shutdown took **2.419** and **0.303 seconds**, with zero
survivors, errors or fallback TERM/KILL signals. Live scheduler status and the
durable stage both reported **SUCCEEDED**, with the stage end timestamp present.

After both sessions, canonical cancellation and controller cleanup verified
remote absence. The owned API container, Kubernetes namespace, tunnels and
temporary local profiles were removed. Independent checks preserved the existing
cluster nodes and unrelated controllers. Workload outputs and private evidence
were retained outside the source checkout.

### Earlier attempts and repairs

The first medium application sequence passed in **102.822 seconds**, but its
workflow failed an immediate descendant-absence check: ten Ray children had
not yet exited after their CLI parent. The pod was subsequently removed and no
GPU orphan remained. The shutdown repair waits for the recorded PID/creation-time
identities, then signals only surviving owned processes and verifies absence.
[Ray 2.46 process cleanup](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/_private/node.py)

A second medium sequence passed in **101.407 seconds**, including artifacts and
process cleanup, but exposed a separate durable-state defect. Image mode's
`exec` replaced the shell owning NPA's EXIT trap, leaving its stage receipt at
RUNNING while live SkyPilot status was SUCCEEDED. Keeping the shell preserves
that finalizer; real Bash regression tests verify success and failure receipts
and exit codes. The final complete medium run above validates both repairs.

Early operator client setup also omitted NumPy for comparison and, separately,
boto3 for the finish-marker write. The missing dependencies were installed and
are now explicit in the recipe; completed GPU work and failed-client evidence
were retained. The isolated local SkyPilot API container later exited with code
137 while Docker reported `OOMKilled=false`. Its exact existing identity was
restarted before status verification. The cause remains unknown; cloud GPU work
was unaffected. A private wrapper also initially omitted the documented caller
identity during status lookup; restoring that environment fixed the lookup.
These operator-side incidents are distinct from application inference failures.

The initial complex sequence already processed **three sets of 16,384 records
across two Kubernetes GPU worker nodes** in **160.523 seconds**, with concurrent actors,
source restoration, committed-shard recovery and cancellation checks passing.
Its artifact checks also passed. That session used the same faulty outer
finalizer. The final complex run above repeated the full workload and verified
the corrected durable completion receipt.

### Source and recovery contract

The client sends source-only temporary directories through upstream
`runtime_env.working_dir`; model weights, inputs, credentials and output files
are outside that payload. Both CPU preprocessing and GPU actors verify the
worker source. A crop-policy edit must change at least 99% of per-record vectors
by an L2 distance above 0.01; restoration must match the baseline within absolute
tolerance 0.00001. Raw model/output hashes and runtime versions belong in private
evidence, alongside sanitized aggregate results suitable for publication.

These results qualify this canonical Workbench CLIP UDF and LanceDB library
recipe on the measured single-node and two-node stack. It does not qualify the published LanceDB service image,
its HTTP backfill API, or a general Ray backend for every Workbench tool.

The recovery check kills one owned actor after a shard is committed, replaces
it, and replays the shard through a verified checkpoint. Its identity includes
source, input, model and runtime hashes. This tests recovery after a durable local
commit, not recovery of an in-flight CUDA kernel or loss of the driver node.
The replacement initializes its model again. A separate job executes real GPU
work before an exact-ID stop; Ray Jobs stop is asynchronous and must reach a
terminal state before session shutdown.
[Ray 2.46 actor fault tolerance](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-core/fault_tolerance/actors.rst),
[Ray 2.46 Jobs SDK](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/dashboard/modules/job/sdk.py)

## Compatibility, security and ownership

The selected application Ray version is **2.46.0**; NPA's SkyPilot version is
**0.12.2**. Keep the application interpreter and package environment separate
from management Ray. Use explicit application GCS and Jobs addresses, separate
component/worker ports and scratch directories, and a private controlled network.
The recipe advertises `config.gpus_per_node` application GPUs per node. Match
that count to the requested accelerator allocation; record actual node/GPU
counts and do not call a one-node run multi-node validation.
[SkyPilot application cluster helper](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky_templates/ray/start_cluster)

The management head's worker range is **11002–65535**: Ray 2.9.3 uses 65535
when a minimum is set without a maximum. The session places every application
service and worker port below 11002, avoiding SkyPilot's fixed ports; its
application workers use 10010–10999. Non-head management workers can request
OS-assigned ports, so startup also verifies the Linux ephemeral range begins
above 10999. Do not rely on a presumed management maximum of 19999.
[Ray 2.9.3 worker-port allocation](https://github.com/ray-project/ray/blob/ray-2.9.3/src/ray/raylet/worker_pool.cc#L122-L138)

Local SkyPilot state isolation does not isolate its host-wide API endpoint.
SkyPilot's API server fixes its own identity for its lifetime and uses it for
the jobs-controller name; individual request user IDs do not replace that
identity. A private development session therefore needs a separately owned API
endpoint as well as isolated state and exact Kubernetes targeting. Do not stop a
shared API server to obtain this isolation.
[SkyPilot 0.12.2 server/controller identity](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/utils/common.py)

Changing only that server's HTTP port is insufficient: the pinned request queue
also uses a fixed port. The upstream Docker deployment supplies a separate
network namespace without modifying SkyPilot internals. Publish its HTTP port
only on host loopback and mount only run-owned state plus the exact required
credential/configuration paths. The container and its state remain session-owned
resources to clean up after managed jobs are terminal.
[SkyPilot 0.12.2 API container deployment](https://github.com/skypilot-org/skypilot/blob/v0.12.2/docs/source/reference/api-server/examples/api-server-in-docker.rst),
[Pinned request queue](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/server/requests/queues/mp_queue.py)

The qualified API setup uses a non-root UID and drops capabilities. The pinned
Kubernetes runner calls `chmod` on its installed rsync helper, so that one file
must belong to the API UID. The reproduction copies the exact helper out and
back with archive ownership, verifies identical bytes, and tests non-root
`chmod`; it changes neither package code nor the image.
[SkyPilot 0.12.2 Kubernetes command runner](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/utils/command_runner.py)

The API container needs the selected Nebius storage profile in its own private
`.aws/credentials` and `.aws/config`, even when AWS credential environment
variables are present. Require `sky check nebius` to enable both compute and
storage before submission. NPA's general S3 health row checks bucket visibility;
the workflow's write preflight remains a separate requirement.
[SkyPilot 0.12.2 Nebius credentials](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/clouds/nebius.py)

The pinned Jobs SDK lets ambient `RAY_ADDRESS` override an explicit client URL.
The client deliberately selects the private application endpoint. Never use
`address="auto"`, `ray stop`, an ambiguous management address, or a second cloud
autoscaler. The recipe stops only its own application Ray CLI after jobs are
terminal and artifacts have been verified in S3. NPA and SkyPilot retain control
of cancellation and infrastructure teardown.
[Ray 2.46 Jobs address handling](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/dashboard/modules/job/sdk.py),
[Ray 2.46 owned-node shutdown](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/_private/node.py#L451)

Ray Jobs permits arbitrary code execution. Bind Dashboard/Jobs to loopback and
reach it through authenticated SSH or Kubernetes forwarding; do not publish
Jobs, Dashboard, Client or other Ray component ports. Namespaces group resources
but do not isolate mutually untrusted customers. `runtime_env` is dependency
configuration, not a sandbox; use separate clusters and external network/access
controls where trust boundaries require them. The existing Cosmos HTTP bearer
token does not secure a general Ray endpoint.
[Ray 2.46 security guidance](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-security/index.md)

The live runs used a dedicated Kubernetes namespace with an ingress
NetworkPolicy selecting all its pods and allowing only same-namespace pod peers.
The operator must provide an enforcing CNI and retain that policy; the session
YAML does not install it. Dashboard loopback alone does not protect Ray's other
pod-interface ports. This configuration permits egress and trusts the underlying
nodes. Policy configuration was verified, without a penetration-test claim.
The [reproduction](../testing/fast-source-iteration.md#prepare-the-session-and-client)
includes the exact policy shape and private receipt commands.
[Kubernetes NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

A Jobs-level runtime environment covers the driver and descendants. Merely
setting `ray.init(runtime_env=...)` does not replace an already-running driver's
environment. Local `working_dir` uploads have a 500 MiB limit; symlinks may pull
in unintended content. Select a reviewed source-only directory. Record your own
SHA-256 manifest and every worker's actual imports: a Ray package URI is useful
cache evidence, not a cryptographic trust attestation.
[Ray 2.46 dependency handling](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-core/handling-dependencies.rst),
[Ray 2.46 packaging implementation](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/_private/runtime_env/packaging.py)

The recipe reuses a model within each actor across batches. A new source job
creates new actors and reloads weights. Retrieving a detached named actor would
return its existing implementation, not hot-reload its class. Persistent Serve
has its own deployment/update contract and should be evaluated separately when
cross-job model residency is the actual requirement.
[Ray 2.46 actor lifetimes](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-core/actors/named-actors.rst)

Compatible Python dependency, packaging-metadata and console-entrypoint changes
require pinned reinstallation or environment re-preparation, followed by workload
validation and refreshed dependency-freeze and hash evidence. They do not
inherently require an image build. Copying source alone does not update installed
metadata or regenerate console-script wrappers.
[PyPA entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)

Rebuild or choose another compatible base image when the change requires different
image-owned Python, Torch/CUDA, native libraries, OS packages or baked vendor
components. Source delivery does not repair native ABI mismatches or SkyPilot
bootstrap incompatibility. Retain normal immutable-image release validation.

This recipe prepares pinned Ray, LanceDB, Transformers and the supporting
application stack in a separate environment once per session, records the
resulting dependency freeze, and leaves model downloads outside the image.
The image digest identifies the base environment; it does not attest the
subsequently installed Python environment. Record both, including actual
installed versions and the submitted UDF hash. Preparation is not an iteration
measurement. Named package pins plus a recorded freeze are not a hash-locked
transitive installation; dependency resolution remains part of its reproducibility
boundary.
The selected image's verified interpreter is `/usr/bin/python3.12`. It provides
pip but lacks `ensurepip`; the workflow creates a system-site-packages venv with
`--without-pip` and uses the base interpreter's `pip --python` option to install
into it. This is standard pip behavior and requires no image build or OS package
change. Startup repairs and cancelled preparation attempts are excluded from
source-iteration measurements.
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/)

## Comparison with native SkyPilot source syncing

| Path | What runs again after an application edit | What remains fixed | Evidence here |
| --- | --- | --- | --- |
| Ray Jobs `working_dir` | Source packaging/upload, job runtime setup, driver and new actors/model loading | Application cluster, image and prepared model files | Complete medium and two-node complex iterations and lifecycles measured above |
| SkyPilot `workdir` + `exec` | Source sync and command execution, including any model load in that command | Allocated cluster and setup environment | Architectural analysis only |
| Local Docker source mount | New container/process and workload | Cached image and fixed dependency directory | Historical CPU measurement below |
| Existing NPA staging/overlay | NPA source transfer/setup, scheduled stage and model initialization | Compatible tool image | Existing implementation inspected; remote performance not measured |

Pinned SkyPilot `exec` synchronizes workdir and executes the command while
skipping provisioning, setup and file-mount synchronization. A changed setup or
file mount needs the corresponding launch/sync path. `.skyignore` replaces the
default `.gitignore`/`.git/info/exclude` filtering, so inspect the actual source
payload. Use NPA's bootstrapped SkyPilot executable for the selected release.
[SkyPilot v0.12.2 execution](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/execution.py#L772),
[SkyPilot source syncing](https://docs.skypilot.ai/en/latest/examples/syncing-code-artifacts.html)

Do not infer a speedup without running the same workload through both paths.
The Jobs SDK submission interval includes package hashing/upload and its submit
request. Keep it separate from environment preparation, model initialization,
synchronized GPU work, aggregation, artifact persistence and total wall time.
Compare durations within each process; do not subtract unsynchronized timestamps
across worker nodes.

## Historical CPU measurement: 2026-09-04

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

## Historical audit-only repository validation

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
suite passed. That audit-only revision changed no product code or tests. Detailed reproduction
and file-identity evidence is retained outside Git.
