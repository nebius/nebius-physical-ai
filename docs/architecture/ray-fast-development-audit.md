# Ray + SkyPilot for source development

Updated 2026-09-05. The customer path is the [Ray Jobs guide](../testing/fast-source-iteration.md):
one SkyPilot development cluster, a private Jobs connection, ordinary Python,
an explicit source edit, inspectable image/vector outputs and exact cleanup.
The [guide comparison and API assessment](ray-development-guide-design.md)
explains the responsibility changes and cites the upstream contracts.

## Recommendation and scope

Use SkyPilot to own Nebius infrastructure and application Ray for tasks, GPU
actors, Jobs submission and `runtime_env.working_dir` source delivery. Keep
`npa.workflow` for durable production composition. Development does not need a
long-lived workflow/controller, a custom submitter or a finish-marker protocol.
The rejected prototype's mandatory versions of those mechanisms are removed.

The scoped example uses the real Workbench CLIP UDF and LanceDB library. It
does not qualify the deployed LanceDB HTTP service or its published service
image. Source redeployment creates new actors and reloads models. Within an
actor, multiple batches reuse resident weights. No hot-reload claim follows.

[Ray Jobs, 2.46.0](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/cluster/running-applications/job-submission/quickstart.rst)
provides the familiar submission/source interface. [SkyPilot's Ray example,
0.12.2](https://github.com/skypilot-org/skypilot/blob/v0.12.2/examples/distributed_ray_train/ray_train.yaml)
provides the separate infrastructure/application-runtime pattern. A finite
`sky exec` path can be simpler for users who want SkyPilot's source syncing;
that path has not been performance-measured here and is not Ray Jobs delivery.

## Existing repository mechanisms

| Mechanism | Source | Boundary |
| --- | --- | --- |
| Editable NPA installation | [Dependencies](../../npa/pyproject.toml) | Local Python development; GPU changes still need compatible real GPU execution. |
| NPA source staging/overlay | [Staging](../../npa/src/npa/orchestration/npa_workflow/src_staging.py), [renderer](../../npa/src/npa/orchestration/npa_workflow/skypilot_render.py) | NPA-owned code distribution for NPA development. It is not used to transfer this Ray application. |
| Existing Cosmos3-Nano Ray Serve | [Service](../../npa/src/npa/workbench/cosmos/ray_server.py), [guide](../workbench/cosmos3-ray-serve.md) | Persistent NVIDIA framework inference; not a general Jobs endpoint or Cosmos3-Super runtime. |
| Ray CLIP development | [Application](../../npa/workflows/workbench/ray-clip-development/embed.py), [cluster task](../../npa/workflows/workbench/ray-clip-development/cluster.yaml) | Native Ray Core/Jobs and source packaging, separate SkyPilot infrastructure, direct Parquet/Lance outputs. |

The redesign retains the isolated SkyPilot launcher relocation fix required by
NPA's client setup. The preceding prototype's image-runtime extension,
managed-controller cleanup changes and associated S3 workflow fixtures are
removed from this branch's final change: the native development path uses none
of them. Their earlier commits and workload evidence remain historical; those
extensions are not advertised as supported features of this final example.

The [May workflow recommendation](workflow-engine-recommendation-20260514T233740Z.md)
is historical. Vendored KubeRay definitions are not an enabled NPA service;
the [Kubernetes renderer](../../npa/src/npa/cluster_backends/mk8s_render.py)
disables those flags. No broad Ray infrastructure migration is needed.

## Redesigned guide: current GPU evidence

After the application readability refactor, an independent reader executed the
basic guide using only its declared platform prerequisites. The first run,
one-line crop edit, restoration, native status/logs, RGB/vector/retrieval
inspection, download/hash verification and exact development-cluster cleanup
passed. The medium loop used one physical Kubernetes worker and one RTX PRO
6000 GPU, with 4,096 images per revision. These are fresh executions of the final
readable application, not the preceding prototype's receipts.

| Medium phase, seconds | Baseline | Changed | Restored |
| --- | ---: | ---: | ---: |
| Ray CLI invocation through exit | 27.674 | 25.560 | 28.637 |
| CLI start through first observed Job | 4.488 | 4.415 | 5.408 |
| Package-creation log through first observed Job | 1.161 | 0.166 | Cached; no creation log |
| Ray server job interval | 17.447 | 16.960 | 17.604 |
| Application, including actor readiness and artifacts | 15.821 | 15.517 | 16.094 |
| Connection and model-actor readiness | 4.901 | 4.506 | 4.796 |
| Model load inside the CUDA actor | 2.406 | 2.241 | 2.344 |
| Preprocessing and inference wall time | 9.669 | 9.831 | 10.050 |
| Aggregation and artifacts | 1.251 | 1.181 | 1.248 |

These are nested timing boundaries, not additive stages. Source delivery is
measured from the client's package-creation log to the first Job visible through
the upstream SDK. It includes packaging, upload, submission and polling/network
observation delay; it is not isolated wire-transfer time. Restoration reused
Ray's cached source package and emitted no creation log, so no separate source
interval is claimed for that revision. CLI startup, connection, status polling
and log drainage are included outside the application; their individual
contributions were not profiled. No image-build-duration comparison was measured.

All 4,096 changed vectors exceeded L2 distance 0.01 (minimum 0.237, mean 0.361).
Restored vectors matched exactly, tighter than the required `1e-5` absolute
tolerance. Imported application/UDF hashes and CPU-worker source hashes matched
each submission. Each Job used seven preprocessing processes and one CUDA actor
for 64 batches. Ray's source-package identity changed and restored; the pod and
immutable image identity remained fixed, with zero image-build or dependency
installation commands in the measured loop. Each submission created a new
model actor and reloaded weights; batches within that actor reused the model.

The reader downloaded 75 files (56,798,868 bytes) across the three revisions and
verified their manifests. Independent artifact review rehashed 72 payload and
JSON-manifest files; the remaining three files are the `SHA256SUMS` manifests.
Decoded RGB inputs/crops, all 4,096 finite normalized 512-dimensional Parquet
vectors per revision, reopened Lance tables and retrieval checks passed. A
second reviewer independently reopened all three Lance tables, verified their
vectors against Parquet, and repeated first/middle/last-row retrieval. The
published preview images match these freshly verified outputs. Procedural
images establish execution/retrieval consistency, not semantic model accuracy.

The medium Jobs all reached `SUCCEEDED`; their status and logs were inspected
through upstream Ray commands. The final advanced sequence also passed on
**two distinct physical GPU workers**, with one RTX PRO 6000 per SkyPilot pod.
Each revision processed 16,384 images in 256 shards with two concurrent CUDA
actors, producing 49,152 qualified embeddings across baseline/change/restore.

| Complex phase, seconds | Baseline | Changed | Restored |
| --- | ---: | ---: | ---: |
| Ray CLI invocation through exit | 53.917 | 55.333 | 64.093 |
| CLI start through submission banner | 6.224 | 3.407 | 10.058 |
| Package-creation log through submission banner | 0.713 | 0.148 | Cached; no creation log |
| Ray server job interval | 42.397 | 48.306 | 49.910 |
| Application, including recovery and output checks | 40.241 | 45.977 | 47.560 |
| Connection and initial actors ready | 5.947 | 6.269 | 6.769 |
| Preprocessing, inference and actor recovery wall time | 30.027 | 30.396 | 31.668 |
| Aggregation into Parquet/Lance and retrieval | 4.249 | 4.410 | 4.280 |

The complex source interval ends at the observed native CLI submission banner;
it includes packaging/upload and submission, not isolated network transfer.
Its endpoint differs from the medium SDK-observation boundary, so the source
rows are not direct performance comparisons. Individual model loads, including
replacement actors, took 3.373–4.120 s; separate source/model fingerprint
verification took 0.434–0.517 s. Changed/restored application totals also include
full-vector comparison. All 16,384 edited vectors changed above L2 0.01;
restoration's maximum error was zero. Imported application, UDF and worker
hashes matched every relevant actor and preprocessor. Runtime Ray-node/pod/
Kubernetes-node mapping verified both physical workers. Coordinator-clock
measurements around CUDA-synchronized calls observed concurrent inference;
these include RPC edges and are not a kernel-profiler trace.

Each revision killed one exact owned actor after committing a real shard,
loaded a replacement and replayed the checkpoint with zero further inference
or writes. Full output IDs were complete and unique. This proves the
application's checkpoint idempotency for actor failure; it does not qualify
driver/node-loss recovery or automatic Ray exactly-once external writes.

A separate active job committed 64 real embeddings, continued CUDA inference,
and reached `STOPPED` through `ray job stop`; the native status command also
confirmed it. The first terminal message arrived 6.239 s after stop-command
start, including CLI startup/polling. This is an observed upper bound, not an
isolated scheduler cancellation latency. All 18 observed application actors
were dead before infrastructure teardown. Application Ray and SkyPilot's
management Ray remained separate, and their core processes survived these
controlled application failures.

Artifact transfer and SHA verification took 74.282 s. The three result trees
contained 1,617 files (376,727,478 bytes), including 1,614 manifest-listed payload
files plus three checksum manifests. All checkpoints, final Parquet vectors,
Lance tables, decoded RGB previews and retrieval checks passed 917 independent
qualification assertions. The initial verifier compared an OCI configuration
digest with the image manifest digest; its failed receipt is preserved. The
corrected check verifies both the pod's requested image and resolved image ID
against the pinned manifest. No application or image was changed for that
observer correction. The first Jobs readiness probe preceded service startup;
the documented Sky log-follow and readiness retry passed without restarting
the service.

Exact service cancellation and named SkyPilot teardown passed for both
customer paths. Complex cleanup took 48.966 s. Their workload pods, services,
PVCs, SkyPilot development clusters and tunnels were absent afterward. After
the GPU loops, recreating the owned API with the final readable Compose
healthcheck took 28.935 s and passed. Final API-container/network/state-volume
removal took 8.519 s; deletion of the owned namespace and NetworkPolicy took
6.017 s. Read-only verification confirmed all owned platform resources absent,
all four pre-existing Kubernetes nodes and seven unrelated running containers
retained, and the original host API healthy before and after cleanup. An
absence-observer message-casing correction required only read-only verification;
no destructive command or GPU workload was repeated for it.

Earlier redesign investigation found two API-readiness failures before GPU
allocation: request-local Kubernetes configuration was lost in a validation
thread, and a shared API lacked a working long-request executor. The exact
pending request was cancelled through the upstream API. The final platform
contract uses an explicitly owned upstream API with a fixed private backend;
no shared API is patched or restarted. Fresh platform qualification verified
that API, Kubernetes permissions, explicit compute enablement and a native
SkyPilot dry run. The separate pre-existing host API was healthy before and
after that setup and was not targeted. The [decision record](ray-development-guide-design.md)
explains why credential checks alone do not prove launch readiness.

## Runtime and reproducibility boundary

The selected existing public image is:

```text
docker.io/pytorch/pytorch@sha256:72f863fa1fe13d5d87a72d00db2c85fb2d43409ee08dd26bc469de4c8a28b427
```

The current GPU runs report Python 3.12.3, Torch 2.12.1+cu130 and CUDA 13.0.
Application preparation pins Ray 2.46.0, LanceDB 0.30.2, PyArrow 23.0.1,
NumPy 2.4.6, Pillow 12.2.0 and Transformers 5.10.2. The CLIP revision is
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; its 605,247,071-byte weights hash is
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
The API/client report SkyPilot 0.12.2. Actual management processes use Ray 2.9.3
on Python 3.10.21, separate from application Ray 2.46.0 and its Jobs endpoint.

The generic credential preflight also reported `NoSuchBucket` for an unused
default S3 destination. That failure is preserved; no bucket or routing was
changed to hide it. This example uses downloaded local artifacts and does not
access S3. Online Nebius authentication, public-model access and the real CUDA
workload passed. This qualifies the selected path, not every configured service.

The image digest does not attest packages installed afterward. Preparation
records a resolved dependency freeze; this is not a fully hash-locked transitive
supply chain. Preparation exposed an inherited, unused `spin 0.18` requirement
for `click<8.4` while the resolved environment contains Click 8.5.0; the selected
Ray/CLIP/Lance paths passed, but the image's unrelated developer tooling is not
qualified as a consistent environment. Pure compatible source changes need no image build. Package
changes require a new preparation/environment record; Python, CUDA, native ABI
or image-owned system changes require a suitable image and possibly a rebuild.

The final medium Sky launch took 95.703 s. Within node preparation, dependency
installation took 23.564 s, model fetch plus weights-hash verification 8.718 s,
and CUDA/environment inspection (including `pip freeze`) 2.171 s. The receipt
records distinct start/end timestamps for each boundary. Sky launch also
includes scheduling, image pull, SSH setup and runtime startup; those components
were not separately profiled. The observed image-pull event was 349 ms on the
existing physical worker. Its image layers were not purged, so this is a fresh
pod/application environment rather than a first-ever image-download benchmark.
The two-worker Sky launch command took 112.263 s. Per-worker dependencies took
23.028–24.699 s, model fetch/hash verification 8.658–8.857 s, and separate
CUDA/environment inspection 2.268–2.406 s. Kubernetes reported image pulls of
342/328 ms; downloaded layer bytes and cache participation were not measured.
Sky launch returned before the application service was ready, so the guide's
readiness check remains necessary.
The original owned-API startup-and-health command took 27.538 s using a cached
image; that platform was reused across the measured development clusters.

Setup does not supply a node rank, so preparation records no inferred rank.
Actual Ray-node, pod and Kubernetes-node evidence establishes placement during
the run. Earlier preparation receipts that included CUDA inspection in the
model-fetch duration remain historical; the final refactored bootstrap measures
those phases separately. These measurements do not bound a slower network.

Model download and dependencies are node-local preparation, separate from
image pull/SkyPilot startup and from the source-iteration loop. Replacement SkyPilot pods
repeat preparation unless a separately configured durable cache supplies weights;
replacement Ray actors reload from the existing pod's model files. Artifacts
are driver files until downloaded and hash-verified on durable local storage;
Ray's object store is not an artifact archive. Production cross-tool workflows
continue to use their declared S3 outputs.

## Historical medium and complex GPU evidence

These results belong to the **superseded session workflow**, completed before
the guide redesign. Its immutable private evidence remains preserved. They must
not be presented as current guide execution evidence.

| Check | Actual scope | Records per revision | Baseline / changed / restored wall seconds |
| --- | --- | ---: | --- |
| Medium | One Kubernetes worker, one RTX PRO 6000 GPU actor | 4,096 | 30.290 / 29.894 / 30.053 |
| Complex | Two distinct Kubernetes workers, one RTX PRO 6000 GPU each | 16,384 | 44.439 / 49.224 / 48.955 |

The previous full sequences took 104.720 s and 156.965 s respectively, including
separate exact-job cancellation checks. All changed vectors exceeded L2 0.01;
restoration's maximum absolute error was zero. Real decoded RGB inputs,
Parquet/Lance row completeness and retrieval passed. Imported source/UDF hashes
were checked on relevant workers. Concurrent actors, controlled replacement,
verified committed-shard reuse without duplicate output, Jobs `STOPPED`, artifact
Put/Get/hash verification and owned cleanup were recorded. Medium retained
416 files (95,567,696 bytes); complex retained 1,568 (375,856,595 bytes).
Both sessions reached live and durable `SUCCEEDED` before their owned resources
were cleaned up. These procedural images validate execution/retrieval consistency,
not semantic CLIP accuracy.

Those runs found real cold-start boundaries: the published non-root LanceDB
image lacked `sudo` required by SkyPilot SSH bootstrap; the selected PyTorch
interpreter has pip but no ensurepip; application and management worker-port
ranges must be disjoint. The old recipe also needed API/controller and S3 finish
repairs. The redesigned path removes the latter protocol rather than asking
developers to reproduce it. Earlier diagnostic evidence remains private.

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

The historical command transcript, source-change proof and artifact hashes remain
in the preserved audit evidence. The current guide describes the redesigned GPU
Ray Jobs path, not this earlier CPU experiment.

## Historical audit-only repository validation

After reconciliation with that audit's then-current main, repository Ruff, 13,479-test
collection, 2,564 combined guardrail/source-staging/SkyPilot-renderer/dataset
tests, CLI documentation drift, and the reproduction's shell syntax/link checks
passed. That audit documentation embedded the executed CPU workload and
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

The same Cosmos failure appeared in that historical base's
[Python 3.12 CI job](https://github.com/nebius/nebius-physical-ai/actions/runs/33911450997/job/101148584343).
Main subsequently fixed it: the isolated test passed after reconciliation with
main revision `0143535e40746bdfe06b0f8cb87fb5e3217bcadc`, without a local test or
environment change. The former failure is not a current validation exemption.

Those historical base/environment failures do not mean the earlier complete
suite passed. That audit-only revision changed no product code or tests.
Detailed reproduction and file-identity evidence remains outside Git; final
redesign repository validation is reported separately after its gates complete.
