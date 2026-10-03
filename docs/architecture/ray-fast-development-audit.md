# Ray + SkyPilot for source development

Updated 2026-09-05. Start with the [Ray Jobs guide](../testing/fast-source-iteration.md):
create a GPU development cluster, embed visible images, change one Python line,
submit again, inspect the vectors and download them before cleanup. The
[comparison and API assessment](ray-development-guide-design.md) explains why
this is simpler than the earlier recipe and identifies the code behind each API.

## Recommendation

Use **Ray Core and Ray Jobs for the application**, with **SkyPilot owning the
infrastructure**. The tested example calls Workbench's actual CLIP UDF inside
CUDA actors and persists Parquet vectors, a Lance table, retrieval results and
RGB previews. Ray packages the application through `--working-dir`; ordinary
`ray job submit/status/logs/stop` controls it. No NPA submitter, finish marker or
application upload Job is required. Keep `npa.workflow` for durable production
composition rather than introducing a workflow/controller into every edit loop.

This follows the recognizable code → submit → inspect journey in the
[Ray Jobs 2.58 quickstart](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/cluster/running-applications/job-submission/quickstart.rst).
The [Ray Data GPU guide](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/data/batch_inference.rst)
also makes model initialization and batch inference visible; Core actors let
this example reuse the existing Workbench UDF without adding another data API.
[SkyPilot's 0.12.2 Ray example](https://github.com/skypilot-org/skypilot/blob/v0.12.2/examples/distributed_ray_train/ray_train.yaml)
provides the infrastructure/application split. These versioned sources were
accessed on 2026-09-05.

## Current medium result: Ray 2.58

An independent reader executed the displayed first run, crop edit and restoration
using the guide's declared platform prerequisites. Each Job embedded **4,096
images on one RTX PRO 6000 GPU on one physical Kubernetes worker**, and all
three reached `SUCCEEDED`, with no undocumented reader rescue steps. Upstream
status/log commands and the displayed
Parquet comparison were executed. A second reviewer independently reopened the
downloaded artifacts rather than trusting the application's success message.

| Medium timing, seconds | Baseline | Changed | Restored |
| --- | ---: | ---: | ---: |
| Ray CLI invocation through exit | 26.859 | 29.111 | 27.963 |
| CLI start through submission banner | 3.385 | 4.196 | 3.238 |
| Package creation through submission banner | 0.685 | 0.115 | Cached; no creation log |
| Ray server Job interval | 18.990 | 19.437 | 19.256 |
| Application, including readiness and output | 16.767 | 17.431 | 17.162 |
| Connection and model-actor readiness | 5.471 | 5.559 | 5.492 |
| Model load inside the CUDA actor | 2.578 | 2.641 | 2.603 |
| Preprocessing and inference wall time | 10.025 | 10.636 | 10.308 |
| Aggregation and artifacts | 1.271 | 1.235 | 1.362 |

These intervals overlap; they are **not additive stages**. The source interval
uses embedded timestamps from the same client's log and includes packaging,
upload and API submission. It does not isolate network transfer. Restoration
reused the original Ray package. CLI overhead includes startup, connection,
status polling and log drainage; the host was shared with other validation work,
so these are observed iteration times, not a controlled throughput benchmark.

All 4,096 edited vectors changed by more than L2 distance 0.01 (minimum 0.237,
mean 0.361). Restored vectors matched exactly, within the guide's `1e-5`
absolute tolerance. Imported application/UDF and preprocessing-worker hashes
matched the delivered source; source package identity changed and restored.
The same pod and immutable image served the loop, with **zero image builds or
dependency installations during source iteration**. Each new Job loaded a new
actor and weights; batches within an actor reused its model.

The three downloaded trees contain 75 files (56,798,871 bytes), including three
checksum manifests. All 12,288 finite, normalized, 512-dimensional Parquet
vectors were checked.
Independent review reopened the three Lance tables, compared every vector with
Parquet, repeated first/middle/last-row retrieval and decoded the actual RGB
inputs, crops and previews. The guide's published images match those outputs.
The current medium qualification passed 525 assertions. Procedural images prove
execution and retrieval consistency; **they do not measure semantic model
accuracy**.

The exact medium tunnel/service/cluster cleanup completed in 76.327 s. Thirteen
read-only ownership/absence checks passed: its workload and tunnel were gone,
while the platform and four pre-existing Kubernetes nodes remained for the
complex check. Two private diagnostic observers needed correction; their failed
receipts remain preserved. No guide command, GPU inference or successful
cleanup was repeated for those observer corrections.

## Current distributed result: Ray 2.58

The optional advanced recipe passed on **two distinct physical Kubernetes GPU
workers**, one RTX PRO 6000 per worker. Each of three native Ray Jobs processed
16,384 images in 256 shards with two concurrent GPU actors. All three reached
`SUCCEEDED`, producing 49,152 independently qualified embeddings. Ray-node,
pod and Kubernetes-node receipts prove placement; requesting two actors or
using `SPREAD` alone would not prove two physical workers.

| Complex timing, seconds | Baseline | Changed | Restored |
| --- | ---: | ---: | ---: |
| Ray CLI invocation through exit | 52.515 | 55.584 | 54.472 |
| CLI start through observed submission banner | 2.755 | 1.809 | 2.038 |
| Package creation log through observed banner | 0.620 | 0.078 | Cached; no creation log |
| Ray server Job interval | 45.790 | 49.664 | 47.279 |
| Application, including recovery and output | 43.103 | 47.050 | 45.340 |
| Connection and initial actor readiness | 6.760 | 6.234 | 5.952 |
| Preprocessing, inference and actor recovery | 31.906 | 31.447 | 30.884 |
| Aggregation into Parquet/Lance and retrieval | 4.409 | 4.516 | 3.924 |

These timings are also nested. The complex source interval ends at the observed
unbuffered CLI banner; the medium table uses its embedded logger timestamp.
Both include packaging/upload/submission rather than isolated network transfer.
Individual model loads, including replacement actors, took 3.201–4.152 s;
separate model/source fingerprint verification took 0.414–0.501 s. Changed and
restored application totals also include full-vector comparison. Coordinator
observations span CUDA-synchronized calls and include RPC edges; they show
concurrent inference, not a kernel-profiler trace.

Each revision killed one exact owned actor after committing a real shard,
loaded a replacement and replayed that checkpoint with zero extra inference
or duplicate output. Ray owns actor termination and scheduling. The application
owns checkpoint identity, integrity checks and deduplication. This qualifies
its controlled actor-failure replay, not driver/node-loss recovery or automatic
Ray exactly-once external writes. All 16,384 edited vectors changed above L2
0.01; restoration's maximum absolute error was zero. Imported application, UDF
and worker hashes matched every relevant actor and preprocessor. Source packages
changed and restored while image identity and core runtime processes stayed
fixed; the measured source loop performed zero image builds.

A separate active Job committed 64 real embeddings, continued CUDA inference,
and reached exactly `STOPPED` through native `ray job stop`; native status and
an independent SDK read confirmed it. The first terminal message arrived
2.719 s after stop-command start, including client startup and polling; this is
an observed upper bound, not isolated scheduler latency. All observed owned GPU
actor instances were accounted for and dead before infrastructure teardown.

Artifact transfer and SHA verification took 34.603 s. The three result trees
contain 1,617 files (376,727,467 bytes), including 1,614 manifest-listed payloads
and three checksum manifests. The qualification passed 919 checks covering
source, placement, concurrency, recovery, cancellation and artifact integrity.
A second reviewer independently compared all 49,152 Lance/Parquet vectors,
repeated 15 retrieval queries and decoded 51 RGB inputs/crops/previews. Exact
Sky service cancellation and named cluster teardown took 50.457 s; 13 cleanup
checks found no remaining workload resources or tunnel and retained the four
pre-existing nodes and healthy platform.

## Preparation and compatibility

The unchanged public image is:

```text
docker.io/pytorch/pytorch@sha256:72f863fa1fe13d5d87a72d00db2c85fb2d43409ee08dd26bc469de4c8a28b427
```

The medium worker reported Python 3.12.3, Torch 2.12.1+cu130 and CUDA 13.0.
Application packages were Ray 2.58.0, LanceDB 0.30.2, PyArrow 23.0.1, NumPy
2.4.6, Pillow 12.2.0 and Transformers 5.10.2. SkyPilot 0.12.2 used its separate
management Ray 2.9.3. Process/interpreter and listener receipts verify that the
application joined its own Ray runtime, not SkyPilot's management cluster.

The model is `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; downloaded weight SHA-256 is
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
Preparation verifies those bytes and records the resolved dependency environment.

The fresh medium Sky launch took 118.391 s. Its preparation receipt separately
measured 25.371 s for dependency preparation, 9.099 s for model download/hash
verification, and 2.463 s for CUDA/environment inspection including `pip freeze`.
These are contained within launch time. The two-worker launch took 77.749 s,
with per-worker dependencies 24.583–27.200 s, model fetch/hash 8.479–8.546 s and
CUDA/environment inspection 2.131–2.427 s. Sky launch returns before Jobs service
readiness; the guide separately checks that endpoint. Starting the reusable
platform API took 19.541 s separately. Node/image/download caches were not reset,
and image layer transfer/cache participation was not measured. These launches are not full cold-image-pull
benchmarks, a controlled scaling comparison or an image-build speed comparison.

The image pins the Python/Torch/CUDA base; later package installation is not a
fully hash-locked transitive dependency supply chain. Recorded resolved versions
make the tested environment inspectable but do not guarantee identical future
resolution. Pure Python edits reuse the prepared worker. Package changes need
re-preparation; incompatible Python, CUDA/native ABI or operating-system changes
require a compatible image. New Jobs reload models; this is source redeployment,
not hot reload of resident actors.

## Ownership, access and persistence

The [one-time platform contract](../../npa/workflows/workbench/ray-clip-development/platform/README.md)
actually provisions and verifies one owned upstream SkyPilot API and namespace.
That service is retained across development clusters; it has not disappeared
behind a helper. The developer receives two standard backend settings and
chooses a cluster name and durable result directory. SkyPilot owns the cluster
and its service task; Ray owns application Job IDs and worker scheduling;
OpenSSH owns one tunnel. The operator owns API/configuration cleanup after all
of its development clusters are gone.

The API's backend is fixed and its credential mounts are read-only. The Jobs HTTP
server binds to loopback and is reached through authenticated SSH; namespace
policy protects worker traffic. A live datapath probe verified that a trusted
worker could reach its head GCS while a non-root, CPU-only pod in a separate
owned namespace could reach the Kubernetes API but failed three attempts to
reach that same GCS endpoint. The probe namespace was removed by its exact
owned identity. This tests one TCP endpoint and namespace boundary, not a
complete port audit or penetration test. Ray 2.58 offers optional
[token authentication](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-security/token-auth.md),
which this recipe does not configure. A Ray namespace or shared token is not
isolation between mutually untrusted users; follow the upstream
[security boundary](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-security/index.md).
The runtime was updated after reviewing the
[Jobs DNS-rebinding](https://github.com/ray-project/ray/security/advisories/GHSA-q279-jhrf-cc6v)
and later [Dashboard DELETE](https://github.com/ray-project/ray/security/advisories/GHSA-q5fh-2hc8-f6rq)
advisories. Using Core actors does not remove Jobs control-endpoint risks.

[Ray working-directory packaging](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-core/handling-dependencies.rst)
is the source-delivery API. The guide visibly copies the canonical Workbench
UDF once; another UDF edit requires repeating that copy. Keep private settings,
credentials, model caches and results outside the submitted source directory.
Local packages have a 500 MiB limit and respect ignore rules.

Application files on worker `/tmp` and Ray's object store are not durable output
storage. The guide downloads and hashes Parquet, Lance, images and reports into
the reader's durable local directory before teardown. It requires no S3 account
or storage credential. Existing Nebius and selected workload preflight passed;
a generic default-S3 check returned `NoSuchBucket` for an unused destination.
That diagnostic was preserved and no storage configuration was changed.

## Fit with the repository and earlier evidence

NPA already has [editable local dependencies](../../npa/pyproject.toml),
[workflow source staging](../../npa/src/npa/orchestration/npa_workflow/src_staging.py)
and [Cosmos3-Nano Ray Serve](../workbench/cosmos3-ray-serve.md). Those mechanisms
serve different purposes; this example does not replace them or use NPA's
source overlay to deliver its application. It qualifies the shared Workbench
CLIP function and LanceDB library, not the deployed LanceDB HTTP service image.
The retained isolated SkyPilot launcher fix supports NPA's existing client
setup; the rejected prototype's runtime/controller/S3 extensions are removed.

Earlier work proved local Docker source edits and then real one-/two-worker
Ray 2.46 GPU jobs. That wrapper already used the native Jobs SDK and
`runtime_env.working_dir` upload. The first GPU recipe was rejected for its
mandatory staging, automatic crop edits, sequencing, finalization and coordinated
state. Those earlier results remain historical; they do not qualify Ray 2.58.
The redesign removes those mechanisms
and separates the basic first-result journey from advanced failure testing.
The former Cosmos baseline test failure was fixed upstream; it is not a current
validation exemption.

A finite `sky launch` / `sky exec` application is a useful alternative for
readers who prefer SkyPilot's source synchronization. It was compared
architecturally, not performance-measured. Do not describe that interface as
Ray Jobs submission or infer a speed advantage from these measurements.

## Final platform cleanup

After both development clusters were gone and outputs preserved, exact owned
Compose teardown took 17.235 s and namespace deletion took 6.659 s; both exited
zero. Twelve final checks confirmed the owned API container, network, state
volume and namespace absent while preserving four pre-existing nodes, seven
unrelated containers and the original API's process identity, configuration
hash and health. Private observer corrections required no repeated cleanup
or GPU operations; their diagnostics remain preserved.

## Bootstrap formatting after GPU qualification

The GPU measurements above precede the final formatting of the startup
port-range guard into a multiline `if`. Application/UDF bytes, model, image,
runtime pins and the customer Jobs commands are unchanged. The bootstrap source
hash changes, so the original GPU/source-package receipts are retained and
their measured identity is not relabelled. Ten paired old/new startup cases
(20 real-shell executions) produced identical exit status, output and Ray argv.
All four head/worker port-boundary cases and 208 focused regressions passed.
This is separate bootstrap verification, not another GPU inference run.

## Repository validation

Ruff, 208 focused tests, 114 onboarding tests, 2,463 guardrail tests and docs
drift passed. The local full suite reported 13,327 passed, 38 skipped, one
expected-failure test passing, and one video-frame extraction failure. An
independent review qualified only that failure as unrelated: its production and
test bytes match the selected main revision, whose Python 3.12 CI explicitly
passed the same test. Running the preserved synthetic inputs directly through
the local FFmpeg 7.0.2 build reproduced an empty video or a native filter
assertion; host FFmpeg 4.4.2 produced the expected eight frames. No Ray code was
in that reproduction, and no test, media argument or timeout was changed.

A separately verified FFmpeg 6.1.3 build stalled on the same compositor and was
rejected as a validation replacement; it is not the distribution build used by
CI. The exact unchanged test passed with host 4.4.2. These private diagnostic
receipts are retained alongside the narrowly scoped exception. The PR's required
checks provide the final published-revision CI result; this exception is not a
waiver for another failure. The guide's CLIP workload processes RGB images and
does not use FFmpeg.
