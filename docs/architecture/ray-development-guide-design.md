# Ray development: guide comparison and API ownership

Accessed 2026-09-05. The design keeps one named SkyPilot development cluster and
uses Ray Jobs directly. The [guide](../testing/fast-source-iteration.md) is the
customer journey; the [audit](ray-fast-development-audit.md) is the evidence
record. This is a tool-specific development example. `npa.workflow` remains
NPA's durable production composition contract.

## What the upstream guides changed

| Official guide | Actual journey and assumptions | Decision for this example |
| --- | --- | --- |
| [Ray Jobs quickstart](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html), checked against [Ray 2.58.0 source](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/cluster/running-applications/job-submission/quickstart.rst) | One inspectable Python file; `ray job submit --working-dir`; streamed output; status/logs/stop. A reachable cluster is assumed. Its small string-returning example does not prepare GPUs, secure a remote server or preserve model outputs. | Show ordinary application code, submit it directly and inspect useful output before fault testing. Keep real GPU preparation and access visible. |
| [Ray Data GPU batch inference](https://docs.ray.io/en/latest/data/batch_inference.html), checked against [2.58.0 source](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/data/batch_inference.rst) | Load data, define a predictor class, run GPU `map_batches`, show/save predictions. Model lifetime is the actor lifetime. Compatible GPU infrastructure is assumed. | Use the same visible model/batch structure with Ray Core around the existing Workbench UDF. No need to introduce Ray Data solely for the example. |
| [SkyPilot Ray example](https://docs.skypilot.ai/en/latest/examples/training/ray.html), checked against the [0.12.2 task](https://github.com/skypilot-org/skypilot/blob/v0.12.2/examples/distributed_ray_train/ray_train.yaml) | One task YAML and a Python trainer, `sky launch`, separate application Ray, visible metrics. A configured backend is assumed. Package pins, private Jobs access and durable download are not its main journey. | Use a development cluster, with visible preparation and Ray startup. Remove the managed production-job controller from the edit loop. |
| [SkyPilot development clusters, 0.12.2](https://github.com/skypilot-org/skypilot/blob/v0.12.2/docs/source/examples/interactive-development.rst) | Named cluster, SSH alias, repeated `sky exec`, exact `sky down`. | Reuse SSH and infrastructure lifecycle. Retain Ray Jobs for source transfer and submission because that is the requested customer contract. |

The application and Jobs client are pinned to Ray 2.58.0, the current stable
release when accessed. Release-pinned sources determine compatibility; the
audit distinguishes completed execution evidence from earlier runtime results.
These guides are useful because they start with a recognizable application and
visible result. Their omitted production responsibilities are
not evidence that those responsibilities disappear.

## What actually became simpler

| Responsibility | Rejected recipe | Final owner and change |
| --- | --- | --- |
| API, backend and network policy | Repeated API/container/config/helper repair within the session | **Platform:** one upstream API/state volume, one fixed namespace/configuration and declared verification/cleanup contract, reused across clusters. Essential, retained. |
| GPU infrastructure | NPA workflow identity plus managed-job controller | **SkyPilot:** one named development cluster and its one Ray service task. The interactive workflow/controller is removed. |
| Application runtime | Separate Ray service plus completion watcher | **Ray:** separate head/workers; essential for Jobs. The external completion watcher is removed. |
| Source and job control | Custom wrapper stages/rewrites revisions, drives Ray Jobs and collects receipts | **Ray Jobs:** visible UDF snapshot, `--working-dir`, submit/status/logs/stop. Custom submission protocol removed. |
| Access | Pod/IP discovery, retained forwarding process and address settings | **OpenSSH:** one Sky-generated alias and one named loopback tunnel. Authentication and ownership remain essential. |
| Useful outputs | S3 copies/profiles, separate upload/finalize Job and marker | **Application + rsync:** first Job writes images/vectors/table; download and hash into one durable local directory. Cloud-storage credentials and finish handshake removed from the developer path. |
| Failure testing | Mandatory medium/complex/fault orchestration | **Optional application recipe:** basic reader reaches a useful result first; checkpoint/restart/cancel validation remains inspectable ordinary Python/Ray commands. |

The developer preserves four identity categories: the named Sky cluster, its
service-task ID, Ray submission IDs and the SSH control socket, plus one durable
results directory. These are responsibility categories, not OS-process counts.
The reader chooses two basic session values (cluster name and result location);
the platform supplies the matching private kubeconfig and API endpoint. Client
binary and current Kubernetes context are derived by displayed commands; image,
model, package and port defaults are declared in checked-in files. The operator
still owns API/volume/configuration and namespace cleanup after all clusters are
gone. Three former application-only mechanisms are removed: the bespoke
submitter, upload/finalize Job and finish-marker watcher. Removing them also
removes their state handoffs; a helper has not merely hidden the same protocol.
The earlier wrapper already used the upstream Jobs SDK and
`runtime_env.working_dir` for packaging/upload. The redesign removes its mandatory
staging, automatic edits, sequencing and receipt orchestration; it does not
replace a proprietary source transport with Ray.

The follow-up readability correction also separates application code by domain:
image preparation, model initialization, inference, output validation and
checkpoint replay. These functions do application work. They do not submit
Jobs, transfer source, run a finish protocol or hide platform cleanup. The
basic guide keeps its ordinary Python example and upstream commands visible;
the more involved recovery implementation remains optional.

## Which APIs own the work

Paths below identify the implementation lines. The [Ray 2.58 Core API](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-core/api/core.rst)
defines tasks, actors, GPU resource requests and actor termination; the Jobs
quickstart above defines source delivery and application control.

| Layer | Implementation | Owning API / honest boundary |
| --- | --- | --- |
| Image rendering and preprocessing | [render_record](../../npa/workflows/workbench/ray-clip-development/worker.py#L39) | Application Python/Pillow; procedural RGB data. |
| Model inference | [ClipModel](../../npa/workflows/workbench/ray-clip-development/embed.py#L136), [canonical UDF:124](../../npa/src/npa/workbench/lancedb/bdd100k_udfs.py#L124) | Workbench's shared `udf_clip_embedding`, Transformers and PyTorch CUDA. This does not qualify the LanceDB HTTP service image. |
| CPU/GPU/distributed scheduling | [basic GPU actors](../../npa/workflows/workbench/ray-clip-development/embed.py#L369), [distributed actors](../../npa/workflows/workbench/ray-clip-development/application.py#L615) | Public Ray Core `ray.init`, `ray.remote`, actor methods and `ray.get`. Actual actor/node receipts establish placement; `SPREAD` alone does not. |
| Source packaging and delivery | [guide submit commands](../testing/fast-source-iteration.md#embed-images-and-see-the-result) | Public Ray Jobs CLI `--working-dir`, which sets Job `runtime_env.working_dir`. SkyPilot mounts only bootstrap files; NPA source-overlay code does not deliver the application. |
| Submit/status/logs/cancel | [guide](../testing/fast-source-iteration.md) | Public `ray job submit/list/status/logs/stop`. No NPA submit/finish protocol. |
| Infrastructure | [cluster.yaml:4](../../npa/workflows/workbench/ray-clip-development/cluster.yaml#L4) | SkyPilot `launch`, `queue`, `cancel`, `down`; existing Kubernetes backend supplies capacity. No competing infrastructure autoscaler. |
| Ray service and isolation | [startup:6](../../npa/workflows/workbench/ray-clip-development/cluster/start.sh#L6), [network policy:1](../../npa/workflows/workbench/ray-clip-development/network-policy.yaml#L1) | Standard `ray start` in a separate environment with reserved ports; Kubernetes policy and authenticated SSH protect access. SkyPilot owns the hosting task/pods. |
| Persistence | [save_results](../../npa/workflows/workbench/ray-clip-development/embed.py#L301), [guide download](../testing/fast-source-iteration.md#embed-images-and-see-the-result) | PyArrow/LanceDB files and standard SSH/rsync download. Ray's object store and worker `/tmp` are not durable artifact storage. |
| Recovery | [actor recovery](../../npa/workflows/workbench/ray-clip-development/application.py#L657), [validation.py](../../npa/workflows/workbench/ray-clip-development/validation.py) | Ray kills/replaces an exact actor; application code owns checkpoint identity, verified reuse and deduplication. This is not automatic exactly-once Ray processing. |

## Boundaries

[Ray runtime environments](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-core/handling-dependencies.rst)
apply the Jobs environment to its driver and descendants. Local source packages
have a 500 MiB limit and respect `.gitignore`; keep secrets, model caches and
results outside the submitted directory. The canonical UDF copy is explicit and
its imported bytes are hashed on relevant workers.

The base image fixes Python/Torch/CUDA, not later package installations.
[Preparation](../../npa/workflows/workbench/ray-clip-development/cluster/prepare.py)
pins application packages and model revision, verifies weight bytes and records
the resolved environment. This is not a fully hash-locked transitive dependency
supply chain. Source edits reuse the prepared environment; package changes need
re-preparation, and incompatible Python/CUDA/native ABI or system changes need
a compatible image. New source Jobs reload models into new actors; repeated
batches within an actor reuse its resident weights.

[Ray 2.58 security guidance](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-security/index.md)
requires trusted code and external isolation. Ray supports optional
[token authentication](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-security/token-auth.md);
this recipe leaves it unconfigured. Loopback Jobs plus authenticated SSH
protects the HTTP endpoint, and network policy also protects GCS/worker traffic.
Ray namespaces and a shared token do not isolate mutually untrusted Jobs.
A live network probe allowed a trusted worker to reach its head GCS and blocked
three attempts from a separate CPU-only namespace that could reach the Kubernetes
API. This checks one endpoint and boundary, not comprehensive penetration testing.
Application ports remain separate from SkyPilot's management Ray. No broad
`ray stop` or process matching is used. The upstream [Sky startup](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky_templates/ray/start_cluster)
and [shutdown](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky_templates/ray/stop_cluster)
are references, not adopted unqualified: their dependency installation and
process matching do not establish this example's isolation contract.

The Ray pin was updated after reviewing the upstream
[Jobs DNS-rebinding advisory](https://github.com/ray-project/ray/security/advisories/GHSA-q279-jhrf-cc6v)
and the later [Dashboard DELETE advisory](https://github.com/ray-project/ray/security/advisories/GHSA-q5fh-2hc8-f6rq).
These affect development/Jobs access, so using only Ray Core does not remove
the need for the update. The selected [2.58.0 release](https://github.com/ray-project/ray/releases/tag/ray-2.58.0)
updates the application and customer client; SkyPilot's isolated management
runtime remains vendor-owned. Fresh one- and two-worker GPU checks now qualify
Ray 2.58 through the literal native Jobs commands, including source changes,
checkpoint replay and exact cancellation; the audit records their measurements.
Earlier 2.46 GPU receipts remain historical.

A finite application launched with `sky launch`, rerun with `sky exec`, and
using `ray.init(address="local")` is a useful simpler alternative when customers
want SkyPilot's source-sync/submission interface. It is **not** Ray Jobs source
delivery. That alternative has not been performance-measured here; no Ray vs
SkyPilot speed claim follows from the architecture comparison.

## A preflight that did not prove launch readiness

During the initial redesign investigation, a local SkyPilot 0.12.2 API answered
short requests but had no working long-request executor. An accepted dry run remained pending and was cancelled
by its exact upstream request ID; no GPU allocation followed. Separately, the
installed `/validate` path spawned threads that lost request-local `KUBECONFIG`
and read the server's default backend. A credential check alone was therefore
insufficient proof of correct placement.

The example uses a fixed backend configured in its owned upstream API service,
with explicit client allowed-context selection. Its preparation contract must
prove actual dry-run completion, launch, namespace and pod ownership. It does
not patch or restart another user's API. This limitation is specific to the
observed pinned runtime; no claim is made that all SkyPilot versions have the
same behavior.
