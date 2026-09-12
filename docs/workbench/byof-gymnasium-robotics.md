# Gymnasium-Robotics BYOF candidate

This integration is a minimal BYOF registry candidate for Farama Foundation
Gymnasium-Robotics. It is not a first-class NPA tool and does not claim policy
training, expert performance, other environment families, or physical-robot
transfer.

## Neutral public-image quarantine

The repository describes a zero-Shadow-payload neutral bootstrap, not a built
image. `npa-gymnasium-robotics` remains in **pre-registration quarantine**: it
has no supported tag, accepted manifest, SM120 compatibility record, registry
object, or anonymous-pull evidence. The candidate layers exclude the upstream
solution tree, Shadow Hand assets, MuJoCo/Python wheels, runtime cache, CUDA or
other vendor runtime, credentials, datasets, checkpoints, and outputs. The
Dockerfile's first executable gate binds the exact neutral Ubuntu/Python
bootstrap closure and corresponding-source metadata before package network
access, and rejects any incomplete or mismatched lock. A URL, credential,
private registry, or consent proxy must not bypass that refusal.

At execution time, the bootstrap reads a repository-pinned manifest before any
fetch, permits only the declared credential-free HTTPS origins, checks exact
redirect targets, sizes, and SHA-256 digests, safely extracts the upstream
source, validates wheel archives, installs offline in private staging, seals an
exact no-follow manifest of every retained path, type, mode, size, and file
hash, and atomically publishes a complete version to an external operator-owned cache.
An incomplete lock, missing anonymous access, malformed archive, mismatched
byte, unsafe cache, or partial existing version is a terminal refusal.
The fetched interpreter is opened and executed by validated file descriptor,
never through the mutable `current` convenience link. Runtime installation and
the capability smoke have AWS credential variables removed. Pod-receipt
validation and artifact bookkeeping stay on the image-baked system Python and
Ubuntu `python3-boto3`; their transitive binary/source/license closure is bound
to the signed immutable Ubuntu snapshot.

The accepted private result at repository head `c308945a` is historical proof
for its exact private digest and workflow bytes only. It is not evidence for
this redesigned source, an executable image, the current head, or a future
public digest. Any later private qualification remains owner-only evidence and
does not establish an accepted or anonymously pullable public image.

## Pinned inputs and licensing

The official source is
[`Farama-Foundation/Gymnasium-Robotics`](https://github.com/Farama-Foundation/Gymnasium-Robotics)
at commit `4d1ebecbc6436806cfbc0e42ebc36f594d05844e`. The commit was
made on September 7, 2026: it is a later maintained main-branch revision, not
the v1.4.2 tag commit. Its package metadata still reports 1.4.2; the current
official v1.4.2 release is from January 2, 2026. The repository root is MIT
licensed.

The Shadow Dexterous Hand files retain
`gymnasium_robotics/envs/assets/LICENSE.md`. That notice attributes the model to
Shadow Robot and Vikash Kumar and includes Apache-2.0 terms. The linked Shadow
Robot `sr_common` `kinetic-devel` source resolves to commit
`59d6bdf35bd9cf53185a20eb63413fdfe57fe77c` and has a GPL-2.0 root
license. The asset provenance is therefore conservatively documented as
GPL-2.0 plus the packaged Apache-2.0 attribution rather than MIT-only. The
historical private image contained the complete pinned Gymnasium-Robotics
source and its notices. The neutral candidate copies no XML, mesh, texture, or
upstream source byte into its layers.

The runtime pins official Google DeepMind MuJoCo 3.12.0 (Apache-2.0), release
commit `13827e9ee56f097f57acf69ae52b078f9839682d`. Its CPython 3.12
x86-64 wheel SHA-256 is
`7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0`.
The wheel embeds the Apache-2.0 file with SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
and its third-party notice bundle with SHA-256
`aec5167579b94d6926340175b4f764b5159f4933657556d89bbfb8238d3b3eb8`.
The base is the Linux amd64 Docker Official Image
`ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61`.
Its config is `sha256:b2b7ea366714195a1e1c5b2b578ece85c0b3920381a8654d038d9684f009613c`
and its sole layer is
`sha256:e51aee9c82ec5dd5ba2add49c45c6d85d460512757e2615b69bcdf9469c7cb58`.
Ubuntu is a package collection under the individual package licenses. The
repository lock now records 142 final binary packages, 102 source packages, 318
source artifacts, and 142 installed copyright records from the signed immutable
snapshot. This source closure permits the separately reviewed private build;
it is not public corresponding-source delivery or release acceptance.

No model, external dataset, gated artifact, or terms-acceptance flag is used.
There is no checkpoint. The official pinned source archive contains the Shadow
assets and is acquired anonymously into the operator cache only after the
runtime lock is complete. The generated JSON is factual operator telemetry.
RGB frames remain in memory and contribute only hashes; no upstream asset byte
is republished as an output.

| Artifact class | Delivery and decision |
| --- | --- |
| Source | Exact official commit and MIT grant are reviewed inputs. The source archive is runtime-cache-only and must never enter candidate layers. |
| Baked runtime | Neutral Ubuntu/Python bootstrap only. Its exact signed-snapshot binary, license/copyright, and corresponding-source mapping is locked. MuJoCo and the Python workload graph are runtime-cache-only; public corresponding-source delivery remains withheld. |
| Weights | None. No model or checkpoint is fetched, baked, mounted, or emitted. |
| Data/assets | No external dataset. Exact Shadow Hand XML/STL/PNG hashes and `assets/LICENSE.md` are recorded. The assets remain runtime-cache-only; their preferred-form, transformation, use, and derivative questions remain unresolved. |
| Runtime cache | External, private to the runtime operator, versioned by the exact fetch-manifest hash, sealed read-only, and atomically published only after full-tree verification. Every reuse rechecks its exact receipt and path/type/mode/size/file-hash manifest. It is neither an image layer nor a rights grant. |
| Outputs | Operator-generated factual JSON telemetry and execution logs. RGB frames are transient; only hashes and measurements are durable. No upstream asset byte is republished. |

Runtime fetch changes delivery only. It does not grant or resolve rights to
use, create derivative works, retain outputs, or provide a hosted service, and
private custody does not change that boundary. The built bytes must still be
scanned to prove the absence of upstream source/assets, MuJoCo/Python workload
runtime, vendor runtime, secrets, datasets, checkpoints, and persistent caches.
The future acceptance scanner binds independently reviewed hashes for every
completed neutral-image lock and the Ubuntu base diff ID; the post-build config
and ordered layer graph remain withheld until real bytes exist. It also checks the exact Docker config, every
raw ordered layer (including header/padding bytes), all retained regular files,
directories, links and metadata, and recursively nested source/package
archives. A locally self-consistent lock or rootfs classification cannot pass.

## Hard-gate capability

[`byof-gymnasium-robotics.yaml`](../../workflows/testing/byof-gymnasium-robotics.yaml)
runs the upstream registered
`HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` environment with the
official pinned Shadow Hand XML, meshes, and textures. A seeded 120-step action
trajectory must prove all of the following:

- real MuJoCo state evolution, finite rewards, physics substeps, and contacts;
- all 92 continuous touch sensors, including nonzero live readings;
- quantitative object-position, object-orientation, and full-state changes;
- actual EGL RGB rendering with at least two distinct frame hashes and an
  observed loaded NVIDIA EGL library path and SHA-256;
- exactly one RTX PRO 6000 Blackwell GPU at compute capability 12.0; and
- equality between the pushed immutable image digest and Kubernetes' pod
  `imageID` observation.

The smoke emits exactly the capability artifact
`$NPA_SMOKE_OUTPUT_DIR/gymnasium-robotics-smoke.json`. Its normalized content
hash, media type, and exact byte size are embedded in the JSON; the uploader's
summary also records the actual uploaded-file SHA-256 and byte size.
Uploads use NPA's standard atomic `If-None-Match: *` object-create contract so
an existing run artifact cannot be overwritten. The target S3-compatible
service must support that contract; failure is terminal and must never be
downgraded to a racy HEAD followed by an unconditional write.

## Run and verify

Cloud execution is allowed only after the manager publishes task-owned runtime
context for the reserved RTX PRO capacity. Do not provision a cluster from this
workflow and never route this EGL workload to B200.

Before submission, preserve the manager-published kubeconfig and derive a
mode-0600 child-local copy whose selected child-specific context is explicitly
bound to the assigned namespace. The task-private SkyPilot config must use the
supported exact-name placement contract with one entry and no IP or label
fallback:

```yaml
kubernetes:
  allowed_nodes:
    names:
      - <manager-assigned-rtx-node>
```

The live gate requires the manager-issued node name, node UID, and provider
node-group identifier at runtime, then verifies that the sole allowed node has
the exact UID, is Ready, exposes one allocatable GPU, carries exactly one label
whose value is that provider group, and advertises the RTX PRO 6000 Blackwell
product. Owner-only preflight must additionally prove that this live provider
node group has fixed/ready/target count one and a `STRICT` reservation policy
containing exactly the one manager-assigned active reservation. These values
and receipts must never enter the repository or PR text.

The live E2E also requires
`NPA_BYOF_GYMNASIUM_ROBOTICS_OUTPUT_ROOT` to be the exact manager-authorized,
task-owned S3 prefix. It fails closed before submission if that value is absent
or is only a bucket. It also requires the exact manager-authorized namespace
and an owner-private local evidence directory. The owner identity must already
be able to list Pods and exec into the exact run Pod. The list is restricted to
the manager-authorized namespace and SkyPilot parent label, then the live gate
matches the full run annotation, immutable spec image, one-GPU request and
limit, assigned `spec.nodeName`, container name, Pod UID, and Kubernetes
`imageID`. It injects that
observation as a mode-0600 receipt. The workload consumes the receipt without
Kubernetes API access. Do not grant the workload service account Pod access or
create a RoleBinding for this integration. The pre-submit `auth can-i` check is
necessarily namespace-wide; exact-Pod scope is established only after Pod
creation by matching its run annotation, UID, container, and image digest. If
the private registry later needs
a run-scoped pull secret, create it only after these checks and ledger its exact
Kubernetes UID, ownership, and cleanup contract before launch. Runner output is
drained directly to owner-private files. Receipt polling, Kubernetes commands,
runner termination, `sky down`, and deletion-to-absence checks are bounded;
every failure attempts exact-run cleanup before returning.

```bash
npa/.venv/bin/npa workbench health preflight --checks nebius,s3 --json
npa/.venv/bin/npa workbench workflow validate-spec \
  workflows/testing/byof-gymnasium-robotics.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec \
  workflows/testing/byof-gymnasium-robotics.yaml \
  --run-id gymnasium-robotics-review --json
```

The workflow uses `base_profile: prebuilt`, an intentionally non-resolving
placeholder, an empty build command, and the fixed neutral entrypoint. A later
qualification transaction must inject an already scanned immutable digest.
The entrypoint may populate only the external operator-owned cache from the
complete immutable runtime lock before invoking the fixed capability script;
it must never modify image layers or treat fetch success as rights acceptance.
Set `NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE` only to that reviewed reference before
enabling the dedicated live E2E gate. Exact registry, storage, cluster, reservation,
pod, and run identifiers belong only in owner-only evidence.

## Deferred scope

RL training sweeps, expert or benchmark scores, other Gymnasium-Robotics
environment families, behavioral claims beyond the single deterministic smoke,
and physical-robot transfer are deliberately deferred. A local import, reset,
container start, CUDA check, or synthetic fixture does not satisfy this
candidate's capability gate.
