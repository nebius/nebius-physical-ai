# LIBERO neutral-bootstrap qualification

LIBERO is an unqualified public-development candidate, not a supported release. Its
planned image is a neutral, non-root bootstrap that contains no LIBERO or GPU
runtime. The trusted workflow may stage this payload-free candidate for inspection;
a future customer-authorized run may materialize one exact runtime into an
operator-owned cache, then exercise one official `libero_spatial` task through
eight upstream BC-RNN optimizer steps, strict checkpoint reload, and a genuine
trajectory-disjoint held-out evaluation on one B200. The path is headless and
never invokes rendering.

The planned managed workflow is
[`workflows/testing/byof-libero.yaml`](../../workflows/testing/byof-libero.yaml).
Its `tool://libero` default deliberately cannot run: execution requires an
explicit qualified `npa-libero@sha256:…` candidate, a short-lived customer/run
authorization, and independent build-lineage evidence. There is no published LIBERO
tag or digest and the public image table remains unchanged.

Image qualification and customer acceptance are separate. The checked-in
qualification binds independently reviewed immutable image and publication
lineage. After reviewing all seven exact terms for one run, the customer signs
the canonical authorization JSON directly with a customer-controlled Ed25519
key. A separately authenticated caller assertion binds that signer fingerprint
to the customer and run. NPA may authenticate the caller, transport the signed
evidence, and validate it, but neither the manager nor control plane accepts,
acknowledges, issues, or signs the customer's terms assertion. The verifier
uses only the public key embedded in that signed evidence and the caller-bound
fingerprint, and requires that key to match an immutable customer-signer
registration provisioned outside the NPA control plane; there is no second
transported public-key file or alternate trust source. A missing or mutable
registration refuses before cache or network effects. Private signing material,
customer identity,
credentials, acceptance records, terms payloads, and runtime payloads are never
baked into the neutral image. Image qualification carries no customer signer
fingerprint; signer trust is supplied owner-private and checked only at runtime.

The customer-run runtime supplies these inputs through its authenticated
handoff; they are not generic BYOF workflow/config inputs, customer acceptance
switches, or manager-issued values and must not be synthesized locally:

| Input | Boundary |
| --- | --- |
| `--libero-customer-runtime-authorization-file` | Owner-private, short-lived, directly customer-signed authorization for this customer, signer, run, workflow digest, immutable source revision, image, manifest, and exact terms. Missing input or a valid signed denial returns the structured terms notification. |
| `NPA_LIBERO_AUTHENTICATED_CALLER_B64` | Short-lived control-plane authentication assertion. It binds customer identity, run, and the customer's signer fingerprint but contains no terms decision and cannot authorize runtime fetch alone. |
| `NPA_LIBERO_CUSTOMER_IDENTITY_SHA256` | Identity hash derived from the verified caller assertion and customer evidence and forwarded to the isolated runtime; customers do not set it directly. |

The authorization file contains no access credential. HF/NGC credentials are
separate upstream-access inputs and never establish terms acknowledgement.

## Six independent boundaries

| Boundary | Phase A contract |
| --- | --- |
| Source | `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT. Source is absent from the image. An authorized runtime sparse-fetch retains only training/config/task-definition paths, verifies the source tree and license hash, and removes `.git`. `libero/libero/assets` is excluded. |
| Baked runtime | The proposed public bytes are the exact linux/amd64 `python:3.10-slim-bookworm` manifest `sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`, snapshot-pinned Debian bootstrap packages, NPA verification material for transport/storage separation, and NPA-owned files. No customer key or private signing key enters the build, image, workflow, or repository. `debian-packages.lock` records every binary and corresponding source. Docker Official Images' immutable in-toto provenance independently binds the base to rootfs material `sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867`. The image contains no PyTorch, CUDA, cuDNN, NCCL, NVIDIA wheel, MuJoCo, robomimic, or robosuite byte. |
| Weights | None are baked. Exact `google-bert/bert-base-cased@cd5ef92a9fb2f889e972770a36d4ed042daf221e` files are runtime-only and Apache-2.0. |
| Data and task inputs | No demonstration or task/render asset is baked. The selected official demonstration is runtime-only: `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`, 508,779,600 bytes, SHA-256 `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`, upstream-declared CC BY 4.0 with LIBERO attribution. The MIT BDDL and initial-state files are fetched only with the sparse source and verified by SHA-256. |
| Runtime cache | `/workspace/.cache/npa/libero/<customer-run-manifest-scope-sha256>` is customer/run/manifest-addressed, atomically completed, sealed group-readable/non-writable, and separate from output. The bootstrap owner and execution UID are distinct, so fetched code cannot restore cache write bits. A shared lock and stable directory descriptor remain held through smoke/upload execution, with full inventory checks before and after. A cold population resolves all seven hash-bound official governing-terms sources after authorization and before the first cache mutation. The cache is never uploaded. Missing, denied, expired, invalid, wrong-customer/run, wrong-image, or wrong-manifest authorization refuses before cache or network effects. Runtime fetch changes delivery, not permission. |
| Outputs | Execution begins in an owner-controlled sticky staging directory, while the bootstrap receipt remains in a separate owner-writable/group-readable cache directory. After the execution UID exits, the supervisor proves that no process under that UID remains, seals the output directory owner-only, and materializes the protected bootstrap/cache receipts itself. Exactly ten named qualification files may then exist beneath a stable, descriptor-opened `$NPA_SMOKE_OUTPUT_DIR`; any missing or additional entry refuses. Every uploadable JSON file is opened with `O_NOFOLLOW`, copied to stable in-memory bytes while inode/size/mtime identity is checked, and the complete local inventory is closed before network output begins. Only four supervisor-defined JSON evidence files are strictly schema-validated, checked for embedded payload fields, canonically serialized, and uploaded; the checkpoint, logs, and GPU probe text remain local and are validated by the local execution envelope. An exact version-bound output lease is retained for the transaction. Each canonical snapshot is conditionally created below a unique immutable transaction prefix, then read back by provider-issued version ID, ETag, checksum, and bytes. The final create-only receipt binds every uploaded object key, version, ETag, size, and digest; pre-existing state refuses without being claimed or deleted. |

The task mentions Google Scanned Objects and a HOPE distractor, but their meshes
and textures are neither fetched nor needed for stored-observation behavior
cloning. Rendered or closed-loop evaluation therefore remains deferred. The
mirror card's license label does not override the upstream LIBERO publisher's
CC BY 4.0 dataset declaration.

## Quarantine and image proof

`npa/docker/workbench/libero/Dockerfile` describes the candidate but qualification
and release remain pending. The non-root `ubuntu` bootstrap user has finite root
sudo grants for generating missing per-Pod SSH host keys with the exact `-A`
argument, for starting or restarting the preinstalled SSH daemon, and for the
preinstalled `NPA_SKYPILOT_APT` guard (only fixed `apt-get update`/`install`
forms are accepted; the guard refuses package acquisition and arbitrary
commands). Its remaining sudo grant invokes the fixed bootstrap entry point as
the separate
unprivileged `npa-libero-exec` user with exactly `execute`. Its environment
excludes storage credentials as well as Python and dynamic-loader injection
variables. After fetched code exits, the owner-controlled uploader strictly
parses the four permitted JSON evidence files, rejects unknown fields and
embedded payload values, and uploads only their canonical allowlisted
representations with separately accepted, run-prefix-only temporary
credentials. Checkpoints, logs, and GPU probe text remain local.
No host key is baked into an image layer. The complete SkyPilot
0.12.2 synchronous package set is
baked, including `curl`, `wget`, and the `fuse3` provider. When every package and
command verifies, the guard bypasses SkyPilot's redundant package-index
transaction. Missing packages fail closed; the deadline wrapper applies TERM
and a kill-after escalation and emits an unambiguous failure sentinel.

This source-only Phase A description does not provide or claim a network sandbox
for the fetched workload. Removing credentials is not an egress boundary: a
materialized workload could still attempt outbound connections unless it is run
inside an independently observed, default-deny network namespace or equivalent
policy. A future qualification must prove that deny policy and its identity,
then grant only a separately trusted uploader access to the exact origin-only
storage endpoint after execution. Until that evidence exists, privacy,
confidentiality, and no-egress claims remain unqualified and no fetched workload
is considered publication-ready.

Before any public byte is disclosed, a separately authorized private stage must
first emit a canonical complete-image inventory. It binds every byte in each
ordered uncompressed layer tar, the resulting flattened-rootfs path records,
and the observed OCI config digest. The checked-in qualification record is only a
descriptive, hash-bound candidate record; it is not an approval or trust boundary
and cannot qualify its own identities. The current source-only branch has no
publication transaction that verifies an external approval receipt. A later
operator-controlled publication transaction must first verify an immutable,
owner-only review receipt bound to the exact source SHA, OCI config/index/layers,
rootfs inventory, reviewer identity, and expiry; an absent or mismatched receipt
keeps LIBERO blocked and unvalidated. Any future trusted publication workflow
must obtain qualified identities only from that verified receipt or qualification
handoff; free-form dispatch values cannot create or widen qualification. Its
pre-push scanner then requires complete image, config, exact Buildx metadata, and
published-base provenance equality. Finite
path and content signatures remain defense in depth, not the proof that
arbitrary renamed, compiled, or subsequently whiteouted bytes are absent.
Before LIBERO's first package-wide visibility change, the private destination
must contain exactly the qualified untagged OCI graph and no unrelated version.
The workflow still refuses the visibility change: repository workflow concurrency
does not exclude other registry writers, so the current source-only
implementation refuses before any visibility PATCH or first publication,
including an absent or empty private destination. A retained private package is
preserved, never deleted merely because its tags or OCI subjects match a
candidate. This is an implementation boundary for the current branch, not a
permanent eligibility requirement: a later customer-scoped transaction may
proceed only after exact graph preflight and must fail closed on any observed
concurrent drift. No public development digest, anonymous pull, or B200
qualification is claimed here.

Requested and failed-build reconciliation may inspect package identity, graph,
tags, and run-bound evidence, but it does not delete package versions or change
visibility. Even an exact final graph read cannot exclude another writer adding
a release tag before a version deletion. Without an atomic identity-and-tag
deletion primitive, reconciliation exits with an explicit refusal and retains
the versions and package configuration for operator recovery. Absence can be
reported only from the existing bounded read-only checks. A failed or cancelled
build remains failed; retained public bytes are not claimed revoked or cleaned.
This source-only boundary is not authorization to publish, repair a registry,
accept customer terms, or perform a live run.
Both build paths derive `SOURCE_DATE_EPOCH` from that exact source commit so the
identity comparison cannot depend on the wall-clock build time. The package
transaction removes APT/dpkg/account logs and normalizes the non-root account's
shadow day to the same epoch in its creation layer.

The following is a future, separately authorized publication runbook; it is not
an executable requirement or completion claim for this source-only phase. No
public digest, anonymous-pull result, or B200 capability is implied until a
customer-scoped authorization and an independently reviewed first-publication
transaction with exact graph preflight and fail-closed drift handling exist. If
that transaction is not available, the
runbook remains deferred and the quarantined candidate remains unvalidated.

When that later transaction is authorized, the trusted build must then:

1. reproduce the accepted development SHA from its hash-bound neutral build-input
   bundle with Buildx SBOM and maximum provenance, exporting an attested OCI
   archive; verify its runtime config plus both subject-bound attestations before
   selecting and importing only its linux/amd64 runtime manifest for complete-byte
   scanning;
2. fetch and hash-verify Docker Official Images' exact published provenance
   metadata, without using Docker history;
3. inspect every OCI config and ordered layer, nested archive, exact runtime
   payload hash, renamed source signature, and an independently exported
   flattened rootfs with `scan_image_libero_payload.py` plus the general
   restricted-payload scanner; require the canonical complete-image inventory
   and config to equal the private-stage identities in the strict checked-in
   qualification manifest; free-form dispatch values are not an authorization
   channel;
4. prove non-root/config/bootstrap identity and independently bind the candidate
   index, platform manifest, config, Buildx metadata, base manifest, rootfs
   material, upstream source revision, and canonical publication/infrastructure
   bundles. The publication bundle recursively binds the complete Python source
   trees used for provider identity, preflight, managed submission, scheduler
   identity, storage readback, signal teardown, and cleanup, plus the exact live
   evidence collector; the checked-in qualification manifest is the only permitted
   path delta after the qualified development SHA, so enforcement/runtime drift
   requires a new candidate and renewed qualification;
5. pass package-license, corresponding-source, Trivy fixed-critical,
   repository Gitleaks, credential, private-infrastructure, cache, data, model,
   output, and confidentiality gates; and
6. push only those inspected bytes to the quarantined development tag, resolve
   an immutable digest, repeat the scanner against pulled bytes, and prove
   anonymous digest equality.

Until those gates and a later exact-digest B200 qualification are independently
accepted, `libero` stays in `UNVALIDATED_PUBLICATION_TOOLS`, has no release
manifest entry, and must not appear in the published-image table. Historical
private r15 bytes were built from an older head; they are private-stage evidence
only and cannot prove public-byte equivalence or live capability.

## Runtime authorization and managed execution

The runtime materializer reads `runtime-manifest.json`, which pins the source
tree, MIT license, sparse task files, CC BY 4.0 demonstration, Apache-2.0 BERT
files, seven official governing-terms documents, and 135 runtime artifacts by
URL, reviewed size, license expression, and SHA-256. The checked-in inventory is
complete at 3,277,640,175 total artifact bytes, but materialization fails closed
until the customer personally acknowledges the exact named terms at their
official URLs and produces direct customer-controlled signed evidence.
`runtime-requirements.txt` is itself hash-bound and must match the ordered
artifact manifest exactly; both bootstrap and remaining installs use
`pip --require-hashes --no-deps --no-index`.

Before authorization, preflight returns a structured
`needs_customer_acceptance` result containing the exact term names, URLs and
content-hash versions, runtime-manifest digest, acknowledgement instructions,
and an explicit refusal path. It also states that HF/NGC credentials establish
upstream access only. The customer may then sign one maximum-24-hour
authorization bound to the authenticated customer and signer, run ID, complete
executable workflow-profile digest, runtime-manifest digest, immutable source
revision, exact term IDs/versions, immutable qualified image, and
acknowledgement/issue/expiry data. The control plane validates and transports
this evidence only. Missing, denied, expired, wrong-customer/run,
wrong-manifest/image, malformed, or improperly signed records refuse before the
first download, install, cache write, or workload operation. No local flag,
`ACCEPT_*` variable, token, access probe, or successful download substitutes for
the authorization, and NPA never automates a vendor acceptance action.

The authorization travels only through the scheduler's redacted secret channel
and is materialized owner-private for bootstrap verification. The neutral image
verifies it with `/usr/bin/ssh-keygen` against the customer public key embedded
in the signed payload and requires its fingerprint to match the separately
authenticated caller assertion. A control-plane-only signature, locally
invented issuer, or matching environment hashes cannot authorize fetch. Only
after this check does bootstrap resolve and hash the official terms into
ephemeral storage; changed or unavailable terms refuse before cache mutation.

The workflow uses `base_profile: prebuilt`, an empty build command, and the
dedicated B200 profile. `run_byof_repo.py` rejects an empty or mutable candidate,
an absent/mismatched customer authorization, an absent independent build-metadata hash, any
identity drift, and `--skip-run` before registry or workload preparation. It
always passes `--no-direct-launch` for LIBERO. `run_byof_container_verify.py`
submits through the managed scheduler, requires its nonempty scheduler job ID,
and polls that ID—not the human run name. Absence remains a failure.

The direct customer-signed authorization, complete short-lived storage credential
triplet, and output-storage authorization bytes travel only through the
scheduler's redacted secret channel, never ordinary rendered workflow state or
persisted prepared YAML. Customer identity remains represented only by its
authenticated hash in redacted runtime audit receipts.
The common SDK submission preflight independently verifies the repository's
image qualification, signed customer/run authorization, exact run ID, immutable
candidate image, and the digest of the profile bytes it is about to submit
before it selects or forwards any LIBERO secret. LIBERO accepts only the Kubernetes managed-jobs
backend with an explicit context. The shared classifier treats the official
`npa-libero` repository in either `resources.image_id` or `BYOF_IMAGE` as a
LIBERO signal, so renaming a raw SDK task cannot bypass this gate. Every signal
requires the canonical task name, solution marker, and the same exact qualified
repository-at-digest image in both fields. The classifier result, rather than a
second name-based guess, controls secret forwarding and the final serialized
profile check. The SDK snapshots the profile once for parsing and hashing, then
requires the final serialized task bytes to retain that digest before any
controller or job operation.
The separate control-plane storage authorization is hash-bound by the prepared execution bundle,
binds the exact run prefix and origin-only HTTPS storage endpoint plus the policy
receipt, and requires short-lived session credentials. The same shared SDK gate
validates the complete credential triplet, authorization digest, hashes, policy,
prefix, endpoint, expiry, and nonce before target resolution or controller
effects; absence is a refusal, not an optional state. Storage secrets are
removed from the fetched-code subprocess; only
the image-owned standard-library uploader receives them. Per-file and aggregate
size budgets apply before reads, PUT is conditional, and GET must return the
service checksum as well as identical bytes. S3 key segments are encoded
individually so slash separators are identical in the SigV4 canonical URI and
the TLS request target. The owner-only worker file is removed immediately after
materialization, and the Base64 secret is unset again at the start of the run
phase.

The payload profile selects `npa-byof-libero-payload`; `default` and
`skypilot-service-account` are forbidden for the payload. The manager-owned
SkyPilot controller remains on its separate engine account. Before a future run,
the exact run-labelled namespace must be empty of Pods and Secrets and contain
only `default`, the reviewed payload and controller ServiceAccounts, their two
exact Roles, and their two exact RoleBindings. The payload Role begins as core
`pods/get` for a no-match `resourceNames` sentinel. After managed submission
returns its exact numeric job ID, the host requires exactly one new Pod, binds
its SkyPilot job label, head identity, service account, and sole container image
to the qualified candidate, then uses a JSON Patch that tests the Role UID and
complete sentinel rule before replacing it with that one Pod's exact
`resourceNames` entry. The workload cannot read any Pod before this binding and
can read only itself afterward. The controller Role has no wildcard: it
enumerates only the namespaced Pod, Pod exec/log, ConfigMap, Secret, and Service
operations needed by this fixed container job. The host verifies both identities and rejects
every workload-granting ClusterRoleBinding or cross-namespace RoleBinding that
reaches the namespace's service accounts before submission, then repeats the complete empty
namespace/payload grant and controller checks after infrastructure preflight
immediately adjacent to submission. It rechecks the bound Role, sole Pod UID,
job label, image, and no-ClusterRoleBinding result immediately after binding and
at every scheduler-status observation, together with the controller identity.
These sampled checks are not a run-long isolation proof: qualification MUST also
bind a dedicated cluster or an admission-enforced exclusive-writer policy that
rejects every non-run-authorized namespace, RBAC, and workload mutation for the
entire execution interval, plus a gap-free Kubernetes watch or audit-log interval
anchored by resource versions before submission and after cleanup. Every unexpected
create, update, or delete event fails qualification. If that enforcement and
continuous event evidence are unavailable, the run stops before submission and
the documentation must not claim run-long isolation.
At a failed terminal status the exact Pod may already be absent, but a successful
qualification must still expose it for the manager-side live-evidence read;
any replacement or foreign Pod is a hard failure. Every
namespace/object UID, inventory, and payload permission is bound to the prepared
execution-integrity record. The isolated namespace inventory includes ConfigMaps as well
as Secrets, Pods, Services, accounts, Roles, and RoleBindings. Kubernetes automatically
injects one `kube-root-ca.crt` ConfigMap; that exact object is the sole allowed
ConfigMap exception. The manager records its namespace/object UID and a canonical
metadata/data SHA-256, requires the expected system-owned shape (no workload owner
reference and only the `ca.crt` data key), and checks it is unchanged at every inventory
boundary. Any additional ConfigMap, changed label/ownership/data, or missing root-CA
object is fail-closed. The canonical
external RBAC inventory binds every allowed broad discovery/self-review grant,
including binding and referenced-role UIDs and rules, so later RBAC drift changes
the accepted infrastructure hash. The payload reads its own Pod and bound JWT
and records the actual service account, Pod UID, node, and
`containerStatuses.imageID`. That self-report is workload output, not
authenticity proof. Before cleanup, the manager-side runner independently reads
the bound Pod status and its exact node, verifies the container's accepted
digest, one-GPU request, B200 node labels, and allocatable GPU state, and emits a
separate manager-live-evidence envelope. The later live gate requires every
corresponding hashed Pod/node/service-account field and image digest in the
uploaded report to equal that independent envelope, and parses the captured
`nvidia-smi -L` output as exactly one B200.

The execution and payload-proof kubeconfig contexts remain distinct but must
flatten to the same cluster identity. A single owner-receipted allowed node and
STRICT B200 reservation are mandatory. All access objects and isolated scheduler
state are ephemeral. After exact job cancellation and cluster absence, cleanup
uses UID preconditions for both controller and payload RoleBindings, Roles, and
ServiceAccounts before the namespace, polls every object to 404, stops the
isolated API, and only then removes the payload kubeconfig and exact-run local
SkyPilot state. Any ambiguity is a failed cleanup and preserves recovery state.

The cluster-wide binding checks cover direct ServiceAccount and User subjects,
the global and namespace `system:serviceaccounts` groups, and resource-bearing
grants to `system:authenticated` or `system:unauthenticated`. Only Kubernetes
non-resource discovery and self-access-review rules are allowed for those broad
public groups, whether a ClusterRoleBinding or namespaced RoleBinding supplies
the grant.

The qualified publication-enforcement bundle includes a closed manifest of known
LIBERO and shared policy tests. A future shared test that protects this contract
must put the exact line `# npa: publication-enforcement=libero` in its source;
the bundle discovers that explicit marker and does not infer policy ownership
from an incidental word match.

## Acceptance artifact

A later live gate owns the managed submission, preserves the manager-side
Pod/node/GPU envelope collected before cleanup, independently retrieves the
four canonical JSON objects plus the final `npa_upload_receipt.json`, verifies
every remote object size/hash/checksum against fresh GET bytes, validates the
checkpoint/log/GPU-probe files from the local execution envelope (they are
never uploaded), checks cleanup/API-stop evidence, and accepts the retrieved
`libero-smoke.json` only with:

- exact source, task, dataset, BERT, runtime-manifest, cache, and license proof;
- 50 official trajectories / 5,068 samples split into 40 train trajectories
  (4,020 samples) and 10 held-out trajectories (1,048 samples), with disjoint
  trajectory-ID hashes;
- upstream `AutoTokenizer`/`AutoModel` BERT `pooler_output` conditioning, never
  a synthetic embedding;
- eight nonzero upstream `Sequential.observe` / `BCRNNPolicy` optimizer steps,
  finite losses, and a changed trainable parameter;
- checkpoint SHA-256, strict upstream reload, full held-out loss, two or more
  held-out forwards, and finite reloaded 7-DoF action shape/dtype/value proof;
- exactly one observed B200, compute capability 10.0 / `sm_100`, the Pod-observed
  immutable candidate digest, actual solution-scoped payload service account,
  and owner-receipted RBAC/node/cluster hashes, all cross-checked against the
  independent manager envelope and interpreted `nvidia-smi` capture;
- the hash of the independent Buildx plus published-base lineage receipt,
  `status: passed`, `exit_status: 0`, and storage readback identity.

Imports, BDDL parsing, startup, synthetic fixtures, dataset inventory, or
zero-step training are not capability evidence. The exact live selector must
collect one test and report one pass with zero skips/failures. Rendering, all 130
tasks, lifelong-algorithm comparisons, closed-loop sweeps, and physical robots
remain deferred.

## Current status

Phase A is packaging and refusal design only: image qualification is
`not_qualified`, runtime license/size inventory review is complete, image unbuilt,
anonymous pull unproven, B200 qualification unrun, and catalog release absent.
No workflow in this change is submitted. Agent-run collection remains disabled unless both
operator-provided collection variables are present and pass the immutable
destination contract.
