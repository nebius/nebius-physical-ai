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
authorization, and independent build-lineage evidence. There is no supported LIBERO release; development digests require independent
qualification and the public release table remains unchanged.

## Customer-operated managed run

The bounded customer path uses the standard NPA managed SkyPilot submission,
status, cancellation and cleanup helpers on one B200. It accepts the same v2
customer signature directly through an owner-private handoff. The customer does
not need to deploy caller or storage signing services. The payload receives no
Kubernetes token or cloud/storage credentials; its controller retrieves only the
sealed, size-limited proof inventory after training with egress denied.

From a reviewed source checkout, prepare the exact image/run/terms/profile packet:

```bash
npa/.venv/bin/python npa/scripts/run_libero_customer.py prepare \
  --image-manifest /private/libero/image-manifest.json \
  --run-id <exact-run-id> --output /private/libero/customer-packet
```

The actual customer reviews `request.json`, including every official terms URL
and content digest, then runs this command in their own interactive terminal:

```bash
npa/.venv/bin/python npa/scripts/run_libero_customer.py authorize \
  --packet /private/libero/customer-packet
```

The prompt requires the customer's identity and `AUTHORIZE <exact-run-id>`.
Declining creates no authorization. An existing Ed25519 key can be selected with
`--signing-key`; otherwise the helper creates the customer's local key only after
that explicit acknowledgement. The customer retains the private signing key and
returns only `customer-authorization.json` and `customer-public-key.b64` through
the private handoff. NPA never runs this acknowledgement on the customer's behalf.

The resource owner supplies a private target JSON with exactly `project`,
`context`, `namespace`, `namespace_uid`, `allowed_node`, `kubeconfig`, `config_path`,
`isolated_config_dir`, `deny_policy_uid`, and `fetch_policy_uid`. The dedicated
namespace bears `npa-libero-run=<exact-run-id>`, has no pre-existing Pods, and uses
the tokenless `npa-byof-libero-payload` account. Its two owner-provisioned policies
are `libero-deny-egress` (all Pods, Egress, no allowed destinations) and
`libero-fetch-egress` (all Pods, Egress, initially all destinations). The controller
checks their UIDs, removes only the exact fetch allowance after verified runtime
materialization, proves the observed egress change, then releases training.

```bash
npa/.venv/bin/python npa/scripts/run_libero_customer.py submit \
  --packet /private/libero/customer-packet \
  --target /private/libero/target.json --output /private/libero/result
```

This profile requires the exact repository `libero-customer-seccomp.json`
installed as `npa-libero-customer-v1.json` under the owned node's kubelet seccomp
root, and `libero-customer-apparmor` loaded as `npa-libero-customer-v1`. Both files
are beside the customer YAML profile. Install them only on the run-owned node and
remove them during that node's cleanup. The seccomp file derives from the resolved
Docker 29.1.3 OCI default with the four permitted capabilities; it contains no
Docker-specific conditional rules. The AppArmor policy derives from
[Moby's default template](https://github.com/moby/profiles/blob/245180c51918481c0525424b3ee025d2b435d46c/apparmor/template.go)
under Apache-2.0. Both retain default denials and permit only the existing builder's
exact user/mount/network/PID namespace operation and scoped sandbox mounts.
Arbitrary namespace flags and mounts outside that sandbox remain denied.

The root filesystem remains read-only. Writable emptyDirs hold workspace, home,
SSH host keys, runtime state and temporary files; no PVC is requested. SETUID and
SETGID support the fixed sudo UID transition, NET_BIND_SERVICE supports SSH port
22, and SYS_CHROOT supports SSH privilege separation. No SYS_ADMIN, CAP_KILL,
privileged container or host namespace is granted. Training sets `no_new_privs`
before executing third-party code. Descendant cleanup uses the existing routine
under the same execution UID through one exact sudo command.

Neutral Docker checks cover these bootstrap and isolation transitions only.
Qualification still requires the real official-data BC-RNN train, checkpoint
reload and complete held-out evaluation on the observed B200 Pod. A successful
image build or neutral test is not GPU evidence.

## Hosted integration

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
registration provisioned outside the NPA control plane. Transport of that
existing registration cannot establish a new signer or alternate trust source. A missing or mutable
registration refuses before cache or network effects. Private signing material,
customer identity,
credentials, acceptance records, terms payloads, and runtime payloads are never
baked into the neutral image. Image qualification carries no customer signer
fingerprint; signer trust is supplied owner-private and checked only at runtime.

Published runtime manifests carry two separate Sigstore OCI artifacts: GitHub
SLSA v1 provenance and an SPDX 2.3 SBOM. Image qualification v2 records each
artifact's manifest, config, bundle and subject descriptors, plus the digest of
its independently verified signature result. Review the exact GitHub workflow,
OIDC issuer, hosted runner, source revision and invocation for both artifacts;
unsigned JSON claims do not establish this evidence. Qualification v1 remains
supported for previously reviewed combined BuildKit attestation records.

The qualification's `development_sha`, build-input bundle and
`publication_enforcement_bundle_sha256` describe the image producer's source.
The separate v2 `operator_enforcement_bundle_sha256` describes the reviewed
current launcher and validators. Verify both source closures independently
before installing the qualification; later operator fixes must not be recorded
as the source that produced an older image. The launcher also requires current
image inputs to remain byte-identical to the producer's build-input bundle.
No qualified record, review validity window or signer identity is inferred from
a successful public push.

The host reads the caller control-plane verification key from
`NPA_LIBERO_AUTHENTICATED_CALLER_PUBLIC_KEY_FILE`. This owner-private key file is
separate from the customer key embedded in the signed authorization and its
independent runtime registration. Storage signature validation uses that
already verified customer key, plus the independently qualified storage signer;
it does not reuse the caller control-plane key as the customer key. Runtime
mounting of the provided caller root and customer registration remains required
before the image can fetch any runtime payload. Supply the independent customer
registration as `NPA_LIBERO_CUSTOMER_SIGNER_REGISTRATION_FILE`, an owner-private
Base64 key file named `<customer-identity-sha256>.b64`. The launcher compares it
with the verified customer signer; it never derives or creates registration
from the authorization.

The run namespace must already contain immutable
`npa-byof-libero-caller-verification` and
`npa-byof-libero-customer-registration` ConfigMaps. Their sole data keys are,
respectively, `authenticated-caller-public-key.b64` and the customer's
`<customer-identity-sha256>.b64`; values must match the independently supplied
key files exactly. The profile mounts them as root-owned, read-only 0444 files.
Render the customer registration destination before the customer signs the
complete profile. The launcher verifies these existing objects after signed
authorization, without creating a caller root or customer registration.

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
| Baked runtime | The proposed public bytes are the exact linux/amd64 `python:3.10-slim-bookworm` manifest `sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`, snapshot-pinned Debian bootstrap packages, NPA-owned bootstrap files and immutable runtime manifests. No customer key or private signing key enters the build, image, workflow, or repository. `debian-packages.lock` records every binary and corresponding source. Docker Official Images' immutable in-toto provenance independently binds the base to rootfs material `sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867`. The image contains no PyTorch, CUDA, cuDNN, NCCL, NVIDIA wheel, MuJoCo, robomimic, or robosuite byte. |
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

The trusted public-development workflow builds the neutral image from the exact
reviewed SHA. Its first local scan records a canonical complete-image inventory
covering ordered uncompressed layer bytes and flattened-rootfs records, verifies
the independent rootfs export and pinned base provenance, and runs the common
SBOM, secret, license, and vulnerability gates. After publication the exact
digest must reproduce the local image/config inventory and pass anonymous pull.
Customer authorization and live workload qualification are not prerequisites
for publishing these neutral bytes. Supported release promotion remains
quarantined until the real exact-digest B200 workload passes.

Storage verification uses a root-owned, mode-0444 control-plane mount at
`/opt/npa/libero/output-storage-authorization-public-key.b64`. The profile mounts the `output-storage-authorization-public-key.b64` entry
from the `npa-byof-libero-storage-verification` ConfigMap read-only using
`subPath`; the control plane must create it with `immutable: true` before submission.
Preflight verifies that its only data entry is the qualified storage key. Missing or
writable key files refuse runtime execution. The neutral image bakes neither this key
nor any customer key or acceptance record. The common publication workflow
cleans only an exact owned development version and refuses a digest carrying
additional tags. Public deletion does not revoke prior downloads.

Both build paths derive `SOURCE_DATE_EPOCH` from that exact source commit so the
identity comparison cannot depend on the wall-clock build time. The package
transaction removes APT/dpkg/account logs and normalizes the non-root account's
shadow day to the same epoch in its creation layer.

The trusted development build must:

1. build the exact reviewed source SHA with Buildx SBOM and maximum provenance,
   verify its attested OCI archive, and import its linux/amd64 runtime image;
2. fetch and verify the pinned official base provenance and SBOM;
3. scan all retained layers and the independently exported rootfs, recording
   the local complete-image inventory and config identity;
4. prove non-root/config/bootstrap identity and pass the common license,
   corresponding-source, vulnerability, credential, and restricted-payload gates;
5. publish only those inspected bytes under the full-SHA development tag,
   resolve the immutable digest, and repeat payload scanning against pulled
   bytes with the recorded local identities; and
6. verify anonymous digest equality and fetch every image layer anonymously.

These image-publication steps neither require nor create customer runtime
acceptance. Runtime qualification remains a separate gate.

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
injects one `kube-root-ca.crt` ConfigMap. Pre-submission inventory permits exactly
that system ConfigMap and the immutable storage-verification ConfigMap. It
requires namespace-bound object UIDs, only the `ca.crt` data key and no workload
owner reference for the root CA, and the exact qualified key for storage
verification. Canonical metadata/data hashes join the pre-submission inventory
receipt. Missing or additional ConfigMaps and invalid data refuse launch. These
sampled records do not themselves prove run-long isolation. The canonical
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
