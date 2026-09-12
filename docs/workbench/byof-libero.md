# LIBERO neutral-bootstrap qualification

LIBERO is a quarantined public-image candidate, not a supported release. Its
planned image is a neutral, non-root bootstrap that contains no LIBERO or GPU
runtime. A future manager-approved run may materialize one exact runtime into an
operator-owned cache, then exercise one official `libero_spatial` task through
eight upstream BC-RNN optimizer steps, strict checkpoint reload, and a genuine
trajectory-disjoint held-out evaluation on one B200. The path is headless and
never invokes rendering.

The planned managed workflow is
[`workflows/testing/byof-libero.yaml`](../../workflows/testing/byof-libero.yaml).
Its `tool://libero` default deliberately cannot run: execution requires an
explicit accepted `npa-libero@sha256:…` candidate, a manager-issued runtime-use
decision, and independent build-lineage evidence. There is no published LIBERO
tag or digest and the public image table remains unchanged.

## Six independent boundaries

| Boundary | Phase A contract |
| --- | --- |
| Source | `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT. Source is absent from the image. An authorized runtime sparse-fetch retains only training/config/task-definition paths, verifies the source tree and license hash, and removes `.git`. `libero/libero/assets` is excluded. |
| Baked runtime | The proposed public bytes are the exact linux/amd64 `python:3.10-slim-bookworm` manifest `sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`, snapshot-pinned Debian bootstrap packages, and NPA-owned files. `debian-packages.lock` records every binary and corresponding source. Docker Official Images' immutable in-toto provenance independently binds the base to rootfs material `sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867`. The image contains no PyTorch, CUDA, cuDNN, NCCL, NVIDIA wheel, MuJoCo, robomimic, or robosuite byte. |
| Weights | None are baked. Exact `google-bert/bert-base-cased@cd5ef92a9fb2f889e972770a36d4ed042daf221e` files are runtime-only and Apache-2.0. |
| Data and task inputs | No demonstration or task/render asset is baked. The selected official demonstration is runtime-only: `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`, 508,779,600 bytes, SHA-256 `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`, upstream-declared CC BY 4.0 with LIBERO attribution. The MIT BDDL and initial-state files are fetched only with the sparse source and verified by SHA-256. |
| Runtime cache | `/workspace/.cache/npa/libero/<runtime-manifest-sha256>` is manifest-addressed, atomically completed, sealed group-readable/non-writable, and separate from output. The bootstrap owner and execution UID are distinct, so fetched code cannot restore cache write bits. A shared lock and stable directory descriptor remain held through smoke/upload execution, with full inventory checks before and after. A cold population resolves all seven hash-bound official governing-terms sources after authorization and before the first cache mutation. The cache is never uploaded. Missing, mismatched, overbroad, expired, or locally invented acceptance decisions refuse before cache or network effects. Runtime fetch changes delivery, not permission. |
| Outputs | Exactly nine named qualification files may exist before receipt creation beneath a stable, descriptor-opened `$NPA_SMOKE_OUTPUT_DIR`; any missing or additional entry refuses. Each regular, single-link file is opened with `O_NOFOLLOW`, copied to stable in-memory bytes while inode/size/mtime identity is checked, hashed, conditionally uploaded with an S3 SHA-256 checksum, then GET/read back byte-for-byte. A tenth receipt records all nine identities and is itself uploaded and read back. Credentials, source, models, packages, input data, and caches are never output artifacts. |

The task mentions Google Scanned Objects and a HOPE distractor, but their meshes
and textures are neither fetched nor needed for stored-observation behavior
cloning. Rendered or closed-loop evaluation therefore remains deferred. The
mirror card's license label does not override the upstream LIBERO publisher's
CC BY 4.0 dataset declaration.

## Quarantine and image proof

`npa/docker/workbench/libero/Dockerfile` describes the candidate but Phase A
does not build or publish it. The non-root `ubuntu` bootstrap user has a finite
root sudo grant for generating missing per-Pod SSH host keys with the exact
`-A` argument and for starting or restarting the preinstalled SSH daemon. Its
only other sudo grant invokes the fixed bootstrap entry point as the separate
unprivileged `npa-libero-exec` user with exactly `execute`. Its environment
excludes storage credentials as well as Python and dynamic-loader injection
variables. After fetched code exits, the image-owned system Python performs the
bounded upload with separately accepted, run-prefix-only temporary credentials.
No host key is baked into an image layer. The complete SkyPilot
0.12.2 synchronous package set is
baked, including `curl`, `wget`, and the `fuse3` provider. When every package and
command verifies, the guard bypasses SkyPilot's redundant package-index
transaction. Missing packages fail closed; the deadline wrapper applies TERM
and a kill-after escalation and emits an unambiguous failure sentinel.

Before any public byte is disclosed, a separately authorized private stage must
first emit a canonical complete-image inventory. It binds every byte in each
ordered uncompressed layer tar, the resulting flattened-rootfs path records,
and the observed OCI config digest. The manager reviews and accepts those exact
identities outside the repository. The trusted public workflow obtains every
accepted identity exclusively from the strict checked-in acceptance record; no
free-form dispatch value can create or widen acceptance. Its pre-push scanner
then requires complete image, config, exact Buildx metadata, and published-base
provenance equality. Finite
path and content signatures remain defense in depth, not the proof that
arbitrary renamed, compiled, or subsequently whiteouted bytes are absent.
Before LIBERO's first package-wide visibility change, the private destination
must contain exactly the accepted untagged candidate digest and no other image
or embedded attestation graph. Other first publications still require a nonexistent
or zero-version destination. The target namespace is globally serialized. A failed first publish
deletes the still-private package only after proving its exact run tag, removing
the image and its referrers before any retry.
Both build paths derive `SOURCE_DATE_EPOCH` from that exact source commit so the
identity comparison cannot depend on the wall-clock build time. The package
transaction removes APT/dpkg/account logs and normalizes the non-root account's
shadow day to the same epoch in its creation layer.

The trusted build must then:

1. reproduce the accepted development SHA from its hash-bound neutral build-input
   bundle with Buildx SBOM and maximum provenance;
2. fetch and hash-verify Docker Official Images' exact published provenance
   metadata, without using Docker history;
3. inspect every OCI config and ordered layer, nested archive, exact runtime
   payload hash, renamed source signature, and an independently exported
   flattened rootfs with `scan_image_libero_payload.py` plus the general
   restricted-payload scanner; require the canonical complete-image inventory
   and config to equal the private-stage identities in the strict checked-in
   acceptance manifest; free-form dispatch values are not an authorization
   channel;
4. prove non-root/config/bootstrap identity and independently bind the candidate
   index, platform manifest, config, Buildx metadata, base manifest, rootfs
   material, upstream source revision, and canonical publication/infrastructure
   bundles. The publication bundle recursively binds the complete Python source
   trees used for provider identity, preflight, managed submission, scheduler
   identity, storage readback, signal teardown, and cleanup, plus the exact live
   evidence collector; the checked-in acceptance manifest is the only permitted
   path delta after the accepted development SHA, so enforcement/runtime drift
   requires a new candidate and renewed manager acceptance;
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
currently incomplete (all sizes and 65 license classifications remain pending),
so both repository acceptance and runtime materialization fail closed.
`runtime-requirements.txt` is itself
hash-bound and must match the ordered artifact manifest exactly; both bootstrap
and remaining installs use `pip --require-hashes --no-deps --no-index`. It
requires an owner-private regular JSON decision whose SHA-256 is fixed by the
checked-in acceptance record, not supplied by the caller. The closed record
binds the official GHCR candidate, OCI/platform/config identities,
complete-image inventory, Buildx/base/publication receipts, exact source/runtime
manifest, reviewed infrastructure, run/namespace, issuer, five boundaries,
nonce, and a maximum 24-hour decision window. It rejects any `ACCEPT_*` proxy.
Only after that decision passes does it resolve and hash all
official terms into ephemeral storage; any changed or unavailable terms source
refuses before cache mutation. Network availability, registry access, a
credential, or a successful download is never treated as permission.

The workflow uses `base_profile: prebuilt`, an empty build command, and the
dedicated B200 profile. `run_byof_repo.py` rejects an empty or mutable candidate,
an absent/mismatched decision, an absent independent build-metadata hash, any
identity drift, and `--skip-run` before registry or workload preparation. It
always passes `--no-direct-launch` for LIBERO. `run_byof_container_verify.py`
submits through the managed scheduler, requires its nonempty scheduler job ID,
and polls that ID—not the human run name. Absence remains a failure.

The decision and output-storage authorization bytes travel only through the
scheduler's redacted secret channel, never ordinary rendered workflow state.
The storage authorization is hash-bound by the checked-in infrastructure bundle,
binds the exact run prefix and policy receipt, and requires short-lived session
credentials. Storage secrets are removed from the fetched-code subprocess; only
the image-owned standard-library uploader receives them. Per-file and aggregate
size budgets apply before reads, PUT is conditional, and GET must return the
service checksum as well as identical bytes. The owner-only worker file is removed
immediately after materialization, and the Base64 secret is unset again at the
start of the run phase.

The payload profile selects `npa-byof-libero-payload`; `default` and
`skypilot-service-account` are forbidden for the payload. The manager-owned
SkyPilot controller remains on its engine-required account. Before a future run,
the exact run-labelled namespace must be empty of Pods and Secrets and contain
only `default` plus the reviewed payload ServiceAccount, core `pods/get`-only
Role, and exact RoleBinding. This makes the sole readable Pod the subsequently
created run Pod. Every namespace/object UID, inventory, and permission is bound
to the checked-in acceptance record. The payload reads its own Pod and bound JWT and records
the actual service account, Pod UID, node, and `containerStatuses.imageID`.

The execution and payload-proof kubeconfig contexts remain distinct but must
flatten to the same cluster identity. A single owner-receipted allowed node and
STRICT B200 reservation are mandatory. All access objects and isolated scheduler
state are ephemeral. After exact job cancellation and cluster absence, cleanup
uses UID preconditions in RoleBinding → Role → ServiceAccount → namespace order,
polls every object to 404, stops the isolated API, and only then removes the
payload kubeconfig and exact-run local SkyPilot state. Any ambiguity is a failed
cleanup and preserves recovery state.

## Acceptance artifact

A later live gate owns the managed submission, independently downloads the
ten-object upload set, verifies every receipt size/hash/checksum against fresh
GET bytes, checks cleanup/API-stop evidence, and accepts the retrieved
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
  and owner-receipted RBAC/node/cluster hashes;
- the hash of the independent Buildx plus published-base lineage receipt,
  `status: passed`, `exit_status: 0`, and storage readback identity.

Imports, BDDL parsing, startup, synthetic fixtures, dataset inventory, or
zero-step training are not capability evidence. The exact live selector must
collect one test and report one pass with zero skips/failures. Rendering, all 130
tasks, lifelong-algorithm comparisons, closed-loop sweeps, and physical robots
remain deferred.

## Current status

Phase A is packaging and refusal design only: repository acceptance is
`not_accepted`, runtime license/size review is incomplete, image unbuilt,
anonymous pull unproven, B200 qualification unrun, and catalog release absent.
No workflow in this change is submitted. Agent-run collection remains disabled unless both
operator-provided collection variables are present and pass the immutable
destination contract.
