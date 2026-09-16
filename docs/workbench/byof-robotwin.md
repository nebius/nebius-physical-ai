# RoboTwin 2.0 BYOF bootstrap candidate

RoboTwin is represented by a public-eligible, zero-vendor-payload bootstrap
candidate and a separately gated live workflow. The candidate version is
`2.0-curobo-v0.7.8-rtfetch-unbuilt`; it is deliberately unbuilt and excluded
from publication. Its neutral base/apt inputs and asset/output classifications
are complete, but its native byte policy, built-byte scans, runtime delivery,
storage/context, and live evidence are not.

The eventual hard gate remains narrow and unchanged: the official
`beat_block_hammer` task with `demo_clean` must find and replay a successful
seed through real SAPIEN/Vulkan on exactly one STRICT-bound RTX PRO 6000
Blackwell, then produce native HDF5 plus decoded MP4/frame evidence. Imports,
registration, renderer startup, CPU refusal, or a planned trajectory do not
pass that gate. B200 is never a valid target.

The workflow is [`byof-robotwin.yaml`](../../workflows/testing/byof-robotwin.yaml),
with deferred prerequisites in its
[`readiness record`](../../workflows/testing/byof-robotwin.readiness.json).

## Six independent boundaries

| Boundary | Phase A treatment |
| --- | --- |
| Source | `RoboTwin-Platform/RoboTwin@96c1feab536306b50c26af200044fcdf126e8904` and `NVlabs/curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (v0.7.8) are identities only. Neither source nor git metadata is baked or currently fetched; a future authorized runtime must fetch the exact revisions directly from their official providers and verify payload bytes before provisioning. |
| Baked runtime | The recipe pins official Ubuntu 22.04 linux/amd64 manifest `sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986` and Ubuntu snapshot `20260912T000000Z`. Its signed `main` closure is 75 exact binary packages and 57 source packages; all 75 installed copyright files are hash-bound. The public Python application lock is complete-empty. CUDA, cuDNN, PyTorch CUDA, SAPIEN, MPLib, Warp, and every Python application package remain absent. The trusted build still refuses before Docker because no reviewed native-content policy or built-byte evidence exists. |
| Weights | Empty. This data-collection gate uses no model or checkpoint. |
| Data/assets | [`TianxingChen/RoboTwin2.0@785feb15aa4a4f532395ad2b1d2be5f28cb561ad`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/785feb15aa4a4f532395ad2b1d2be5f28cb561ad) is public and ungated, and its repository card declares MIT for the two exact locked members: `embodiments.zip` (219,859,313 bytes, SHA-256 `6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6`) and `objects.zip` (3,737,778,549 bytes, SHA-256 `6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27`). No archive or extracted asset is fetched or baked. A future authorized runtime must anonymously probe and fetch the exact revision-bound bytes from the official provider and verify both locked sizes and hashes before provisioning. |
| Runtime cache | Empty and disabled while runtime delivery is unapproved and has no complete artifact lock. The planned default is single-customer, single-workload node-local ephemeral storage keyed by provider, artifact, immutable revision/digest, and format. Population uses owner-only staging, verification, and receipt-last atomic rename. Durable reuse remains disabled until its rights and tenant isolation are approved. |
| Outputs | No outputs exist. No inspected authoritative term imposes a generated-output restriction on the declared native HDF5 action/state data, decoded MP4, rendered frames, smoke JSON, or summary JSON; CuRobo execution remains noncommercial research/evaluation. A future run may write only the destination derived from the manager-authorized root and run ID; source staging is a separate control-plane input, never a second output destination. |

The operator has recorded the exact statement `noncommercial` once for this
bounded manager run; its separate intended activity is containerization plus
technical workload validation and evaluation. That record expires with this
run and is neither global nor permanent. It is compatible with CuRobo v0.7.8's
noncommercial research/evaluation field-of-use limit, but it does not authorize
a hosted service, broaden derivative/output rights, or silently upgrade CuRobo.
Before any CUDA 12.8.1, cuDNN 9.8.0, or CuRobo v0.7.8 fetch, install, or cache
mutation, an authenticated customer control plane must issue and consume once
the run-scoped assertion described below. An unsigned file, filesystem owner,
environment value, manager context, or self-declared provenance does not prove
customer origin. NPA and the infrastructure manager do not accept or sign
those terms for the customer. The exact public asset revision needs no
token or local acceptance flag. If a later artifact is token-gated, only the
customer's vendor-side entitlement and exact payload probe may gate its runtime
delivery.
Runtime fetch, a credential, a private registry, successful scanning, or
writable storage is not permission.

## Runtime-fetch delivery contract

The selected shape is whole-source/SDK runtime fetch. The public bootstrap is
credential-free. Public upstream artifacts use an exact-revision payload-byte
probe and need no token. If an exact selected artifact is gated, the customer
supplies and controls the provider credential and the client must complete an
exact payload probe bound to provider, customer credential fingerprint,
artifact, immutable revision or digest, and terms revision before provisioning.
Token presence alone is not evidence and NPA adds no generic acceptance flag.
Any required token crosses only the existing runtime secret-value channel; it
must never enter argv, YAML, plans, logs, Git, image layers, cache metadata,
outputs, or PR text. A provider/artifact/revision/terms change requires a new
probe. For terms without an upstream token gate, credential presence is not
entitlement evidence. Phase A accepts no upstream credential and performs no
probe or fetch.

The runtime lock separates the complete public bootstrap from the disabled
workload delivery. The exact asset members and declared outputs are classified,
but the runtime artifact lock and exact payload probes remain absent, so the
guarded path refuses before network or provisioning. The eventual implementation
must download into a
unique owner-only temporary path, verify the exact expected file set, sizes,
hashes, source identity, and notices, and publish an atomic ready marker only
after verification. A restart may reuse only the same verified identity within
the same customer's entitlement; otherwise it fails closed. Fetched bytes and
populated caches never become image or output payloads.

## Fail-closed submit and runtime

The exact RoboTwin normal-submit contract reads the owner-owned resource
context with byte bounds and `O_NOFOLLOW` before config discovery, image work,
storage, networking, scheduling, or GPU activity. The manager-issued
resource context binds the workflow hash, source/CuRobo/asset revisions,
runtime-lock hash, immutable bootstrap digest, one STRICT RTX reservation,
resource coordinates, opaque customer scope, and one output root/run ID. It
contains no legal-acceptance booleans. The repository deliberately implements
no customer identity, key, trust root, signature scheme, or local successful
acceptance path. Normal local submit returns a structured
`needs_customer_acceptance` notice. A future authenticated control-plane
implementation must pass a typed assertion that binds verified issuer,
customer scope, run ID, runtime-lock SHA-256
`dda9bfebe81250247d25259d655589f8f3b95af7d8629d31b49c59a6af3150ee`,
issuance, expiry, exact intended
activity, exact terms, assertion identity, and replay-resistant nonce. Missing,
declined, stale, replayed, unauthenticated, malformed, or wrongly bound
assertions fail before config reads or any external side effect.

The only public RoboTwin context secret name is
`NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT`. The retired
`NPA_BYOF_ROBOTWIN_CUSTOMER_ENTITLEMENT` local-file channel is rejected even
for a same-UID owner-only file. To make an informed choice, the customer
reviews the exact
[CUDA 12.8.1 EULA](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html),
[cuDNN 9.8.0 SLA](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/eula.html),
and [CuRobo v0.7.8 license](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE).
Declining or leaving the authenticated customer flow incomplete performs no
runtime work. A future customer control plane may resume only by authenticating
the customer and atomically consuming the exact run-scoped assertion; the
current repository has only a typed integration boundary and hermetic mocks.
No assertion or receipt belongs in argv, plans, logs, Git, or PR text. This
authorization does not satisfy the separate technical artifact-lock,
payload-probe, native-content, built-image, storage/context, or live gates.

Only validated bytes cross the existing secret-value transport. The CPU worker
materializes temporary context/authenticated-authorization-receipt/config files
below a 0700 directory as exclusive 0600 files, scrubs context variables from
child environments, and cleans them on every exit. The public YAML and plan
retain `tool://robotwin`,
`example-bucket`, and the context variable name—not any value. The outer task is
CPU-only and the inner profile contains the sole accelerator request:
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`.

A future live submit would have to receive an already verified, content-addressed
`NPA_SRC_S3_URI` from the operator environment. Its final path component must
equal the current local source fingerprint, and its bucket must differ from the
manager-authorized workload-output bucket. RoboTwin rejects saved source
coordinates and `--stage-src`: it neither uploads nor persists a source URI.
The current code also refuses this input because the existing source-stage
metadata does not authenticate every remote path, mode, and digest. A separately
reviewed manifest verification design is required before any source download or
editable install can occur. If that design is later approved, the URI value may
cross only SkyPilot's secret-value transport; submitted YAML and setup logs keep
only the variable name and a generic sync message.
The confidential path also bypasses the generic first-run, submission, launch,
and provisioning journals, so the manager-selected project, context, and
destination remain in process memory and are absent from CLI results and local
submission state. Non-RoboTwin workflows retain the normal durable-state path.

Direct RoboTwin BYOF CLI/script invocation is inert even when a caller supplies
an internal-looking transport value. The CPU outer-to-worker bridge is also
disabled until runtime-lock completion, source-byte verification, and an
independently attestable outer identity are designed and reviewed. Non-RoboTwin
BYOF behavior is unchanged.

The image bootstrap independently validates the transported context and
run/image/output bindings. It then exits 78 with
`ROBOTWIN_RUNTIME_REFUSED:runtime-delivery-technical-gates-incomplete` before it
can create source, asset, cache, or output paths or access the network. Its CPU golden eval runs
`robotwin-runtime assert-refusal`; this is packaging/refusal evidence only.

The RoboTwin byte scanner is likewise unavailable before the first reviewed
private candidate-byte population. Path and regex
denials remain defense in depth, but they cannot prove absence after arbitrary
renaming or re-encoding. Even a completed-looking policy file is rejected until
a separately reviewed implementation binds the shared fresh-native exact-content
scanner, detector identities, and reviewed candidate-byte population. No image
absence or publication claim exists yet.

## Local validation

Use the repository Python 3.12 environment:

```bash
npa/.venv/bin/python --version
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/byof-robotwin.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/byof-robotwin.yaml --run-id robotwin-plan --json
npa/.venv/bin/python -m pytest -q \
  npa/tests/docker/test_robotwin_runtime_bootstrap.py \
  npa/tests/docker/test_robotwin_public_image_contract.py \
  npa/tests/docker/test_robotwin_image_payload_scan.py \
  npa/tests/workflows/test_byof_robotwin.py
```

Do not build while the native-content policy remains unresolved; do not publish,
pull, submit, or attempt live qualification while the readiness and runtime-use
gates remain incomplete. The resolved bootstrap inputs do not authorize any
RoboTwin, CuRobo, CUDA, cuDNN, asset, cache, or output byte. After separate
authorization, the sequence would be private-stage build and exact-byte scans,
exact-digest RTX gate, trusted public-development build, anonymous pull proof,
and a repeat of the same capability on byte-identical public bytes. Only then
may an accepted-image manifest, GPU digest inventory, or public catalog row be
added.

The full 50-task sweep, policy training/evaluation, durable shared cache,
additional object families, and physical-robot deployment remain deferred.
