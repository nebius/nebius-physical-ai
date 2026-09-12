# RoboTwin 2.0 BYOF bootstrap candidate

RoboTwin is represented by a public-eligible, zero-vendor-payload bootstrap
candidate and a separately gated live workflow. The candidate version is
`2.0-curobo-v0.7.8-rtfetch-unbuilt`; it is deliberately unbuilt and excluded
from publication until its base, apt, runtime, legal, byte-scan, and live
evidence gates are complete.

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
| Baked runtime | The candidate may eventually contain a digest-pinned Ubuntu 22.04 bootstrap and an exact snapshot/version-locked OS closure. Phase A leaves these locks incomplete and refuses every build. CUDA, cuDNN, PyTorch CUDA, SAPIEN, MPLib, Warp, and Python application packages are absent. |
| Weights | Empty. This data-collection gate uses no model or checkpoint. |
| Data/assets | `TianxingChen/RoboTwin2.0@785feb15aa4a4f532395ad2b1d2be5f28cb561ad` is recorded only as an identity. No archive or extracted asset is currently fetched or baked. A future authorized runtime must probe and fetch the two exact revision-bound archives from the official provider and verify their recorded sizes and SHA-256 values. |
| Runtime cache | Empty and disabled while the runtime lock is incomplete. The planned default is single-customer, single-workload node-local ephemeral storage keyed by provider, artifact, immutable revision/digest, and format. Population uses owner-only staging, verification, and receipt-last atomic rename. Durable reuse remains disabled until its rights and tenant isolation are approved. |
| Outputs | No Phase A outputs exist. A future run may write only the destination derived from the manager-authorized root and run ID; source staging is a separate control-plane input, never a second output destination. |

The operator has recorded the exact statement `noncommercial` once for this
bounded manager run; its separate intended activity is containerization plus
technical workload validation and evaluation. That record expires with this
run and is neither global nor permanent. It is compatible with CuRobo v0.7.8's
noncommercial research/evaluation field-of-use limit, but it does not authorize
a hosted service, broaden derivative/output rights, or silently upgrade CuRobo.
CUDA and cuDNN delivery/use, CuRobo's service/output boundary, and aggregate
RoboTwin asset/output treatment remain genuine manager/human decisions.
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
probe. Phase A accepts no upstream credential and performs no probe or fetch.

The runtime lock is the immutable delivery manifest. Its source, asset, runtime,
cache, and output records remain incomplete, so the guarded path refuses before
network or provisioning. The eventual implementation must download into a
unique owner-only temporary path, verify the exact expected file set, sizes,
hashes, source identity, and notices, and publish an atomic ready marker only
after verification. A restart may reuse only the same verified identity within
the same customer's entitlement; otherwise it fails closed. Fetched bytes and
populated caches never become image or output payloads.

## Fail-closed submit and runtime

The exact RoboTwin normal-submit contract reads one owner-owned 0600 context
file with a byte bound and `O_NOFOLLOW` before config discovery, image work,
storage, networking, scheduling, or GPU activity. Its closed schema binds the
workflow hash, source/CuRobo/asset revisions, runtime-lock hash, one immutable
bootstrap image digest, one STRICT RTX reservation, the run-scoped
noncommercial binding plus the remaining independent decisions, and one output
root/run ID. Phase A deliberately rejects even a well-formed context: no
manager-approved receipt format, verifier key, or immutable evidence binding
exists, so its decision fields cannot self-certify authority or access.

Only validated bytes cross the existing secret-value transport. The CPU worker
materializes temporary context/config files below a 0700 directory as exclusive
0600 files, scrubs context variables from child environments, and cleans them
on every exit. The public YAML and plan retain `tool://robotwin`,
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
disabled until genuine manager receipt, source-byte verification, and an
independently attestable outer identity are designed and reviewed. Non-RoboTwin
BYOF behavior is unchanged.

The image bootstrap independently validates the transported context and
run/image/output bindings. Phase A then exits 78 with
`ROBOTWIN_RUNTIME_REFUSED:runtime-lock-incomplete` before it can create source,
asset, cache, or output paths or access the network. Its CPU golden eval runs
`robotwin-runtime assert-refusal`; this is packaging/refusal evidence only.

The RoboTwin byte scanner is likewise unavailable in Phase A. Path and regex
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

Do not build, publish, pull, submit, or attempt live qualification while the
locks/readiness record remain incomplete. After separate authorization, the
sequence would be private-stage build and exact-byte scans, exact-digest RTX gate,
trusted public-development build, anonymous pull proof, and a repeat of the
same capability on byte-identical public bytes. Only then may an accepted-image
manifest, GPU digest inventory, or public catalog row be added.

The full 50-task sweep, policy training/evaluation, durable shared cache,
additional object families, and physical-robot deployment remain deferred.
