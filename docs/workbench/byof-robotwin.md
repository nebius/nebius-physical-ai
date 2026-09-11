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
| Source | `RoboTwin-Platform/RoboTwin@96c1feab536306b50c26af200044fcdf126e8904` and `NVlabs/curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (v0.7.8) are identities only. Neither source nor git metadata is baked or fetched. |
| Baked runtime | The candidate may eventually contain a digest-pinned Ubuntu 22.04 bootstrap and an exact snapshot/version-locked OS closure. Phase A leaves these locks incomplete and refuses every build. CUDA, cuDNN, PyTorch CUDA, SAPIEN, MPLib, Warp, and Python application packages are absent. |
| Weights | Empty. This data-collection gate uses no model or checkpoint. |
| Data/assets | `TianxingChen/RoboTwin2.0@785feb15aa4a4f532395ad2b1d2be5f28cb561ad` is recorded only as an identity. No archive or extracted asset is fetched or baked. |
| Runtime cache | Empty and disabled while the runtime lock is incomplete. A future cache inherits all source/runtime restrictions and must use owner-only staging, verification, and receipt-last atomic rename. |
| Outputs | No Phase A outputs exist. A future run may write only the destination derived from the manager-authorized root and run ID; source staging is a separate control-plane input, never a second output destination. |

CuRobo v0.7.8's noncommercial research/evaluation field-of-use restriction is
binding on runtime use and service claims even though its source redistribution
terms include notice obligations. CUDA and cuDNN acceptance and delivery,
CuRobo purpose/service use, and aggregate RoboTwin asset/output rights each
require a genuine manager/human decision. Runtime fetch, a credential, a
private registry, successful scanning, or writable storage is not permission.

## Fail-closed submit and runtime

The exact RoboTwin normal-submit contract reads one owner-owned 0600 context
file with a byte bound and `O_NOFOLLOW` before config discovery, image work,
storage, networking, scheduling, or GPU activity. Its closed schema binds the
workflow hash, source/CuRobo/asset revisions, runtime-lock hash, one immutable
bootstrap image digest, one STRICT RTX reservation, four independent decisions,
and one output root/run ID.

Only validated bytes cross the existing secret-value transport. The CPU worker
materializes temporary context/config files below a 0700 directory as exclusive
0600 files, scrubs context variables from child environments, and cleans them
on every exit. The public YAML and plan retain `tool://robotwin`,
`example-bucket`, and the context variable name—not any value. The outer task is
CPU-only and the inner profile contains the sole accelerator request:
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`.

The image bootstrap independently validates the transported context and
run/image/output bindings. Phase A then exits 78 with
`ROBOTWIN_RUNTIME_REFUSED:runtime-lock-incomplete` before it can create source,
asset, cache, or output paths or access the network. Its CPU golden eval runs
`robotwin-runtime assert-refusal`; this is packaging/refusal evidence only.

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
sequence is private-stage build and exact-byte scans, exact-digest RTX gate,
trusted public-development build, anonymous pull proof, and a repeat of the
same capability on byte-identical public bytes. Only then may an accepted-image
manifest, GPU digest inventory, or public catalog row be added.

The full 50-task sweep, policy training/evaluation, durable shared cache,
additional object families, and physical-robot deployment remain deferred.
