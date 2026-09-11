# RoboTwin 2.0 BYOF validation

RoboTwin 2.0 is an NPA BYOF registry candidate for native bimanual simulation
and data collection. Its acceptance gate is intentionally narrow: the official
`beat_block_hammer` task under `demo_clean` must search for one successful seed,
replay it through the real SAPIEN/Vulkan simulation, and produce a non-empty
native RoboTwin HDF5 episode plus a decoded MP4 on exactly one RTX PRO 6000
Blackwell (`sm_120`). Imports, task registration, renderer startup, and planned
trajectories do not pass this gate on their own.

The workflow is [`byof-robotwin.yaml`](../../workflows/testing/byof-robotwin.yaml),
with execution prerequisites tracked separately in its
[`readiness record`](../../workflows/testing/byof-robotwin.readiness.json). It
uses the packaged `byof-solution-smoke-robotwin-rtxpro-gpu` resource profile and
the existing `workbench.byof.repo` toolRef.

## Immutable inputs and licensing

| Input | Immutable identity | Packaging boundary |
| --- | --- | --- |
| RoboTwin source | `RoboTwin-Platform/RoboTwin@96c1feab536306b50c26af200044fcdf126e8904` | MIT source is cloned into the private BYOF image. |
| RoboTwin assets | `TianxingChen/RoboTwin2.0@785feb15aa4a4f532395ad2b1d2be5f28cb561ad` | Only `embodiments.zip` and `objects.zip` are fetched in the GPU pod and checked against recorded sizes and SHA-256 hashes. The official `update_embodiment_config_path.py` then materializes the embodiment paths. No asset bytes are baked. `background_texture.zip` is unnecessary for `demo_clean`. |
| CuRobo | `NVlabs/curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (`v0.7.8`) | Compiled for `sm_120` in the image. NVIDIA's license limits this version to noncommercial research/evaluation, so the image is restricted and private. |
| CUDA base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:61f6c08f…` (linux/amd64 manifest) | Upstream NVIDIA base, selected by immutable digest. |

The build installs an embedded, exact 148-distribution Python 3.10/Linux lock
with SHA-256
`5a2b78949b5be3e30089b930e2dfd2197a505f89a3b66cf26def2572306af5b8`.
Its resolved license closure is MIT/BSD/Apache/ISC/PSF, file-level MPL-2.0,
and NVIDIA's CUDA/cuDNN runtime terms. CuRobo preserves its own NVIDIA license,
`LICENSE_ASSETS` covers its bundled Franka, Kinova, UR, Jaco, IIWA/Allegro and
Techman files, and the two embedded Robotiq notices remain in their source
directories. A build is therefore allowed only after an authorized operator
records acceptance of the current NVIDIA CUDA EULA and cuDNN SLA and confirms
that this is noncommercial research/evaluation in the owner-only runtime
context. The agent does not accept those terms on the operator's behalf.

The Hugging Face dataset card labels the aggregate asset repository MIT, while
the upstream asset documentation also describes custom RoboTwin-OD, Objaverse,
and PartNet-Mobility origins. The smoke uses the custom `020_hammer` and the
ALOHA-AgileX embodiment, but the runtime `objects.zip` archive is an aggregate.
Keep all downloaded bytes runtime-only, preserve upstream notices, and review
the terms of any additional object selected by a future task. This candidate is
not eligible for the public NPA image catalog; do not push it to public GHCR.

## Validate and plan

Create the repository environment once as described in `CONTRIBUTING.md`, then:

```bash
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/byof-robotwin.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/byof-robotwin.yaml --run-id robotwin-plan --json
```

Planning proves the `npa.workflow/v0.0.1` schema and `workbench.byof.repo`
mapping. It does not prove registry access, reserved capacity, Vulkan, task
success, or artifact integrity.

## Live run

Before spending GPU time, verify credentials and gated access boundaries:

```bash
npa/.venv/bin/npa workbench health preflight --checks nebius,s3 --json
```

Use only a manager-authorized, run-scoped private registry and unique writable
S3 prefix. The dedicated live test requires an owner-only runtime-context JSON
path in `NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT`; it rejects missing ownership
provenance, a different project/context, anything but one RTX PRO 6000, and any
reservation policy other than `STRICT` before it builds or submits. The record
also supplies the exact private registry, bucket, output root, and run ID, none
of which belong in the repository. Its SHA-256 is passed through SkyPilot's
secret environment channel and retained in the smoke artifact without exposing
the private values. The selected Kubernetes worker group must already be bound
to that reservation; the repository deliberately contains no tenant capacity-
block or node-group identifiers. Submit through the normal NPA/SkyPilot route
and retain the resolved image digest:

```bash
npa/.venv/bin/npa workbench workflow submit workflows/testing/byof-robotwin.yaml \
  --run-id <unique-run-id> \
  --var bucket=<authorized-bucket>
```

The BYOF runner builds and pushes the source image, resolves the pushed tag to
an immutable digest, and materializes that digest as the pod image. The
dedicated resource profile captures `nvidia-smi` and `vulkaninfo`, runs the
solution smoke, uploads every run output with create-only S3 writes, verifies
the summary and primary artifact with `HeadObject`, then returns the upstream
exit status. Cancel the run after terminal evidence; never destroy shared
manager infrastructure.

## Required acceptance artifact

The hard-gate file is exactly `$NPA_SMOKE_OUTPUT_DIR/robotwin-smoke.json`. A
passing record includes:

- `solution`, primary `capability`, and all `capabilities_exercised`;
- exact source, asset, and CuRobo revisions;
- task `beat_block_hammer`, config `demo_clean`, the successful seed, and the
  seed-search attempt count;
- `task_success: true`, positive action and decoded-frame counts, and native
  HDF5 structure (`source_format=RoboTwin`, `source_path=native_collection`);
- HDF5 and MP4 relative paths, sizes, SHA-256 hashes, and video dimensions;
- successful NVIDIA Vulkan discovery plus SAPIEN device/RT-renderer evidence;
- one observed RTX PRO 6000, `sm_120`, compute capability 12.0;
- the pod materialized immutable image reference/digest and `exit_status: 0`.

The video frame count must equal the HDF5 action count plus the final
observation. Missing files, failed decoding, the wrong GPU, a mutable image
reference, incorrect archive bytes, or a failed task replay all make the smoke
exit nonzero.

## Deferred work

This candidate does not claim the full 50-task sweep, policy training or
evaluation, randomized-background coverage, other embodiments or object
licenses, nor physical-robot deployment. Those are separate workflows and
acceptance campaigns, not extensions of this smoke result.
