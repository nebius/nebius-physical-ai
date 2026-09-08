# RoboCasa service image

This image runs RoboCasa kitchen simulation through the NPA CLI, SDK and
authenticated HTTP service. It installs the complete NPA package, pinned
RoboCasa/robosuite source and a hash-locked CUDA 13 runtime. The source-bundled
assets and their licenses differ from the large kitchen archives fetched on
first use; see [the packaging record](REDISTRIBUTION.md) and
[asset attributions](ASSET-NOTICES.md).

Build committed inputs with `build.sh --registry <authorized-registry>`.
The helper snapshots the exact Git commit, stages its workflow catalog and
creates only `dev-<full-sha>`. It refuses dirty build inputs, a different tag,
and direct `--push`; the trusted publication workflow owns registry pushes
after all image gates pass.

The service listens on port 8791 as the non-root `ubuntu` user. Its default
`ROBOCASA_AUTH_MODE=token` requires a run-scoped `ROBOCASA_TOKEN` for protected
operations. The `/health` endpoint remains available for health probes.
The image grants no passwordless sudo and is a service deployment, while
workflow toolRefs invoke it from the standard task image. Deploy an exact
development digest through `npa workbench robocasa deploy --image <image@digest>`.

Run `kitchen_trajectory_export` against the deployed service to produce actual
robot state/action arrays and MP4 recordings. A meaningful publication workload
must run simulation steps, decode the resulting videos, check finite array
values and matching step counts, and verify uploaded artifacts from the exact
image digest. Random rollouts do not establish successful manipulation or
trained-policy quality. Separate ACT training/evaluation and other GPU families
remain unqualified until their own real workloads pass.

The primary environment intentionally supplies the existing LeRobot ACT import
subset instead of LeRobot's full training dependency graph. Its exact dependency
overrides and limitations are recorded in `REDISTRIBUTION.md`; the image does
not claim a compatible whole-distribution `pip check` result.

## Verified public development image

`ghcr.io/nebius/nebius-physical-ai/npa-robocasa:dev-8493d5af4c20eb8fec4bb4949dd8e29b5ed8a096`
resolves to
`sha256:538c531f26e282af9463f149a6d93ab9a8cd92b24004b38cac381c0289ffe257`.
The [trusted build](https://github.com/nebius/nebius-physical-ai/actions/runs/34185701752)
passed its complete-layer, licensing, security and bootstrap gates. On
2026-09-08, that exact image ran a 256-step `PickPlaceCounterToCabinet`
random-action trajectory through the authenticated NPA SDK/service on one
RTX PRO 6000. It exported finite state/action arrays, two 256-frame camera
arrays and a fully decoded 256-frame, 20 fps, 12.8-second MP4. All nine generated
objects passed storage readback. Final reward was zero; this is simulation and
export evidence, not successful manipulation or ACT qualification.

The measured runtime was Torch 2.13.0+cu130, MuJoCo 3.3.1, RoboCasa 1.0.0,
robosuite 1.5.2 and Gymnasium 0.29.1, with compute capability 12.0. It used the
image's installed NPA service files without a source overlay. The same digest
first failed on a managed-driver node that exposed CUDA but lacked NVIDIA
EGL/GLX libraries. Select a qualified NVIDIA graphics runtime as described in
the [driver strategy](../../../../docs/workbench/mk8s-gpu-driver-strategy.md);
an RTX product name or `torch.cuda.is_available()` alone is insufficient.
The successful run used an existing Operator-enabled cluster with an isolated
qualified worker. It does not justify changing shared driver management or
deploying GPU Operator on NVSwitch systems.
