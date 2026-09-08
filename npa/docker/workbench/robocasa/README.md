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
