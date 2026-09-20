---
name: curobo
description: Use when running, validating or reviewing NVIDIA cuRobo V2 Franka pose planning and complete MotionBenchMaker/MPiNets benchmarks on CUDA, with factual joint/FK Rerun artifacts.
---

# cuRobo V2 motion planning

The image candidate remains `0.8.0-cuda13-b300-unbuilt` and publication-quarantined
until built-image checks and real GPU validation pass. Build from committed inputs;
`build.sh` checks scoped source cleanliness and archives the exact commit for Docker.
The Dockerfile pins the full Ubuntu package closure to an exact snapshot and
rejects a build-arg override; a snapshot change requires a rebuilt-image scan
and fresh independent native-byte policy review. The tag family does not
establish B300 validation.

For image changes, inspect actual dependency wheels and base layers. The cuDNN
development base carries headers; deleting inherited files in a later layer
does not remove those bytes. This image uses a non-cuDNN CUDA base and filters
the locked cuDNN wheel inside its installation RUN, retaining only shared
runtime libraries and notices. The bundled older and current cuDNN supplements
differ on headers; preserve both evidence and use the common runtime boundary.
Keep the filter's version/inventory checks, then inspect every built layer before
publication. Passing its source tests does not establish clean image bytes.

Run the GPU workflow on Nebius through the ordinary workflow submit path:

```bash
npa workbench workflow validate-spec workflows/testing/curobo-benchmark.yaml
npa workbench workflow submit workflows/testing/curobo-benchmark.yaml --var bucket=<your-bucket>
```

This is the complete benchmark: both pinned datasets, in kinematic and 3 kg
payload dynamics modes. `curobo_mode=kinematic|dynamics` selects one complete
configuration. Do not silently replace either dataset with the upstream demo
or add problem/time/job limits. The golden evaluation separately qualifies one
real pose; it does not prove full-benchmark completion.

## Exact implementation and limits

- Source is cuRobo **V2** at
  `8e734f3ced1df898990bcd92de40abce475907db`, using `MotionPlanner` and
  `MotionPlannerCfg`. V1 `MotionGen` examples are incompatible.
- Raw benchmark datasets are robometrics
  `81e3d1d605de84100d8ab880b43096aba221a48b`. V2 source and Franka assets are
  Apache-2.0; dataset/source MIT and MotionBenchMaker BSD notices are retained.
  No weights, gated access or model-acceptance switch is required.
- The benchmark calls upstream's configuration loader, including its relaxed
  joint limits (0.2 radians), obstacle-to-OBB conversion and optimizer settings.
  Negative `collision_buffer_ik` inputs remain recorded as invalid.
- Every input contributes to the full denominator. Eligible success is reported
  separately. Failed solves remain failed; inverse-dynamics errors fail the job
  instead of becoming zero energy. FK path lengths are computed from actual
  tool positions, not upstream's placeholder end-effector metrics.
- Every solved benchmark row retains the exact optimized joint trajectory,
  torque samples, payload mass and torque limits used for Pinocchio dynamics.
  Validation is deliberately separate from producer validation: a digest-pinned
  GPU replays FK position/orientation from retained joints, while Pinocchio
  replays inverse dynamics CPU-side in the same image. The audit recomputes
  energy, torque limits, path measures, timeline duration, jerk, exact
  identities and per-dataset/mode rates from durable bytes. Treat this replay
  as a downstream dynamics consumer, not robot safety.
- A matching total count is insufficient evidence. Validate exact problem
  identities and invalid indices against `benchmark_inventory.py`, independently
  derived from the pinned YAML with file hashes. The runner checks those bytes,
  and report validation requires the known metrics for every status plus sample
  timeline consistency. Do not accept a self-consistent but invented journal.
- Energy is a Pinocchio inverse-dynamics proxy on the optimized joint trajectory.
  Planner success is upstream feasibility, not independent collision
  certification or authorization to move physical hardware.

## Artifacts and diagnostics

The workflow writes its recipe, `results/problems.jsonl`, `results/result.json`,
`validation.json`, `reports/planning.rrd` and `reports/rrd-manifest.json` under the
same run prefix. S3 publication reads back and hashes every object. The mandatory
RRD contains actual joint traces, tool paths from FK, goal markers,
timing/pose/dynamics metrics and every problem status. Its manifest binds the
run, journal, result and RRD hashes plus logged sample/status/goal counts and a
streamed full-decode receipt. The receipt checks decoded row counts for every
status, goal, FK position, FK quaternion and joint position/velocity/
acceleration/jerk entity; a producer-authored coverage label is not evidence.
The validator also regenerates a journal-derived reference RRD, drops only
nondeterministic `log_tick`/`log_time`, and requires unordered semantic equality
of every remaining value and factual timeline. It contains no invented robot
meshes.

The golden command writes a non-overwriting run directory beneath the required
absolute `NPA_SMOKE_OUTPUT_DIR`; direct invocation has no shared `/tmp` fallback.
The serverless runner supplies `/workspace/npa-golden` in the image-owned
workspace. Retain its input, journal, result, independent validation, RRD, full
`rerun rrd print -vv` output, control record and artifact manifest. Acceptance
requires the feasible pose to succeed and the separately declared goal-blocked
pose to fail; malformed manifest rejection is additional failure evidence.
After private build and byte review, freeze the candidate digest in the
post-build plan amendment. For serverless golden acceptance pass the selected
digest through `--tag sha256:<hex>` and the separately reviewed value through
`--expected-image-digest sha256:<hex>`. Missing, mutable or
different identities are rejected before credential/provider access. The job
receives both identities and checks equality again. The command requires
`NPA_OUTPUT_PATH`, uploads every declared file, reads each object back and emits
a separate upload receipt. Workload failures upload partial artifacts and a
redacted failure receipt; partial upload failures are retained locally and the
receipt is published when S3 remains reachable.

On a GPU or upload failure the runtime retains a mode-0700 working directory
and the already flushed problem journal. Inspect that evidence before any
retry. Do not repeat a successful GPU run to repair telemetry. CUDA/Warp caches
are node-local ephemeral state unless explicitly mounted by the workflow.
After both result artifacts pass S3 readback verification, the runtime removes
only that call's working directory. A cleanup failure emits a fixed warning and
preserves the successful result; it must not trigger another GPU run.

Decode artifacts with `rerun rrd verify` and `rerun rrd print -vv`; compare
problem and trajectory counts against the journal. A file extension or viewer
opening does not establish correctness. Follow `emit-reviewable-rrd` for live
artifact discovery and readback handoff.

## Operator inputs and access surfaces

`npa workbench curobo plan --input-path <s3-manifest> --output-path <s3-prefix>
--run-id <run-id>` accepts a strict `npa.curobo.plan.v1` manifest: `robot` is
`franka.yml`, and `problems` carries unique simple ids, seven-joint `start`,
`goal_pose.position_xyz`, normalized `goal_pose.quaternion_wxyz`, and optional
named `cuboids` (`dims` plus xyz/wxyz `pose`). Arbitrary robot YAML paths or
executable configs are rejected. The SDK module is `npa.sdk.workbench.curobo`.

The optional service exposes the same operations. Configure `CUROBO_TOKEN`
and `CUROBO_ALLOWED_S3_ROOTS`; it refuses missing auth, cross-root S3 writes,
and concurrent GPU requests. It owns no deployment resources or worker launcher.

## GPU and publication gates

CUDA 13 and driver 580 or later are required by the pinned upstream runtime.
B200 and RTX PRO 6000 need separate real validation because SM100 and SM120 are
different CUDA majors. The headless solver/FK recording needs no RT cores.
The image remains publication-quarantined until exact-byte scans and physical
GPU validation pass; a checked-in Dockerfile is not a published capability.
Read `health-preflight`, `gpu-selection`, `secure-image-build` and
`solution-licensing` before building/provisioning/submitting.

## Verify source changes

```bash
npa/.venv/bin/python -m pytest npa/tests/workbench/test_curobo.py npa/tests/workbench/test_curobo_audit.py npa/tests/workbench/test_curobo_report_contract.py npa/tests/cli/test_curobo_cli.py npa/tests/workflows/test_curobo_workflow.py -q
```

Then run applicable `pre-pr-validation`, container, catalog, skill and live-submit
gates. Never report mocked unit tests as GPU or benchmark evidence.
