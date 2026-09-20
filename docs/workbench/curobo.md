# cuRobo V2 motion planning

[Workbench docs](README.md)

The image candidate remains `0.8.0-cuda13-b300-unbuilt` and publication-quarantined
until built-image checks and real GPU validation pass. Build from committed inputs;
`build.sh` checks scoped source cleanliness and archives the exact commit for Docker.
The image also locks the Ubuntu package closure to an immutable snapshot; changing
that closure requires a new built-byte policy review. The tag family does not
establish B300 validation.

cuRobo adds GPU motion planning to the workbench: operator-defined Franka
start/goal problems and full MotionBenchMaker/MPiNets benchmarks produce actual
joint trajectories, pose/timing metrics and Rerun recordings.

The four-stage `curobo-benchmark.yaml` workflow prepares a complete recipe,
runs NVIDIA's V2 planner, validates the recorded results, and builds factual
joint/FK recordings. Use the normal workflow submission command and your own
S3 destination. The default evaluates both kinematic and 3 kg dynamics
configurations. B200 and RTX PRO 6000 require separate GPU qualification.

```bash
npa workbench workflow submit workflows/testing/curobo-benchmark.yaml --var bucket="<your-bucket>"
```

All input problems remain in the success denominator, including invalid queries
that upstream excludes. The eligible success rate is also reported. Results
include measured plan/solve times, pose errors, joint and FK tool path lengths,
motion duration, jerk, inverse-dynamics energy proxy and torque violations.
The exact optimized trajectory and torque samples behind dynamics metrics are
retained. The digest-pinned GPU validation stage independently replays FK from
joint samples while Pinocchio recomputes per-sample torque, energy and limit
violations CPU-side in the same image. This is a measured downstream consumer,
not hardware-execution permission. No upstream published performance number is
presented as a Nebius measurement. Planner feasibility is not independent
collision certification.

For custom inputs, write a `npa.curobo.plan.v1` JSON manifest to S3:

```json
{
  "schema_version": "npa.curobo.plan.v1",
  "robot": "franka.yml",
  "problems": [{
    "id": "reach",
    "start": [0, -1.3, 0, -2.5, 0, 1, 0],
    "goal_pose": {
      "position_xyz": [0.5, 0, 0.3],
      "quaternion_wxyz": [1, 0, 0, 0]
    },
    "cuboids": {}
  }]
}
```

Invoke `npa workbench curobo plan`, then `validate` and `visualize` using the
same run id. Each command accepts `--input-path` and `--output-path` S3 handoffs.
The SDK exposes the same operations. The optional API requires `CUROBO_TOKEN`
and an explicit `CUROBO_ALLOWED_S3_ROOTS` allowlist.

The image is a publication candidate until exact-image scans and real hardware
results are accepted. Source/robot assets and benchmark datasets have separate
Apache-2.0/MIT/BSD notices; no model weights or gated data are required. See the
[packaging record](../../npa/docker/workbench/curobo/REDISTRIBUTION.md) and
[operator skill](../../skills/tools/curobo/SKILL.md) for exact revisions and limits.

The real-GPU golden command preserves its input, journal, result, independent
audit, RRD, full `rerun rrd print -vv` output, controls and SHA-256 manifest
under `NPA_SMOKE_OUTPUT_DIR`. It requires one feasible pose to succeed and a
separately declared valid-but-goal-blocked pose to fail; malformed manifest
rejection is retained as a second negative control. RRD manifests bind the run,
journal and result hashes and verify decoded status/goal entities plus actual
FK-position, FK-quaternion and per-joint sample chunk counts for every successful
trajectory instead of trusting producer-authored coverage labels.

After the private build, freeze its reviewed digest separately from the selected
image and pass both values. Admission compares them before credentials or provider
access:

```bash
npa workbench golden-eval run curobo --serverless \
  --registry "<candidate-registry>" \
  --tag "sha256:<candidate-digest>" \
  --expected-image-digest "sha256:<independently-frozen-candidate-digest>"
```

Every declared artifact is uploaded to the run-scoped S3 prefix and read back
before success. A workload failure uploads its input, partial outputs, redacted
failure record and receipt; an upload failure retains and, when S3 remains
reachable, publishes the per-object failure receipt.
