# cuRobo V2 motion planning

The image candidate remains `0.8.0-cuda13-b300-unbuilt` and publication-quarantined
until built-image checks and real GPU validation pass. Build from committed inputs;
`build.sh` checks scoped source cleanliness and archives the exact commit for Docker.
The tag family does not establish B300 validation.

cuRobo adds GPU motion planning to the workbench: operator-defined Franka
start/goal problems and full MotionBenchMaker/MPiNets benchmarks produce actual
joint trajectories, pose/timing metrics and Rerun recordings.

The four-stage `curobo-benchmark.yaml` workflow prepares a complete recipe,
runs NVIDIA's V2 planner, validates the recorded results, and builds factual
joint/FK recordings. Use the normal workflow submission command and your own
S3 destination. The default evaluates both kinematic and 3 kg dynamics
configurations. B200 and RTX PRO 6000 require separate GPU qualification.

```bash
npa workbench workflow submit npa/workflows/workbench/npa-workflows/curobo-benchmark.yaml --var bucket=<your-bucket>
```

All input problems remain in the success denominator, including invalid queries
that upstream excludes. The eligible success rate is also reported. Results
include measured plan/solve times, pose errors, joint and FK tool path lengths,
motion duration, jerk, inverse-dynamics energy proxy and torque violations.
No upstream published performance number is presented as a Nebius measurement.
Planner feasibility is not independent collision certification.

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
