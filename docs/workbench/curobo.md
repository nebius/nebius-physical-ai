# cuRobo V2 motion planning

[Workbench docs](README.md)

The image candidate remains `0.8.0-cuda13-blackwell-unbuilt` and publication-quarantined
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

A benchmark manifest may request just `kinematic` or just `dynamics`. Validation
requires both complete datasets for each requested mode (2,600 inputs, including
10 upstream exclusions), and rejects missing datasets or unrequested modes.
Each mode retains its original frozen failure and torque-violation gates; choosing
a single mode does not relax its acceptance threshold.

```bash
npa workbench workflow submit workflows/testing/curobo-benchmark.yaml --var bucket="<your-bucket>"
```

All input problems remain in the success denominator, including invalid queries
that upstream excludes. The eligible success rate is also reported. Results
include measured plan/solve times, pose errors, joint and FK tool path lengths,
motion duration, jerk, inverse-dynamics energy proxy and torque violations.
The exact optimized trajectory and torque samples behind dynamics metrics are
retained. Both journal validators bind every retained joint position and derivative
to that dynamics series, accounting for the pinned cubic B-spline interpolation
and derivative retiming. Equal endpoints or self-consistent energy alone cannot
bind a different interior path. The digest-pinned GPU validation stage independently replays FK from
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

If the planner subprocess exits nonzero, the operation still fails but publishes
read-back-verified diagnostics below `_failures/<run-id>/` in the requested
output prefix. That namespace contains `runtime.log`, `failure.json`, and, when
the runner created a partial journal, `partial-problems.jsonl`; the receipt binds
the log and exact raw journal hash. It records physical lines separately from
complete nonblank JSON-object records, so a blank, scalar, malformed, or
truncated line is never reported as a completed record. It never contains an
accepted `result.json`, and `validate` and `visualize` reject the failure
namespace. If evidence publication fails, the subprocess failure remains
primary and reports the secondary failure type without exposing its potentially
sensitive message.

Set `NPA_CUROBO_WORK_DIR` to an existing private writable volume for the runner,
CPU artifact processing, full decoded text and regenerated comparison RRDs.
Visualization is not lightweight: observed 5,200-input populations each retained
approximately 540 MB of journal, 550 MB of RRD and 925 MB of decoded text, plus a
regenerated journal/recording and normalization copies. Plan at least 8 GiB of
available scratch, separately from image storage, for that population. The
reference CPU state requests 16 GiB RAM; larger operator inputs require measured,
proportionately larger capacity. These are capacity planning values, not input,
time or job limits. Parsed JSON rows and comparison state remain memory-resident;
this is not a constant-memory validator. The `cpu` selector does not provision a
persistent scratch volume or reserve its disk capacity. Measure free space and
resource requests before choosing the node. Journal regeneration and failed-run
log/journal publication stream bytes, including read-back hashing, without
truncating or discarding failed records. Temporary CPU scratch is removed at
operation exit; failed runner working directories and durable failure artifacts
remain available for diagnosis.

The image is a publication candidate until exact-image scans and real hardware
results are accepted. Source/robot assets and benchmark datasets have separate
Apache-2.0/MIT/BSD notices. Pinocchio's distro `libgomp1` runtime and matching
GCC source/license identities are frozen with the Ubuntu snapshot; no model
weights or gated data are required. See the
[packaging record](../../npa/docker/workbench/curobo/REDISTRIBUTION.md) and
[operator skill](../../skills/tools/curobo/SKILL.md) for exact revisions and limits.

The real-GPU golden command preserves its input, journal, result, independent
audit, RRD, full `rerun rrd print -vv` output, controls and SHA-256 manifest
under `NPA_SMOKE_OUTPUT_DIR`. Direct smoke invocation must set this to an
absolute writable path; there is no shared `/tmp` fallback. Serverless golden
jobs set it to `/workspace/npa-golden` in the image-owned workspace. The command
requires one feasible pose to succeed and a separately declared
valid-but-goal-blocked pose to fail; malformed manifest rejection is retained as
a second negative control. RRD manifests bind the run, journal and result hashes
and verify decoded status/goal entities plus actual FK-position, FK-quaternion
and per-joint sample chunk counts for every successful trajectory instead of
trusting producer-authored coverage labels. A separately regenerated recording
from the durable journal is normalized only for
nondeterministic log clock timelines and compared semantically, including all
joint/FK values, factual timelines, goals, statuses and metrics.

After the private build, freeze its reviewed digest separately from the selected
image and pass both values. Admission compares them before credentials or provider
access:

```bash
npa workbench golden-eval run "${NPA_GOLDEN_TOOL:-curobo}" --serverless \
  --registry "<candidate-registry>" \
  --tag "sha256:<candidate-digest>" \
  --expected-image-digest "sha256:<independently-frozen-candidate-digest>"
```

Every declared artifact is uploaded to the run-scoped S3 prefix and read back
before success. A workload failure uploads its input, partial outputs, redacted
failure record and receipt; an upload failure retains and, when S3 remains
reachable, publishes the per-object failure receipt.

The independent digest option is serverless-only; dry-run and local execution
reject it rather than imply an identity check. Generic `golden-eval run-all`
does not supply an independently frozen per-image digest and cannot qualify this
quarantined candidate. Use the explicit single-image command above; a batch
failure is not a reason to remove its digest admission gate.
