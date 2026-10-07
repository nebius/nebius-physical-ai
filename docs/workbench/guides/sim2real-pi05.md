# pi0.5 released-surface Sim2Real variant

`workflows/testing/sim2real-pi05.yaml` is a separate declarative adaptation of
the compositional Sim2Real workflow. It does not change
`workflows/main/sim2real.yaml`, replace its PPO policy, or compare its airborne
goal task directly with released placement.

## What the graph proves

The workflow first runs a differential-IK Franka expert in Isaac physics. The
expert must establish bilateral finger/object contact, lift and transport the
object, open the fingers, retreat, and leave the object supported and stable on
the table for three consecutive control steps. It never writes object pose or
velocity. Failed attempts and their captured frames are retained but excluded
from training.

Every admitted row binds a causal exterior image, a real camera mounted below
`panda_hand`, seven measured joints, measured gripper width in the DROID
convention (`0=open`, `1=closed`), the controller's computed absolute joint
target, and simulator timestamps for image capture, command application, and
resulting state. The exporter forms 15-step targets only within episode
boundaries, computes normalization from the train split, and writes both the
OpenPI dataset contract and native LeRobot v3 data. Train, validation, and gold
hold out object morphology and scene identity.

The base and adapted policies use the pinned OpenPI source, Polaris config, and
checkpoint documented in [the OpenPI pi0.5 guide](../openpi-pi05-polaris.md).
They are evaluated with the same gold object, scenario seeds, instruction,
15 Hz absolute-joint controller, and eight-action receding-horizon prefix. A
continuous policy gripper value is explicitly thresholded at `>0.5`, matching
the upstream DROID execution example, then checked against both physical finger
controller targets. The final report permits an improvement claim only when the
two gold protocol digests and checkpoint train/reload hashes agree.

## Required operator inputs

The checked-in spec is intentionally non-runnable until an operator supplies:

- a writable bucket and fresh prefix;
- exact immutable controller, data, Isaac, OpenPI, and viewer images carrying
  the selected NPA source revision;
- the read-only Isaac cache PVC and scoped service account/RBAC;
- healthy RTX PRO 6000 capacity for Isaac and B200 capacity for OpenPI; and
- an explicit task-scoped `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` receipt before
  any OpenPI image build, weight fetch, inference, or training.

The Gemma decision is not inferred from previous runs or general authorization.
The Isaac expert can be qualified independently under the existing Isaac
product routing.

Validate and render without provisioning:

```bash
npa workbench workflow validate-spec workflows/testing/sim2real-pi05.yaml
npa workbench workflow plan-spec workflows/testing/sim2real-pi05.yaml \
  --run-id '<run-id>' --json --check-render \
  --var bucket='<owned-bucket>' --var source_sha='<full-source-sha>' \
  --var controller_image='<immutable-image>' --var data_image='<immutable-image>' \
  --var isaac_image='<immutable-image>' --var viewer_image='<immutable-image>' \
  --var isaac_cache_pvc='<read-only-cache-pvc>'
```

Run health and exact access checks before submission. Use the standard runtime
submit path; the spec itself owns service start, Isaac clients, cleanup, durable
S3 handoffs, and resume behavior. Do not use local `run-spec --execute` as a
substitute for a SkyPilot/Kubernetes acceptance run.

## Outputs and claim boundary

The graph declares physics collection reports, dense OpenPI and LeRobot data,
the trained checkpoint manifest, exact reload evaluation, base/adapted gold
reports, MP4, Rerun recording, and a final factual report. MP4 and Rerun are
decoded after writing. Successful orchestration is reported separately from
task success.

Physical robot deployment remains an explicit operator seam. DROID-shaped
Franka observations are useful for integration, but neither simulation success
nor Polaris loading establishes safe zero-shot transfer to an unqualified robot.
