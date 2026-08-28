# Canonical Sim2Real RobotSpec input

The canonical 14-stage workflow accepts one optional `config.robot_spec_uri`.
Leave it empty for the unchanged stock Franka path. Set it to an exact `s3://`
object containing `npa.sim2real.robot_spec.v1` to run a custom articulated robot.

## Prepare the input

Upload the complete URDF package, not only the `.urdf`. Preserve the package
directory so `package://` mesh references resolve. For the runnable Panda example:

```bash
git clone https://github.com/moveit/moveit_resources.git
git -C moveit_resources checkout c55b102711fc0aebe80c6952d2ce97c38110abba

# Upload moveit_resources/panda_description as the directory
# moveit_resources_panda_description below your operator-owned S3 prefix.
# Then replace YOUR_BUCKET in the example and upload the JSON as robot-spec.json.
```

The complete example is
[`robot-spec-panda-urdf.json`](../../../npa/workflows/workbench/npa-workflows/examples/robot-spec-panda-urdf.json).
It pins the public MoveIt resources commit and records the package's bundled
Apache-2.0 license (and the legacy BSD declaration in `package.xml`). Both source
license records are preserved in the private consumed contract.

The ordered `joint_names` list is arm joints followed by gripper joints. The
`gripper_joint_names` list must exactly equal that trailing segment. Stage 2 also
requires valid `base_link`, `ee_link`, two finger links, home pose, per-joint
stiffness/damping/effort bounds, and open/close targets. The canonical task is
currently a two-finger parallel-jaw Lift task; unsupported tasks or grippers fail
before GPU work begins.

## Validate, plan, and submit

Use the standard runtime path and the same RobotSpec URI in each command:

```bash
SPEC=npa/workflows/workbench/npa-workflows/sim2real.yaml
ROBOT_SPEC_URI=s3://YOUR_BUCKET/robots/panda/robot-spec.json
RUN_ID=robot-spec-proof

npa workbench workflow validate-spec "$SPEC"
npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" \
  --var robot_spec_uri="$ROBOT_SPEC_URI" \
  --assume-decision loop_back
npa workbench workflow submit "$SPEC" --runtime \
  --run-id "$RUN_ID" \
  --var robot_spec_uri="$ROBOT_SPEC_URI" \
  --var bucket="$NPA_BUCKET" \
  --var controller_image="$CONTROLLER_IMAGE" \
  --var transfer_image="$TRANSFER_IMAGE" \
  --var envgen_image="$ENVGEN_IMAGE" \
  --var isaac_image="$ISAAC_IMAGE" \
  --var viewer_image="$VIEWER_IMAGE" \
  --var isaac_cache_pvc="$ISAAC_CACHE_PVC" \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env HF_TOKEN
```

Run `npa workbench health preflight` and `npa workbench health access sim2real`
before submit. Image values must be registry-qualified immutable digests.

## Integrity and observability

Stage 2 downloads and validates the RobotSpec, URDF graph, named links/joints, and
every referenced mesh. It republishes the source tree and normalized contract at
SHA-256-addressed URIs. The first Isaac rollout converts URDF to USD with Isaac
Lab and publishes the resolved USD plus a digest manifest. PPO, validation, and
gold evaluation fetch those exact bytes; each checks the embodiment digest and
the derived action/observation dimensions before accepting a checkpoint.

Inspect these private run artifacts:

- `stage_02_assets/consumed_robot_spec.json`: requested input, normalized source,
  content hashes, frames, joints, controls, task config, dimensions, and target USD.
- `stage_02_assets/robot/resolved/<embodiment-digest>/manifest.json`: converter and
  exact USD SHA-256.
- `inner_loop/outer-XX/evidence.json`: training and selected-checkpoint embodiment.
- `eval/gold-heldout/outer-XX/report.json`: checkpoint SHA and eval parity.
- `reports/sim2real-report.json`, `reports/sim2real.rrd`, and `reports/sim2real.mcap`:
  final human-readable and viewer artifacts.

Missing source objects, dependencies, links, joints, conversion output, or parity
mismatches fail closed. A non-empty `robot_spec_uri` never falls back to Franka.
