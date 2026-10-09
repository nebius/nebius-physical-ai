# Collect warehouse observations with a simulated rover

Run [marble-warehouse-rover.yaml](../../../workflows/testing/marble-warehouse-rover.yaml)
to generate a warehouse with World Labs and collect a wheel-driven inspection
rollout on one Nebius RTX PRO 6000. The generated Gaussian splats provide the
appearance; its triangle mesh provides the floor, collisions, and depth.

| Stage | Execution | Saved output |
| --- | --- | --- |
| Acquire | World Labs hosted API, CPU download | SPZ splats, GLB collider, hashes, provider scale |
| Collect | PyBullet CPU dynamics; gsplat and NVIDIA Warp CUDA on one GPU | RGB, depth, wheel commands, contacts, actual rigid-body poses |
| Report | CPU | One standalone HTML file with recorded observer, RGB, depth, and telemetry |

The embodiment is a small four-wheel skid-steer inspection rover defined in the
code. It is not a Spot model or a physical robot connection. Wheel velocity
motors drive the rigid body against the actual static concave warehouse mesh.
No plane substitutes for missing geometry. A feedback controller follows a
straight, collision-checked aisle route of up to eight meters. Generation can
produce unsuitable geometry; collection fails when no supported route exists,
the rover moves less than one meter, or ground contact is insufficient.

## Run headlessly

Use a Linux operator with the NPA checkout, configured Nebius project, exact
cluster context, owned S3 bucket, and the pinned SkyPilot runtime. The native
runtime verifies process identity through Linux `/proc`; run it on Linux rather
than submitting the runtime directly from macOS.

Create an API key at [World Labs](https://platform.worldlabs.ai/api-keys), fund its
API credits, and expose **`WLT_API_KEY`** in the operator environment or save it
under `tokens.WLT_API_KEY` in the private NPA credentials file. Browser login is
unnecessary after setup. A Marble web subscription does not supply API credits.

```bash
export WLT_API_KEY="$(cat /secure/worldlabs-api-key)"
npa workbench health preflight --project <project-alias> --checks nebius,s3 --json
npa skypilot verify --cluster <exact-context> --output-format json
npa workbench workflow validate-spec workflows/testing/marble-warehouse-rover.yaml
npa workbench workflow submit workflows/testing/marble-warehouse-rover.yaml \
  --project <project-alias> \
  --infra k8s/<exact-context> \
  --run-id warehouse-rover-collection \
  --var bucket=<owned-bucket> \
  --stage-src --runtime --max-wait-seconds 0 \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --isolated-config-dir /var/lib/npa/warehouse-rover-sky
```

Replace the placeholders. Submit resolves the named secrets from the environment
or selected project's NPA credentials. No secret belongs in the YAML or HTML.
Keep the same Linux operator and isolated state directory for status, resume,
and cleanup. Use `--resume-run warehouse-rover-collection` after an interrupted
runtime instead of starting a new run. The acquisition journal retains the
accepted World Labs operation and refuses uncertain duplicate generation.

The default collection contains 240 synchronized observations at 12 Hz and
960 × 540 pixels. Set `--var frames=...`, `width=...`, `height=...`, or
`sensor_hz=...` to change the dataset. The sensor frequency must divide the
240 Hz physics clock. The rover settles for two simulated seconds before the
recorded episode. The reported observation timestamps begin at the collection
start and describe the state reached by the associated wheel command.

The standalone CLI is `npa workbench marble rover-collect --input-path
s3://<owned-bucket>/world/ --output-path s3://<owned-bucket>/collection/
--run-id <run-id>`. The SDK exposes `RoverRequest` and `rover_collect` from
`npa.sdk.workbench.marble`; the workflow toolRef is
`workbench.marble.rover_collect`. All three use the same implementation.

## Inspect the evidence

Download `runs/<run-id>/marble-rover/report/index.html` from the selected bucket
and open it directly. It contains all replay images, needs no internet access,
and does not invoke World Labs or Nebius. The observer composites a CPU-rendered
rover at its actual simulated pose over a CUDA-rendered Gaussian background;
Warp mesh depth determines occlusion. The onboard RGB and depth panels show the
dataset observations.

The `results/` prefix also includes:

- `frames/*.jpg`: onboard RGB from gsplat CUDA.
- `depth.npz`: ray distances, intrinsics, and camera-to-world matrices. No return
  is NaN. Distance is along a unit ray, not optical-axis Z depth.
- `trajectory.json`: timestamps, Bullet XYZ positions and XYZW orientations,
  wheel commands and joint states, contacts, and both camera pose streams.
- `rover.urdf`: the exact body and wheel model used in simulation.
- `result.json`: actual GPU model, CUDA versions, per-frame CUDA events, physics
  summary, source manifest, and artifact SHA-256 hashes.

World coordinates are Y-up; Bullet coordinates are Z-up. Conversion is
`world = [bullet.x, bullet.z, -bullet.y]`. Camera axes are OpenCV. Both downloaded
assets receive the same provider-estimated metric scale, Y/Z axis flip, and
ground offset. These are approximate generated meters, not a surveyed site
calibration. The collector uses SH degree-zero colors for splats.

## Manufacturing and RL scope

This collection is useful for prototyping camera/depth ingestion, visual
odometry, mapping, and action-conditioned model inputs in a warehouse setting.
The generated environment is not a measured digital twin, and it has no
verified defect, pallet, or safety labels. Evaluate perception changes on
held-out real site data before claiming a manufacturing improvement.

This rollout does **not** train an RL policy. To use the environment for RL,
add task rewards, resets, terminated/truncated signals, randomized environments,
and a trainer/evaluation loop. The saved action-state-observation sequence is
an integration foundation, not evidence of learned navigation or transfer to
a real rover. Real site calibration, dynamics identification, and robot control
integration are separate work.
