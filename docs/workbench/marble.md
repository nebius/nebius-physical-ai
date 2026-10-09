# Marble worlds in Workbench

Start with [From World Labs Marble to a Nebius GPU](guides/marble-to-nebius.md)
for the file handoff, account setup, one-command headless submission, and HTML
download. This page contains the implementation and validation reference.

Marble is a hosted world generator. Workbench calls the official World API on
CPU, persists its SPZ splats and GLB collider in S3, and schedules downstream
CUDA work through the normal `npa.workflow` runtime. Marble weights are not
available for self-hosting in this integration.

The capture and scan references below are joined by the
[manufacturing pallet-detection experiment](marble-manufacturing.md), which
requires hosted generation and compares real-image detector accuracy.

| Workflow | Actual GPU work | Inspectable outputs |
| --- | --- | --- |
| [World Capture](../../workflows/testing/marble-world-capture.yaml) | gsplat Gaussian rasterization | RGB camera dataset, camera matrices, CUDA timings, interactive HTML |
| [Spatial Scan](../../workflows/testing/marble-spatial-scan.yaml) | NVIDIA Warp ray/triangle intersection | raw depth NPZ, sampled hit cloud, camera matrices, CUDA timings, interactive HTML |
| [Warehouse Rover](../../workflows/testing/marble-warehouse-rover.yaml) | gsplat RGB and Warp depth along actual wheel-driven motion | synchronized observations, actions, joints, contacts, poses, and standalone offline HTML |

Both use `acquire → capture/scan → report`, S3 handoffs, and one RTX PRO 6000
GPU. The CPU acquisition/report stages request no GPU. The GPU consumers reuse
the existing EnvGen CUDA/compiler image with its attested SkyPilot bootstrap.
The specs enable source overlay for staged NPA code and install pinned runtime
dependencies into a writable environment that inherits the baked CUDA libraries.
Capture and scan execute gsplat or Warp. The rover combines both consumers
with PyBullet CPU rigid-body dynamics in the same one-GPU worker; see the
[headless collection guide](guides/marble-warehouse-rover.md).
No Marble image or model weights are published.

## Which workloads benefit

The execution boundary is `World API → exported world in S3 → Nebius GPU
consumers → measured results`. The API supplies scene assets. Subsequent camera
rendering, simulation steps, and model optimization operate on those assets;
they do not need a World API request for every frame or robot action. The
[official API](https://docs.worldlabs.ai/api) exposes SPZ splats, a GLB collider,
and per-world scale/ground metadata.

| Use case | Path after World API generation | What establishes usefulness | Workbench status |
| --- | --- | --- | --- |
| More camera rendering | SPZ → gsplat CUDA → RGB views and camera poses | Views cover the target camera distribution and pass image/pose checks | Capture consumer GPU-validated on the explicit sample; current path is a fixed exploratory sweep |
| Depth and geometry inspection | GLB → Warp CUDA raycasts → depth and hit cloud | Geometry, coverage, and sensor conventions match the intended test | Scan consumer GPU-validated on the explicit sample; no planner or robot is evaluated |
| Manufacturing perception | SPZ → CUDA backgrounds → labeled training composites → CUDA detector training/evaluation | Held-out real-image AP improves against the equal-update real-only baseline | Implemented testing workflow; funded API generation and real-data benchmark not yet run |
| Factory mobile-robot RL | Aligned visual/collision scene + robot + goal task → simulation and action-dependent GPU sensors → PPO → unseen-world evaluation | Goal success, collision rate, and path efficiency improve against the same task without Marble variation | Wheel-driven collection is implemented; task rewards, resets, PPO training, and policy evaluation remain extensions |
| Manipulation RL or imitation learning | Marble surroundings + validated robot/fixtures/parts → GPU simulation and rendering → PPO, or recorded demonstrations → policy training | Pick/place success improves on held-out scenes and real trials | Proposed extension; asset integration and episode recording are not implemented |

For RL, a fixed camera sweep is insufficient. An implementation must import
the scene into a simulator and define robot dynamics, observations, actions,
rewards, termination, and reset behavior. These are the explicit contracts in
[Isaac Lab's RL environments](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.html).
The existing Workbench Isaac Lab trainer does not automatically connect Marble
assets to one of those tasks.

Before calling either proposed policy path useful:

1. Establish a common metric frame for splats, collider, robot, cameras, and
   task objects. Verify ground alignment, collision surfaces, valid reset
   states, and reachable goals. Export metadata alone is not a validation of
   factory dimensions or contact behavior.
2. Make each camera observation follow the robot's current state. For visual
   policies, render the robot and dynamic objects with consistent depth and
   occlusion against the scene. A state-only policy cannot benefit from changes
   to background pixels; it needs meaningful geometry or dynamics variation.
3. Use validated physical assets for manipulable objects, fixtures, joints,
   friction, mass, and inertia. Marble supplies scene context; generated splats
   do not supply the contact model of a gripper or manufactured part.
4. For imitation learning, record synchronized observations, executed actions,
   timestamps, and episode boundaries from teleoperation or a working teacher.
   Camera images alone are not action-labeled demonstrations.
5. Split evaluation by world/capture, keep the task and training-update budget
   comparable, run multiple seeds, and report task success and failures on
   unseen scenes. Real trials are needed before claiming real-factory gains.

My recommended RL extension is a camera-based mobile-robot navigation task in
varied factory aisles. It exercises both the generated appearance and validated
layout while leaving precision assembly contacts to measured assets. This is
a proposed experiment, not a measured learning result.

There is an upstream precedent: World Labs documents Marble warehouse scenes
combined with separately supplied robots, conveyors, and boxes in Isaac Sim,
and a Lightwheel pipeline that adds SimReady assets, tasks, and evaluation.
Those examples support the integration approach; they do not validate this
repository's missing RL adapter or establish a training improvement here.
See [robotics demonstrations](https://www.worldlabs.ai/case-studies/1-robotics)
and [Lightwheel integration](https://www.worldlabs.ai/case-studies/2-lightwheel).

## Generate your own world

Create a key and fund API credits at the [World Labs platform](https://platform.worldlabs.ai/).
Marble web credits and World API credits are separate. Supply `WLT_API_KEY` in
the submitting process environment or `tokens.WLT_API_KEY` in the private NPA
credentials file. Never put it in workflow YAML, source, or HTML.
Native submission automatically requires and forwards `WLT_API_KEY` for
generation stages, even when `--secret-env WLT_API_KEY` is omitted. Missing
credentials fail before provisioning; explicit sample imports do not require it.

Use the [complete headless submission command](guides/marble-to-nebius.md#2-submit-the-complete-workflow-headlessly)
after setting the project, context, bucket, and key as shown there.

Use `marble-spatial-scan.yaml` for the second workflow. Defaults are
`world_source=generate`, `world_model=marble-1.1`, and 120 views. Capture defaults
to 960×540; scan defaults to 480×270. Override `world_prompt`, `world_model`,
`frames`, `width`, and `height` with `--var`. These are output dimensions, not
spending limits. Both require source staging for this new integration.

Generation records the provider operation before polling. A saved operation is
resumed only for the same run and request. A conditional S3 write claims the
generation intent before the POST so concurrent submissions cannot both start
generation at the same prefix. An uncertain POST must be reconciled in the
provider account before retrying the same output prefix, preventing an automatic
duplicate purchase.

## Validate without a new generation purchase

Add `--var world_source=sample-hobbit` and omit `--secret-env WLT_API_KEY`.
This explicitly imports the real Marble example from the MIT-licensed
[Spark Physics repository](https://github.com/bmild/spark-physics), pinned at
`308fbf8d0d9a336112c57697a0b0499d16b31504`. The workflow still executes actual GPU
work. The report records `generated_this_run: false`, source revision, byte
hashes, and the upstream license. It does not claim a new API generation.

## Artifact contract

- `world/`: `world.json`, `world.spz`, `collider.glb`, and the sample license.
- `results/`: verified source assets, `frames/`, and `result.json` containing
  actual GPU model/CUDA version, per-frame CUDA-event timings, camera poses,
  and file hashes. Scan additionally writes `depth.npz` and `scan-points.json`.
- `report/`: a portable copy of the verified result bundle plus `index.html`.

Download the complete report prefix and serve it over HTTP. The recorded
camera frames and metadata are local artifacts. World exploration additionally
loads pinned Three.js and Spark JavaScript from their public CDNs. The report
has no API key and makes no World API generation requests.

The Python SDK uses the same typed operations as the CLI:

```python
from npa.sdk.workbench.marble import RunRequest, capture

capture(RunRequest(
    input_path="s3://your-bucket/run/world/",
    output_path="s3://your-bucket/run/results/",
    run_id="your-run-id",
))
```

This is a stateless API client and GPU artifact processor, with no persistent
service to deploy. Workflow scheduling owns the runtime lifecycle.

## Interpretation and validation

SPZ decoding uses the pinned upstream Niantic library. Capture uses gsplat
1.5.3 and SH degree-zero colors; higher-order appearance is not evaluated.
Scan uses Warp 1.17.0 and real collision triangles. No-return depth is NaN,
not fabricated range. Sample distances are upstream scene units, not calibrated
meters. Generated splat scale/ground metadata follows the
[official conversion](https://docs.worldlabs.ai/api/rendering-spz). Generated
colliders receive the same scale, ground offset, and axis conversion as splats.
These are provider-estimated meters; real-site calibration remains necessary.

The camera sweep is exploratory. Neither workflow trains a robot, proves a
collision-free path, or certifies simulation-ready physics. Generated meshes
may contain holes or approximate surfaces.

The live-submit matrix selects the explicit imported example and real GPU
execution. `world_source=generate` requires a separate funded-key live check.
Tests reject missing CUDA evidence, missing frames, wrong identities, path
traversal, and mismatched source/output hashes.

The warehouse rover passed a separate real World API generation and native
Nebius run on one RTX PRO 6000 Blackwell Server Edition. It collected 240
RGB/depth pairs at 960 × 540 along 7.98 meters of simulated travel, with ground
contacts in every recorded state. All 727 result assets passed download hash
verification. The standalone replay passed offline Chromium playback, scrubbing,
trajectory download, and mobile layout checks. This validates synthetic
collection, not a physical robot or a trained policy. Warm CUDA event medians
were 0.85 ms for RGB and 0.36 ms for depth; first-use intervals include compilation
and initialization and do not measure active GPU residency.

## Live validation — 2026-10-03

Both CUDA consumers executed through the native workflow runtime on one
NVIDIA RTX PRO 6000 Blackwell Server Edition, with CUDA 13.0 and Torch
2.9.0+cu130. These runs used `world_source=sample-hobbit`; hosted World API
generation was not executed.

| Consumer | Actual input/work | Output | Median CUDA event time per frame |
| --- | --- | --- | --- |
| gsplat 1.5.3 | 2,500,000 Gaussian primitives | 120 RGB frames, 960×540 | 1.0199 ms |
| NVIDIA Warp 1.17.0 | 15,552,000 ray queries against 50,000 triangles | 120 depth frames, 480×270 | 0.2648 ms |

The scan returned 12,934,026 hits (83.1663%). Its raw depth, sampled hit
positions, cameras, and per-frame timings are retained in the report bundle.
CUDA event medians describe the measured consumer operations, not total
workflow latency; provisioning, dependency setup, encoding, and S3 transfers
are separate. First-launch compilation is retained in the timing trace.

The imported world remains unchanged. HTML presentation can be rebuilt from
the verified result bundle without repeating GPU execution. The opening view
selects recorded frame 60 of the complete 120-frame exploratory sweep.

Both native submissions finished successfully, including all six acquisition,
GPU, and report stages. All 240 JPEG frames decoded at the expected dimensions
and every asset hash verified. The raw depth contained exactly the reported
12,934,026 finite hits; the browser cloud displays 30,344 sampled hit positions.
Chrome checks passed for desktop/mobile layouts, playback, scrubbing, Spark
world exploration, point-cloud controls, and downloads, with no JavaScript errors.
The final worker-pod audit found no remaining Marble workers.

Validation also passed 22 focused Marble tests and 25 shared CLI/SDK/Workbench
contract checks, both workflow validation/planning commands, and scoped lint.
The workflow readiness records retain the funded-key requirement for the
default generation mode.

The validation workstation's standalone `workflow status` lookup selected a
different controller and returned `VERIFICATION_UNAVAILABLE`. The native submit
runtime used the intended isolated controller and recorded all stages as
successful. Completion evidence uses those terminal records, verified S3
artifacts, and the exact-context worker audit; the standalone status lookup
still needs separate controller-context reconciliation.
