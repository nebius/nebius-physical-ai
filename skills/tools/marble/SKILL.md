---
name: marble
description: Generate or import World Labs Marble worlds into NPA and run native CUDA capture or spatial-scan workflows with factual interactive HTML reports.
---

# Marble

For detailed quadruped collection, use
`workflows/testing/marble-warehouse-quadruped.yaml` and
`docs/workbench/guides/marble-warehouse-quadruped.md`. Reuse an existing generated
warehouse: no new World API request is needed. The runtime verifies pinned
official Unitree Go1 meshes and a Google MuJoCo Playground ONNX policy. PyBullet
torque dynamics and ONNX inference execute on CPU; gsplat RGB, Warp depth, and
Cycles CUDA articulated rendering use one RTX PRO 6000. OpenImageDenoise runs
on CPU; CPU path tracing is rejected. This is pretrained-policy collection, not new RL training or a
physical robot connection. Preserve upstream licenses, actual link poses,
per-foot contact forces, raw depth, and per-frame renderer evidence in the
standalone HTML and collection bundle. Verify the completed run and browser
playback before claiming live acceptance.

For GPU learning beyond rendering, use `workflows/testing/marble-navigation-rl.yaml`
and `docs/workbench/guides/marble-navigation-rl.md`. The
`workbench.marble.navigation_prepare` adapter preserves a generated world's
collision triangles, measures supported cases with disjoint goal locations,
and seals inputs for the existing native ANYmal/Isaac/RSL-RL trainer. Training
and evaluation use standard Workbench/SkyPilot stages and immutable S3 handoffs.
Require an operator-qualified immutable Isaac image; do not default to the
quarantined public release. This adapter's CPU preparation is verified against
the real warehouse, but native GPU PPO and held-out evaluation remain unverified.
The earlier sensor collection cannot establish navigation learning acceptance.
`WLT_API_KEY` is needed for upstream generation, not for reusing its world bundle.

For embodied warehouse collection, use
`workflows/testing/marble-warehouse-rover.yaml` and
`docs/workbench/guides/marble-warehouse-rover.md`. It requires real World API
generation and aligned mesh/splat transforms. `workbench.marble.rover_collect`
drives a four-wheel rover through PyBullet CPU rigid-body dynamics, then renders
RGB with gsplat CUDA and raycasts depth with Warp CUDA on one Nebius GPU. It
saves actions, joints, contacts, poses, raw depth, and a standalone HTML replay.
The observer uses a CPU-rendered rover at recorded poses and CUDA mesh depth
occlusion. Do not describe this as physical Spot collection or RL training.
Run native runtime submission from a Linux operator and preserve its isolated
SkyPilot identity for resume and cleanup.

For a straightforward user walkthrough, use
`docs/workbench/guides/marble-to-nebius.md`: World Labs generates the scene,
the CPU acquisition stage downloads files to S3, and a Nebius GPU worker loads
those files for rendering or scanning. The report replays completed outputs.

For manufacturing, use `workflows/testing/marble-manufacturing-pallet-detection.yaml`.
It requires `WLT_API_KEY` and a real pallet dataset manifest, fails before
generation on missing auth or invalid/overlapping data, and never substitutes
the sample world. Native submit automatically resolves and forwards this token
from the environment or private NPA credentials. See
`docs/workbench/marble-manufacturing.md` for the fully headless command.
The experiment renders factory backgrounds on CUDA, composites training-only
RGBA pallet cutouts, and invokes the existing Faster R-CNN trainer/evaluator
on CUDA. Both arms receive matched optimizer-update counts and use the same
held-out real evaluation partition. Report negative AP changes honestly.
This is a 2D background-augmentation experiment, not a physics simulation or
production accuracy claim. New manufacturing execution is unverified until a
funded API credential and the operator's real dataset are supplied.

Use the native specs `workflows/testing/marble-world-capture.yaml` and
`workflows/testing/marble-spatial-scan.yaml`. Their stages are acquisition on
CPU, gsplat rendering or Warp raycasting on CUDA, and report publication on CPU.
Submit through `npa workbench workflow submit --stage-src --runtime` with the
operator's explicit project, exact context, and S3 destination.

Marble generation is hosted by World Labs; do not claim its weights run on
Nebius. `WLT_API_KEY` plus API credit is required for `world_source=generate`.
Pass the key as a secret environment variable, never in YAML or browser code.
The web subscription does not fund API requests.

For an explicitly disclosed existing-world demo, use
`--var world_source=sample-hobbit`. This downloads a pinned, MIT-licensed real
Marble example. Reports must retain `generated_this_run: false`, source
revision, file hashes, and `license.txt`. It proves the GPU consumer, not a new
generation request. Never switch to this mode while claiming prompt generation.

The consumers reuse the EnvGen CUDA/compiler image and its attested SkyPilot bootstrap without invoking Genesis.
Capture requires upstream Niantic SPZ decoding and gsplat; scan uses NVIDIA
Warp. Both fail without CUDA and record real device identity and CUDA events.
Reports reject incomplete frame sets and hash mismatches. Do not replace absent
GPU output with illustrative imagery or invented metrics.

SPZ scale and coordinate conversion are explicit in `world.json`. The sample
uses upstream scene units; do not call those calibrated meters. The sweep is
not a robot policy rollout or a proven navigable route. Collision meshes remain
approximate geometry requiring calibration before simulation use.

See `docs/workbench/marble.md` for CLI/SDK examples, artifact contracts, and
provider links. Serve the complete downloaded report prefix over HTTP; Spark
and Three.js are loaded from pinned CDN URLs. Check frame playback, world
exploration, and scan point-cloud controls in a real browser before handoff.
