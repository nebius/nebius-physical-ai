---
name: openarm
description: Use when packaging, running, validating, or troubleshooting Enactic OpenArm workloads in MuJoCo or NVIDIA Isaac Sim/Isaac Lab through the NPA workbench.
---

# Enactic OpenArm

Use the native `npa workbench openarm` surface and the
`workflows/testing/openarm-simulators.yaml` reference workflow. Do not route
OpenArm to the generic Isaac Lab or RoboCasa tools: the OpenArm tool owns the
upstream source pins, robot assets, task IDs, artifact schemas, and dual-simulator
qualification contract.

## Legal and runtime boundary

- Pin `enactic/openarm_mujoco` release 2.2.0 to commit
  `a8c979629f2591ad035d99d338ce114969e6cddc`.
- Pin the untagged `enactic/openarm_isaac_lab` repository to commit
  `bad82e23716e6941c2de78ccb978f57c78b37734`.
- Both simulator repositories and their included model assets are Apache-2.0.
  Do not copy hardware/CAD payload from the separate OpenArm repository.
- The public image may bake OpenArm and MuJoCo OSS bytes. It must never bake
  Isaac Sim, Isaac Lab, Omniverse Kit, acceptance, credentials, caches, or data.
  Load and follow `skills/atomic/third-party-eula-preflight/SKILL.md` and
  `skills/atomic/solution-licensing/SKILL.md` before building or executing.
- Invoke Isaac only through `${ISAAC_LAB_PYTHON:-/isaac-sim/python.sh}`. Never
  run this shim from a Dockerfile `RUN` instruction.
- Classify the CUDA base and CUDA/cuDNN Python runtime libraries as conditional
  NVIDIA redistributable components, not OSS. Retain their notices, remove
  separately installed SDK headers/static archives, and review their exact
  built-image files and SBOM before publication.

## Supported workloads

MuJoCo uses the real upstream `openarm_demo_xml`, `JointResolver`, and
`mujoco.mj_step`. A successful run uploads `result.json` and
`mujoco_trajectory.npz`; `--render` adds a simulator-rendered MP4.

Isaac rollout imports `openarm.tasks`, creates the requested upstream OpenArm
Gymnasium task, and performs real vectorized steps. Isaac training calls the
pinned upstream RSL-RL script and fails closed unless a `model_*.pt` checkpoint
exists. `Isaac-Reach-OpenArm-v0` is the default qualification task. Treat lift,
drawer, bimanual, imitation, teleoperation, and sim-to-real as unqualified until
each has exact-image live evidence; upstream itself says the latter three are
under development.

The reference workflow must end with `workbench.openarm.qualify`. That stage
downloads the common run root, validates all three terminal results, rejects
unsafe/missing/empty artifacts, hashes the MuJoCo trace/video, Isaac rollout
trace, and every upstream checkpoint, then emits
`npa.openarm.qualification.v1`. A successful simulator process without this
artifact-level gate is not complete workflow evidence.

Use an L40S or RTX PRO 6000 for Isaac execution. H100, H200, B200, and B300 are
not valid render-capable qualification targets.

Direct service deployment must mount an operator-owned PVC for
`/opt/isaac-cache`; never reintroduce an `emptyDir` fallback for the proprietary
runtime. Service deletion retains that claim.

## Ordered gate

1. Run `npa workbench health preflight`, including `--checks nebius` before
   provisioning and S3 before submission.
2. Build only a clean, commit-locked `dev-<full-sha>` image.
3. Run the packaging, license, full-filesystem/layer, secret, and SBOM scans.
4. Push the dev tag, resolve its digest, and prove anonymous pullability.
5. Run the MuJoCo golden eval and the complete reference workflow against that
   exact digest. Inspect the uploaded result, NPZ, and training checkpoint.
6. Record accepted evidence without concrete infrastructure identifiers, then
   promote the already validated digest. Never rebuild a release tag.

## Accepted release

Release `2.2.0-isaac0.1.0-rtfetch` is bound to development revision
`01fbf3a554cb7b15066283fd171c5b81f6207eda` and OCI digest
`sha256:c30da0d55de0b1b0528b1481a318bf43ad9d95c7128ae44b5d434203e7d1543a`.
Those exact bytes passed the ordered gate above and the complete four-stage
workflow on RTX PRO 6000. The accepted scope is the 500-step rendered MuJoCo
rollout, the 64-environment × 100-step upstream reach rollout, and one upstream
RSL-RL training iteration with checkpoint plus independent artifact
qualification. Do not extend that evidence to other tasks, policy convergence,
physical hardware, L40S, or non-RT GPUs.

## Tests

```bash
npa/.venv/bin/python -m pytest npa/tests/workbench/test_openarm.py -q
npa/.venv/bin/python -m pytest npa/tests/docker/test_packaging_contract.py -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_golden_eval_manifest.py -q
```
