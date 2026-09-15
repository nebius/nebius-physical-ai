# Enactic OpenArm

`npa workbench openarm` provides one supported interface for two genuine
OpenArm simulation stacks: the upstream OpenArm v2 bimanual MJCF in MuJoCo and
the upstream OpenArm reinforcement-learning environments in NVIDIA Isaac
Sim/Isaac Lab.

## Source and licensing boundary

The image uses `enactic/openarm_mujoco` 2.2.0 at immutable commit
`a8c979629f2591ad035d99d338ce114969e6cddc` and
`enactic/openarm_isaac_lab` at immutable commit
`bad82e23716e6941c2de78ccb978f57c78b37734`. Both simulator repositories are
Apache-2.0 and include the MJCF/USD assets used by these workloads. NPA does not
copy from the separately licensed OpenArm hardware/CAD repository.

The public image bakes MuJoCo and the two OpenArm source trees. It does not bake
Isaac Sim, Isaac Lab, Omniverse Kit, an EULA decision, credentials, runtime
caches, checkpoints, or user data. The first Isaac command fetches exact pinned
vendor wheels into `/opt/isaac-cache` through the shared NPA runtime bootstrap.
Unset `ACCEPT_EULA` takes the repository's documented non-interactive default;
an empty or recognized negative value refuses before network access. Privacy
and telemetry remain disabled.

The CUDA base and the exact CUDA/cuDNN runtime distributions required by
PyTorch are proprietary redistributable components rather than OSS. Publication
is conditioned on NVIDIA's container, CUDA, and cuDNN distribution terms. The
image removes separately installed vendor SDK headers and static archives,
retains license files, and records their hashes; no patent rights are inferred.

The complete legal decision, six artifact boundaries, delivery identity, cache
policy, and release gates are recorded in
`npa/docker/workbench/openarm/ONBOARDING_CONTRACT.md`. The built image also
contains its exact Python distribution inventory and retains Ubuntu package
copyright records. A generated inventory identifies bytes; it does not replace
review of the cited upstream terms.

## Run MuJoCo

```bash
npa workbench openarm run \
  --simulator mujoco \
  --output-path s3://<bucket>/openarm/<run>/mujoco/ \
  --steps 500 --render --output-format json
```

This loads `openarm_mujoco.v2.openarm_demo_xml()`, resolves both arms with the
upstream `JointResolver`, applies bounded position commands, calls `mj_step`,
and uploads finite joint/command/velocity-energy arrays. `--render` also encodes
the frames produced by MuJoCo itself; it is not stock or copied media.

## Run Isaac Lab

Use an L40S or RTX PRO 6000 because Isaac Sim rendering and PhysX validation
need RT-capable infrastructure. The default reach task uses no external
customer asset:

```bash
npa workbench openarm run \
  --simulator isaac-lab --isaac-mode rollout \
  --task Isaac-Reach-OpenArm-v0 --num-envs 64 --steps 100 \
  --output-path s3://<bucket>/openarm/<run>/isaac-rollout/ \
  --output-format json
```

For real upstream RSL-RL training:

```bash
npa workbench openarm run \
  --simulator isaac-lab --isaac-mode train \
  --task Isaac-Reach-OpenArm-v0 --num-envs 64 --max-iterations 1 \
  --output-path s3://<bucket>/openarm/<run>/isaac-training/ \
  --output-format json
```

Training fails if the upstream script exits non-zero or writes no `model_*.pt`
checkpoint. NPA does not substitute random actions or a synthetic checkpoint.
Other upstream-registered OpenArm tasks may be selected explicitly, but a task
is supported only after exact-image live qualification.

## Service and workflow

Set `OPENARM_TOKEN`, then deploy the authenticated RTX service. Supply a PVC to
reuse the first-run Isaac cache; without one the deployment uses an ephemeral
50 GiB volume:

```bash
npa workbench openarm deploy --project "$NPA_PROJECT" --gpu-type rtxpro6000 \
  --isaac-cache-pvc "$OPENARM_CACHE_PVC" --dry-run --output-format json
npa workbench openarm deploy --project "$NPA_PROJECT" \
  --isaac-cache-pvc "$OPENARM_CACHE_PVC"
```

Service clients add `--service --endpoint <url> --wait`. `status`, `list`, and
`system-info` are read-only. `delete` removes only the named service,
deployment, and secret; it deliberately retains the cache PVC.

The complete reference is `workflows/testing/openarm-simulators.yaml`. Its four
real stages use `workbench.openarm.mujoco_rollout`,
`workbench.openarm.isaac_rollout`, `workbench.openarm.isaac_train`, and
`workbench.openarm.qualify`. The final stage downloads the preceding artifacts,
rejects missing, empty, non-finite, or path-escaping payloads, hashes each real
trace/video/checkpoint, and writes `qualification/qualification.json`.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/workbench/test_openarm.py -q
npa/.venv/bin/python -m npa workbench workflow validate-spec \
  workflows/testing/openarm-simulators.yaml --json
npa/.venv/bin/python -m npa workbench workflow plan-spec \
  workflows/testing/openarm-simulators.yaml --json
```

Before publication, scan the built and pushed exact digest with
`npa/scripts/scan_image_omniverse_payload.py`, review its SBOM/license
inventory, verify anonymous pullability, and run both simulators on the exact
digest.
