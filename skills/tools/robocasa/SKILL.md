---
name: robocasa
description: Use to run RoboCasa kitchen-task simulation and held-out policy evaluation as a first-class NPA workbench tool through the npa-robocasa service.
---

# RoboCasa (kitchen-task simulation)

RoboCasa is an Apache-2.0 kitchen-task simulation framework built on robosuite
and MuJoCo. This tool promotes the accepted RoboCasa BYOF candidate into a
first-class workbench tool: a dedicated `npa-robocasa` container with a FastAPI
service, CLI, SDK, and workflow toolRefs that exercise the real upstream
capabilities.

The upstream repo is `robocasa/robocasa` at the pinned `v1.0` tag, with
robosuite at a pinned commit and the exact MuJoCo 3.3.1 / Gymnasium 0.29.1 /
CUDA 12.4 closure from the live-accepted BYOF evidence. Kitchen assets
(textures, fixtures, objects) are NOT baked and download at run time under the
operator's own network access.

## Capabilities

| Capability | What it proves |
| --- | --- |
| `kitchen_task_registration` | Gymnasium `robocasa/PickPlaceCounterToCabinet` is registered |
| `kitchen_asset_availability` | The kitchen assets root exists and is populated |
| `kitchen_egl_env_reset` | A headless `MUJOCO_GL=egl` env creates and resets |
| `kitchen_random_rollout` | A real random rollout runs and writes a video artifact |
| `kitchen_trajectory_export` | Causal observation/action rows and terminal review frames are exported |
| `kitchen_policy_eval` | Matched-seed ACT and random-baseline episodes retain native outcomes and videos |

## Two execution modes

Direct (the default) runs the work from your CLI invocation. Service mode calls a
deployed Kubernetes endpoint:

```bash
npa workbench robocasa run --capability kitchen_random_rollout \
  --output-path s3://<bucket>/robocasa/<id>/ \
  --iterations 1 --num-envs 1
npa workbench robocasa run --capability kitchen_random_rollout \
  --output-path s3://<bucket>/robocasa/<id>/ --service --endpoint <url> \
  --expected-image-source-sha <40-hex-source-sha> \
  --expected-image-manifest-digest sha256:<manifest-digest>
```

Deploy the service when you want a persistent endpoint several runs share:

```bash
npa workbench robocasa deploy \
  --project <alias> --cluster-name <name> \
  --image '<registry>/npa-robocasa@sha256:<manifest-digest>' \
  --expected-image-source-sha <40-hex-source-sha> \
  --output-path s3://<bucket>/robocasa/ \
  --gpu-type l40s \
  --dry-run
npa workbench robocasa deploy --project <alias> --destroy
```

`--gpu-type` is `h100`, `l40s`, `rtx6000`, or `rtxpro6000`. Auth defaults to
`token` (the token comes from the variable named by `--token-env`, default
`ROBOCASA_TOKEN`); `--insecure-no-auth` exists but should not be used. Always
`--dry-run` first and read the manifest. While the image is quarantined, deploy
requires an explicit immutable `@sha256` image plus its exact baked source SHA.
The default remains Kubernetes' existing `default` namespace, preserving
upgrade/destroy continuity with earlier releases without requiring permission
to create cluster-scoped resources. A custom namespace must already exist
unless the operator explicitly passes `--create-namespace`; destroy
deliberately leaves it in place. When selecting a custom namespace, also
override `robocasa_endpoint` in the workflow. For a private image, create the
named pull secret in the selected namespace and pass `--image-pull-secret`.
The pod and container both enforce `runAsNonRoot`.

The `0.1.1` image uses a CUDA 12.4 base with the pinned PyTorch 2.13.0+cu129
runtime; this is the first available CUDA 12 wheel set that clears the declared
Torch dependency vulnerabilities. LeRobot 0.5.1 is installed without dependency
resolution: its older Torch, setuptools, Gymnasium, OpenCV, and Diffusers bounds
are asserted unchanged, fixed Diffusers 0.38 remains for LeRobot's eager policy
package import, and the selected ACT path must pass real construction, queued
inference, and exact checkpoint save/load before deployment. Use L40S for
pixel-bearing EGL runs.
RTX PRO 6000, B200, and B300 are
unverified for this exact image until its wheel architecture set and real EGL
path are measured; do not infer support or a blocker from the base-image tag or
generic rendering capability alone.

## Run

```bash
npa workbench robocasa run \
  --capability kitchen_random_rollout \
  --env-id robocasa/PickPlaceCounterToCabinet \
  --output-path s3://<bucket>/robocasa/runs/<id>/ \
  --service --endpoint <url> \
  --expected-image-source-sha <40-hex-source-sha> \
  --expected-image-manifest-digest sha256:<manifest-digest> \
  --iterations 1 --num-envs 1 \
  --wait --poll-seconds 30 --timeout-seconds 3600 \
  --output json
```

`--capability` and `--output-path` are required. The legacy `--output-uri`
spelling remains as an alias. `--wait` polls `/status` until
the run completes **and fails if it does not** — without it, the command returns
as soon as the run is accepted. Service mode additionally requires both expected
runtime-identity flags. The service compares them with the deployment-provided
manifest digest and with the source SHA baked into the running image before it
creates a run record or schedules GPU work.

## Status, system-info, list

```bash
npa workbench robocasa status --run-id <id> --service --endpoint <url>
npa workbench robocasa system-info --service --endpoint <url>
npa workbench robocasa list --service --endpoint <url>
```

`system-info` reports the RoboCasa, robosuite, MuJoCo, and Gymnasium versions,
the exact LeRobot/Torch/TorchVision policy stack, CUDA availability, registered
env count, exact NPA image source revision, and deployment manifest digest.
The official image sets `NPA_IMAGE_SOURCE_SHA` and
`ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=1`; do not override them. A direct local run
without an image-bound revision reports `source_identity: local_unbound`.

## In workflows

There are six per-capability toolRefs:
`workbench.robocasa.task_registration`, `.asset_availability`,
`.egl_env_reset`, `.random_rollout`, `.trajectory_export`, and `.policy_eval`.
The first four appear together in `robocasa-smoke.yaml`; the latter two drive
the data-policy workflow's real rollout export and held-out evaluation.
Every RoboCasa toolRef passes the expected source SHA and manifest digest. The
checked-in workflows use all-zero non-runnable placeholders; override both with
the exact values used by `deploy`.
Trajectory rows use the observation-before-action convention required by
behavior cloning. The checked-in workflow deliberately labels its source as a
seeded random-action baseline rather than expert demonstrations.

These toolRefs are HTTP clients and run on CPU resources; assigning them a GPU
would compete with the separately deployed service for its exclusive device.
The smoke path needs one L40S for the service plus a CPU worker. The data-policy
path needs two L40S devices/nodes plus a CPU worker: one L40S remains held by the
service while the second runs ACT training before held-out evaluation returns to
the service.

Policy evaluation runs one unambiguous exact ACT checkpoint and a random-action
arm on the same held-out task IDs and reset seeds. It fails when paired initial
workspace frames or robot-state hashes do not match, when native success
signals disagree, when actions or states are non-finite, or when either arm does
not produce its MP4. Finite policy outputs outside the simulator bounds are
clipped at the environment boundary; each episode records the raw and applied
action hashes, clipped-step count, and maximum bound violation. The split
record verifies the checkpoint's content-bound
`training_dataset_provenance.json` against the exact ordered training task IDs,
then proves the held-out set is disjoint. Use
`success_rate_delta` and the paired win/loss/tie counts for comparison; do not
turn a zero or negative delta into a policy-improvement claim.

Every run uploads `result.json` and `provenance.json`. Rollout provenance names
the RoboCasa → MuJoCo execution path, exact image source revision, and hashes
each generated MP4, with machine-readable `rrd: false` and `mcap: false` fields;
this tool does not emit RRD or MCAP recordings.

The service owns one GPU, so accepted runs execute through one process-wide GPU
gate. Each run has a bounded registry slot and executes in a killable process
group under its requested `timeout_seconds`; timeout cleanup removes parent-owned
checkpoint, output, and asset-staging scratch. If the process group or scratch
cannot be cleaned up, the gate stays poisoned and new runs receive HTTP 503
instead of overlapping a possibly live GPU worker. A duplicate request for the
same active deterministic run ID returns the existing queued/running record and
does not enqueue a second copy. SDK service calls must pass
`expected_image_source_sha` and `expected_image_manifest_digest`, just like CLI
service calls.

## Gotchas

- **Without `--wait`, "started" is not "succeeded".** Check `status` before
  reporting a result.
- **Kitchen assets download at run time.** The first run on a fresh image needs
  network access. Fetches are pinned to
  `robocasa/robocasa-assets@1b92c3d02ca4354984fec961357db0bff7b32166`
  and
  `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF@420a04af939c34873e6839a586b70844baf28aab`.
  Population is locked, bounded before ZIP metadata allocation, extracted
  without following links into validated staging trees, and published with
  per-archive content-hash/file-count receipts. Installed bytes are rehashed
  before a receipt can suppress a fetch. A failed or mutated archive remains
  retryable and fails the capability instead of being logged and ignored.
- **`--service` needs both a reachable `--endpoint` and the token variable set.**
  A missing token presents as an auth failure from the endpoint, not as a CLI
  validation error.
- **Gymnasium must stay pinned at 0.29.1.** Newer wrappers drop `__getattr__`
  and break `env.sim` video capture.
- **The data-policy reference trains on random actions.** It proves the
  data/training/evaluation path and exposes a matched baseline; it does not
  establish useful imitation learning without successful demonstrations and a
  measured held-out benefit.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
