# Isaac Lab Bring Your Own Fork Cookbook

This cookbook shows how to run a custom Isaac Lab fork, custom RSL-RL fork, or
custom training wrapper on Workbench without changing the checked-in platform
workflow. The worked example in this directory uses a synthetic image layer and
`custom_train.py` to prove the BYOF mechanism.

The W10 validation exercised both override surfaces:

- image override through `resources.image_id`, exposed by
  `npa/scripts/run_isaac_lab_rl.py --image`;
- command override through the SkyPilot YAML `run:` block, passed to the same
  runner with `--yaml`.

The reference workflow remains `npa/src/npa/workflows/skypilot/isaac-lab-rl-train.yaml`;
the reference runner remains `npa/scripts/run_isaac_lab_rl.py`.

## What This Cookbook Covers

Use this guide when you want to:

- keep Workbench's Isaac Lab S3 artifact layout;
- run a non-default Isaac Lab container image;
- layer a forked Isaac Lab checkout or forked `rsl_rl` package into that image;
- invoke a custom entrypoint while preserving the output contract;
- verify that the custom image ran and training still produced a checkpoint.

The example image does not contain customer code. It only adds:

- a marker label;
- `/opt/byof/custom_train.py`;
- a digest-pinned Workbench Isaac Lab base image.

That keeps the validation focused on the mechanism, not on a particular fork.

## Prerequisites

Before using this cookbook, complete
[../../getting-started.md](../../getting-started.md). That guide is the
canonical setup path for the local NPA install, Nebius credentials, AWS profile,
S3 endpoint, workbench environment variables, Kubernetes context, registry pull
secret, and isolated SkyPilot runtime.

Isaac Lab requires RT cores. Use L40S for this eu-north1 workflow validation.
RTX Pro 6000 is the expected US Central target when capacity is available.
Do not route Isaac Lab training to H100 or H200.

Before you start, verify the three live dependencies:

```bash
aws s3 ls "s3://${NPA_S3_BUCKET}/" --endpoint-url "${AWS_ENDPOINT_URL}"
"${NPA_SKYPILOT_BIN}" check
npa skypilot status
```

The SkyPilot version must be `0.12.2` for this validation lineage.

SkyPilot 0.12.2 does not interpolate environment variables inside YAML `envs`
blocks at submission time. Use the runner script
(`npa/scripts/run_isaac_lab_rl.py`), which materializes endpoint values before
submission, or substitute the literal endpoint value in your YAML.

## Files In This Example

This directory contains `Dockerfile.example`, `custom_train.py`, and this
README. The Dockerfile starts from the digest-pinned Workbench Isaac Lab image
and copies `custom_train.py` to `/opt/byof/custom_train.py`.

`custom_train.py` accepts the same argument shape used by the upstream RSL-RL
training command:

```bash
--task Isaac-Cartpole-v0 \
--num_envs 64 \
--max_iterations 1 \
--headless \
--experiment_name npa_byof \
--run_name "${RUN_ID}" \
agent.save_interval=1
```

At startup it writes `/workspace/output/byof_sentinel.json`. When
`NPA_ISAAC_LAB_OUTPUT_DIR` is set, it also writes
`${NPA_ISAAC_LAB_OUTPUT_DIR}/byof_sentinel.json`; the second path is uploaded by
the existing workflow artifact uploader.

## Building Your Image

Export the registry namespace and choose an image tag:

```bash
export NPA_REGISTRY_ID=<your-registry-id>
export BYOF_BUILD_ID="w10-byof-image-$(date -u +%Y%m%dT%H%M%SZ)"
export BYOF_IMAGE="cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/isaac-lab-byof-test:${BYOF_BUILD_ID}"
```

Build from the repo root:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg NPA_REGISTRY_ID="${NPA_REGISTRY_ID}" \
  --build-arg BYOF_RUN_ID="${BYOF_BUILD_ID}" \
  -f docs/workbench/cookbooks/byof-isaac-lab/Dockerfile.example \
  -t "${BYOF_IMAGE}" \
  docs/workbench/cookbooks/byof-isaac-lab
```

The example base image digest is
`sha256:dc1dd94e64c1e970ec74dccf152180f739cfd457125996100069f688b7911fca`,
the validated `npa-isaac-lab:2.3.2.post1` Workbench image used by W10. Refresh
it only when the platform base image is intentionally rebuilt.

Common customizations:

- copy a forked Isaac Lab tree into `/workspace/isaaclab`;
- install a forked RSL-RL package with `/isaac-sim/python.sh -m pip install -e`;
- add task config defaults under your fork;
- copy private assets into a known path;
- add internal provenance labels;
- keep `/isaac-sim/python.sh` and the artifact upload contract available.

## Pushing To The Registry

Push the image:

```bash
docker push "${BYOF_IMAGE}"
```

Inspect the pushed digest:

```bash
docker buildx imagetools inspect "${BYOF_IMAGE}"
```

W10 pushed
`cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/isaac-lab-byof-test:w10-byof-image-20260520T223706Z`.
The pushed manifest-list digest was
`sha256:c3e104601e31afaa833e3e73558ec9f0c6478f1dce59261fa45073a4d03518bf`;
the linux/amd64 platform digest was
`sha256:d9abdad36137a2f7cb38a6dbd85313ab0c5594582c248e174c9c4a13883d399c`.
Both differ from the vanilla base digest above.

## Running With Image Override Only

This path proves that `--image` replaces the default `image_id` while the
upstream training command still runs.

```bash
export RUN_ID_A="w10-byof-image-only-$(date -u +%Y%m%dT%H%M%SZ)"
export NPA_S3_BUCKET=<your-bucket>
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud

NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN}" \
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py \
  --yaml npa/src/npa/workflows/skypilot/isaac-lab-rl-train.yaml \
  --image "${BYOF_IMAGE}" \
  --task Isaac-Cartpole-v0 \
  --iterations 1 \
  --run-id "${RUN_ID_A}" \
  --output-root "s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof" \
  --cleanup
```

W10 validated this surface with:

```text
Run ID: w10-byof-image-only-20260520T232650Z
GPU: L40S
Output: s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-only-20260520T232650Z/
Manifest train_script: /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py
```

Expected artifacts are `npa_isaac_lab_checkpoint.pt`,
`npa_isaac_lab_checkpoint_manifest.json`, `npa_isaac_lab_train_summary.json`,
`isaac_lab_train.log`, and RSL-RL run logs under `logs/rsl_rl/`.

## Running With Image Plus Command Override

The runner does not have a `--run-cmd` flag. The command override surface is the
SkyPilot YAML `run:` block, selected through `--yaml`.

Create a customer YAML variant outside the platform workflow:

```bash
cp npa/src/npa/workflows/skypilot/isaac-lab-rl-train.yaml /tmp/isaac-lab-byof-command.yaml
```

In that temporary YAML, replace only the training script assignment in `run:`:

```bash
TRAIN_SCRIPT="/opt/byof/custom_train.py"
```

Keep the rest of the run block intact so checkpoint discovery, manifest
creation, and S3 upload still use the platform contract. The command should
still resolve to this shape:

```bash
"${PYTHON_BIN}" \
  /opt/byof/custom_train.py \
  --task "${ISAAC_LAB_TASK}" \
  --num_envs "${ISAAC_LAB_NUM_ENVS}" \
  --max_iterations "${ISAAC_LAB_ITERATIONS}" \
  --headless \
  --experiment_name "${ISAAC_LAB_EXPERIMENT_NAME}" \
  --run_name "${ISAAC_LAB_RUN_NAME}" \
  agent.save_interval=1
```

Launch with the custom YAML and image:

```bash
export RUN_ID_B="w10-byof-image-and-cmd-$(date -u +%Y%m%dT%H%M%SZ)"

NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN}" \
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py \
  --yaml /tmp/isaac-lab-byof-command.yaml \
  --image "${BYOF_IMAGE}" \
  --task Isaac-Cartpole-v0 \
  --iterations 1 \
  --run-id "${RUN_ID_B}" \
  --output-root "s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof" \
  --cleanup
```

W10 validated this surface with:

```text
Run ID: w10-byof-image-and-cmd-20260520T233113Z
GPU: L40S
Output: s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-and-cmd-20260520T233113Z/
Manifest train_script: /opt/byof/custom_train.py
Sentinel: byof_sentinel.json
```

The validation sentinel included:

```json
{
  "byof": true,
  "script": "custom_train.py",
  "run_id": "w10-byof-image-and-cmd-20260520T233113Z",
  "task": "Isaac-Cartpole-v0",
  "num_envs": "64",
  "max_iterations": "1",
  "headless": true,
  "hydra_args": ["agent.save_interval=1"]
}
```

This proves that a non-default image and a non-default entrypoint can run while
still producing a normal Isaac Lab checkpoint.

## Platform Guarantees And Image Responsibilities

The platform guarantees:

- a SkyPilot Kubernetes task;
- L40S accelerator routing in the reference YAML;
- image replacement through `resources.image_id`;
- run id injection through `NPA_ISAAC_LAB_RUN_ID`;
- task and iteration injection through `ISAAC_LAB_TASK` and `ISAAC_LAB_ITERATIONS`;
- output prefix injection through `S3_OUTPUT_PREFIX`;
- log capture to `isaac_lab_train.log`;
- checkpoint discovery under `logs/rsl_rl/`;
- stable checkpoint upload as `npa_isaac_lab_checkpoint.pt`;
- manifest upload as `npa_isaac_lab_checkpoint_manifest.json`;
- cleanup through the runner's existing SkyPilot cleanup path.

Your image must provide:

- `/isaac-sim/python.sh`, or a compatible Python fallback;
- Isaac Lab and RSL-RL dependencies for the requested task;
- the upstream training script if your wrapper delegates to it;
- `boto3` availability or installability during setup;
- write access to `/workspace/isaaclab/npa-runs`;
- headless Isaac Lab behavior;
- any custom assets or packages your fork requires.

Your custom command must preserve `--task`, `--num_envs`, `--max_iterations`,
`--headless`, `--experiment_name`, `--run_name`, and Hydra override passthrough.

## Verifying Your Run

List the output prefix:

```bash
aws s3 ls \
  "s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/${RUN_ID_B}/" \
  --recursive \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Fetch the manifest:

```bash
aws s3 cp \
  "s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/${RUN_ID_B}/npa_isaac_lab_checkpoint_manifest.json" \
  - \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Check:

- `status` is `success`;
- `checkpoint_count` is at least `1`;
- `run_id` matches the submitted run;
- `task` matches the requested task;
- `train_script` is the expected upstream or custom path.

For command override runs, fetch the sentinel:

```bash
aws s3 cp \
  "s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/${RUN_ID_B}/byof_sentinel.json" \
  - \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

Check:

- `byof` is `true`;
- `script` is `custom_train.py`;
- `run_id` matches the submitted run;
- `argv` contains the task, environment count, iteration count, and Hydra args.

Confirm SkyPilot cleanup:

```bash
"${NPA_SKYPILOT_BIN}" status --refresh
"${NPA_SKYPILOT_BIN}" jobs queue --refresh
```

There should be no live run cluster and no in-progress managed job after a
successful `--cleanup` run. The shared jobs controller may remain up.

## Known Constraints

- Isaac Lab requires RT-core GPUs.
- Batch jobs must run headless.
- The current path does not run Omniverse interactive rendering.
- Omniverse rendering support is roadmap work, not part of this BYOF smoke.
- The runner exposes image override directly but command override through YAML.
- Nebius registry pull secrets can expire and may need refresh.
- S3 access must be configured for the target bucket and endpoint.
- The example uses a synthetic image, not a real customer image.
- Multi-GPU and multi-iteration validation are separate prompts.

## Where To Get Help

Open an issue in this repository with the run id, image tag and digest,
SkyPilot job id, manifest JSON, sentinel JSON for command override runs, and
the relevant `sky jobs logs <job-id>` excerpt.

For managed Nebius environments, include the same artifacts when contacting the
support or account team.
