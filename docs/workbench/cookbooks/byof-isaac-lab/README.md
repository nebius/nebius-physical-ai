# Run a custom Isaac Lab image or training entrypoint

[Cookbooks](../README.md) · [Workbench setup](../../getting-started.md)

This example adds a small training wrapper to the NPA Isaac Lab image, runs
Cartpole training, and checks that the wrapper and checkpoint actually reached
S3. Use it to learn the two customization surfaces before packaging your own
fork: `--image` selects container bytes; `--yaml` selects the task's `run:` block.

| Input | Output |
| --- | --- |
| Compatible RT-core GPU, project storage, and candidate image | Training log, checkpoint, and checkpoint manifest |
| Optional `custom_train.py` entrypoint | `byof_sentinel.json` proving the wrapper ran |

The current base is Isaac Lab 3 beta / Isaac Sim 6. The
[historical W10 measurements](historical-validation.md) used generation 2 and
are retained separately; they are not new validation of the current base.

## 1. Prepare the environment

Complete [Workbench setup](../../getting-started.md) for an **L40S or RTX PRO
6000** cluster. Isaac requires RT cores. Use the
[contributor Python environment](../../../../npa/README.md#developing-and-testing-npa)
for the scripts below, and run from the repository root.

You also need Docker Buildx, a registry you control, and S3 write access.
Configure exact-host authentication for a private registry. Export S3 credentials
privately; the runner forwards `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
when present. An AWS profile used only by your local CLI is not enough to
supply the remote task.

```bash
export PROJECT_ALIAS='<your-project-alias>'
export KUBE_CONTEXT='<verified-kubernetes-context>'
export NPA_S3_BUCKET='<your-bucket>'
export AWS_ENDPOINT_URL='<your-bucket-endpoint>'
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"

npa workbench health preflight --checks nebius,s3 --json
npa skypilot verify --cluster "$KUBE_CONTEXT" --output-format json
```

Select the profile matching your GPU. The first example below uses L40S;
for RTX PRO 6000 select `isaac-lab-rl-train-rtxpro.yaml` instead:

```bash
export BYOF_PROFILE=npa/src/npa/workflows/byof/profiles/isaac-lab-rl-train.yaml
```

## 2. Build and publish your candidate

[Dockerfile.example](Dockerfile.example) adds only labels and
[`custom_train.py`](custom_train.py) to a digest-pinned NPA base. The wrapper
writes a sentinel, then delegates to the upstream RSL-RL training script with
the original arguments.

```bash
export BYOF_BUILD_ID="byof-$(date -u +%Y%m%dT%H%M%SZ)"
export BYOF_REGISTRY='<your-registry>/<namespace>'
export BYOF_IMAGE="$BYOF_REGISTRY/isaac-lab-byof-test:$BYOF_BUILD_ID"

docker build --platform linux/amd64 \
  --build-arg "BYOF_RUN_ID=$BYOF_BUILD_ID" \
  -f docs/workbench/cookbooks/byof-isaac-lab/Dockerfile.example \
  -t "$BYOF_IMAGE" docs/workbench/cookbooks/byof-isaac-lab
docker push "$BYOF_IMAGE"
docker buildx imagetools inspect "$BYOF_IMAGE"
```

Record the pushed digest and use that immutable reference for validation.
The base fetches Isaac at runtime under the operator's accepted terms. Do not
invoke `/isaac-sim/python.sh` during an image build: that would fetch and bake
the runtime. Follow [BYOF onboarding](../../../../skills/workflows/byof-onboard/SKILL.md)
and [packaging](../../container-packaging.md) when adding your own fork,
dependencies, or assets. Public publication requires separate image acceptance.

## 3. Preview an image-only run

```bash
export RUN_ID_A="byof-image-$(date -u +%Y%m%dT%H%M%SZ)"
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT" \
  --yaml "$BYOF_PROFILE" --image "$BYOF_IMAGE" \
  --task Isaac-Cartpole-v0 --iterations 1 --run-id "$RUN_ID_A" \
  --output-root "s3://$NPA_S3_BUCKET/checkpoints/isaac-lab-byof" \
  --cleanup --render-only
```

The JSON reports the rendered YAML path and output locations. Inspect its
image, accelerator, command, endpoint, and run prefix. `--render-only` creates
no cloud job. Remove only `--render-only` to submit this training smoke.
A completed one-iteration run proves checkpoint production, not convergence.

## 4. Override the training entrypoint

Keep the original profile intact. Make a private copy:

```bash
export BYOF_PRIVATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-byof.XXXXXX")"
cp "$BYOF_PROFILE" "$BYOF_PRIVATE_DIR/task.yaml"
```

In the copy's `run:` block, change only the `TRAIN_SCRIPT` assignment to:

```bash
TRAIN_SCRIPT="/opt/byof/custom_train.py"
```

Keep checkpoint discovery, manifest creation, and S3 upload unchanged. The
wrapper must preserve task, environment count, iteration count, experiment/run
names, visualization flags, and Hydra argument passthrough. Generation 3 uses
`--visualizer none`; `--headless` is only for generation 2 compatibility images.
The runner has no `--run-cmd` option.

Preview with the new YAML and a fresh run ID:

```bash
export RUN_ID_B="byof-entrypoint-$(date -u +%Y%m%dT%H%M%SZ)"
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT" \
  --yaml "$BYOF_PRIVATE_DIR/task.yaml" --image "$BYOF_IMAGE" \
  --task Isaac-Cartpole-v0 --iterations 1 --run-id "$RUN_ID_B" \
  --output-root "s3://$NPA_S3_BUCKET/checkpoints/isaac-lab-byof" \
  --cleanup --render-only
```

Verify the custom script in the rendered command, then remove `--render-only`
to execute. Keep the private YAML until the run and any recovery are complete.

## 5. Verify the result and cleanup

After success, inspect the run prefix with AWS CLI using the same endpoint and
credentials. For the custom-entrypoint run:

```bash
export BYOF_RUN_URI="s3://$NPA_S3_BUCKET/checkpoints/isaac-lab-byof/$RUN_ID_B"
aws s3 ls "$BYOF_RUN_URI/" --recursive --endpoint-url "$AWS_ENDPOINT_URL"
aws s3 cp "$BYOF_RUN_URI/npa_isaac_lab_checkpoint_manifest.json" - \
  --endpoint-url "$AWS_ENDPOINT_URL"
aws s3 cp "$BYOF_RUN_URI/byof_sentinel.json" - \
  --endpoint-url "$AWS_ENDPOINT_URL"
```

| Evidence | Required result |
| --- | --- |
| Checkpoint manifest | `status: success`, at least one checkpoint, matching run ID and task, expected `train_script` |
| Checkpoint and logs | `npa_isaac_lab_checkpoint.pt`, `npa_isaac_lab_train_summary.json`, `isaac_lab_train.log`, and `logs/rsl_rl/` |
| Custom sentinel | `byof: true`, `script: custom_train.py`, matching run ID, expected arguments |
| Runner summary | Terminal job status and successful cleanup for this run |

The image-only run uses the upstream script and does not need a custom sentinel.
`--cleanup` removes the run's compute; the shared jobs controller may remain.
Use [teardown](../../../teardown.md) for separately owned infrastructure.

If setup fails, check image pull access, GPU spelling, the runtime's Isaac
bootstrap, and the exact S3 endpoint. If training completes without the sentinel,
verify the rendered `TRAIN_SCRIPT` and uploaded output directory. When reporting
an issue, include sanitized versions, image digest, status, and log excerpts;
keep credentials and live infrastructure identifiers private.
