# SONIC Locomotion Fine-Tuning

This cookbook describes the SkyPilot workflow at
`npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml`.

The workflow composes three Workbench stages:

1. Retarget source motion artifacts into the SONIC embodiment schema with
   `npa workbench retargeting run`.
2. Fine-tune or smoke-validate the SONIC locomotion checkpoint on L40S with the
   baked SONIC image variant.
3. Evaluate the resulting checkpoint with MJLab metrics through
   `npa workbench mjlab eval`.

The workflow logic is in YAML. There is no Python runner script for this path.
The same YAML can be used three ways:

- raw SkyPilot, after editing the YAML literals yourself;
- SDK submission through `npa.sdk.workbench.sonic.submit_workflow`;
- CLI submission through `npa workbench workflow submit`.

## Required Inputs

Prepare these S3 prefixes before submission:

- Source motions: `s3://<your-bucket-name>/motions/source/`
- Per-run output root: `s3://<your-bucket-name>/sonic-locomotion/<run-id>/`

SkyPilot 0.12.2 does not expand same-block environment variables inside `envs`.
For raw `sky` runs, replace `<your-bucket-name>`, `<run-id>`,
`<your-registry-id>`, and image tags with literal values before launch. For CLI
or SDK submission, pass the values to the SONIC materializer and it writes the
literal YAML before calling SkyPilot.

Retargeting uses `NPA_RETARGETING_IMAGE`, which defaults to the CPU
`npa-retargeting:0.1.0` preprocess image. That image includes the `npa` CLI,
CPU Python dependencies, and the pinned upstream
`NVlabs/GR00T-WholeBodyControl` data-process scripts. MJLab still uses
`NPA_WORKBENCH_IMAGE`, which defaults to the pushed generic Workbench image
`cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-genesis:0.4.6`.

## Tool Templates

The individual tool templates are available when you want to validate stages
separately:

```bash
npa workbench retargeting workflow
npa workbench mjlab workflow
```

Those commands point to:

- `npa/workflows/workbench/skypilot/retargeting.yaml`
- `npa/workflows/workbench/skypilot/mjlab-eval.yaml`

## Raw SkyPilot

Copy the YAML, edit literals, then run it directly with SkyPilot:

```bash
cp npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml /tmp/sonic.yaml
# Edit /tmp/sonic.yaml so image_id, AWS_ENDPOINT_URL, and all s3:// prefixes
# contain concrete values. Do not leave ${VAR} in envs or image_id fields.
sky jobs launch \
  --name sonic-locomotion-<run-id> \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  --yes \
  /tmp/sonic.yaml
```

The `sonic-finetune` stage uses the first-party SONIC image. The retargeting
stage uses the CPU preprocess image and does not request accelerators. MJLab
uses the generic helper image that contains the `npa` CLI and this repository's
Workbench package.

## CLI Submission

Bootstrap the pinned SkyPilot environment, then submit the YAML. The SONIC
materializer resolves the first-party SONIC image from the manifest and fills
literal S3 values before SkyPilot sees the YAML:

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"

npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --run-id sonic-locomotion-<run-id> \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --npa-image cr.eu-north1.nebius.cloud/<registry-id>/npa:<tag> \
  --gpu-target l40s \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --s3-prefix sonic-locomotion/<run-id> \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Use the Kubernetes controller backend unless you explicitly need the Nebius VM
fallback:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --run-id sonic-locomotion-<run-id> \
  --controller-backend kubernetes
```

For RTX PRO 6000 Kubernetes targets, use:

```bash
--gpu-target gpu-rtx6000 \
--accelerators RTXPRO-6000-BLACKWELL-SERVER-EDITION:1
```

This resolves the SONIC stage to `npa-sonic:0.1.2-k8s-runtime`. L40S resolves to
`npa-sonic:0.1.2`.

For Kubernetes targets that pull private images, pass a SkyPilot config with the
namespace's registry pull secret:

```yaml
kubernetes:
  pod_config:
    spec:
      imagePullSecrets:
        - name: <registry-pull-secret>
```

Use that file with `--config-path`. Only add `serviceAccountName` when the
service account has the Kubernetes node and pod discovery permissions required
by SkyPilot.

## SDK Submission

```python
from pathlib import Path
from npa.sdk.workbench import sonic

sonic.submit_workflow(
    Path("npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml"),
    run_id="sonic-locomotion-run",
    registry="cr.eu-north1.nebius.cloud/<registry-id>",
    npa_image="cr.eu-north1.nebius.cloud/<registry-id>/npa:<tag>",
    gpu_target="l40s",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    s3_prefix="sonic-locomotion/sonic-locomotion-run",
    secret_envs=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
```

## Stage Contract

The retargeting stage writes:

```text
s3://<your-bucket-name>/sonic-locomotion/<run-id>/retargeted/**/*.pkl
s3://<your-bucket-name>/sonic-locomotion/<run-id>/retargeted/retargeting_result.json
```

The `.pkl` files are real SONIC motion-lib artifacts with fields such as
`root_trans_offset`, `pose_aa`, `dof`, `root_rot`, and `fps`. The metadata JSON
is only a sidecar. For raw `bvh` inputs, the stage writes SOMA skeleton PKLs;
upstream SONIC does not bundle the external SOMA Retargeter/GMR step required
to turn raw BVH into G1 robot motion-lib data.

The SONIC stage writes training artifacts under:

```text
s3://<your-bucket-name>/sonic-locomotion/<run-id>/training/
```

Expected SONIC smoke artifacts are:

- `sonic_smoke_result.json`
- `sonic_train_summary.json`
- `checkpoint_smoke.json`

The MJLab stage writes:

```text
s3://<your-bucket-name>/sonic-locomotion/<run-id>/mjlab/mjlab_eval.json
```

## Resources

Retargeting is CPU-only. The first-party SONIC stage requests L40S through
SkyPilot and uses the baked image variant from the SONIC image manifest. MJLab
evaluation remains on H100.

```yaml
resources:
  cloud: kubernetes
  accelerators: L40S:1
```

For RTX PRO 6000 Blackwell Kubernetes targets, switch the accelerator and image
tag together using `sonic-k8s-host-mounted` from
`docs/workbench/sonic-image-catalog.md`.

## Dry Validation

The tool CLIs support `NPA_DRY_RUN=1`, which validates inputs and returns the
artifact schema without writing outputs:

```bash
NPA_DRY_RUN=1 npa workbench retargeting run \
  --input-path s3://bucket/motions/source/ \
  --output-path s3://bucket/sonic-locomotion/run-1/retargeted/ \
  --source-format bones-seed-csv \
  --frame-rate 30 \
  --source-frame-rate 120 \
  --output json

NPA_DRY_RUN=1 npa workbench mjlab eval \
  --input-path s3://bucket/sonic-locomotion/run-1/retargeted/ \
  --checkpoint s3://bucket/sonic-locomotion/run-1/training/checkpoint_smoke.json \
  --output-path s3://bucket/sonic-locomotion/run-1/mjlab/ \
  --output json
```
