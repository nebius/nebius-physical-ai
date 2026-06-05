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

The workflow logic is in YAML. There is no Python runner script for this path;
submit the checked-in YAML directly through the generic SkyPilot submit command.

## Required Inputs

Prepare these S3 prefixes before submission:

- Source motions: `s3://<your-bucket-name>/motions/source/`
- Per-run output root: `s3://<your-bucket-name>/sonic-locomotion/<run-id>/`

The committed YAML uses explicit placeholders because SkyPilot 0.12.2 does not
expand same-block environment variables inside `envs`. Replace
`<your-bucket-name>`, `<run-id>`, `<your-registry-id>`, and image tags before a
live run.

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

## Full Submission

Bootstrap the pinned SkyPilot environment, then submit the YAML:

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"

npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --run-id sonic-locomotion-<run-id>
```

Use the Kubernetes controller backend unless you explicitly need the Nebius VM
fallback:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --run-id sonic-locomotion-<run-id> \
  --controller-backend kubernetes
```

## Stage Contract

The retargeting stage writes:

```text
s3://<your-bucket-name>/sonic-locomotion/<run-id>/retargeted/retargeting_manifest.json
```

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
  --output json

NPA_DRY_RUN=1 npa workbench mjlab eval \
  --input-path s3://bucket/sonic-locomotion/run-1/retargeted/ \
  --checkpoint s3://bucket/sonic-locomotion/run-1/training/checkpoint_smoke.json \
  --output-path s3://bucket/sonic-locomotion/run-1/mjlab/ \
  --output json
```
