# GR00T N1.7 training

Use the checked-in `groot-1-7-finetune.yaml` workflow to fine-tune NVIDIA GR00T
N1.7 on a GR00T-format LeRobot dataset. The workflow runs NVIDIA's real
`gr00t/experiment/launch_finetune.py` from the `npa-groot:0.1.0` image; it is
not a manifest-only training substitute.

## Reproducible contract

The workbench image and command pin and verify:

- base model `nvidia/GR00T-N1.7-3B` at Hugging Face revision
  `2fc962b973bccdd5d8ce4f67cc63b264d6886495`;
- GR00T package version `0.1.0`;
- Isaac-GR00T source revision
  `3df8b3825d67f755e69141446f4315f281b9b7e6`.

The trainer writes `npa_groot_finetune_manifest.json` beside the uploaded
checkpoints after successful training. It records those pins, the run ID,
embodiment, GPU count, batch size, training steps, input dataset, and output
URI.

For single-node hosts where NCCL peer-to-peer and shared-memory transports are
not viable, the workflow exposes `nccl_transport=socket`. This sets
`NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` inside the trainer and records the
choice in the manifest. Keep the default `auto` on hosts whose native NCCL
transport passes a collective smoke; the socket mode is a compatibility
fallback and trades intra-node bandwidth for stability.

## Validate and inspect

```bash
SPEC=npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml
RUN_ID=groot-n1-7-example

npa workbench workflow validate-spec "$SPEC"
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID"
```

Preview a four-GPU render without launching it:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var gpu_count=4 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --plan-only
```

The plan must show both `accelerators: H100:4` and `--num-gpus 4`. Inside the
stage, the GR00T command converts counts above one to
`torchrun --nproc_per_node=4` and still passes upstream `--num-gpus 4`.

## Submit training

The dataset must already use GR00T's LeRobot layout, including the modality
metadata required for the selected embodiment. Provide the operator's Hugging
Face and S3 credentials through the normal secret environment mechanism.

Single GPU:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var gpu_count=1 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Multi-GPU uses the same command with any positive count:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var gpu_count=4 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

On RTX PRO 6000 Blackwell hosts where a minimal two-rank NCCL probe fails over
both P2P and SHM, add `--var nccl_transport=socket`. Validate that fallback with
a small collective before downloading the full model.

`gpu_count` must be a positive integer. It controls the H100 allocation, the
SkyPilot task resources, and the trainer world size. `global_batch_size`
defaults to the GPU count; if overridden, it must be a positive multiple of
`gpu_count`, as required by the GR00T N1.7 trainer.

## Find the run and outputs

Successful text output includes `run_id: <value>`; JSON automation can add
`--output-format json` and read the top-level `run_id` field. To rediscover
runs later from their persisted NPA manifests:

```bash
npa workbench workflow list \
  --s3-bucket <bucket> \
  --workflow-s3-prefix groot-1-7-finetune \
  --json
```

With the default prefix, checkpoints and provenance are under:

```text
s3://<bucket>/groot-1-7-finetune/<run-id>/checkpoints/
s3://<bucket>/groot-1-7-finetune/<run-id>/checkpoints/npa_groot_finetune_manifest.json
```

If `prefix` or `checkpoint_uri` is overridden, use those resolved locations
and pass the corresponding parent prefix to `workflow list`.
