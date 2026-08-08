# GR00T N1.7 training

Use the checked-in `groot-1-7-finetune.yaml` workflow to fine-tune NVIDIA GR00T
N1.7 on a GR00T-format LeRobot dataset. The workflow runs NVIDIA's real
`gr00t/experiment/launch_finetune.py` and `Gr00tPolicy.get_action` surfaces from
the `npa-groot:0.1.0` image; it is not a manifest-only training or scoring
substitute. Its serial stage graph is:

```text
prepare_split -> baseline_eval -> finetune -> posttrain_eval ->
compare_learning -> emit_mcap -> emit_rrd -> publish
```

The split is deterministic at episode granularity, stores content hashes, uses
training-only normalization statistics, and proves zero episode overlap. Both
evaluations retain aligned predicted and expert actions for every held-out
sample. The primary action-MSE gate must improve after one complete pass over
the configured training cohort. RRD, MCAP, and comparison video are synchronized
from those held-out outputs and explicitly labelled offline evaluation, not a
closed-loop policy rollout.

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
URI. It also records per-rank/world-size evidence, distinct visible GPU UUIDs,
an NCCL all-reduce result, a finite training loss, optimizer-step evidence,
and actual checkpoint object/byte counts plus a real loss trajectory. The
`compare_learning` stage fails unless model-forward proof, exact sample/action
alignment, identical held-out provenance, training coverage, and a factual
primary-metric improvement are all present. Per-dimension regressions remain
visible even when the primary gate passes.

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

Preview the validated eight-GPU render without launching it:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var gpu_count=8 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --plan-only
```

The plan must show the episode split and two real GPU evaluation phases around
`finetune`, plus the compare/replay/publish phases. It must show both
`accelerators: H100:8` and `--num-gpus 8` on `finetune`. Inside that stage, the
GR00T command converts counts above one to `torchrun --nproc_per_node=8` and
still passes upstream `--num-gpus 8`.

## Submit training

The dataset must already use GR00T's LeRobot layout, including the modality
metadata required for the selected embodiment. Provide the operator's Hugging
Face and S3 credentials through the normal secret environment mechanism.

Single GPU:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var source_data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var gpu_count=1 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Multi-GPU uses the same command with any positive count (the checked-in example defaults to 8):

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var source_data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var gpu_count=8 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

On RTX PRO 6000 Blackwell hosts where a minimal two-rank NCCL probe fails over
both P2P and SHM, add `--var nccl_transport=socket`. Validate that fallback with
a small collective before downloading the full model.

`gpu_count` must be a positive integer. It controls the trainer allocation and
world size. `global_batch_size` must be a positive multiple of `gpu_count`, as
required by GR00T. Recalculate `max_steps=ceil(train_samples/global_batch_size)`
when the cohort or batch changes so the stated coverage remains a complete
training-cohort pass rather than an arbitrary demo duration.

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
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/split/manifest.json
s3://<bucket>/groot-1-7-finetune/<run-id>/eval/baseline/evaluation.json
s3://<bucket>/groot-1-7-finetune/<run-id>/checkpoints/posttrain/
s3://<bucket>/groot-1-7-finetune/<run-id>/eval/posttrain/evaluation.json
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/learning-report.json
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/offline-heldout-comparison.mp4
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/groot-learning.mcap
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/groot-learning.rrd
s3://<bucket>/groot-1-7-finetune/<run-id>/reports/publish-manifest.json
s3://<bucket>/groot-1-7-finetune/<run-id>/workflow.yaml
```

MCAP uses `foxglove.CompressedImage` on `/camera/<name>`, `foxglove.Log` on
`/log`, plus `/policy/predicted_action`, `/expert/action`,
`/metrics/action_error`, `/metrics/heldout_before`,
`/metrics/heldout_after`, and `/metrics/train_loss`. RRD opens with a camera,
predicted-versus-expert action plots, error panels, before/after metrics,
training loss, and provenance on the `dataset_time` timeline. Dataset index/FPS
is not robot wall-clock time; every output sets `is_robot_capture_time=false`.

The terminal `publish` stage downloads and independently parses both recording
formats. It succeeds only after their run IDs, schemas/topics, Rerun identity,
timeline/entities, provenance, nonzero sizes, and the exact submitted workflow
artifact have all passed validation.

If `prefix` or `checkpoint_uri` is overridden, use those resolved locations
and pass the corresponding parent prefix to `workflow list`.

## Kubernetes image prerequisites

The image keeps its non-root `ubuntu` runtime while enabling the repository's
SkyPilot Kubernetes prerequisite layer. That layer supplies the system Python,
rsync/SSH client, netcat, and passwordless `sudo` required by SkyPilot's in-pod
bootstrap. If a historical image exits before setup with `sudo: command not
found`, build the additive repair tag with
`npa/docker/workbench/groot/Dockerfile.k8s-prereqs`; do not clear the workflow's
image pin or switch the task to root.

Historical `0.1.0` images also removed `linux-libc-dev` after build while
retaining `libc6-dev`, which makes later `apt` operations fail dependency
resolution. The derived recipe repairs that package relationship before adding
the prerequisites. Fresh source builds retain the current header package so
SkyPilot can install any remaining SSH runtime packages during bootstrap.
