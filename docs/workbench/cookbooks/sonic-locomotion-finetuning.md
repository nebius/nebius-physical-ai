# SONIC locomotion workflow

[Cookbooks](README.md) · [G1 guide](../guides/g1-humanoid-walk-sonic.md)

[`sonic-locomotion-finetuning.yaml`](../../../workflows/testing/sonic-locomotion-finetuning.yaml)
is an `npa.workflow/v0.0.1` reference for **retarget → train → MJLab**.
Prepare its resource and artifact contracts before execution. It is not raw
SkyPilot YAML and must be submitted through `npa workbench workflow submit`.

## Inspect the reference

```bash
npa workbench workflow validate-spec workflows/testing/sonic-locomotion-finetuning.yaml
npa workbench workflow plan-spec workflows/testing/sonic-locomotion-finetuning.yaml \
  --run-id sonic-preview --json
```

Validation and planning do not launch a workload. The reference contains an
example bucket and an H100 GPU profile, which does not match the active
first-party SONIC training image.

## Prepare resources and inputs

Copy the spec to an operator-owned path and complete
[Workbench setup](../getting-started.md). For first-party training, use the
[active RTX PRO 6000 Kubernetes image](../sonic-image-catalog.md) with its required
host driver and cache mounts. Give training and MJLab separate resource profiles
if they require different GPU/image combinations. Changing `--gpu-target` alone
does not rewrite the spec's resource profiles.

Check these config values against the actual planned commands:

| Key | Consumer / purpose |
| --- | --- |
| `bucket`, `prefix` | Writable project storage and a unique run prefix |
| `motion_uri` | Source motions for retargeting and MJLab |
| `retargeted_uri` | Retargeting output |
| `data_uri` | SONIC trainer input; not automatically replaced by `retargeted_uri` |
| `checkpoint_uri` | Shared by training and MJLab in the reference |
| `training_uri` | Training output prefix |
| `mjlab_uri` | Evaluation output prefix |
| `train_iterations`, `episodes` | Training and evaluation settings |

The reference's `checkpoint_uri` points inside the training output. A real
fine-tune needs a valid starting checkpoint, and MJLab must consume the resulting
compatible checkpoint. Set those inputs per stage in the prepared spec, for
example with state `params` overlays. Verify the trainer's real output path;
`checkpoint.json` is a manifest, not policy weights.

Retargeting writes `retargeting_result.json` and motion artifacts. Raw BVH
conversion produces SOMA skeleton PKLs; upstream SONIC does not bundle the
external SOMA/GMR step needed to convert those into G1 motion-library data.
A successful conversion does not establish trainer compatibility.

## Submit the prepared workflow

Validate and plan the final copy with your actual overrides, then follow the
[workflow submission sequence](../npa-workflow-guide.md#quick-start), including
model access, GPU discovery, and exact-image checks. Use the same project,
cluster, bucket, and run ID throughout. Public images need no custom registry.

Use the generic workflow command for this spec. SONIC's legacy raw-YAML
materializer, uppercase `SONIC_*` environment overrides, and direct
`sky jobs launch` examples belong to a different format.

## Verify outputs

Inspect each stage's artifacts and the actual checkpoint consumed by MJLab:

- Retargeting: `retargeting_result.json` and compatible motion files.
- Training: weights plus the summary and checkpoint manifest.
- MJLab: `mjlab_eval.json`, including the backend and measured metrics.

Import checks or smoke manifests alone do not establish policy learning.
For a single training stage, use the [training runbook](sonic-train-runbook.md).
For ONNX export/evaluation, use the [export runbook](sonic-eval-runbook.md).
Finish with [teardown](../../teardown.md) for owned resources.
