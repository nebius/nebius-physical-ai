# Unitree G1 with SONIC

[Guides](README.md) · [SONIC image catalog](../sonic-image-catalog.md)

Use NVIDIA GEAR-SONIC checkpoints for Unitree G1 locomotion. Choose training,
ONNX evaluation, or MuJoCo checkpoint evaluation first: they require different
inputs and image capabilities.

## Choose the path

| Goal | Path | Requirements |
| --- | --- | --- |
| Train or fine-tune | [SONIC training](../cookbooks/sonic-train-runbook.md) | Compatible motion data; active RTX PRO 6000 Kubernetes image and host driver mounts |
| Retarget, train, and score | [Locomotion workflow](../cookbooks/sonic-locomotion-finetuning.md) | Source motions, prepared GPU profiles, and MJLab-compatible checkpoint |
| Export and evaluate ONNX | [Export/eval runbook](../cookbooks/sonic-eval-runbook.md) | Checkpoint, ONNX metadata, and the selected evaluator's runtime |
| Evaluate a checkpoint in MuJoCo | [MuJoCo image status](../cookbooks/sonic-mvp-g1-mujoco.md) | Active `sonic-mujoco-runtime-fetch` image and its checkpoint input contract |

The old combined H100 fine-tune/MuJoCo image is quarantined. The active public
MuJoCo replacement advertises **evaluation only**; it cannot satisfy a training
stage. Check the [image catalog](../sonic-image-catalog.md) before choosing a GPU.

## Inspect the workflow

From an installed checkout, these commands validate and plan without launching:

```bash
npa workbench sonic --help
npa workbench workflow validate-spec workflows/testing/sonic-locomotion-finetuning.yaml
npa workbench workflow plan-spec workflows/testing/sonic-locomotion-finetuning.yaml \
  --run-id g1-preview --json
```

The current spec contains **retarget → train → MJLab**, with an example bucket
and H100 resource profile. That profile does not satisfy the active first-party
SONIC training image. Follow the [locomotion runbook](../cookbooks/sonic-locomotion-finetuning.md)
to prepare the spec before submission. It is not the retired two-stage
fine-tune/MuJoCo template.

## Verify the result

Inspect the actual checkpoint and evaluation report. A smoke manifest proves
only the checks it records. The public MuJoCo release records a B200 acceptance
run of 64 finite simulation steps and zero falls; this does not establish
long-horizon walking, training convergence, or mesh fidelity.

For results you can share, use [Rerun](../rerun-sharing.md) or
[Foxglove / MCAP](../foxglove-export.md) when the run produces those artifacts.
