# SONIC G1 MuJoCo image status

[Cookbooks](README.md) · [SONIC images](../sonic-image-catalog.md) · [G1 guide](../guides/g1-humanoid-walk-sonic.md)

The active public `sonic-mujoco-runtime-fetch` image evaluates a Unitree G1
checkpoint in headless MuJoCo. The old combined
`sonic-mujoco-h100-mvp` / `npa-sonic-mujoco:0.1.3-mvp` image is quarantined;
its historical H100 fine-tune-and-evaluate instructions no longer apply.

## Supported capability

The [image manifest](../../../npa/src/npa/deploy/sonic_image_manifest.json)
records the active tag, immutable digest, supported GPUs, and `mujoco-eval`
workload. The replacement is built independently from the quarantined image
and advertises evaluation only. It is not a training image.

The evaluator reads a checkpoint through `SONIC_EVAL_CHECKPOINT_PATH` using
the container's `mujoco-eval` entrypoint. The released warm-start checkpoint is
`nvidia/GEAR-SONIC:sonic_release/last.pt`. ONNX evaluation through
`npa workbench sonic eval --backend container` has a different ONNX/metadata
contract and requests the Isaac-render workload; changing its container argument
to `mujoco-eval` does not convert those inputs.

## Recorded validation

The manifest records real B200 acceptance: **64 finite simulation steps, zero
falls, and verified metrics**. The public checkout omits Git-LFS assets. When
mesh paths are pointers, the evaluator uses primitive collision proxies while
retaining the G1 joints, actuators, masses, and inertias. Metrics record
`geometry_mode=primitive-proxy-no-lfs-payload`.

This establishes the measured checkpoint-to-dynamics path. It does not prove
mesh fidelity, long-horizon walking, or fine-tuning convergence.

## Training and workflow composition

[`sonic-locomotion-finetuning.yaml`](../../../workflows/testing/sonic-locomotion-finetuning.yaml)
is an `npa.workflow/v0.0.1` spec with **retarget → train → MJLab** stages.
It is not the old raw SkyPilot fine-tune/MuJoCo template. Follow the
[locomotion runbook](sonic-locomotion-finetuning.md) for its input and resource
preparation, or the [training runbook](sonic-train-runbook.md) for a single stage.

For packaging and publication, use the [SONIC image catalog](../sonic-image-catalog.md#build-and-publication-commands).
A runtime EULA flag does not change the redistribution status of existing bytes.
