# SONIC training runbook

[Cookbooks](README.md) · [SONIC images](../sonic-image-catalog.md)

Run Isaac-backed SONIC training on RTX PRO 6000 Kubernetes using the active
`sonic-k8s-host-mounted` image. It requires NVIDIA GPU Operator driver mounts
and acquires Isaac dependencies at runtime. Complete
[Workbench setup](../getting-started.md) and the
[runtime cache setup](../model-weight-cache.md) first.

## Prepare the spec

The checked-in training spec uses `sonic_runtime: local`: the trainer runs inside
the GPU task's container. Its H100 resource default does **not** match the active
first-party training image. Make an operator copy:

```bash
cp workflows/testing/sonic-train.yaml /tmp/sonic-train.yaml
```

In that copy, set `resources.gpu.accelerators` to the compatible RTX PRO 6000
name reported by `npa workbench workflow gpus`. Retain the CPU and memory
requirements and configure the required host driver/cache mounts. Use the
[image catalog](../sonic-image-catalog.md) and [GPU driver guide](../mk8s-gpu-driver-strategy.md)
for the exact runtime contract.

Set these `config` values for your run, either in the copy or through `--var`:

| Key | Meaning |
| --- | --- |
| `bucket` | Your writable project bucket |
| `data_uri` | Compatible SONIC motion data in S3 |
| `checkpoint_uri` | The selected starting checkpoint |
| `training_uri` | A new output prefix for this run |
| `train_iterations` | Training iterations requested by this spec |

Then validate and plan:

```bash
npa workbench workflow validate-spec /tmp/sonic-train.yaml
npa workbench workflow plan-spec /tmp/sonic-train.yaml --run-id "<run-id>" --json
```

Inspect the resolved data, checkpoint, output, and GPU profile. The plan is not
proof that the training runtime or inputs are usable.

## Submit and inspect

Follow the [workflow submission sequence](../npa-workflow-guide.md#quick-start)
with your prepared spec and the same project, cluster, and run ID. Run the
SONIC-specific model-access and image checks before submission. Public images
need no private registry override. Isaac execution defaults to acceptance;
`--no-accept-eula` refuses the runtime fetch.

Inspect checkpoint weights and the training summary after completion.
`checkpoint.json`, `checkpoint_smoke.json`, or a successful import alone does
not establish a learned policy. Evaluate the checkpoint against the intended
task; runtime success and policy quality are separate outcomes.

## Serverless limitation

There is no built-in compute-only training image for the default serverless
L40S/H100/H200 path. The historical baked variants are quarantined. Serverless
training requires an independently validated compute-only `--image`; the active
Kubernetes image cannot substitute because it requires host driver mounts.
The public MuJoCo image serves evaluation only.

For retargeting and downstream evaluation, see the
[locomotion workflow](sonic-locomotion-finetuning.md). Finish with
[owned-resource teardown](../../teardown.md).
