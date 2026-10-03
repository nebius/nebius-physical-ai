# GR00T runtime image

[Workbench docs](../../../../docs/workbench/README.md) · [Training cookbook](../../../../docs/workbench/cookbooks/groot-1-7-training.md)

This image runs NVIDIA Isaac-GR00T N1.7 inference and fine-tuning on Linux
x86_64. For a first workload, follow the training cookbook and use an accepted
image from the [public catalog](../../../../docs/workbench/container-image-catalog.md).
Build a candidate when changing the runtime or adding your own code.

## Build a candidate

Run from the **repository root** with Docker Buildx available:

```bash
npa/docker/workbench/groot/build.sh --help
npa/docker/workbench/groot/build.sh --tag my-groot-candidate
```

The local build produces `npa-groot:my-groot-candidate`. To build and push to a
registry you control, authenticate to that registry first, then run:

```bash
npa/docker/workbench/groot/build.sh \
  --registry '<your-registry>/<namespace>' --tag my-groot-candidate --push
```

Use a distinct candidate tag and validate its immutable digest before selecting
it for workloads. Publishing to the official public channel follows the
[contribution and image-validation procedure](../../../../docs/workbench/container-packaging.md).

## What is installed and what is fetched

| Component | Contract |
| --- | --- |
| Isaac-GR00T | Source pinned to `3df8b3825d67f755e69141446f4315f281b9b7e6`; package version `0.1.0` |
| Cosmos Reason2 dependency | Model revision configured by the Dockerfile; weights fetched at runtime |
| Isaac Sim / Isaac Lab | Fetched only when `/isaac-sim/python.sh` is first used for simulation; **not baked into this image** |
| Models, datasets, checkpoints, and caches | Kept under the mounted `/opt/groot-data` and Isaac cache paths |

Standard inference and fine-tuning need Hugging Face access to the selected
GR00T and Cosmos Reason2 weights. They do not need Isaac simulation or its
runtime download. Isaac simulation additionally needs an RT-core GPU and the
operator's accepted NVIDIA terms; see the
[GR00T operating guidance](../../../../skills/tools/groot/SKILL.md).

## Run and verify

Use [`groot-1-7-finetune.yaml`](../../../../workflows/testing/groot-1-7-finetune.yaml)
with the cookbook's dataset, checkpoint, and image settings. Its `gpu_count`
controls a single-node allocation and the real trainer; counts above one use
`torchrun`. Inspect the training report and emitted checkpoint. Image build
success alone does not prove GPU execution or learned policy quality.
