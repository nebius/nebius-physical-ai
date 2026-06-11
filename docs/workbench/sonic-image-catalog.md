# SONIC Image Catalog

The machine-readable source of truth is
`npa/src/npa/deploy/sonic_image_manifest.json`. `npa.deploy.images` loads that
manifest to resolve first-party SONIC image tags for CLI, SDK, and workflow
paths.

This manifest is SONIC-scoped. Other Workbench images resolve through
`npa/src/npa/deploy/images.py` and `[tool.npa.supported-tools]` in
`npa/pyproject.toml`; do not add non-SONIC tools to this manifest unless the
catalog is intentionally expanded to all Workbench solutions.

SONIC uses two image variants. They differ only in how NVIDIA graphics and
driver-coupled userspace are provided.

| Variant | Tag | Driver provisioning | Use for | Why |
| --- | --- | --- | --- | --- |
| `sonic-l40s-baked` | `npa-sonic:0.1.2` | `baked` | L40S VM or compute-only host driver targets | The host does not mount the NVIDIA graphics userspace needed by Isaac Lab, so the image carries the matching NVML, GL, and Vulkan libraries. |
| `sonic-k8s-host-mounted` | `npa-sonic:0.1.2-k8s-runtime` | `host-mounted` | RTX PRO 6000 Blackwell on Kubernetes with the NVIDIA GPU Operator | The GPU Operator mounts driver-matched NVML, GL, and Vulkan libraries from the node, so the image must not carry conflicting driver libraries. |

Use `${NPA_REGISTRY}/npa-sonic:<tag>` for a concrete registry reference:

```bash
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
```

## Selection

The default SONIC image is `sonic-l40s-baked`. For RTX PRO 6000, Blackwell, or
`sm_120` Kubernetes targets, the resolver selects `sonic-k8s-host-mounted`.

CLI example:

```bash
npa workbench sonic train \
  --runtime serverless \
  --gpu-type rtx6000 \
  --image-variant sonic-k8s-host-mounted
```

SDK example:

```python
from npa.sdk.workbench import sonic

sonic.train(
    runtime="serverless",
    gpu_type="rtx6000",
    image_variant="sonic-k8s-host-mounted",
)
```

SkyPilot YAMLs expose the same selectors through env vars such as
`SONIC_GPU_TYPE`, `SONIC_GPU_TARGET`, `SONIC_IMAGE_VARIANT`,
`SONIC_EVAL_CONTAINER_GPU_TARGET`, and
`SONIC_EVAL_CONTAINER_IMAGE_VARIANT`.

For standard workflow submission, the same selector is available on the generic
workflow command. The submitted YAML is materialized with literal env values
before SkyPilot sees it:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-train-standalone.yaml \
  --registry "${NPA_REGISTRY}" \
  --gpu-target l40s \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

SDK users call the same materializer:

```python
from pathlib import Path
from npa.sdk.workbench import sonic

sonic.submit_workflow(
    Path("npa/workflows/workbench/skypilot/sonic-train-standalone.yaml"),
    run_id="sonic-smoke",
    registry="cr.eu-north1.nebius.cloud/<registry-id>",
    gpu_target="gpu-rtx6000",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    secret_envs=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
```

## Related BYO Images

The VLM eval workflows use `NPA_VLM_IMAGE` for the serving image. The committed
default is `cr.eu-north1.nebius.cloud/<your-registry-id>/npa-cosmos:1.0.9`,
a pushed CUDA/PyTorch Workbench image; set `NPA_VLM_IMAGE` to a prebuilt VLM or
vLLM image when you need pinned serving dependencies.

The retargeting workflow uses `NPA_RETARGETING_IMAGE` for the CPU preprocess
image. The committed default is
`cr.eu-north1.nebius.cloud/<your-registry-id>/npa-retargeting:0.1.0`, a pushed
image that installs this repository's `npa` package, CPU preprocess
dependencies, and pinned upstream SONIC data-process scripts.

MJLab workflows use `NPA_WORKBENCH_IMAGE` for the generic Workbench CLI image.
The committed default remains
`cr.eu-north1.nebius.cloud/<your-registry-id>/npa-genesis:0.4.6`.

## Build Commands

Baked L40S variant:

```bash
npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push --variant baked
```

Kubernetes host-mounted variant:

```bash
npa/docker/workbench/sonic/build.sh \
  --registry "${NPA_REGISTRY}" \
  --push \
  --variant k8s \
  --tag 0.1.2-k8s-runtime
```

Do not overwrite existing `0.1.2`, `0.1.2-k8s`, `0.1.1`, or `0.1.0` tags. New
compatibility variants must use additive tags.
