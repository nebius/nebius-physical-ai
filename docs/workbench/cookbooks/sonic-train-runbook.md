# SONIC Train Runbook

This runbook covers the `npa workbench sonic train --runtime serverless` path
for Isaac Lab-based SONIC training and smoke validation.

## Purpose

The serverless train command submits a Nebius Serverless Job using the active
runtime-fetch SONIC image. The job validates that Isaac Lab and SONIC are
available in the same container, writes `sonic_smoke_result.json`, and uploads
artifacts to the requested S3 prefix.

The image contains no Isaac or NVIDIA driver userspace. It defaults the run-scoped
Isaac route to `ACCEPT_EULA=Y` (with `--no-accept-eula` as an explicit opt-out),
then acquires pinned Isaac dependencies and uses
`gear_sonic` from `NVlabs/GR00T-WholeBodyControl`. The default build
skips optional C++ deploy compilation; use `BUILD_SONIC_DEPLOY=1` only when
validating the TensorRT/ONNX Runtime deploy path.

Full SONIC training on BONES-SEED is a large multi-GPU workload. The Workbench
smoke target is intentionally minimal: it validates environment integration and
the training entry path, not convergence.

## Minimal Smoke

```bash
SMOKE_TS=$(date -u +%Y%m%dT%H%M%SZ)
npa workbench sonic -p eu-north1 -n w7sonic train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --gpu-type rtx6000 \
  --gpu-count 1 \
  --embodiment unitree-g1 \
  --steps 10 \
  --output-path s3://${NPA_S3_BUCKET}/w7sonic-smoke/$SMOKE_TS/ \
  --job-name sonic-smoke-$SMOKE_TS \
  --timeout 3600 \
  --poll-interval 15
```

When validating an unpromoted build, pass the pushed image explicitly:

```bash
--image "${NPA_REGISTRY}/npa-sonic:cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
```

## Standalone SkyPilot YAML

The raw SkyPilot training smoke is
`npa/workflows/workbench/npa-workflows/sonic-train.yaml`. It has literal
editable defaults because SkyPilot 0.12.2 does not interpolate `${VAR}` inside
`envs` or `resources.image_id`.

For a zero-NPA raw SkyPilot run, copy the YAML, replace these literals, and
launch it directly:

| YAML field | L40S value | RTX PRO 6000 Kubernetes value |
| --- | --- | --- |
| `resources.image_id` and `POLICY_IMAGE` | `<your-registry>/<namespace>/npa-sonic:0.1.2` | `<your-registry>/<namespace>/npa-sonic:0.1.2-k8s-runtime` |
| `SONIC_GPU_TYPE` | `l40s` | `gpu-rtx6000` |
| `SONIC_IMAGE_VARIANT` | `sonic-l40s-baked` | `sonic-k8s-host-mounted` |
| `S3_ENDPOINT_URL` | your S3-compatible endpoint | your S3-compatible endpoint |
| `S3_BUCKET` / `SONIC_OUTPUT_PREFIX` | your artifact destination | your artifact destination |

```bash
cp npa/workflows/workbench/npa-workflows/sonic-train.yaml /tmp/sonic-train.yaml
# Edit /tmp/sonic-train.yaml with concrete image and S3 values.
sky jobs launch \
  --name sonic-train-smoke \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  --yes \
  /tmp/sonic-train.yaml
```

The same YAML can be submitted through the CLI, which materializes the image and
S3 values before SkyPilot submission:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sonic-train.yaml \
  --run-id sonic-train-smoke \
  --registry "${NPA_REGISTRY}" \
  --gpu-target l40s \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --s3-prefix sonic-train/sonic-train-smoke \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

For Kubernetes targets that pull from a private registry, pass a SkyPilot config
with the namespace's registry pull secret:

```yaml
kubernetes:
  pod_config:
    spec:
      imagePullSecrets:
        - name: <registry-pull-secret>
```

Then add `--config-path /path/to/skypilot-kubernetes.yaml` to the submit command.
Do not set `serviceAccountName` unless that account can also list Kubernetes
nodes and pods for SkyPilot prechecks.

When `SONIC_PAYLOAD_MODE=docker`, the default `SONIC_DOCKER_GPU_REQUEST=all`
uses Docker's legacy `--gpus all` path. On Kubernetes sidecars where the NVIDIA
runtime is configured for CDI, set
`SONIC_DOCKER_GPU_REQUEST: nvidia.com/gpu=all` and ensure the SkyPilot runtime
has `nvidia-ctk` from `nvidia-container-toolkit`. The workflow generates
`/etc/cdi/nvidia.yaml` immediately before `docker run`, then starts the payload
with `--runtime=nvidia`, `NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all`, and
`NVIDIA_DRIVER_CAPABILITIES=all`. If the sidecar is unprivileged or lacks Docker
and `nvidia-ctk`, use the direct Kubernetes host-mounted SONIC image path rather
than the nested Docker payload.

SDK equivalent:

```python
from pathlib import Path
from npa.sdk.workbench import sonic

sonic.submit_workflow(
    Path("npa/workflows/workbench/npa-workflows/sonic-train.yaml"),
    run_id="sonic-train-smoke",
    registry="<your-registry>/<namespace>",
    gpu_target="l40s",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    s3_prefix="sonic-train/sonic-train-smoke",
    secret_envs=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
```

Use the exact supported host-mounted image from the manifest for RTX PRO 6000
Blackwell on Kubernetes with the NVIDIA GPU Operator:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
docker manifest inspect \
  "${NPA_REGISTRY}/npa-sonic:cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
```

The L40S baked variant is quarantined and must not be rebuilt or pushed as an
NPA-owned public image. Compute-only deployments require an independently
licensed, operator-built and validated BYOF image.

Expected output artifacts:

- `sonic_smoke_result.json`
- `sonic_train_summary.json`
- `checkpoint_smoke.json`

## GPU Selection

Prefer RT-core GPUs because Isaac Lab simulation workloads use rendering and
physics paths that are best validated on RT-capable hardware:

- `l40s`
- `gpu-rtx-pro-6000`

H100/H200/B200/B300 may be useful for model training throughput, but they are not
the preferred smoke target for Isaac Lab simulation validation.

The image compatibility catalog is `docs/workbench/sonic-image-catalog.md`; the
runtime resolver reads `npa/src/npa/deploy/sonic_image_manifest.json`.

## Parameters

- `--embodiment`: defaults to `unitree-g1`, mapped to `UNITREE_G1_SONIC`.
- `--checkpoint`: defaults to `nvidia/GEAR-SONIC:sonic_release/last.pt`.
- `--data-path`: optional path or URI for training data.
- `--sample-data`: explicitly uses the SONIC sample data path.
- `--steps` / `--max-iterations`: minimal smoke iteration count.
- `--output-path`: required S3 prefix for serverless artifacts.
- `--submit-only`: submit and return without polling.

If `--data-path` is omitted, the command treats the run as a sample-data smoke.

## Failure Classification

- `PASS`: serverless Job succeeds and S3 artifacts are present.
- `FAIL_TRAINING`: the job starts but SONIC or Isaac Lab fails.
- `FAIL_PLATFORM`: image pull, subnet, auth, or scheduler failure before SONIC
  code runs.
- `FAIL_NER`: capacity or quota blocks scheduling.

Do not fall back to `npa-sonic:0.1.2`: it is quarantined because its built bytes
inherit restricted NVIDIA content and baked driver libraries.

## Known Limitations

- GR00T+SONIC orchestration is not part of this runbook.
- Additional embodiments beyond Unitree G1 are exposed as tags but not
  qualified by Workbench smoke.
- NIM distribution was not confirmed in discovery; the supported path is
  Hugging Face plus the upstream SONIC repository.
- W7 build-fix local validation passed for build, imports, and entrypoint smoke.
  The first pushed-image L40S smoke reached Nebius Job `ERROR` before
  `started_at` and produced no container logs, so that result is classified as
  `FAIL_PLATFORM`, not a SONIC runtime failure.
