# sm_120 Image Catalog

This catalog records the first-party images used for RTX PRO 6000 Blackwell
(`sm_120`) validation. Published redistributable releases use the public GHCR
channel; rebuilds may target an operator-controlled registry via `NPA_REGISTRY`.

Manifest source: `npa/docker/workbench/sm120-images.json`.

## Required Images

| Image | Tag | Purpose |
| --- | --- | --- |
| `npa-base` | `cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest` | CUDA 13 / PyTorch 2.9 base with an asserted sm_80/sm_90/sm_100/sm_103/sm_120 source-build contract and a baked same-major `sm_100` → `sm_103` SASS validator. |
| `npa-genesis` | `cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Genesis and Sim2Real base layered on the widened CUDA 13 runtime. |
| `npa-envgen` | `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Sim2Real environment generation. |
| `npa-reference-policy` | `cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Reference BYO-compatible action policy. |
| `npa-loop-eval` | `cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Sim2Real evaluation. |
| `npa-lerobot-vlm-rl` | `cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | LeRobot VLM/RL runtime layered on the widened Genesis base. |
| `npa-cosmos3-reason` | `cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Cosmos3 reasoning on the widened CUDA 13 base. |
| `npa-sonic` | `cuda13-b300-0.1.2-k8s-runtime-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | SONIC Kubernetes runtime using GPU-operator-mounted NVIDIA driver libraries. |

## Build Commands

Build the base image:

```bash
npa/docker/workbench/base/cuda13-b300/build.sh \
  --registry "${NPA_REGISTRY}" \
  --tag sm80-sm90-sm100-sm103-sm120-<timestamp> \
  --push
```

Build the Genesis sm_120 image:

```bash
npa/docker/workbench/genesis/build_sm120.sh \
  --base-image "${NPA_REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest" \
  --registry "${NPA_REGISTRY}" \
  --tag 0.4.6-sm80-sm90-sm100-sm103-sm120-<timestamp> \
  --push
```

Build the Sim2Real and Cosmos3 images:

```bash
BASE_IMAGE="${NPA_REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest" \
GENESIS_IMAGE="${NPA_REGISTRY}/npa-genesis:cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
npa/docker/workbench/sim2real-build.sh --registry "${NPA_REGISTRY}" --push
```

For a generic operator-owned BYOF registry, build the SONIC RTX PRO 6000
Kubernetes runtime with:

```bash
npa/docker/workbench/sonic/build.sh \
  --registry "${NPA_REGISTRY}" \
  --variant k8s \
  --tag 0.1.2-k8s-runtime \
  --push
```

This command is not the NPA release path. Official GHCR development builds and
promotions use `.github/workflows/publish-public-images.yml` and immutable
source-SHA tags.

## Live Smoke

Live validation should run on a Kubernetes cluster with schedulable RTX PRO 6000
GPUs and assert `torch.cuda.get_device_capability() == (12, 0)` for the images
that carry PyTorch. SONIC uses `/isaac-sim/python.sh` for the same torch check.
After validation, explicitly tear down the SkyPilot job cluster and confirm no
clusters, managed jobs, or services remain.
