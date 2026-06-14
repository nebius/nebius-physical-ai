# sm_120 Image Catalog

This catalog records the first-party images used for RTX PRO 6000 Blackwell
(`sm_120`) validation. The registry IDs can be replaced with a customer registry
when rebuilding the same Dockerfiles.

Manifest source: `npa/docker/workbench/sm120-images.json`.

## Required Images

| Image | Tag | Purpose |
| --- | --- | --- |
| `npa-base` | `cuda13-b300-sm80-sm90-sm120-latest` | CUDA 13 / PyTorch 2.9 base with sm_120-capable runtime dependencies. |
| `npa-genesis` | `0.4.6-sm80-sm90-sm120-latest` | Genesis and Sim2Real base layered on the sm_120 runtime. |
| `npa-sim2real-envgen` | `0.1.1` | Sim2Real environment generation. |
| `npa-sim2real-reference-policy` | `0.1.1` | Reference BYO-compatible action policy. |
| `npa-sim2real-eval` | `0.1.0` | Sim2Real evaluation. |
| `npa-lerobot-vlm-rl` | `0.1.0` | LeRobot VLM/RL runtime layered on the sm_120 Genesis base. |
| `npa-cosmos3-reason` | `3.0.0` | Cosmos3 reasoning on the sm_120 CUDA 13 base. |
| `npa-sonic` | `0.1.2-k8s-runtime` | SONIC Kubernetes runtime using GPU-operator-mounted NVIDIA driver libraries. |

## Build Commands

Build the base image:

```bash
npa/docker/workbench/base/cuda13-b300/build.sh \
  --registry "${NPA_REGISTRY}" \
  --tag sm80-sm90-sm120-<timestamp> \
  --push
```

Build the Genesis sm_120 image:

```bash
npa/docker/workbench/genesis/build_sm120.sh \
  --base-image "${NPA_REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm120-latest" \
  --registry "${NPA_REGISTRY}" \
  --tag 0.4.6-sm80-sm90-sm120-<timestamp> \
  --push
```

Build the Sim2Real and Cosmos3 images:

```bash
BASE_IMAGE="${NPA_REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm120-latest" \
GENESIS_IMAGE="${NPA_REGISTRY}/npa-genesis:0.4.6-sm80-sm90-sm120-latest" \
npa/docker/workbench/sim2real-build.sh --registry "${NPA_REGISTRY}" --push
```

Build the SONIC RTX PRO 6000 Kubernetes runtime:

```bash
npa/docker/workbench/sonic/build.sh \
  --registry "${NPA_REGISTRY}" \
  --variant k8s \
  --tag 0.1.2-k8s-runtime \
  --push
```

## Live Smoke

Live validation should run on a Kubernetes cluster with schedulable RTX PRO 6000
GPUs and assert `torch.cuda.get_device_capability() == (12, 0)` for the images
that carry PyTorch. SONIC uses `/isaac-sim/python.sh` for the same torch check.
After validation, explicitly tear down the SkyPilot job cluster and confirm no
clusters, managed jobs, or services remain.
