# npa-sonic-export: lean SONIC ONNX-export runtime.
#
# The full npa-sonic image is a self-contained Isaac Lab stack used for SONIC
# training and simulation. ONNX *export* (and ONNX parity eval) only needs the
# torch + onnx toolchain — the policy is loaded on CPU (map_location="cpu") and
# traced — so this image carries just the `npa[sonic]` export dependencies on a
# slim, driver-independent base. That unblocks `npa workbench sonic export` /
# `sonic eval` on Kubernetes GPU nodes (host-mounted driver) without the
# upstream-blocked cuda13-b300 Blackwell base.
#
# npa itself is synced at runtime from NPA_SRC_S3_URI (matching the workflow
# convention), so this image is npa-version independent.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU torch keeps the image lean and avoids CUDA/driver coupling; SONIC ONNX
# export loads the checkpoint on CPU and traces the MLP policy. The onnx stack
# mirrors the `npa[sonic]` extra in npa/pyproject.toml.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.12.1" \
    && pip install \
        "onnx>=1.16" \
        "onnxscript>=0.5" \
        "onnxruntime>=1.18" \
        "boto3>=1.34" \
        "numpy>=1.26"

RUN useradd -m -u 1000 ubuntu
USER ubuntu
WORKDIR /home/ubuntu

LABEL npa.tool="sonic" \
      npa.variant="sonic-export-lean" \
      npa.purpose="SONIC ONNX export / parity eval (torch + onnx, CPU load)" \
      npa.driver_provisioning="none"
