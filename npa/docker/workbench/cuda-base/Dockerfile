# Shared CUDA + PyTorch base for GPU workbench service images.
#
# Both npa-lancedb (CLIP embeddings) and npa-detection-training (detector
# train/eval) build on the same CUDA 13 / torch 2.12 runtime plus a common
# FastAPI service + data (lancedb / pyarrow / boto3 / pillow) stack. Factoring
# that shared layer into one base keeps the per-tool images thin and composable
# and lets them share the (large) CUDA + torch layers in the registry.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# libgl/libglib are needed by pillow/opencv-style image decoding paths used by
# both the CLIP embedding and detection dataloaders.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Common service + data dependencies. torch / CUDA come from the base image.
RUN pip install --no-cache-dir \
    "fastapi==0.136.1" \
    "uvicorn[standard]==0.38.0" \
    "lancedb==0.30.2" \
    "pyarrow==23.0.1" \
    "pydantic==2.13.4" \
    "pillow==12.2.0" \
    "boto3==1.42.9" \
    "numpy"

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu

LABEL npa.image="workbench-cuda-base" \
      npa.base="pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime" \
      npa.purpose="Shared CUDA+torch+service base for GPU workbench images"

# Default to non-root; child images switch back to root for their build steps
# and drop to a non-root user again for runtime.
USER ubuntu
