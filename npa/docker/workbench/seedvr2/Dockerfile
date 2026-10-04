# Apache-2.0 SeedVR2 source/runtime only; public model weights stay runtime-only.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:ae7f650405a3964972dacfa889273bf8e3fbe9709899afd187da01c4cdff3105 AS seedvr2-build

ARG SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185
ARG SEEDVR2_SOURCE_SHA256=aeaed7c661f4c3804c71542bbf7ce6c45c103e5f62caf917c2d2ab6e8a29dc98
ARG APEX_SOURCE_REF=8a6508aaad6e75a2b939e33f308cd63d745d97f1
ARG APEX_SOURCE_SHA256=59b3e915823a2f41fec7f8cb20b4a558c9658839709f171953c1f6c16a8975f8
ARG FLASH_ATTN_VERSION=2.8.3.post1
ARG FLASH_ATTN_SHA256=55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348
ARG FLASH_ATTN_MAX_JOBS=2
ARG FLASH_ATTN_NVCC_THREADS=1
ARG APEX_MAX_JOBS=2
ARG TORCH_CUDA_ARCH_LIST=9.0

LABEL npa.tool="seedvr2" \
      npa.version="0.1.0-cu130-unbuilt" \
      npa.source.repo="https://github.com/ByteDance-Seed/SeedVR" \
      npa.source.ref="${SEEDVR2_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.build.flash-attn.max-jobs="${FLASH_ATTN_MAX_JOBS}" \
      npa.build.flash-attn.nvcc-threads="${FLASH_ATTN_NVCC_THREADS}" \
      npa.build.apex.max-jobs="${APEX_MAX_JOBS}" \
      npa.build.torch.cuda-arch-list="${TORCH_CUDA_ARCH_LIST}" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-Toolkit"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    SEEDVR2_PYTHON=/opt/seedvr2-venv/bin/python \
    NPA_SEEDVR2_WORK_DIR=/workspace/seedvr2-runs \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_LIGHT_WORKBENCH_TOOL=seedvr2 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH=/opt/npa-venv/bin:/opt/seedvr2-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

RUN rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 \
      ninja-build openssh-server python3.12 python3.12-dev python3.12-venv \
      rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /opt/nvidia/nsight-compute \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

COPY docker/workbench/seedvr2/requirements.lock /tmp/seedvr2-requirements.lock
COPY --chmod=0555 docker/workbench/seedvr2/filter_cudnn_runtime.py /opt/npa-tools/filter_cudnn_runtime.py
COPY --chmod=0555 docker/workbench/seedvr2/filter_nvshmem_runtime.py /opt/npa-tools/filter_nvshmem_runtime.py
ADD --checksum=sha256:1f5b7ada702926bc73327e6eb02dc2d41facc844cc4512ac900451bda06a459e \
    https://raw.githubusercontent.com/NVIDIA/nvshmem/v3.4.5-0/License.txt \
    /opt/npa-legal/NVSHMEM-License-v3.4.5-0.txt
RUN python3.12 -m venv /opt/seedvr2-venv \
    && pip install --no-cache-dir --upgrade 'pip==26.2.1' \
    && pip install --no-cache-dir --require-hashes -r /tmp/seedvr2-requirements.lock \
    && install -d -m 0755 /opt/seedvr2-build-reports \
    && /opt/seedvr2-venv/bin/python /opt/npa-tools/filter_cudnn_runtime.py \
      > /opt/seedvr2-build-reports/cudnn-runtime.json \
    && /opt/seedvr2-venv/bin/python /opt/npa-tools/filter_nvshmem_runtime.py \
      > /opt/seedvr2-build-reports/nvshmem-runtime.json

ARG FLASH_ATTN_CUDA_ARCHS=90
COPY --chmod=0555 docker/workbench/seedvr2/cuda_arches.py /opt/npa-tools/cuda_arches.py
RUN /opt/seedvr2-venv/bin/python /opt/npa-tools/cuda_arches.py \
      --flash-arches "${FLASH_ATTN_CUDA_ARCHS}" \
      --torch-arches "${TORCH_CUDA_ARCH_LIST}" >/dev/null

RUN curl --fail --location \
      "https://files.pythonhosted.org/packages/source/f/flash-attn/flash_attn-${FLASH_ATTN_VERSION}.tar.gz" \
      -o /tmp/flash-attn.tar.gz \
    && echo "${FLASH_ATTN_SHA256}  /tmp/flash-attn.tar.gz" | sha256sum -c - \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE \
      FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
      MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
      NVCC_THREADS="${FLASH_ATTN_NVCC_THREADS}" \
      pip install --no-cache-dir --no-index --no-build-isolation --no-deps \
      /tmp/flash-attn.tar.gz \
    && rm -f /tmp/flash-attn.tar.gz

LABEL npa.build.flash-attn.cuda-archs="${FLASH_ATTN_CUDA_ARCHS}"

RUN curl --fail --location \
      "https://codeload.github.com/NVIDIA/apex/tar.gz/${APEX_SOURCE_REF}" \
      -o /tmp/apex.tar.gz \
    && echo "${APEX_SOURCE_SHA256}  /tmp/apex.tar.gz" | sha256sum -c - \
    && mkdir /tmp/apex \
    && tar -xzf /tmp/apex.tar.gz --strip-components=1 -C /tmp/apex \
    && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="${APEX_MAX_JOBS}" \
      pip install --no-cache-dir --no-index --no-build-isolation --no-deps \
      /tmp/apex \
    && rm -rf /tmp/apex /tmp/apex.tar.gz

RUN pip check && python - <<'PY'
import os
import torch
import torchvision
from apex.normalization import FusedLayerNorm, FusedRMSNorm
from flash_attn import flash_attn_varlen_func
assert torch.version.cuda == "13.0", torch.version.cuda
expected_arches = {
    "sm_" + value.replace(".", "")
    for value in os.environ["TORCH_CUDA_ARCH_LIST"].split(";")
}
assert expected_arches <= set(torch._C._cuda_getArchFlags().split())
assert torchvision.__version__.startswith("0.28.0")
assert FusedLayerNorm and FusedRMSNorm and flash_attn_varlen_func
PY

COPY --chmod=0555 scripts/measure_extension_arches.py /opt/npa-tools/measure_extension_arches.py
RUN ARCH_ARGS="$(/opt/seedvr2-venv/bin/python /opt/npa-tools/cuda_arches.py \
      --flash-arches "${FLASH_ATTN_CUDA_ARCHS}" \
      --torch-arches "${TORCH_CUDA_ARCH_LIST}")" \
    && read -r -a ARCH_ARGS <<< "$ARCH_ARGS" \
    && install -d -m 0755 /usr/share/doc/npa-seedvr2/extension-arches \
    && /opt/seedvr2-venv/bin/python /opt/npa-tools/measure_extension_arches.py \
      --distribution flash-attn \
      --min-size-mb 0 --skip-no-fatbin "${ARCH_ARGS[@]}" --json \
      > /usr/share/doc/npa-seedvr2/extension-arches/flash-attn.json \
    && /opt/seedvr2-venv/bin/python /opt/npa-tools/measure_extension_arches.py \
      --distribution apex \
      --min-size-mb 0 --skip-no-fatbin "${ARCH_ARGS[@]}" --json \
      > /usr/share/doc/npa-seedvr2/extension-arches/apex.json

COPY --chmod=0444 docker/workbench/seedvr2/npa_seedvr2_video_io.py /opt/npa-tools/npa_seedvr2_video_io.py
COPY --chmod=0555 docker/workbench/seedvr2/patch_video_io.py /opt/npa-tools/patch_video_io.py
RUN curl --fail --location \
      "https://codeload.github.com/ByteDance-Seed/SeedVR/tar.gz/${SEEDVR2_SOURCE_REF}" \
      -o /tmp/seedvr2.tar.gz \
    && echo "${SEEDVR2_SOURCE_SHA256}  /tmp/seedvr2.tar.gz" | sha256sum -c - \
    && mkdir /opt/seedvr2 \
    && tar -xzf /tmp/seedvr2.tar.gz --strip-components=1 -C /opt/seedvr2 \
    && test -s /opt/seedvr2/LICENSE \
    && printf '%s\n' "${SEEDVR2_SOURCE_REF}" > /opt/seedvr2/NPA_SOURCE_REVISION \
    && /usr/bin/python3.12 -I -S /opt/npa-tools/patch_video_io.py \
      --entrypoint /opt/seedvr2/projects/inference_seedvr2_3b.py \
      --adapter /opt/npa-tools/npa_seedvr2_video_io.py \
      > /opt/seedvr2-build-reports/video-io-compat.json \
    && rm -rf /opt/seedvr2/.git \
    && rm -f \
      /opt/seedvr2/neg_emb.pt \
      /opt/seedvr2/pos_emb.pt \
      /tmp/seedvr2.tar.gz \
    && test -z "$(find /opt/seedvr2 -type f \
      \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.safetensors' \
         -o -iname '*.ckpt' -o -iname 'pytorch_model*.bin' \) -print -quit)"

RUN ffmpeg -hide_banner -loglevel error -y \
      -f lavfi -i testsrc=size=32x16:rate=5 -frames:v 3 \
      -c:v libx264 -pix_fmt yuv420p /tmp/seedvr2-video-io-smoke.mp4 \
    && PYTHONPATH=/opt/seedvr2 /opt/seedvr2-venv/bin/python - <<'PY'
from pathlib import Path

from npa_seedvr2_video_io import read_video
from torchvision.io import read_image

path = Path("/tmp/seedvr2-video-io-smoke.mp4")
assert read_image
video, audio, info = read_video(path, output_format="TCHW")
assert tuple(video.shape) == (3, 3, 16, 32), video.shape
assert str(video.dtype) == "torch.uint8", video.dtype
assert tuple(audio.shape) == (1, 0), audio.shape
assert info == {"video_fps": 5.0}, info
source = Path("/opt/seedvr2/projects/inference_seedvr2_3b.py").read_text()
assert "from npa_seedvr2_video_io import read_video" in source
assert "from torchvision.io.video import read_video" not in source
path.unlink()
PY

# The ML installer is a build dependency, not an inference dependency. Remove
# it before exporting this environment so its vendored code is absent from
# every final-image layer, not merely hidden by a later whiteout.
FROM seedvr2-build AS seedvr2-ml-runtime
RUN /opt/seedvr2-venv/bin/python -m pip uninstall --yes pip \
    && /opt/seedvr2-venv/bin/python -c \
      'import importlib.util; assert importlib.util.find_spec("pip") is None'

# Build-only upstream tooling never enters the final image. The source-identified
# derivative also seeds ordinary venv creation through Ubuntu's ensurepip path.
FROM seedvr2-build AS seedvr2-bootstrap
COPY docker/workbench/common/secure_pip /opt/secure-pip
RUN /usr/bin/python3.12 /opt/secure-pip/build.py \
      --work-dir /tmp/secure-pip-build --output-dir /opt/secure-pip-wheels
# The limited pkg_resources donor is not a full setuptools bootstrap seed.
COPY docker/workbench/seedvr2/setuptools-seed.lock /opt/setuptools-seed.lock
RUN /opt/seedvr2-venv/bin/python -m pip --isolated download \
      --no-cache-dir --no-deps --only-binary=:all: --require-hashes \
      --index-url https://pypi.org/simple \
      --requirement /opt/setuptools-seed.lock --dest /opt/secure-pip-wheels

FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:4d242f206abc4b9588a6506cce2d88932cc879849395aae3785075179718cc49 AS seedvr2-runtime-producer

ARG SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185
ARG FLASH_ATTN_MAX_JOBS=2
ARG FLASH_ATTN_NVCC_THREADS=1
ARG FLASH_ATTN_CUDA_ARCHS=90
ARG APEX_MAX_JOBS=2
ARG TORCH_CUDA_ARCH_LIST=9.0

LABEL npa.tool="seedvr2" \
      npa.version="0.1.0-cu130-unbuilt" \
      npa.source.repo="https://github.com/ByteDance-Seed/SeedVR" \
      npa.source.ref="${SEEDVR2_SOURCE_REF}" \
      npa.weights="operator-runtime-fetch" \
      npa.build.flash-attn.max-jobs="${FLASH_ATTN_MAX_JOBS}" \
      npa.build.flash-attn.nvcc-threads="${FLASH_ATTN_NVCC_THREADS}" \
      npa.build.flash-attn.cuda-archs="${FLASH_ATTN_CUDA_ARCHS}" \
      npa.build.apex.max-jobs="${APEX_MAX_JOBS}" \
      npa.build.torch.cuda-arch-list="${TORCH_CUDA_ARCH_LIST}" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-Toolkit"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    SEEDVR2_PYTHON=/opt/seedvr2-venv/bin/python \
    NPA_SEEDVR2_WORK_DIR=/workspace/seedvr2-runs \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_LIGHT_WORKBENCH_TOOL=seedvr2 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH=/opt/npa-venv/bin:/opt/seedvr2-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root

RUN --mount=type=bind,from=seedvr2-bootstrap,source=/opt/secure-pip-wheels,target=/opt/secure-pip-wheels,ro \
    rm -f /etc/apt/sources.list.d/*.list \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg libgl1 libglib2.0-0 openssh-server \
      python3.12 python3.12-venv rsync sudo \
    && echo '51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670  /opt/secure-pip-wheels/setuptools-84.0.0-py3-none-any.whl' | sha256sum -c - \
    && find /usr/share/python-wheels -maxdepth 1 -type f \
      \( -name 'pip-*.whl' -o -name 'setuptools-*.whl' \) -delete \
    && install -m 0644 /opt/secure-pip-wheels/*.whl /usr/share/python-wheels/ \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /opt/nvidia/nsight-compute \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

COPY --from=seedvr2-ml-runtime /opt/seedvr2-venv /opt/seedvr2-venv
COPY --from=seedvr2-build /opt/seedvr2 /opt/seedvr2
COPY --from=seedvr2-build /usr/share/doc/npa-seedvr2/extension-arches /usr/share/doc/npa-seedvr2/extension-arches
COPY --from=seedvr2-build /opt/seedvr2-build-reports/cudnn-runtime.json /usr/share/doc/npa-seedvr2/cudnn-runtime.json
COPY --from=seedvr2-build /opt/seedvr2-build-reports/nvshmem-runtime.json /usr/share/doc/npa-seedvr2/nvshmem-runtime.json
COPY --from=seedvr2-build /opt/seedvr2-build-reports/video-io-compat.json /usr/share/doc/npa-seedvr2/video-io-compat.json
COPY --from=seedvr2-build --chmod=0444 /opt/npa-legal/NVSHMEM-License-v3.4.5-0.txt /usr/share/doc/npa-seedvr2/NVSHMEM-License-v3.4.5-0.txt
COPY --from=seedvr2-bootstrap --chmod=0444 /opt/secure-pip-wheels/build-receipt.json /usr/share/doc/npa-seedvr2/secure-pip-build.json

# Keep the final installed-dependency gate, with its installer mounted only for
# this build instruction. The separate NPA service installer remains available
# for standard workflow bootstrap; the frozen ML environment does not use it.
RUN --mount=type=bind,from=seedvr2-build,source=/opt/seedvr2-venv/lib/python3.12/site-packages/pip,target=/opt/pip-check/pip,ro \
    /opt/seedvr2-venv/bin/python /opt/pip-check/pip check \
    && /opt/seedvr2-venv/bin/python - <<'PY'
import importlib.util
import os
import torch
import torchvision
import sys
from apex.normalization import FusedLayerNorm, FusedRMSNorm
from flash_attn import flash_attn_varlen_func
sys.path.insert(0, "/opt/seedvr2")
from npa_seedvr2_video_io import read_video
from torchvision.io import read_image
assert torch.version.cuda == "13.0", torch.version.cuda
expected_arches = {
    "sm_" + value.replace(".", "")
    for value in os.environ["TORCH_CUDA_ARCH_LIST"].split(";")
}
assert expected_arches <= set(torch._C._cuda_getArchFlags().split())
assert torchvision.__version__.startswith("0.28.0")
assert FusedLayerNorm and FusedRMSNorm and flash_attn_varlen_func
assert read_video and read_image
assert importlib.util.find_spec("pip") is None
PY

COPY --chmod=u=rwX,go=rX pyproject.toml README.md /opt/npa-src/
COPY --chmod=u=rwX,go=rX src/npa /opt/npa-src/src/npa
COPY docker/workbench/seedvr2/service-requirements.lock /tmp/seedvr2-service.lock
COPY --chmod=0555 docker/workbench/seedvr2/verify_no_weight_payloads.py /opt/npa-tools/verify_no_weight_payloads.py
RUN python3.12 -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/python -c \
      'import pip; from pip._vendor import urllib3; assert pip.__version__ == "26.2.1+npa.1"; assert urllib3.__version__ == "2.8.0"' \
    && /opt/npa-venv/bin/pip install --no-cache-dir --require-hashes \
      -r /tmp/seedvr2-service.lock \
    && /opt/npa-venv/bin/pip install --no-cache-dir --no-deps \
      --no-build-isolation /opt/npa-src \
    && /opt/npa-venv/bin/pip check \
    && /opt/npa-venv/bin/npa workbench seedvr2 restore \
      --input-path s3://dry-run/input.mp4 \
      --output-path s3://dry-run/output/ \
      --run-id image-build --dry-run >/dev/null \
    && find /opt/seedvr2-venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && rm -rf /root/.cache /tmp/apex /tmp/*.tar.gz /tmp/pip-* \
    && /usr/bin/python3.12 -I -S /opt/npa-tools/verify_no_weight_payloads.py \
      > /usr/share/doc/npa-seedvr2/weight-payload-scan.json

ARG NPA_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
RUN [[ "$NPA_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] \
    && [[ "$SOURCE_DATE_EPOCH" =~ ^[1-9][0-9]*$ ]] \
    && printf '%s\n' "$NPA_SOURCE_SHA" > /opt/npa-source-revision
LABEL org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

COPY --chmod=0444 docker/workbench/seedvr2/REDISTRIBUTION.md /usr/share/doc/npa-seedvr2/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/seedvr2/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-seedvr2/THIRD_PARTY_NOTICES.md
COPY --chmod=0444 docker/workbench/seedvr2/MCAP-1.4.0-LICENSE.txt /usr/share/doc/npa-seedvr2/MCAP-1.4.0-LICENSE.txt
COPY --chmod=0755 docker/workbench/seedvr2/entrypoint.sh /usr/local/bin/seedvr2-entrypoint

RUN install -d -m 0755 -o ubuntu -g ubuntu \
      /workspace /workspace/.cache /workspace/.cache/huggingface \
      /workspace/seedvr2-runs \
    && install -d -m 0700 -o ubuntu -g ubuntu /workspace/tmp \
    && chown -R ubuntu:ubuntu /workspace

# Export the repaired filesystem, not the producer's superseded ancestor layers.
# This is a canonical build stage, not a repack of historical image bytes.
# Restore the measured NVIDIA and NPA runtime config explicitly: COPY does not
# carry environment, launch, user, volume or healthcheck metadata across stages.
FROM scratch AS seedvr2-runtime
ARG NPA_SOURCE_SHA
ARG SEEDVR2_SOURCE_REF=e4de8c24441a67e1b7df56abea10645059bb1185
ARG FLASH_ATTN_MAX_JOBS=2
ARG FLASH_ATTN_NVCC_THREADS=1
ARG FLASH_ATTN_CUDA_ARCHS=90
ARG APEX_MAX_JOBS=2
ARG TORCH_CUDA_ARCH_LIST=9.0
COPY --from=seedvr2-runtime-producer / /
LABEL com.nvidia.cudnn.version="9.14.0.64-1" \
      npa.build.apex.max-jobs="${APEX_MAX_JOBS}" \
      npa.build.flash-attn.cuda-archs="${FLASH_ATTN_CUDA_ARCHS}" \
      npa.build.flash-attn.max-jobs="${FLASH_ATTN_MAX_JOBS}" \
      npa.build.flash-attn.nvcc-threads="${FLASH_ATTN_NVCC_THREADS}" \
      npa.build.torch.cuda-arch-list="${TORCH_CUDA_ARCH_LIST}" \
      npa.source.ref="${SEEDVR2_SOURCE_REF}" \
      npa.source.repo="https://github.com/ByteDance-Seed/SeedVR" \
      npa.tool="seedvr2" \
      npa.version="0.1.0-cu130-unbuilt" \
      npa.weights="operator-runtime-fetch" \
      org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause AND LicenseRef-NVIDIA-CUDA-Toolkit" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai"

ENV PATH="/opt/npa-venv/bin:/opt/seedvr2-venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    NVARCH="x86_64" \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576" \
    NV_CUDA_CUDART_VERSION="13.0.96-1" \
    CUDA_VERSION="13.0.2" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64" \
    NVIDIA_VISIBLE_DEVICES="all" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    NV_CUDA_LIB_VERSION="13.0.2-1" \
    NV_NVTX_VERSION="13.0.85-1" \
    NV_LIBNPP_VERSION="13.0.1.2-1" \
    NV_LIBNPP_PACKAGE="libnpp-13-0=13.0.1.2-1" \
    NV_LIBCUSPARSE_VERSION="12.6.3.3-1" \
    NV_LIBCUBLAS_PACKAGE_NAME="libcublas-13-0" \
    NV_LIBCUBLAS_VERSION="13.1.0.3-1" \
    NV_LIBCUBLAS_PACKAGE="libcublas-13-0=13.1.0.3-1" \
    NV_LIBNCCL_PACKAGE_NAME="libnccl2" \
    NV_LIBNCCL_PACKAGE_VERSION="2.28.3-1" \
    NCCL_VERSION="2.28.3-1" \
    NV_LIBNCCL_PACKAGE="libnccl2=2.28.3-1+cuda13.0" \
    NVIDIA_PRODUCT_NAME="CUDA" \
    NV_CUDNN_VERSION="9.14.0.64-1" \
    NV_CUDNN_PACKAGE_NAME="libcudnn9-cuda-13" \
    NV_CUDNN_PACKAGE="libcudnn9-cuda-13=9.14.0.64-1" \
    DEBIAN_FRONTEND="noninteractive" \
    PYTHONUNBUFFERED="1" \
    HF_HOME="/workspace/.cache/huggingface" \
    HF_HUB_CACHE="/workspace/.cache/huggingface/hub" \
    SEEDVR2_PYTHON="/opt/seedvr2-venv/bin/python" \
    NPA_SEEDVR2_WORK_DIR="/workspace/seedvr2-runs" \
    NPA_SKIP_EAGER_IMPORTS="1" \
    NPA_LIGHT_WORKBENCH_TOOL="seedvr2" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

VOLUME ["/workspace/.cache/huggingface"]
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
ENTRYPOINT ["seedvr2-entrypoint"]
CMD ["serve"]
