# npa-hy-world is a zero-Tencent-payload bootstrap, not a packaged HY-World
# distribution. Tencent's Community License covers both its source/inference
# code and its model weights, so they are fetched only by an eligible operator
# into a mounted runtime cache after an external terms/access preflight.
#
# This is deliberately a CUDA *devel* image: the released upstream worldgen
# path compiles its own gsplat, NavMesh, PyTorch3D, and FlashAttention extensions
# during the authorized operator runtime bootstrap.  It contains no Tencent or
# model component bytes.
# Digest pinned 2026-10-04. Tag: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed

ARG NPA_SOURCE_COMMIT=unknown
ARG NPA_RUNTIME_USER=ubuntu
ARG UV_VERSION=0.11.15

LABEL npa.tool="hy-world" \
      npa.version="2.0-rtfetch-unbuilt" \
      npa.source.repo="https://github.com/Tencent-Hunyuan/HY-World-2.0" \
      npa.source.ref="df9988efb87bfc0f4947eb3889411cf957478b06" \
      npa.source.provisioning="operator-runtime-fetch" \
      npa.weights="runtime-only" \
      npa.license.upstream="Tencent HY-WORLD 2.0 Community License Agreement (2026-04-15)" \
      npa.license.payload="none-baked" \
      npa.security.runtime-user="uid-1000-with-skypilot-bootstrap-sudo" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    NPA_HY_WORLD_RUNTIME_CACHE=/workspace/model-cache/huggingface/hy-world/runtime \
    HF_HOME=/workspace/model-cache/huggingface \
    UV_PYTHON_INSTALL_DIR=/workspace/model-cache/huggingface/hy-world/runtime/uv-python \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake ffmpeg git libgl1 libglib2.0-0 \
      ninja-build pkg-config python3-pip \
      openssh-server procps rsync sudo util-linux \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u "${NPA_RUNTIME_USER}" >/dev/null 2>&1 \
       || useradd -m -s /bin/bash -u 1000 "${NPA_RUNTIME_USER}") \
    && printf '%s\n' "${NPA_RUNTIME_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

# uv and huggingface_hub are generic bootstrap clients. They contain no Tencent,
# Qwen, WorldStereo, CUDA, PyTorch, or checkpoint payload.
RUN python3 -m pip install --no-cache-dir \
      "uv==${UV_VERSION}" "huggingface-hub[cli]==0.36.0" \
    && rm -rf /root/.cache

COPY --chmod=0444 src/npa/workbench/hy_world/asset_contract.py /opt/npa/hy-world/asset_contract.py
COPY --chmod=0555 docker/workbench/hy-world/validate_scene.py /opt/npa/hy-world/validate_scene.py
COPY --chmod=0555 docker/workbench/hy-world/hy_world_report.py /opt/npa/hy-world/hy_world_report.py
COPY --chmod=0755 docker/workbench/hy-world/hy_world_runtime.sh /usr/local/bin/hy-world-runtime
COPY --chmod=0755 docker/workbench/hy-world/entrypoint.sh /usr/local/bin/hy-world-entrypoint
COPY --chmod=0755 docker/workbench/hy-world/golden_smoke.sh /usr/local/bin/npa-hy-world-golden-smoke
COPY --chmod=0444 docker/workbench/hy-world/REDISTRIBUTION.md /usr/share/doc/npa-hy-world/REDISTRIBUTION.md

RUN install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace /workspace/model-cache/huggingface/hy-world/runtime /opt/byof \
    && printf '%s\n' \
      '{"source":"operator-runtime-fetch","repo":"https://github.com/Tencent-Hunyuan/HY-World-2.0","ref":"df9988efb87bfc0f4947eb3889411cf957478b06"}' \
      > /opt/byof/npa_source_metadata.json \
    && chown "${NPA_RUNTIME_USER}:${NPA_RUNTIME_USER}" /opt/byof/npa_source_metadata.json \
    && chmod 0444 /opt/byof/npa_source_metadata.json \
    && chmod 0755 /opt/npa /opt/npa/hy-world /usr/share/doc/npa-hy-world \
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'hy-world-runtime health' \
    && su "${NPA_RUNTIME_USER}" -s /bin/bash -c 'hy-world-runtime payload-absence' \
    && /opt/npa/hy-world/validate_scene.py --help >/dev/null \
    && /opt/npa/hy-world/hy_world_report.py --help >/dev/null \
    && test -z "$(find /workspace -type f -print -quit)" \
    && test -z "$(find / -xdev -name '*.safetensors' -print -quit)" \
    && test -z "$(find / -xdev \( -name 'torch' -o -name 'nvidia' \) -type d -path '*site-packages*' -print -quit)"

VOLUME ["/workspace/model-cache/huggingface"]
WORKDIR /workspace
USER ${NPA_RUNTIME_USER}
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["hy-world-runtime", "health"]
ENTRYPOINT ["hy-world-entrypoint"]
