# MIT flex-pi source/runtime only. Checkpoints and public observation bytes are
# fetched into an operator-owned runtime cache and never baked into this image.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9 AS upstream-source

ARG FLEX_PI_SOURCE_URL=https://github.com/geyan21/flex-pi.git
ARG FLEX_PI_SOURCE_REF=20c1b2b71ea35a415d5d47c39b04443cfadad7a1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none --no-checkout "${FLEX_PI_SOURCE_URL}" /tmp/flex-pi-source \
    && git -C /tmp/flex-pi-source checkout --detach "${FLEX_PI_SOURCE_REF}" \
    && test "$(git -C /tmp/flex-pi-source rev-parse HEAD)" = "${FLEX_PI_SOURCE_REF}" \
    && test -s /tmp/flex-pi-source/LICENSE

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

ARG FLEX_PI_SOURCE_URL=https://github.com/geyan21/flex-pi.git
ARG FLEX_PI_SOURCE_REF=20c1b2b71ea35a415d5d47c39b04443cfadad7a1
ARG NPA_SOURCE_COMMIT=unknown

LABEL npa.tool="flex-pi" \
      npa.version="0.1.0-cu128" \
      npa.source.repo="${FLEX_PI_SOURCE_URL}" \
      npa.source.ref="${FLEX_PI_SOURCE_REF}" \
      npa.source.commit="${NPA_SOURCE_COMMIT}" \
      npa.weights="operator-runtime-fetch" \
      npa.dataset="operator-runtime-fetch" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/.cache/modelscope \
    DIFFSYNTH_MODEL_BASE_PATH=/workspace/.cache/flex-pi/diffsynth \
    IMAGEIO_FFMPEG_EXE=/opt/conda/bin/ffmpeg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/npa-venv/bin:/opt/conda/bin:${PATH}

SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git openssh-server patch rsync sudo \
    && apt-get install -y --only-upgrade linux-libc-dev \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && rm -rf /opt/nvidia/nsight-compute/2025.1.0 \
    && conda install -y -c conda-forge 'ffmpeg=7' \
    && conda clean -afy \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime

RUN install -d /opt/flex-pi/experiments/robotwin/flexpi_policy
COPY --from=upstream-source /tmp/flex-pi-source/LICENSE /tmp/flex-pi-source/pyproject.toml /opt/flex-pi/
COPY --from=upstream-source /tmp/flex-pi-source/src /opt/flex-pi/src
COPY --from=upstream-source /tmp/flex-pi-source/configs /opt/flex-pi/configs
COPY --from=upstream-source /tmp/flex-pi-source/experiments/robotwin/flexpi_policy/deploy_policy.py /opt/flex-pi/experiments/robotwin/flexpi_policy/deploy_policy.py
COPY docker/workbench/flex-pi/pins/dino-checkpoint.patch /tmp/dino-checkpoint.patch
COPY docker/workbench/flex-pi/pins/security-dependencies.patch /tmp/security-dependencies.patch

RUN patch --directory=/opt/flex-pi --strip=1 --input=/tmp/dino-checkpoint.patch \
    && patch --directory=/opt/flex-pi --strip=1 --input=/tmp/security-dependencies.patch \
    && rm /tmp/dino-checkpoint.patch /tmp/security-dependencies.patch \
    && pip install -e /opt/flex-pi --extra-index-url https://download.pytorch.org/whl/cu128 \
    && pip install --force-reinstall --no-deps --index-url https://pypi.org/simple torchcodec==0.5 \
    && rm -f /opt/conda/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 \
    && python -c 'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/opt/conda/bin/ffmpeg"' \
    && pip check \
    && test -z "$(find /opt/flex-pi -type f \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname '*.pt' -o -iname '*.mp4' -o -iname '*.parquet' \) -print -quit)" \
    && python - <<'PY'
import torch
flags = set(torch._C._cuda_getArchFlags().split())
assert 'sm_120' in flags, flags
PY

COPY pyproject.toml README.md /opt/npa-src/
COPY src/npa /opt/npa-src/src/npa
RUN python -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/pip install 'uvicorn==0.41.0' /opt/npa-src \
    && /opt/npa-venv/bin/pip check

COPY --chmod=0555 docker/workbench/flex-pi/inference.py /opt/flex-pi/npa/inference.py
COPY --chmod=0444 docker/workbench/flex-pi/public_robotwin_sample.json /opt/flex-pi/npa/public_robotwin_sample.json
COPY --chmod=0555 docker/workbench/flex-pi/entrypoint.sh /usr/local/bin/flex-pi-entrypoint
COPY --chmod=0444 docker/workbench/flex-pi/REDISTRIBUTION.md /usr/share/doc/npa-flex-pi/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/flex-pi/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-flex-pi/THIRD_PARTY_NOTICES.md

RUN install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/.cache \
      /workspace/.cache/huggingface /workspace/.cache/modelscope /workspace/.cache/flex-pi \
    && chown -R ubuntu:ubuntu /workspace \
    && /opt/npa-venv/bin/npa workbench flex-pi infer \
      --input-path /opt/flex-pi/npa/public_robotwin_sample.json \
      --output-path s3://dry-run.invalid/flex-pi-plan/ --dry-run >/dev/null

VOLUME ["/workspace/.cache"]
WORKDIR /workspace
USER ubuntu
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["/opt/npa-venv/bin/python", "-c", "import urllib.request; r=urllib.request.Request('http://127.0.0.1:8080/health', headers={'Authorization':'Bearer '+__import__('os').environ.get('NPA_FLEX_PI_TOKEN','')}); urllib.request.urlopen(r, timeout=3)"]
ENTRYPOINT ["flex-pi-entrypoint"]
CMD ["serve"]
