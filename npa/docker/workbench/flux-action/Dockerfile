# Native Apache-2.0 FLUX Action source; model weights and datasets are runtime-only.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134
ARG NPA_SOURCE_SHA
ARG SOURCE_DATE_EPOCH
ARG FLUX_SOURCE_REF=e2dd1d8dbc5977b54315d61f7548c63c043d6d4f
LABEL npa.tool="flux-action" \
      npa.version="0.1.0-cuda12-unbuilt" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV PYTHONUNBUFFERED=1 \
    PATH=/opt/npa-venv/bin:${PATH} \
    NPA_BAKED_PYTHON=/opt/npa-venv/bin/python \
    NPA_FLUX_ACTION_ROOT=/opt/flux-action \
    HF_HOME=/workspace/.cache/huggingface
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ffmpeg openssh-server rsync sudo \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime
RUN git clone --filter=blob:none --no-checkout https://github.com/black-forest-labs/flux-action.git /opt/flux-action \
    && git -C /opt/flux-action checkout --detach "${FLUX_SOURCE_REF}" \
    && test "$(git -C /opt/flux-action rev-parse HEAD)" = "${FLUX_SOURCE_REF}" \
    && printf '%s\n' "${FLUX_SOURCE_REF}" > /opt/flux-action/SOURCE_REVISION \
    && rm -rf /opt/flux-action/.git \
    && python3 -m venv /opt/uv \
    && /opt/uv/bin/pip install --no-cache-dir 'uv==0.11.15' \
    && cd /opt/flux-action \
    && /opt/uv/bin/uv sync --locked --no-dev --extra encoders --extra data \
    && /opt/uv/bin/uv pip install --python .venv/bin/python \
       'natten==0.21.6+torch2100cu128' --find-links https://whl.natten.org/
COPY pyproject.toml README.md /opt/npa/
COPY src/npa /opt/npa/src/npa
RUN python3 -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/pip install --no-cache-dir /opt/npa \
    && rm -rf /root/.cache /tmp/uv-cache \
    && install -d -o ubuntu -g ubuntu /workspace/.cache/huggingface
COPY docker/workbench/flux-action/REDISTRIBUTION.md /usr/share/doc/npa-flux-action/REDISTRIBUTION.md
ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}
WORKDIR /workspace
USER ubuntu
CMD ["npa", "workbench", "flux-action", "--help"]
