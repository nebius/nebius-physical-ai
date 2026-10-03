# Neutral OpenVLA-OFT bootstrap: no OFT source, model, adapter, RLDS data,
# CUDA/Torch runtime, credentials, or caches are image bytes. A minimal NPA
# control path is included solely to execute hash-verified artifact handoffs.
FROM python:3.10-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc

ARG NPA_SOURCE_COMMIT=unknown
ARG NPA_RUNTIME_USER=ubuntu

LABEL npa.tool="openvla-oft" \
      npa.version="runtime-fetch-unvalidated" \
      npa.source.repo="https://github.com/moojink/openvla-oft.git" \
      npa.source.ref="e4287e94541f459edc4feabc4e181f537cd569a8" \
      npa.weights="runtime-only" \
      npa.data="runtime-only" \
      org.opencontainers.image.revision="${NPA_SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    NPA_OPENVLA_OFT_RUNTIME_ROOT=/workspace/.cache/npa/openvla-oft \
    PYTHONPATH=/opt/npa-runtime \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates ffmpeg git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u "${NPA_RUNTIME_USER}" >/dev/null 2>&1 \
        || useradd -m -s /bin/bash -u 1000 "${NPA_RUNTIME_USER}") \
    && install -d -m 0755 -o "${NPA_RUNTIME_USER}" -g "${NPA_RUNTIME_USER}" \
      /workspace /workspace/.cache/npa/openvla-oft /opt/npa/openvla-oft \
    && install -d -m 0755 -o root -g root /usr/share/doc/npa-openvla-oft

COPY docker/workbench/openvla-oft/control-requirements.lock /tmp/openvla-oft-control-requirements.lock
RUN python3 -m pip install --no-cache-dir --require-hashes \
      -r /tmp/openvla-oft-control-requirements.lock \
    && rm -f /tmp/openvla-oft-control-requirements.lock \
    && mkdir -p \
      /opt/npa-runtime/npa/clients \
      /opt/npa-runtime/npa/workflows/byof \
      /opt/npa-runtime/npa/workbench/cosmos \
      /opt/npa-runtime/npa/workbench/dataset \
      /opt/npa-runtime/npa/workbench \
    && touch \
      /opt/npa-runtime/npa/__init__.py \
      /opt/npa-runtime/npa/clients/__init__.py \
      /opt/npa-runtime/npa/workflows/__init__.py \
      /opt/npa-runtime/npa/workflows/byof/__init__.py \
      /opt/npa-runtime/npa/workbench/__init__.py \
      /opt/npa-runtime/npa/workbench/cosmos/__init__.py \
      /opt/npa-runtime/npa/workbench/dataset/__init__.py
COPY src/npa/clients/storage.py /opt/npa-runtime/npa/clients/storage.py
COPY src/npa/workflows/byof/openvla_pipeline.py /opt/npa-runtime/npa/workflows/byof/openvla_pipeline.py
COPY src/npa/workbench/cosmos/policy_artifacts.py /opt/npa-runtime/npa/workbench/cosmos/policy_artifacts.py
COPY src/npa/workbench/dataset/storage.py /opt/npa-runtime/npa/workbench/dataset/storage.py
COPY src/npa/workbench/storage_scope.py /opt/npa-runtime/npa/workbench/storage_scope.py
COPY --chmod=0555 docker/workbench/openvla-oft/entrypoint.py /opt/npa/openvla-oft/entrypoint.py
COPY --chmod=0444 docker/workbench/openvla-oft/REDISTRIBUTION.md /usr/share/doc/npa-openvla-oft/REDISTRIBUTION.md
COPY --chmod=0444 docker/workbench/openvla-oft/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-openvla-oft/THIRD_PARTY_NOTICES.md
RUN chown -R "${NPA_RUNTIME_USER}:${NPA_RUNTIME_USER}" /opt/npa-runtime \
    && find /opt/npa-runtime -type d -exec chmod 0755 {} + \
    && find /opt/npa-runtime -type f -exec chmod 0644 {} +

WORKDIR /workspace
USER ${NPA_RUNTIME_USER}
ENTRYPOINT ["python3", "/opt/npa/openvla-oft/entrypoint.py"]
CMD ["python3", "--version"]
