# Immutable Python base: python:3.11-slim-trixie as resolved on 2026-06-02.
# This is the CPU-only cloud evidence worker, never the Jetson TensorRT runtime.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG NPA_SOURCE_SHA

LABEL org.opencontainers.image.title="NPA MolmoAct2 Jetson-Thor evidence worker" \
      org.opencontainers.image.description="Private cloud verifier for native Thor artifacts; it contains no policy or TensorRT payload" \
      org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      npa.solution="molmoact2-jetson-thor-edge" \
      npa.redistribution="unvalidated-private-byof" \
      npa.target-runtime="operator-installed-native-jetson-only"

ENV PATH=/opt/venv/bin:$PATH \
    NPA_BAKED_PYTHON=/opt/venv/bin/python \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/npa/src \
    PYTHONUNBUFFERED=1

RUN test "${#NPA_SOURCE_SHA}" -eq 40 \
    && case "${NPA_SOURCE_SHA}" in *[!0-9a-f]*) exit 64 ;; esac \
    && useradd --create-home --shell /bin/bash --uid 1000 ubuntu \
    && install -d --mode=0755 --owner=ubuntu --group=ubuntu /opt/npa /opt/venv /work

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu docker/workbench/molmoact2-jetson-thor/THIRD_PARTY_NOTICES.md \
    /usr/share/doc/molmoact2-jetson-thor/THIRD_PARTY_NOTICES.md
COPY --chown=ubuntu:ubuntu docker/workbench/molmoact2-jetson-thor/REDISTRIBUTION.md \
    /usr/share/doc/molmoact2-jetson-thor/REDISTRIBUTION.md

USER ubuntu
WORKDIR /work

RUN python -m venv /opt/venv \
    && python -m pip install --no-cache-dir \
      boto3==1.43.108 \
      numpy==2.2.6 \
      rerun-sdk==0.38.1 \
    && python -m pip check \
    && python -c 'import boto3, numpy, rerun; from npa.workflows import molmoact2_jetson_thor; assert callable(molmoact2_jetson_thor.main)'

CMD ["/opt/venv/bin/python", "-m", "npa.workflows.molmoact2_jetson_thor", "--help"]
