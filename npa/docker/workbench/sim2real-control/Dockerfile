# Digest pinned 2026-06-02. Tag: python:3.11-slim-trixie.
FROM python:3.11-slim-trixie@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

ARG SIM2REAL_CONTROL_VERSION=0.1.0
ARG NPA_SOURCE_SHA

LABEL npa.tool="sim2real-control" \
      npa.version="${SIM2REAL_CONTROL_VERSION}" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_SKIP_EAGER_IMPORTS=1 \
    PYTHONPATH=/opt/npa/src \
    PYTHONUNBUFFERED=1

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /opt/npa

WORKDIR /workspace

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-controller-requirements.txt \
    /opt/npa/sim2real-controller-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip install --no-cache-dir --no-deps \
      -r /opt/npa/sim2real-controller-requirements.txt \
    && python -m pip check \
    && python -m py_compile \
      /opt/npa/src/npa/workflows/sim2real/workflow_io.py \
      /opt/npa/src/npa/workflows/sim2real/workflow_stage.py \
    && python -c "import boto3; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)" \
    && chown -R ubuntu:ubuntu /workspace

USER ubuntu

ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
