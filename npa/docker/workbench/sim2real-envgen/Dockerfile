ARG BASE_IMAGE=npa-genesis:cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
LABEL npa.tool="envgen" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

WORKDIR /opt/npa
USER root
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py

ENV PATH=/opt/npa/venv/bin:$PATH \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    PYTHONPATH=/opt/npa/src

USER ubuntu
ENTRYPOINT ["/bin/bash"]
