ARG BASE_IMAGE=npa-genesis:0.4.6-sm80-sm90-sm120-latest
FROM ${BASE_IMAGE}

LABEL npa.tool="envgen"

WORKDIR /opt/npa
USER root
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa

RUN python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py

ENV PATH=/opt/npa/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src

USER ubuntu
ENTRYPOINT ["/bin/bash"]
