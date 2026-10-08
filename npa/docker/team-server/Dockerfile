# Package the CPU-only team gateway and its isolated, pinned scheduler client.
ARG KUBECTL_IMAGE=registry.k8s.io/kubectl:v1.36.5
FROM ${KUBECTL_IMAGE} AS kubectl
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
COPY --from=kubectl /bin/kubectl /usr/local/bin/kubectl
COPY npa/pyproject.toml /opt/npa/pyproject.toml
COPY npa/src /opt/npa/src
COPY workflows/main /opt/workflows/main
COPY workflows/testing /opt/workflows/testing
COPY workflows/partners /opt/workflows/partners
RUN python -m venv /opt/npa-venv \
    && /opt/npa-venv/bin/pip install --no-cache-dir /opt/npa 'uvicorn>=0.52.4' \
    && python -m venv /opt/sky \
    && /opt/sky/bin/pip install --no-cache-dir 'skypilot[kubernetes]==0.12.2' \
    && useradd --uid 1000 --create-home workbench
ENV PATH=/opt/npa-venv/bin:/usr/local/bin:/usr/bin:/bin
USER 1000:1000
ENTRYPOINT ["npa", "workbench", "team", "serve"]
CMD ["--config", "/etc/npa-team/team.yaml", "--host", "0.0.0.0"]
