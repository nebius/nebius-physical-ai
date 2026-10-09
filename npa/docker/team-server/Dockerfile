# Package the CPU-only team gateway and its isolated, pinned scheduler client.
ARG KUBECTL_IMAGE=registry.k8s.io/kubectl:v1.36.5@sha256:0667e3f3e6ef435d7cd4c16684244703ebd67314c17511c2a9f87ea44f6cdf6b
FROM ${KUBECTL_IMAGE} AS kubectl
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG NPA_REVISION
LABEL org.opencontainers.image.revision=${NPA_REVISION}
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
      'kubernetes>=20.0.0,!=32.0.0,<36' \
    && useradd --uid 1000 --create-home workbench
ENV PATH=/opt/npa-venv/bin:/usr/local/bin:/usr/bin:/bin
USER 1000:1000
ENTRYPOINT ["npa", "workbench", "team", "serve"]
CMD ["--config", "/etc/npa-team/team.yaml", "--host", "0.0.0.0"]
