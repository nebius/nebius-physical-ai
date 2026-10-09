# Package the CPU-only team gateway and its isolated, pinned scheduler client.
ARG KUBERNETES_SOURCE_REVISION=ad950d1cc78b0183c476bd4d3f1934c104229727
FROM golang:1.26.9-bookworm@sha256:d9c68c2c51161e12fd77e4c6320687c9cd86e1af1e3ad6e6cd63ff970641453c AS kubectl
ARG KUBERNETES_SOURCE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/* \
    && git init /src \
    && git -C /src remote add origin https://github.com/kubernetes/kubernetes.git \
    && git -C /src fetch --depth=1 origin refs/tags/v1.36.5 \
    && git -C /src checkout FETCH_HEAD \
    && test "$(git -C /src rev-parse HEAD)" = "$KUBERNETES_SOURCE_REVISION" \
    && make -C /src WHAT=cmd/kubectl
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG NPA_REVISION
LABEL org.opencontainers.image.revision=${NPA_REVISION}
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git openssh-client rsync socat netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=kubectl /src/_output/local/go/bin/kubectl /usr/local/bin/kubectl
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
    && useradd --uid 1000 --create-home workbench \
    && chown workbench:workbench \
      /opt/sky/lib/python3.12/site-packages/sky/utils/kubernetes/rsync_helper.sh
ENV PATH=/opt/npa-venv/bin:/usr/local/bin:/usr/bin:/bin
USER 1000:1000
ENTRYPOINT ["npa", "workbench", "team", "serve"]
CMD ["--config", "/etc/npa-team/team.yaml", "--host", "0.0.0.0"]
