# Neutral bootstrap: SAM source, weights and CUDA packages are runtime fetches.
FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ARG NPA_SOURCE_SHA=unknown
LABEL npa.tool="sam3" \
      npa.version="3.1-rtfetch" \
      npa.source.provisioning="operator-runtime-fetch" \
      npa.weights="runtime-only" \
      npa.license.payload="none-baked" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NPA_SAM3_CACHE=/workspace/.cache/npa/sam3

COPY docker/workbench/sam3/bootstrap-requirements.lock /opt/npa/sam3/bootstrap-requirements.lock
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git openssh-server procps rsync sudo util-linux \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && useradd -m -s /bin/bash -u 1000 ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && python -m pip install --no-cache-dir --require-hashes -r /opt/npa/sam3/bootstrap-requirements.lock \
    && rm -rf /root/.cache

COPY docker/workbench/sam3/runtime.py /opt/npa/sam3/runtime.py
COPY docker/workbench/sam3/segment.py /opt/npa/sam3/segment.py
COPY docker/workbench/sam3/entrypoint.sh /opt/npa/sam3/entrypoint.sh
COPY docker/workbench/sam3/pins.json /opt/npa/sam3/pins.json
COPY docker/workbench/sam3/requirements.in /opt/npa/sam3/requirements.in
COPY docker/workbench/sam3/requirements.lock /opt/npa/sam3/requirements.lock
COPY docker/workbench/sam3/README.md /opt/npa/sam3/README.md
COPY docker/workbench/sam3/REDISTRIBUTION.md /opt/npa/sam3/REDISTRIBUTION.md
RUN chmod 0755 /opt/npa /opt/npa/sam3 /opt/npa/sam3/runtime.py /opt/npa/sam3/entrypoint.sh \
    && ln -s /opt/npa/sam3/runtime.py /usr/local/bin/sam3-runtime \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace/.cache/npa/sam3 \
    && su ubuntu -s /bin/bash -c 'sam3-runtime health' \
    && su ubuntu -s /bin/bash -c 'sam3-runtime assert-refusal' \
    && test -z "$(find /workspace -type f -print -quit)"

VOLUME ["/workspace/.cache/npa/sam3"]
WORKDIR /workspace
USER ubuntu
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["sam3-runtime", "health"]
ENTRYPOINT ["/opt/npa/sam3/entrypoint.sh"]
CMD ["health"]
