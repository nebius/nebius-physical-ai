# Public SkyPilot wrapper with runtime-fetched CUDA, vLLM, and model payloads.
ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-serving@sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180
FROM ${BASE_IMAGE}

ARG VLLM_OMNI_REVISION=eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c
ARG VLLM_OMNI_SOURCE_SHA256=a0380c02b403f3c71ba4360fd23fc2461ea8a84c28d5543b8ac4c08a31a97a3f
ARG COSMOS3_CLOSURE_SHA256=c13b5b13868fa45bb125e0f7f32ec718bcb10e554ce34b85af944dbd4bafc6aa
ARG DEBIAN_SNAPSHOT=20260907T000000Z
ENV NPA_COSMOS3_CLOSURE_SHA256=${COSMOS3_CLOSURE_SHA256} \
    NPA_VLLM_OMNI_REVISION=${VLLM_OMNI_REVISION} \
    NPA_VLLM_OMNI_SOURCE_SHA256=${VLLM_OMNI_SOURCE_SHA256}

USER root
# The accepted parent's older runtime recipe is replaced by these reviewed files.
COPY --chmod=0644 docker/workbench/sonic/packaging-requirements.txt /opt/npa-cosmos3-serving/packaging-requirements.txt
COPY --chmod=0644 docker/workbench/cosmos3-serving/requirements.lock /opt/npa-cosmos3-serving/requirements.lock
COPY --chmod=0755 docker/workbench/cosmos3-serving/runtime_bootstrap.sh \
     docker/workbench/cosmos3-serving/entrypoint.sh \
     docker/workbench/cosmos3-serving/smoke_serving.sh /opt/npa-cosmos3-serving/
COPY --chmod=0644 docker/workbench/cosmos3-serving/access_preflight.py \
     docker/workbench/cosmos3-serving/hf_snapshot_pin.py \
     docker/workbench/cosmos3-serving/prepare_guardrail_runtime.py \
     docker/workbench/cosmos3-serving/backport_setuptools_manifest.py \
     docker/workbench/cosmos3-serving/verify_env.py /opt/npa-cosmos3-serving/

ARG NPA_SOURCE_SHA=unknown

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      npa.tool="cosmos3-super-benchmark" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      npa.source.commit="${NPA_SOURCE_SHA}" \
      npa.runtime="operator-fetch" \
      npa.vllm_omni.revision="${VLLM_OMNI_REVISION}" \
      npa.version="0.1.0" \
      npa.weights="runtime-fetch-only" \
      npa.redistribution="public"

USER root
RUN cp /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py \
       /opt/npa-cosmos3-serving/prepare_guardrail_payload.py \
    && sed -ri "s|(https?://snapshot.debian.org/archive/debian(-security)?/)[0-9TZ]+|\\1${DEBIAN_SNAPSHOT}|g" \
       /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-server rsync sudo util-linux \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
       -r /opt/npa-cosmos3-serving/packaging-requirements.txt \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && (id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu) \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/npa-runtime \
    && chmod 0440 /etc/sudoers.d/npa-runtime \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
       > /etc/ssh/sshd_config.d/99-npa.conf \
    && dpkg-query -W openssh-server rsync sudo \
    && command -v service >/dev/null

COPY --chmod=0755 docker/workbench/cosmos3-super-benchmark/entrypoint.sh /usr/local/bin/npa-cosmos3-super-benchmark-entrypoint
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/prepare_guardrail_runtime.py /opt/npa-cosmos3-serving/prepare_guardrail_runtime.py
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/bootstrap-source-sha256s.txt /opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/THIRD_PARTY_NOTICES.md /opt/npa-cosmos3-serving/THIRD_PARTY_NOTICES.md
COPY --chmod=0644 docker/workbench/cosmos3-super-benchmark/LICENSE-APACHE-2.0 /opt/npa-cosmos3-serving/LICENSE-APACHE-2.0

ENV HOME=/home/ubuntu \
    NPA_COSMOS3_SERVE_GUARDRAILS=off \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
WORKDIR /workspace
USER ubuntu
RUN test -w /opt/npa-cosmos3-serving/runtime \
    && sha256sum -c /opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt \
    && python /opt/npa-cosmos3-serving/verify_env.py
RUN /usr/local/bin/npa-cosmos3-super-benchmark-entrypoint /bin/sh -ec \
    'test -w /tmp; test -w "$HOME"; command -v rsync; command -v service; \
     test -x /usr/sbin/sshd; sudo -n true; \
     test "$(/bin/sh -c "printf %s forwarded" sentinel)" = forwarded; \
     sudo install -d -m 0755 /run/sshd; sudo ssh-keygen -A; \
     sudo /usr/sbin/sshd -t; sudo service ssh start; sudo service ssh status; \
     sudo service ssh stop; sudo rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub'
ENTRYPOINT ["/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint"]
