# syntax=docker/dockerfile:1.7
# Neutral public bootstrap candidate. The build gate precedes every package
# network operation and binds the independently reviewed Ubuntu bootstrap binary
# and corresponding-source closure before APT can run.
ARG BASE_IMAGE=ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61
FROM ${BASE_IMAGE} AS runtime

WORKDIR /opt/npa/gymnasium-robotics
COPY docker/workbench/gymnasium-robotics/apt-runtime.lock.json ./apt-runtime.lock.json
COPY docker/workbench/gymnasium-robotics/corresponding-source.lock.json ./corresponding-source.lock.json
COPY docker/workbench/gymnasium-robotics/build.sh ./build.sh
RUN chmod 0755 ./build.sh && ./build.sh verify-bootstrap-locks
RUN --mount=type=secret,id=npa_host_ca_bundle,required=true,target=/run/npa-host-ca-bundle.crt,mode=0444 \
    ./build.sh install-bootstrap

ARG SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.nebius.npa.delivery="neutral-runtime-bootstrap" \
      org.nebius.npa.phase="pre-registration-quarantine" \
      org.nebius.npa.redistribution="public-eligible-unvalidated" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      org.nebius.npa.skypilot-bootstrap-packages="openssh-server,rsync,sudo"

# These are NPA bootstrap/provenance bytes only. In particular, no upstream
# solution tree, Shadow asset, MuJoCo/Python wheel, venv, runtime cache, CUDA or
# NVIDIA runtime, credential, dataset, checkpoint, or prior output is copied.
COPY --chmod=0444 docker/workbench/gymnasium-robotics/asset-lock.json ./asset-lock.json
COPY --chmod=0444 docker/workbench/gymnasium-robotics/source-lock.json ./source-lock.json
COPY --chmod=0444 docker/workbench/gymnasium-robotics/requirements.lock ./requirements.lock
COPY --chmod=0444 docker/workbench/gymnasium-robotics/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md
COPY --chmod=0444 docker/workbench/gymnasium-robotics/REDISTRIBUTION.md /usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md
COPY --chmod=0555 docker/workbench/gymnasium-robotics/runtime-bootstrap.py ./runtime-bootstrap.py
COPY --chmod=0555 docker/workbench/gymnasium-robotics/capability_smoke.py ./capability_smoke.py
COPY --chmod=0555 docker/workbench/gymnasium-robotics/verify_image.py ./verify_image.py
COPY --chmod=0555 docker/workbench/gymnasium-robotics/build.sh /usr/local/bin/npa-gymnasium-entrypoint

RUN test "$(id -u ubuntu)" = 1000 \
    && test "$(id -g ubuntu)" = 1000 \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace \
    && install -d -m 0755 /run/sshd /etc/ssh/sshd_config.d \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' \
      > /etc/sudoers.d/99-npa-skypilot-runtime \
    && chmod 0440 /etc/sudoers.d/99-npa-skypilot-runtime \
    && printf '%s\n' 'PasswordAuthentication no' 'PermitRootLogin no' \
      > /etc/ssh/sshd_config.d/99-npa-runtime.conf \
    && sed -i '/^  start)$/a\    ssh-keygen -A' /etc/init.d/ssh \
    && grep -Fq 'ssh-keygen -A' /etc/init.d/ssh \
    && rm -f /etc/ssh/ssh_host_*

ENV HOME=/home/ubuntu
WORKDIR /workspace
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-gymnasium-entrypoint"]
CMD ["/bin/sh"]
