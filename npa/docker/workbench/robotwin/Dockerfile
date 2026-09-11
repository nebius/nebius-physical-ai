# syntax=docker/dockerfile:1.7
# Deliberately unbuildable Phase A candidate: build.sh refuses before Docker and
# this invalid digest prevents an accidental direct build from resolving a base.
FROM ubuntu:22.04@sha256:UNRESOLVED-PHASE-A AS runtime

ARG NPA_SOURCE_SHA
ENV HOME=/home/ubuntu \
    PATH=/opt/npa/robotwin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The eventual complete recipe may install only the version/snapshot-locked
# public OS bootstrap closure. Its SkyPilot contract requires openssh-server,
# rsync, and sudo; all three remain unresolved in apt-packages.lock, so Phase A
# performs no package installation. It must never install Python/vendor bytes.
RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

COPY --chmod=0644 docker/workbench/robotwin/runtime-lock.json \
  docker/workbench/robotwin/runtime-requirements.in \
  docker/workbench/robotwin/runtime-requirements.lock \
  docker/workbench/robotwin/apt-packages.lock \
  docker/workbench/robotwin/REDISTRIBUTION.md \
  docker/workbench/robotwin/THIRD_PARTY_NOTICES.md /opt/npa/robotwin/
COPY --chmod=0755 docker/workbench/robotwin/robotwin_runtime.py /opt/npa/robotwin/robotwin-runtime
COPY --chmod=0755 docker/workbench/robotwin/entrypoint.sh /opt/npa/robotwin/entrypoint.sh
COPY --chmod=0755 docker/workbench/robotwin/smoke.sh /opt/npa/robotwin/smoke.sh

LABEL org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1" \
      org.nebius.npa.source-sha="${NPA_SOURCE_SHA}" \
      org.nebius.npa.payload="zero-vendor-payload" \
      org.nebius.npa.publication-state="phase-a-unbuilt-quarantined"

RUN env -u NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1 \
      /opt/npa/robotwin/robotwin-runtime assert-refusal

USER ubuntu
WORKDIR /workspace
ENTRYPOINT ["/opt/npa/robotwin/entrypoint.sh"]
CMD ["/bin/bash"]
