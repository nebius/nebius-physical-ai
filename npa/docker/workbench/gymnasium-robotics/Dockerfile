# syntax=docker/dockerfile:1.7
# Phase A source only. This Dockerfile MUST fail before network access while
# any evidence lock is incomplete; it is not an image-build authorization.
ARG BASE_IMAGE=ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61
FROM ${BASE_IMAGE} AS evidence-gate

WORKDIR /opt/npa/gymnasium-robotics
COPY npa/docker/workbench/gymnasium-robotics/source-lock.json ./source-lock.json
COPY npa/docker/workbench/gymnasium-robotics/apt-runtime.lock.json ./apt-runtime.lock.json
COPY npa/docker/workbench/gymnasium-robotics/corresponding-source.lock.json ./corresponding-source.lock.json
COPY npa/docker/workbench/gymnasium-robotics/requirements.lock ./requirements.lock
COPY npa/docker/workbench/gymnasium-robotics/build.sh ./build.sh
RUN chmod 0755 ./build.sh && ./build.sh verify-locks

# No later instruction is reachable in Phase A. A separately authorized lock
# closure will replace the refusal-only install mode without changing the base.
RUN ./build.sh install

FROM ${BASE_IMAGE} AS runtime
ARG SOURCE_SHA
LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.nebius.npa.phase="pre-registration-quarantine" \
      org.nebius.npa.redistribution="public-eligible-unvalidated"

COPY --from=evidence-gate /opt/venv /opt/venv
COPY --from=evidence-gate /usr/share/source/npa-gymnasium-robotics /usr/share/source/npa-gymnasium-robotics
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/asset-lock.json /opt/npa/gymnasium-robotics/asset-lock.json
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/source-lock.json /opt/npa/gymnasium-robotics/source-lock.json
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/apt-runtime.lock.json /opt/npa/gymnasium-robotics/apt-runtime.lock.json
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/corresponding-source.lock.json /opt/npa/gymnasium-robotics/corresponding-source.lock.json
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/requirements.lock /opt/npa/gymnasium-robotics/requirements.lock
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/THIRD_PARTY_NOTICES.md /usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md
COPY --chmod=0444 npa/docker/workbench/gymnasium-robotics/REDISTRIBUTION.md /usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md
COPY --chmod=0555 npa/docker/workbench/gymnasium-robotics/capability_smoke.py /opt/npa/gymnasium-robotics/capability_smoke.py
COPY --chmod=0555 npa/docker/workbench/gymnasium-robotics/verify_image.py /opt/npa/gymnasium-robotics/verify_image.py
COPY --chmod=0555 npa/docker/workbench/gymnasium-robotics/build.sh /usr/local/bin/npa-gymnasium-entrypoint

USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-gymnasium-entrypoint"]
CMD ["/bin/sh"]
