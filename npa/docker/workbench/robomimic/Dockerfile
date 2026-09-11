# syntax=docker/dockerfile:1.7
FROM python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

ARG ROBOMIMIC_REPOSITORY=https://github.com/ARISE-Initiative/robomimic.git
ARG ROBOMIMIC_REVISION=d309eaecc18acf4152a830a895a6984b8ac71b05
ARG ROBOMIMIC_LICENSE_SHA256=7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.title="NPA neutral robomimic bootstrap candidate" \
      org.opencontainers.image.description="Pinned MIT robomimic source without CUDA, PyTorch, weights, data, cache, or outputs" \
      org.nebius.npa.redistribution="candidate-unvalidated" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/robomimic:/opt/robomimic-deps \
    NPA_ROBOMIMIC_RUNTIME_ROOT=/opt/npa-runtime/robomimic

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates git openssh-server procps rsync sudo \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && useradd --create-home --uid 1000 --shell /bin/bash ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' >/etc/sudoers.d/90-npa-skypilot \
    && chmod 0440 /etc/sudoers.d/90-npa-skypilot \
    && rm -f /etc/ssh/ssh_host_* \
    && mkdir -p /opt/byof /opt/robomimic /opt/robomimic-deps /opt/npa/robomimic \
        /opt/npa-runtime/robomimic /workspace/byof-inputs /workspace/byof-runs

RUN git init /opt/robomimic \
    && git -C /opt/robomimic remote add origin "${ROBOMIMIC_REPOSITORY}" \
    && git -C /opt/robomimic fetch --depth 1 origin "${ROBOMIMIC_REVISION}" \
    && git -C /opt/robomimic checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/robomimic rev-parse HEAD)" = "${ROBOMIMIC_REVISION}" \
    && test "$(sha256sum /opt/robomimic/LICENSE | cut -d' ' -f1)" = "${ROBOMIMIC_LICENSE_SHA256}" \
    && tree_sha="$(git -C /opt/robomimic archive --format=tar HEAD | sha256sum | cut -d' ' -f1)" \
    && printf '{"schema":"npa.robomimic.source.v1","repo":"%s","ref":"%s","observed_head":"%s","tree_archive_sha256":"%s","license_sha256":"%s"}\n' \
        "${ROBOMIMIC_REPOSITORY}" "${ROBOMIMIC_REVISION}" "${ROBOMIMIC_REVISION}" \
        "${tree_sha}" "${ROBOMIMIC_LICENSE_SHA256}" >/opt/byof/npa_source_metadata.json \
    && rm -rf /opt/robomimic/.git

COPY baked-requirements.lock /opt/npa/robomimic/baked-requirements.lock
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir \
        --only-binary=:all: --no-deps --require-hashes \
        --target /opt/robomimic-deps \
        --requirement /opt/npa/robomimic/baked-requirements.lock

COPY runtime-requirements.lock runtime_bootstrap.sh entrypoint.sh smoke.py verify_image.py \
     REDISTRIBUTION.md THIRD_PARTY_NOTICES.md /opt/npa/robomimic/
RUN chmod 0555 /opt/npa/robomimic/runtime_bootstrap.sh \
        /opt/npa/robomimic/entrypoint.sh /opt/npa/robomimic/smoke.py \
        /opt/npa/robomimic/verify_image.py \
    && ln -s /opt/npa/robomimic/runtime_bootstrap.sh /usr/local/bin/robomimic-runtime \
    && ln -s /opt/npa/robomimic/entrypoint.sh /usr/local/bin/robomimic-entrypoint \
    && python3 /opt/npa/robomimic/verify_image.py image \
    && su ubuntu --command 'robomimic-runtime assert-refusal' \
    && chmod -R a-w /opt/byof /opt/robomimic /opt/robomimic-deps /opt/npa/robomimic \
    && chown -R ubuntu:ubuntu /opt/npa-runtime /workspace

USER ubuntu
WORKDIR /workspace
ENTRYPOINT ["/opt/npa/robomimic/entrypoint.sh"]
