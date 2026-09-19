# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

LABEL org.opencontainers.image.source="https://github.com/nebius/nebius-physical-ai" \
      org.opencontainers.image.title="NPA neutral robomimic bootstrap candidate" \
      org.opencontainers.image.description="Pinned MIT robomimic source without CUDA, PyTorch, weights, data, cache, or outputs" \
      org.nebius.npa.redistribution="candidate-unvalidated" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/robomimic \
    NPA_ROBOMIMIC_RUNTIME_ROOT=/opt/npa-runtime/robomimic \
    NPA_ROBOMIMIC_RUNTIME_FETCH=0

# Exact lock roots: ca-certificates openssh-server procps rsync sudo.
COPY debian-packages.lock source-manifest.json verify_image.py /opt/npa/robomimic/
RUN --mount=type=bind,source=build-inputs,target=/mnt/robomimic-build-inputs,readonly \
    umask 022 \
    && python3 -I -S -B /opt/npa/robomimic/verify_image.py build-inputs \
        --input-root /mnt/robomimic-build-inputs \
        --debian-lock /opt/npa/robomimic/debian-packages.lock \
        --source-manifest /opt/npa/robomimic/source-manifest.json \
    && mkdir -p /opt/byof /opt/robomimic \
        /opt/npa-runtime/robomimic /workspace/byof-inputs /workspace/byof-runs \
    && tar --extract --no-same-permissions --file /mnt/robomimic-build-inputs/source/robomimic.tar \
        --directory /opt/robomimic \
    && python3 -I -S -B /opt/npa/robomimic/verify_image.py source-tree \
        --source-root /opt/robomimic \
        --source-archive /mnt/robomimic-build-inputs/source/robomimic.tar \
        --source-manifest /opt/npa/robomimic/source-manifest.json \
    && cp /opt/npa/robomimic/source-manifest.json \
        /opt/byof/npa_source_metadata.json \
    && dpkg --unpack /mnt/robomimic-build-inputs/debian/*.deb \
    && DEBIAN_FRONTEND=noninteractive dpkg --configure --pending \
    && python3 -I -S -B /opt/npa/robomimic/verify_image.py debian-install \
        --debian-lock /opt/npa/robomimic/debian-packages.lock \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && useradd --create-home --uid 1000 --shell /bin/bash ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' >/etc/sudoers.d/90-npa-skypilot \
    && chmod 0440 /etc/sudoers.d/90-npa-skypilot \
    && rm -f /etc/ssh/ssh_host_*

# Python distributions, including CUDA/PyTorch/vendor wheels, are deliberately
# absent from this public bootstrap.  The customer-owned runtime inventory is
# fetched and RECORD-verified by runtime_bootstrap.sh after entitlement checks.
COPY runtime-requirements.lock runtime_bootstrap.sh entrypoint.sh smoke.py \
     REDISTRIBUTION.md THIRD_PARTY_NOTICES.md /opt/npa/robomimic/
RUN chmod 0555 /opt/npa/robomimic/runtime_bootstrap.sh \
        /opt/npa/robomimic/entrypoint.sh /opt/npa/robomimic/smoke.py \
        /opt/npa/robomimic/verify_image.py \
    && ln -s /opt/npa/robomimic/runtime_bootstrap.sh /usr/local/bin/robomimic-runtime \
    && ln -s /opt/npa/robomimic/entrypoint.sh /usr/local/bin/robomimic-entrypoint \
    && su ubuntu --command 'robomimic-runtime assert-refusal' \
    && chmod -R a-w /opt/byof /opt/robomimic /opt/npa/robomimic \
    && chown -R ubuntu:ubuntu /opt/npa-runtime /workspace

USER ubuntu
WORKDIR /workspace
VOLUME ["/opt/npa-runtime/robomimic"]
ENTRYPOINT ["/opt/npa/robomimic/entrypoint.sh"]
