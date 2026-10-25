# Digest pinned 2026-05-12. Tag: nvidia/cuda:12.4.1-devel-ubuntu22.04
# To update: re-resolve via registry API and update digest below.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

ARG LEROBOT_VERSION=0.5.1

LABEL npa.tool="lerobot" \
      npa.version="${LEROBOT_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_LEROBOT_HOME=/opt/lerobot/hf_cache \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    NPA_SERVER_HOST=0.0.0.0 \
    NPA_SERVER_PORT=8080 \
    NPA_CHECKPOINT_DIR=/opt/lerobot/checkpoints \
    NPA_JOB_STATUS_DIR=/opt/lerobot/job_status \
    NPA_LOG_DIR=/var/log/npa-lerobot \
    NPA_LEROBOT_VERSION=${LEROBOT_VERSION} \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

COPY docker/workbench/lerobot/install_immutable_system_packages.sh \
    /usr/local/bin/install-immutable-lerobot-system-packages
RUN chmod 0755 /usr/local/bin/install-immutable-lerobot-system-packages \
    && /usr/local/bin/install-immutable-lerobot-system-packages

RUN useradd -m -s /bin/bash -u 1000 ubuntu \
    && install -d -m 0755 -o ubuntu -g ubuntu \
      /opt/lerobot \
      /opt/lerobot/checkpoints \
      /opt/lerobot/data \
      /opt/lerobot/dataset_cache \
      /opt/lerobot/checkpoint_cache \
      /opt/lerobot/hf_cache \
      /opt/lerobot/job_status \
      /opt/lerobot/logs \
      /opt/lerobot/benchmarks \
      /opt/npa/src/npa \
      /var/log/npa-lerobot \
    && touch /opt/lerobot/.env \
    && chown ubuntu:ubuntu /opt/lerobot/.env \
    && chmod 600 /opt/lerobot/.env

USER ubuntu
WORKDIR /opt/lerobot

COPY --chown=ubuntu:ubuntu docker/workbench/curobo/remove_scikit_image_recipe.py \
    /opt/lerobot/remove-scikit-image-recipe.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/default-runtime-requirements.txt \
    /opt/lerobot/default-constraints.txt
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/smoke_native_cpu.py \
    /opt/lerobot/smoke-native-cpu.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/prepare_secure_wheel.py \
    /opt/lerobot/prepare-secure-wheel.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/prepare_secure_wheel_060.py \
    /opt/lerobot/prepare-secure-wheel-060.py
COPY --chown=ubuntu:ubuntu docker/workbench/lerobot/bootstrap-requirements.lock \
    docker/workbench/lerobot/requirements-051.lock \
    docker/workbench/lerobot/requirements-060.lock \
    /opt/lerobot/

ENV PIP_NO_CACHE_DIR=1

# 0.5.1 is an explicitly versioned, source-verified NPA integration: the upstream
# model code and license are retained, and dependency changes face native gates.
# 0.6.0 uses lean feature extras. Its secure candidate closure explicitly replaces
# the upstream torch ceiling with a hash-locked security remediation; it is not an
# acceptance claim for the separately pinned, previously qualified 0.6.0 image.
# Diffusers and transformers remain behind extras and are enforced in each policy's
# __init__, so diffusion and smolvla must be requested explicitly or
# --policy.type=diffusion fails inside make_policy() at run time, not at build.
# The runtime locks intentionally include hash-pinned source distributions. Keep
# their build backends on the separately hash-pinned bootstrap toolchain instead
# of letting pip create an unpinned isolated build environment.
# The locked CMake wheel ships one Windows-only signing template. Remove that
# exact inert path and prove the virtualenv tree has no PKCS#12 path before
# the dependency-install layer commits.
RUN python3.12 -m venv /opt/lerobot/venv \
    && /opt/lerobot/venv/bin/python -m pip install \
         --no-cache-dir --disable-pip-version-check --only-binary=:all: \
         --require-hashes --no-deps -r /opt/lerobot/bootstrap-requirements.lock \
    && if [ "${LEROBOT_VERSION}" = "0.6.0" ]; then \
         /opt/lerobot/venv/bin/python /opt/lerobot/prepare-secure-wheel-060.py \
           --output-dir /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/pip install \
           --no-cache-dir --disable-pip-version-check --no-build-isolation \
           --find-links /opt/lerobot/secure-wheel \
           --require-hashes --no-deps -r /opt/lerobot/requirements-060.lock \
         && cp /opt/lerobot/secure-wheel/source-integration.json \
           /opt/lerobot/source-integration.json \
         && rm -rf /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/python /opt/lerobot/remove-scikit-image-recipe.py \
           --site-packages /opt/lerobot/venv/lib/python3.12/site-packages \
           --capability lerobot \
           > /opt/lerobot/dependency-source-correction.json; \
       elif [ "${LEROBOT_VERSION}" = "0.5.1" ]; then \
         /opt/lerobot/venv/bin/python /opt/lerobot/prepare-secure-wheel.py \
           --output-dir /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/pip install \
           --no-cache-dir --disable-pip-version-check --no-build-isolation \
           --find-links /opt/lerobot/secure-wheel \
           --require-hashes --no-deps -r /opt/lerobot/requirements-051.lock \
         && cp /opt/lerobot/secure-wheel/source-integration.json \
           /opt/lerobot/source-integration.json \
         && rm -rf /opt/lerobot/secure-wheel \
         && /opt/lerobot/venv/bin/python /opt/lerobot/remove-scikit-image-recipe.py \
           --site-packages /opt/lerobot/venv/lib/python3.12/site-packages \
           --capability lerobot \
           > /opt/lerobot/dependency-source-correction.json; \
       else \
         echo 'Unsupported LeRobot package version' >&2; exit 1; \
       fi \
    && test -f /opt/lerobot/venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx \
    && rm /opt/lerobot/venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx \
    && test ! -e /opt/lerobot/venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx \
    && test ! -L /opt/lerobot/venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.1/Templates/Windows/Windows_TemporaryKey.pfx \
    && test -z "$(find /opt/lerobot/venv \( -iname '*.p12' -o -iname '*.pfx' \) -print -quit)" \
    && /opt/lerobot/venv/bin/pip check \
    && find /opt/lerobot/venv -type f \
         -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test "$(/opt/lerobot/venv/bin/python -c \
         'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
         = /usr/bin/ffmpeg \
    && test -z "$(find /opt/lerobot/venv -type f \
         -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)"

USER root

# Dependency backends can write outside the ubuntu-owned venv. This root-owned
# audit immediately follows that layer and fails before any later COPY can
# create a successful image with a PKCS#12 path in an ancestor layer.
RUN test -z "$(find / -xdev \( -iname '*.p12' -o -iname '*.pfx' \) -print -quit)"

# These provenance receipts feed the version smoke and release audit. Keep the
# runtime user from rewriting their source-integration or dependency claims.
RUN install -d -m 0755 -o root -g root /usr/share/doc/npa-lerobot \
    && if [ -f /opt/lerobot/source-integration.json ]; then \
         install -o root -g root -m 0644 \
           /opt/lerobot/source-integration.json \
           /usr/share/doc/npa-lerobot/source-integration.json; \
       fi \
    && install -o root -g root -m 0644 \
      /opt/lerobot/dependency-source-correction.json \
      /usr/share/doc/npa-lerobot/dependency-source-correction.json \
    && rm -f /opt/lerobot/source-integration.json \
      /opt/lerobot/dependency-source-correction.json

USER ubuntu

ENV PATH=/opt/lerobot/venv/bin:$PATH \
    PYTHONPATH=/opt/npa/src \
    NPA_SKIP_EAGER_IMPORTS=1

COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
# Keep this image-local SDK intentionally partial. The stage's own NPA install
# must supply npa.workbench; shipping it here would make the repair's import
# probe pass against the baked tree.
COPY --chown=ubuntu:ubuntu src/npa/__init__.py /opt/npa/src/npa/__init__.py
COPY --chown=ubuntu:ubuntu src/npa/clients/__init__.py /opt/npa/src/npa/clients/__init__.py
COPY --chown=ubuntu:ubuntu src/npa/clients/storage.py /opt/npa/src/npa/clients/storage.py
COPY --chown=ubuntu:ubuntu src/npa/server /opt/npa/src/npa/server
COPY --chown=ubuntu:ubuntu src/npa/smoke /opt/npa/src/npa/smoke

RUN python -m npa.smoke.test_lerobot_env

USER root

# SkyPilot-on-Kubernetes runtime prerequisites are part of the immutable system
# package closure above. Keep setup and host-key removal here, before hardening.
# SkyPilot prerequisites, so this image can host a workflow stage.
# SkyPilot's bootstrap runs inside the task container; without these, provisioning fails with
# "container not found (ray-node)" (see Dockerfile.k8s-prereqs and
# npa/tests/guardrails/test_workbench_image_k8s_prereqs.py).
#
# The PATH line flips the vendor venv OFF the front of PATH, which SkyPilot requires. The
# vendor python stays reachable by absolute path, and the renderer's
# TOOL_REF_VENDOR_INTERPRETERS installs npa into it and records it as the stage interpreter —
# without that half, a tool importing `lerobot` would run under the system python and fail.
# Install every bootstrap package before the hardening step. That step deliberately leaves
# APT unable to resolve packages needing linux-libc-dev, so deferring sshd to SkyPilot's
# runtime bootstrap would make an otherwise non-root image unschedulable. Remove the
# package-generated host keys in this same layer; SkyPilot creates per-task keys later.
# The system python3 the bootstrap needs is already present, so it is asserted rather than
# installed.
RUN rm -f /etc/ssh/ssh_host_* \
    && test -z "$(find /etc/ssh -maxdepth 1 -type f -name 'ssh_host_*' -print -quit)" \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' ubuntu \
         > /etc/sudoers.d/99-npa-runtime-user \
    && chmod 0440 /etc/sudoers.d/99-npa-runtime-user \
    && test -x /usr/bin/python3 \
    && /usr/bin/python3 --version

ENV PATH=/usr/bin:$PATH

USER root

RUN packages='dirmngr gnupg gnupg-utils gnupg2 gpg gpg-agent nodejs software-properties-common' \
    && installed='' \
    && for package in $packages; do \
         if dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null | grep -qx installed; then \
           installed="$installed $package"; \
         fi; \
       done \
    && if [ -n "$installed" ]; then apt-get purge -y --auto-remove $installed; fi \
    && dpkg --purge --force-depends linux-libc-dev


USER ubuntu

# Exercise the final hardened 0.5.1 integration and declared server interpreter.
# The separately qualified 0.6.0 recipe retains its existing acceptance path.
RUN if [ "${LEROBOT_VERSION}" = "0.5.1" ]; then \
      /opt/lerobot/venv/bin/python /opt/lerobot/smoke-native-cpu.py; \
    fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD /opt/lerobot/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)" || exit 1

ENTRYPOINT ["/opt/lerobot/venv/bin/python", "-m", "npa.server.app"]
