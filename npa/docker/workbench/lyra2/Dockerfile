# Lyra 2 reconstruction bootstrap. CUDA compiler and NPA are baked; Lyra source,
# its Python/CUDA inference environment, and model weights are runtime fetches.
# Retain the scanned CUDA/Ubuntu closure from rtfetch1. Pin the repair packages
# separately so required startup tools do not depend on Ubuntu snapshots.
FROM ghcr.io/nebius/nebius-physical-ai/npa-lyra2:2.0-rtfetch1@sha256:cd76116ffac0f930bc8a4645ad92ea0848662426f3fe34c498050ae5bc5afc4d

ARG NPA_SOURCE_SHA

LABEL npa.tool="lyra2" \
      npa.version="2.0-rtfetch2" \
      npa.weights="operator-runtime-fetch" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NPA_SKIP_EAGER_IMPORTS=1 \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_BAKED_PYTHON=/opt/npa/.venv/bin/python \
    NPA_LYRA_BASE_PYTHON=/usr/bin/python3.12 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    CUDA_HOME=/usr/local/cuda \
    PYTHONPATH=/opt/npa/src \
    PATH=/opt/npa/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

USER root
COPY docker/workbench/lyra2/bootstrap-packages.lock /tmp/bootstrap-packages.lock
RUN set -eu; mkdir /tmp/lyra2-debs; \
    while read -r checksum url; do \
      package="/tmp/lyra2-debs/${url##*/}"; \
      curl --fail --location --retry 3 "$url" --output "$package"; \
      printf '%s  %s\n' "$checksum" "$package" | sha256sum --check -; \
    done < /tmp/bootstrap-packages.lock; \
    dpkg -i /tmp/lyra2-debs/*.deb; \
    rm -rf /tmp/lyra2-debs /tmp/bootstrap-packages.lock /opt/npa/src/npa /opt/npa/workflows

COPY pyproject.toml README.md /opt/npa/
COPY src/npa /opt/npa/src/npa
COPY workflows /opt/npa/workflows
COPY --chmod=0755 docker/workbench/lyra2/entrypoint.sh /usr/local/bin/lyra2-entrypoint
COPY --chmod=0644 docker/workbench/lyra2/REDISTRIBUTION.md /usr/share/doc/npa-lyra2/REDISTRIBUTION.md

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && /opt/npa/.venv/bin/python -m pip install --no-cache-dir --no-deps /opt/npa \
    && /opt/npa/.venv/bin/python -m pip check \
    && python3.12 -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-source.pth').write_text('/opt/npa/src\\n')" \
    && env -u PYTHONPATH python3.12 -c 'import npa' \
    && /opt/npa/.venv/bin/python -m npa.workflows.lyra_stage --help \
    && /opt/npa/.venv/bin/python -c "import importlib.util; assert importlib.util.find_spec('torch') is None" \
    && install -d -o ubuntu -g ubuntu /workspace /home/ubuntu/.cache \
    && chmod -R a+rX /opt/npa \
    && rm -rf /root/.cache /tmp/pip-*

WORKDIR /workspace
USER ubuntu
ENV HOME=/home/ubuntu
# Exercise the contract as the final user before attesting to its behavior.
RUN /usr/local/bin/lyra2-entrypoint /bin/sh -c \
    'set -eu; test -w /tmp; test -w "$HOME"; command -v rsync; command -v service; command -v wget; command -v nc; command -v fusermount; command -v lspci; test -x /usr/sbin/sshd; sudo -n true; dpkg --audit' \
    && test "$(/usr/local/bin/lyra2-entrypoint /usr/bin/printf '%s' 'argv with spaces')" = 'argv with spaces'
ENTRYPOINT ["/usr/local/bin/lyra2-entrypoint"]
