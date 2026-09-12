# Public Isaac Lab-Arena worker. The exact Apache-2.0 Arena source is baked;
# NVIDIA Isaac Sim/Lab remains an operator-authorized runtime fetch inherited
# from npa-isaac-lab. No model, dataset, generated result, or populated cache is
# present in the image.
FROM ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6

ARG ISAAC_ARENA_VERSION=0.3.0
ARG ISAAC_ARENA_REVISION=ed0fd12be862078be316c73eb7cf423ba9b1c5cd
ARG ISAAC_ARENA_ARCHIVE_SHA256=4e62ddbd7edc40fb47e62a0d5ba523eebc129482b4ce083612f17c79f1fc40a8
ARG NPA_SOURCE_SHA

LABEL npa.tool="isaac-arena" \
      npa.version="${ISAAC_ARENA_VERSION}" \
      npa.isaac_arena_revision="${ISAAC_ARENA_REVISION}" \
      npa.isaac_arena_release_channel="alpha" \
      npa.isaac_provisioning="runtime-fetch" \
      npa.architecture="arena-source-no-baked-omniverse-kit" \
      org.opencontainers.image.source="https://github.com/isaac-sim/IsaacLab-Arena" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_LIGHT_WORKBENCH_TOOL=isaac-arena \
    ISAAC_ARENA_VERSION=${ISAAC_ARENA_VERSION} \
    ISAAC_ARENA_REVISION=${ISAAC_ARENA_REVISION} \
    ISAAC_ARENA_ROOT=/opt/isaac-arena \
    PYTHONPATH=/opt/isaac-arena:/opt/npa/src

USER root
SHELL ["/bin/bash", "-lc"]

# Replace the parent image's NPA source with the exact branch source, then fetch
# and checksum the official immutable Arena archive. Documentation, upstream
# tests, sample checkpoints, HDF5 demonstrations, and example media are removed;
# none is needed for supported evaluation and their separate provenance is not
# part of this public image's redistribution claim.
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/isaac-arena/npa-cli-requirements.txt /opt/npa/docker/workbench/isaac-arena/npa-cli-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh /usr/local/bin/npa-workflow-entrypoint
COPY docker/workbench/isaac-arena/REDISTRIBUTION.md /opt/npa/docker/workbench/isaac-arena/REDISTRIBUTION.md
COPY docker/workbench/isaac-arena/THIRD_PARTY_NOTICES.md /opt/npa/docker/workbench/isaac-arena/THIRD_PARTY_NOTICES.md
COPY --chmod=0755 docker/workbench/isaac-arena/smoke_functional.sh /opt/npa/docker/workbench/isaac-arena/smoke_functional.sh

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && dpkg-query -W openssh-server rsync sudo >/dev/null \
    && command -v sshd >/dev/null \
    && grep -Rq 'NOPASSWD' /etc/sudoers /etc/sudoers.d \
    && install -d -m 0755 /opt/isaac-arena \
    && curl --fail --silent --show-error --location \
      "https://github.com/isaac-sim/IsaacLab-Arena/archive/${ISAAC_ARENA_REVISION}.tar.gz" \
      --output /tmp/isaac-arena.tar.gz \
    && printf '%s  %s\n' "${ISAAC_ARENA_ARCHIVE_SHA256}" /tmp/isaac-arena.tar.gz | sha256sum -c - \
    && tar -xzf /tmp/isaac-arena.tar.gz --strip-components=1 -C /opt/isaac-arena \
    && rm -f /tmp/isaac-arena.tar.gz \
    && rm -rf \
      /opt/isaac-arena/.agents \
      /opt/isaac-arena/.claude \
      /opt/isaac-arena/.github \
      /opt/isaac-arena/docs \
      /opt/isaac-arena/submodules \
      /opt/isaac-arena/isaaclab_arena/tests \
      /opt/isaac-arena/isaaclab_arena_gr00t/tests \
      /opt/isaac-arena/isaaclab_arena_examples/sensitivity_analysis/example_results \
    && find /opt/isaac-arena -type d -exec chmod 0755 {} + \
    && find /opt/isaac-arena -type f -exec chmod 0644 {} + \
    && test "$(sed -n 's/^version = "\([^"]*\)"/\1/p' /opt/isaac-arena/pyproject.toml | head -1)" = "${ISAAC_ARENA_VERSION}" \
    && grep -q 'Apache License' /opt/isaac-arena/LICENSE.md \
    && grep -q 'Version 2.0, January 2004' /opt/isaac-arena/LICENSE.md \
    && test ! -e /opt/isaac-arena/isaaclab_arena/tests \
    && test -z "$(find /opt/isaac-arena -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.hdf5' -o -name '*.mp4' \) -print -quit)" \
    && /opt/npa/sim/venv/bin/python -m py_compile \
      /opt/npa/src/npa/workbench/isaac_arena/runtime.py \
      /opt/npa/src/npa/cli/workbench/isaac_arena.py \
    && /opt/npa/sim/venv/bin/python -m pip install --no-cache-dir --no-deps \
      --require-hashes -r /opt/npa/docker/workbench/isaac-arena/npa-cli-requirements.txt \
    && env -u ACCEPT_EULA /usr/local/bin/npa workbench isaac-arena evaluate \
      --output-path /tmp/npa-arena-build-probe --dry-run --output-format json \
      | grep -q '"status": "dry_run"' \
    && test ! -e /tmp/npa-arena-build-probe \
    && chown -R ubuntu:ubuntu /opt/isaac-arena /opt/npa/docker/workbench/isaac-arena \
    && install -d -m 0755 -o ubuntu -g ubuntu /workspace/isaac-arena

WORKDIR /workspace/isaac-arena
USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
