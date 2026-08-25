ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-genesis@sha256:80cd1c4c7f7a5466533de29ee5cad1213202f348545346fea9ed6746fa03b1ca
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20260801T053000Z
LABEL npa.tool="envgen" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

WORKDIR /opt/npa
USER root
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY docker/workbench/common/sim2real-control-requirements.txt \
    /opt/npa/sim2real-control-requirements.txt
COPY docker/workbench/common/sim2real-genesis-requirements.txt \
    /opt/npa/sim2real-genesis-requirements.txt
COPY --chmod=0755 docker/workbench/common/workflow_runtime_entrypoint.sh \
    /usr/local/bin/npa-workflow-entrypoint
COPY --chmod=0755 docker/workbench/common/install_workflow_runtime_prereqs.sh \
    /usr/local/bin/install-workflow-runtime-prereqs

# The public Genesis parent includes imageio-ffmpeg's wheel-bundled static
# FFmpeg, which is not redistributable under this image's public packaging
# contract. Genesis declares MoviePy at runtime, so retain that package while
# replacing imageio-ffmpeg with its hash-pinned pure-Python sdist and bind it to
# snapshot-pinned Ubuntu FFmpeg.
RUN /usr/local/bin/install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip uninstall -y transformers imageio-ffmpeg \
    && python -m pip install --no-deps \
      'imageio-ffmpeg @ https://files.pythonhosted.org/packages/44/bd/c3343c721f2a1b0c9fc71c1aebf1966a3b7f08c2eea8ed5437a2865611d6/imageio_ffmpeg-0.6.0.tar.gz#sha256=e2556bed8e005564a9f925bb7afa4002d82770d6b08825078b7697ab88ba1755' \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && python -m pip check \
    && IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg python -c \
      'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && test -z "$(find /opt/npa/venv -type f -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && python -c 'import importlib.metadata as m; assert not any(d.metadata["Name"].lower() == "transformers" for d in m.distributions())' \
    && python -m pip install --no-deps -e /opt/npa \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py

ENV PATH=/opt/npa/venv/bin:$PATH \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    PYTHONPATH=/opt/npa/src

USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
