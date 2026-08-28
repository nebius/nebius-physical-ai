ARG BASE_IMAGE=npa-genesis:cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA
ARG UBUNTU_SNAPSHOT=20260801T053000Z
LABEL npa.tool="envgen" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}" \
      org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"

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

RUN /usr/local/bin/install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" \
    && test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip uninstall -y npa \
    && python -m pip uninstall -y transformers \
    && python -m pip install --no-deps \
      -r /opt/npa/sim2real-control-requirements.txt \
      -r /opt/npa/sim2real-genesis-requirements.txt \
    && find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -delete \
    && test -z "$(find /opt/npa/venv -type f \
      -path '*/imageio_ffmpeg/binaries/ffmpeg*' -print -quit)" \
    && IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg python -c \
      'import imageio_ffmpeg; assert imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' \
    && python -m pip check \
    && python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'npa-exact-source.pth').write_text('/opt/npa/src\\n', encoding='utf-8')" \
    && env -u PYTHONPATH /opt/npa/venv/bin/python -c \
      'import boto3; from npa.workflows.sim2real.workflow_stage import main; assert callable(main)' \
    && python -c 'import importlib.metadata as m; assert not any(d.metadata["Name"].lower() == "transformers" for d in m.distributions())' \
    && python -m py_compile /opt/npa/src/npa/workflows/sim2real_envgen.py

ENV PATH=/opt/npa/venv/bin:$PATH \
    NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA} \
    NPA_BAKED_PYTHON=/opt/npa/venv/bin/python \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    PYTHONPATH=/opt/npa/src

USER ubuntu
ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]
