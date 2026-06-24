ARG RERUN_SDK_VERSION=0.31.4
FROM python:3.11-slim-bookworm

LABEL npa.tool="rerun-viewer"
LABEL npa.tool.legacy="sim2real-rerun-viewer"

ARG RERUN_SDK_VERSION
RUN python -m pip install --no-cache-dir "rerun-sdk==${RERUN_SDK_VERSION}"

EXPOSE 9090
USER nobody
ENTRYPOINT ["rerun"]
