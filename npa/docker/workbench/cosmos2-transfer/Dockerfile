# Golden-eval wrapper: add a minimal CUDA PyTorch venv to the transfer runtime.
# The upstream 2.5.0 image ships cosmos-transfer2.5 sources but does not bake a
# flash-attn-compatible uv venv; golden eval only needs torch+cuda proof.
ARG BASE_IMAGE=npa-cosmos2-transfer:2.5.0
FROM ${BASE_IMAGE}

USER root
WORKDIR /opt/cosmos/cosmos-transfer2.5

ENV UV_LINK_MODE=copy

RUN uv python install 3.10 \
    && UV_PYTHON=3.10 uv venv .venv \
    && UV_PYTHON=3.10 uv pip install --python .venv/bin/python \
         "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 \
    && .venv/bin/python -c "import torch; print('torch', torch.__version__)"

COPY --chmod=755 docker/workbench/cosmos2-transfer/smoke_functional.sh /opt/cosmos2-transfer/smoke_functional.sh

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /opt/cosmos2-transfer

USER ubuntu
