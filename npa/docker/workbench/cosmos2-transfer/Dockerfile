# Capability wrapper for npa-cosmos2-transfer.
#
# Golden eval in this repo means "does this container actually do its job" — a
# real capability test, not a CUDA probe. So this wrapper bakes the REAL
# Cosmos-Transfer2.5 inference environment (Python 3.10 + torch cu128 +
# flash-attn + all deps via `uv sync`) on top of the upstream transfer runtime,
# and ships a golden-eval script that runs an actual video-to-video transfer on a
# bundled robot control example and asserts a generated video is produced.
#
# The upstream 2.5.x image ships the cosmos-transfer2.5 sources but does not bake
# a flash-attn-compatible venv, and pins .python-version=3.13 while the flash-attn
# wheel is cp310-only — both are corrected here.
ARG BASE_IMAGE=npa-cosmos2-transfer:2.5.0
FROM ${BASE_IMAGE}

USER root
WORKDIR /opt/cosmos/cosmos-transfer2.5

ENV UV_LINK_MODE=copy

# Build the real inference env. `uv sync --extra=cu128` resolves torch cu128 +
# flash-attn (cp310) + the full transfer2.5 dependency set from the upstream lock.
RUN uv python install 3.10 \
    && echo "3.10" > .python-version \
    && uv sync --extra=cu128 --python 3.10 \
    && .venv/bin/python -c "import torch, flash_attn; print('cosmos-transfer2.5 env torch', torch.__version__, 'flash_attn', flash_attn.__version__, 'cuda-build', torch.version.cuda)"

COPY --chmod=755 docker/workbench/cosmos2-transfer/smoke_functional.sh /opt/cosmos2-transfer/smoke_functional.sh

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /opt/cosmos2-transfer /opt/cosmos/cosmos-transfer2.5/.venv

USER ubuntu
