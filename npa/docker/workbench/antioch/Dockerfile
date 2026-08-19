# Digest pinned 2026-07-30. Tag: python:3.12-slim-bookworm.
FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG NPA_VERSION=0.1.1-cli0.3.47-streaming-20260819-r1
ARG NPA_SOURCE_SHA

LABEL npa.tool="antioch" \
      npa.version="${NPA_VERSION}" \
      npa.vendor_runtime="none-baked-in; antioch-sim is fetched at runtime" \
      org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPA_ANTIOCH_RUNTIME_CACHE=/workspace/.cache/npa/antioch

RUN test -n "${NPA_SOURCE_SHA}" \
    && test "$(printf '%s' "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 npa

WORKDIR /app
COPY docker/workbench/antioch/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import importlib.metadata as m; assert not any(d.metadata.get('Name','').lower() == 'antioch-sim' for d in m.distributions())" \
    && mkdir -p /workspace/.cache/npa/antioch \
    && chown -R npa:npa /workspace /app
RUN mkdir -p /app/npa/workbench /app/npa/clients /app/npa/adapter /app/npa/smoke \
    && touch /app/npa/__init__.py /app/npa/workbench/__init__.py /app/npa/clients/__init__.py /app/npa/adapter/__init__.py /app/npa/smoke/__init__.py
COPY src/npa/workbench/antioch /app/npa/workbench/antioch
COPY src/npa/clients/storage.py /app/npa/clients/storage.py
COPY src/npa/adapter/sim_to_lerobot.py /app/npa/adapter/sim_to_lerobot.py
COPY src/npa/smoke/test_antioch_functional.py /app/npa/smoke/test_antioch_functional.py
RUN chown -R npa:npa /app

USER 10001
EXPOSE 8789

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8789/docs', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.antioch.service:app", "--host", "0.0.0.0", "--port", "8789"]
