FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ARG NPA_REVISION=unknown
LABEL org.opencontainers.image.revision="${NPA_REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPA_ANTIOCH_RUNTIME_CACHE=/workspace/.cache/npa/antioch

RUN apt-get update \
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
# The public adapter carries only reviewed NPA bootstrap/scenario source.  The
# proprietary Antioch CLI and simulator image are fetched at runtime under the
# operator's own accepted terms.
COPY examples/antioch-openpi-live /opt/npa/antioch-openpi-live
RUN chown -R npa:npa /app /opt/npa/antioch-openpi-live

# Build-time proof that the workload-identity resolver is self-contained and
# does not depend on the host-only npa.clients.config graph.
RUN AWS_EC2_METADATA_DISABLED=true python -c "from npa.workbench.antioch.storage_config import resolve_storage_client; assert resolve_storage_client().s3"

USER 10001
EXPOSE 8789

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8789/docs', timeout=3)" || exit 1

CMD ["uvicorn", "npa.workbench.antioch.service:app", "--host", "0.0.0.0", "--port", "8789"]
