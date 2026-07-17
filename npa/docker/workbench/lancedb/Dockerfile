FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    LANCEDB_PORT=8686

WORKDIR /app

COPY docker/workbench/lancedb/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/npa/workbench/lancedb/bdd100k_import.py /app/npa_lancedb_bdd100k_import.py
COPY src/npa/workbench/lancedb/bdd100k_udfs.py /app/npa_lancedb_bdd100k_udfs.py
COPY src/npa/workbench/lancedb/backfill.py /app/npa_lancedb_backfill.py
COPY src/npa/workbench/lancedb/views.py /app/npa_lancedb_views.py
COPY src/npa/workbench/lancedb/server.py /app/npa_lancedb_server.py
# Golden-eval smoke (standalone; the image does not install the npa package).
COPY src/npa/smoke/test_lancedb_functional.py /app/smoke_functional.py
COPY docker/workbench/lancedb/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# The smoke starts uvicorn against the flattened server module in /app.
ENV LANCEDB_SMOKE_APP=npa_lancedb_server:app

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app

EXPOSE 8686

USER ubuntu

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.environ.get('LANCEDB_PORT','8686'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3)" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
