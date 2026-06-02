FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANCEDB_PORT=8686

WORKDIR /app

COPY docker/workbench/lancedb/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/npa/workbench/lancedb/bdd100k_import.py /app/npa_lancedb_bdd100k_import.py
COPY src/npa/workbench/lancedb/bdd100k_udfs.py /app/npa_lancedb_bdd100k_udfs.py
COPY src/npa/workbench/lancedb/backfill.py /app/npa_lancedb_backfill.py
COPY src/npa/workbench/lancedb/views.py /app/npa_lancedb_views.py
COPY src/npa/workbench/lancedb/server.py /app/npa_lancedb_server.py
COPY docker/workbench/lancedb/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN id -u ubuntu >/dev/null 2>&1 || useradd -m -s /bin/bash -u 1000 ubuntu \
    && chown -R ubuntu:ubuntu /app

EXPOSE 8686

USER ubuntu

ENTRYPOINT ["/entrypoint.sh"]
