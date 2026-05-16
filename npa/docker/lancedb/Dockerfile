FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANCEDB_PORT=8686

WORKDIR /app

COPY docker/lancedb/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/npa/workbench/lancedb/bdd100k_import.py /app/npa_lancedb_bdd100k_import.py
COPY src/npa/workbench/lancedb/bdd100k_udfs.py /app/npa_lancedb_bdd100k_udfs.py
COPY src/npa/workbench/lancedb/backfill.py /app/npa_lancedb_backfill.py
COPY src/npa/workbench/lancedb/server.py /app/npa_lancedb_server.py
COPY docker/lancedb/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8686

ENTRYPOINT ["/entrypoint.sh"]
