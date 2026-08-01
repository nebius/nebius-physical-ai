# npa-foxglove-embed — static host for the Foxglove embedding SDK and MCAP data.
#
# Serves three things over plain HTTP (see Caddyfile):
#   /sdk/    the pinned @foxglove/embed browser ESM build (MIT, fetched + sha512
#            verified at build time by install-sdk.sh)
#   /app/    the shared NPA glue module, identical to the one the agent UI loads
#   /data/   MCAP/bag recordings mounted by the operator, with CORS + byte ranges
#            so Foxglove `remote-file` data sources can stream them (no directory
#            listing; set FOXGLOVE_DATA_BROWSE=browse to enable one)
# plus `/` — a standalone host page — and `/healthz`.
#
# The Foxglove *application* is not redistributed here: point the viewer at
# https://embed.foxglove.dev/ (Foxglove-hosted) or at a self-hosted deployment
# with `?src=`. Build context is the `npa/` directory.
ARG FOXGLOVE_EMBED_VERSION=0.58.0
ARG FOXGLOVE_EMBED_INTEGRITY=sha512-hNxqEQWPk2Wm0KmDlNs3Y0TTEl9Wm+4CuppBZcLzK8j8m2EcwbbCVWg43oCsf5HJgwXt7KYorIdoMO7CICQ7Vg==
ARG FOXGLOVE_EMBED_REGISTRY=https://registry.npmjs.org

FROM alpine:3.21@sha256:48b0309ca019d89d40f670aa1bc06e426dc0931948452e8491e3d65087abc07d AS sdk
ARG FOXGLOVE_EMBED_VERSION
ARG FOXGLOVE_EMBED_INTEGRITY
ARG FOXGLOVE_EMBED_REGISTRY
RUN apk add --no-cache bash ca-certificates curl openssl tar
COPY docker/workbench/foxglove-embed/install-sdk.sh /usr/local/bin/install-foxglove-sdk.sh
RUN chmod +x /usr/local/bin/install-foxglove-sdk.sh \
    && /usr/local/bin/install-foxglove-sdk.sh \
        --dest /out/sdk \
        --version "${FOXGLOVE_EMBED_VERSION}" \
        --integrity "${FOXGLOVE_EMBED_INTEGRITY}" \
        --registry "${FOXGLOVE_EMBED_REGISTRY}"

FROM caddy:2.11-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648
ARG FOXGLOVE_EMBED_VERSION

LABEL npa.tool="foxglove-embed" \
      npa.version="${FOXGLOVE_EMBED_VERSION}" \
      org.opencontainers.image.title="NPA Foxglove embedded viewer host" \
      org.opencontainers.image.description="Static host for the @foxglove/embed SDK, the NPA glue module, and MCAP recordings (CORS + byte ranges)." \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"

# Caddy's default XDG dirs are root-owned; the image runs unprivileged.
ENV FOXGLOVE_EMBED_VERSION=${FOXGLOVE_EMBED_VERSION} \
    XDG_CONFIG_HOME=/tmp/caddy-config \
    XDG_DATA_HOME=/tmp/caddy-data

COPY --from=sdk /out/sdk /srv/sdk
COPY src/npa/cli/assets/foxglove/npa-foxglove-host.js /srv/app/npa-foxglove-host.js
COPY src/npa/cli/assets/foxglove/index.html /srv/index.html
COPY docker/workbench/foxglove-embed/Caddyfile /etc/caddy/Caddyfile
COPY docker/workbench/foxglove-embed/smoke.sh /usr/local/bin/npa-foxglove-smoke.sh

RUN mkdir -p /srv/data /tmp/caddy-config /tmp/caddy-data \
    && chmod -R a+rX /srv \
    && chmod +x /usr/local/bin/npa-foxglove-smoke.sh \
    # /srv/data is a read-only drop point for mounted recordings; it is owned by
    # the runtime user (not world-writable) so a reachable service cannot be used
    # to plant files. The golden eval writes its range probe here as that user.
    && chown -R 65534:65534 /srv/data \
    && chmod 0755 /srv/data \
    && chmod 1777 /tmp/caddy-config /tmp/caddy-data \
    && caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

EXPOSE 8099

USER nobody

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8099/healthz || exit 1

ENTRYPOINT ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
