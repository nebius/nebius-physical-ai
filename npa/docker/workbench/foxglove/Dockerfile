# Foxglove OSS web viewer (Lichtblick).
#
# OSS source: lichtblick-suite/lichtblick (LICENSE: MPL-2.0), the actively
# maintained community fork of the archived, relicensed Foxglove Studio. This
# image ships ONLY the permissively licensed OSS web build; it never requires a
# Foxglove account or any proprietary/closed SaaS component. Pinned to tag
# v${LICHTBLICK_VERSION}.
#
# Digest pinned 2026-07-23.
#   Build stage tag:   node:22-bookworm
#   Runtime stage tag: caddy:2.11.4-alpine
# To update: bump LICHTBLICK_VERSION, then re-resolve both digests via the
# registry API and update the @sha256 pins below.

ARG LICHTBLICK_VERSION=1.26.0

# ---- build stage: compile the static web bundle from OSS source ----
# Full (non-slim) node image so native yarn/node-gyp deps build like upstream.
FROM node:22-bookworm@sha256:5647be709086c696ff32edaaf1c70cd26d1da6ab2b39c32f3c7b4c4a31957e37 AS build

ARG LICHTBLICK_VERSION
ENV LICHTBLICK_VERSION=${LICHTBLICK_VERSION} \
    DEBIAN_FRONTEND=noninteractive \
    CI=true

WORKDIR /src

# Shallow-clone exactly the pinned OSS tag, then assert the checked-out source
# version matches the pinned ARG before spending time on the web build.
RUN git clone --depth 1 --branch "v${LICHTBLICK_VERSION}" \
      https://github.com/lichtblick-suite/lichtblick.git . \
    && node -e "const v=require('./package.json').version; if (v !== process.env.LICHTBLICK_VERSION) { throw new Error('expected lichtblick '+process.env.LICHTBLICK_VERSION+', found '+v); } console.log('LICHTBLICK_SOURCE_VERSION_OK '+v);"

RUN corepack enable \
    && yarn install --immutable \
    && yarn run web:build:prod

# ---- runtime stage: minimal static file server ----
FROM caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648

ARG LICHTBLICK_VERSION

LABEL npa.tool="foxglove" \
      npa.version="${LICHTBLICK_VERSION}"

ENV FOXGLOVE_VERSION=${LICHTBLICK_VERSION}

WORKDIR /srv

# Static web bundle (index.html + wasm/js) served by caddy on :8080.
COPY --from=build /src/web/.webpack ./
# Keep the source version manifest for the golden-eval / build-time assertion.
COPY --from=build /src/package.json /opt/npa/foxglove/package.json

# Build-time assertion: the served bundle version matches the pinned ARG. Mirrors
# the fiftyone metadata.version() check, adapted to the web asset's version file.
RUN grep -q "\"version\": \"${LICHTBLICK_VERSION}\"" /opt/npa/foxglove/package.json \
    && test -s /srv/index.html \
    && printf '%s\n' "${LICHTBLICK_VERSION}" > /opt/npa/foxglove/VERSION \
    && echo "FOXGLOVE_BUILD_VERSION_OK ${LICHTBLICK_VERSION}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1

USER nobody
ENTRYPOINT ["caddy"]
CMD ["file-server", "--listen", ":8080", "--root", "/srv"]
