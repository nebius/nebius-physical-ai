# npa-foxglove-embed

Static host for the [Foxglove embedding SDK](https://docs.foxglove.dev/docs/embed/typescript-sdk)
(`@foxglove/embed`), the shared NPA glue module, and MCAP/bag recordings.

| Path | Contents |
| --- | --- |
| `/` | Standalone host page (`?src=`, `?org=`, `?mcap=`, `?ws=`, `?layout=`, `?theme=`, `?autoplay=1`) |
| `/sdk/` | Pinned `@foxglove/embed` browser ESM build (MIT), sha512-verified at build time |
| `/app/npa-foxglove-host.js` | Glue module — the same file the NPA agent UI loads |
| `/data/` | Operator-mounted recordings, served with CORS + byte ranges (no directory listing; `FOXGLOVE_DATA_BROWSE=browse` opts in) |
| `/healthz` | `{"ok":true,...}` liveness/readiness probe |

## What this image is *not*

It does not contain the Foxglove application. The SDK creates an iframe pointing at
a Foxglove deployment you control:

- `https://embed.foxglove.dev/` — Foxglove-hosted; requires a Foxglove organization
  on a plan that allows embedding (Pro / Enterprise / Academic), and users sign in there.
- `https://foxglove.internal.example/` — your self-hosted Foxglove deployment.

Without `?src=` (or `NPA_FOXGLOVE_EMBED_SRC` on the agent) the host page reports that
it is not configured instead of rendering an empty viewer.

## Build

```bash
docker build -t npa-foxglove-embed:0.58.0 \
  -f npa/docker/workbench/foxglove-embed/Dockerfile npa
```

Build args: `FOXGLOVE_EMBED_VERSION`, `FOXGLOVE_EMBED_INTEGRITY` (npm `dist.integrity`),
`FOXGLOVE_EMBED_REGISTRY` (mirror / air-gapped cache). Version and integrity defaults are
kept in sync with `npa.workbench.foxglove` by `npa/tests/docker/test_foxglove_image.py`.

## Run

```bash
docker run --rm -p 8099:8099 \
  -v /path/to/recordings:/srv/data:ro \
  npa-foxglove-embed:0.58.0

# then open, e.g.
# http://localhost:8099/?src=https://embed.foxglove.dev/&org=my-org&mcap=/data/run.mcap
```

Foxglove requires a [secure context](https://developer.mozilla.org/docs/Web/Security/Secure_Contexts):
`localhost` works for local runs; serve it over HTTPS (or behind an HTTPS ingress) anywhere else.

## Security posture

- Runs as `nobody`, no admin API (`admin off`), no automatic HTTPS/ACME.
- No authentication of its own: expose it on a cluster-internal address or behind an
  authenticating proxy. `/data/` is intentionally readable without credentials because the
  cross-origin Foxglove iframe cannot send them — mount only recordings you are willing to
  serve to whoever can reach the port. Directory listing is **off** by default, and
  `/srv/data` is owned by the runtime user rather than world-writable, so a reachable
  service can neither enumerate nor accept files.
- `/data/*` is never compressed so HTTP Range playback keeps working.

## Attribution

`@foxglove/embed` is Copyright Foxglove Technologies, MIT licensed, fetched unmodified from
the npm registry at build time. Caddy is Apache-2.0.
