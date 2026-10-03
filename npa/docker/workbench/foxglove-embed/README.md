# npa-foxglove-embed

[Foxglove guide](../../../../docs/workbench/foxglove-export.md) · [Workbench docs](../../../../docs/workbench/README.md)

Static host for the [Foxglove embedding SDK](https://docs.foxglove.dev/docs/embed/typescript-sdk)
(`@foxglove/embed`), the shared NPA glue module, and MCAP/bag recordings.

| Path | Contents |
| --- | --- |
| `/` | Standalone host page (`?src=`, `?org=`, `?mcap=`, `?ws=`, `?layout=`, `?theme=`, `?autoplay=1`) |
| `/sdk/` | Pinned `@foxglove/embed` browser ESM build (MIT), sha512-verified at build time |
| `/app/npa-foxglove-host.js` | Glue module — the same file the NPA agent UI loads |
| `/data/` | Operator-mounted recordings, served with CORS + byte ranges (no directory listing; `FOXGLOVE_DATA_BROWSE=browse` opts in) |
| `/healthz` | `{"ok":true,...}` liveness/readiness probe |

## Prerequisites

The SDK embeds a separately hosted Foxglove application. Before starting the
container, choose a deployment that your users can access:

- `https://embed.foxglove.dev/` — Foxglove-hosted; requires a Foxglove organization
  on a plan that allows embedding (Pro / Enterprise / Academic), and users sign in there.
- `https://foxglove.internal.example/` — your self-hosted Foxglove deployment.

Without `?src=` (or `NPA_FOXGLOVE_EMBED_SRC` on the agent) the host page reports that
it is not configured instead of rendering an empty viewer.

## Build

From the repository root with Docker installed:

```bash
docker build -t npa-foxglove-embed:0.58.0 \
  -f npa/docker/workbench/foxglove-embed/Dockerfile npa
```

Build args: `FOXGLOVE_EMBED_VERSION`, `FOXGLOVE_EMBED_INTEGRITY` (npm `dist.integrity`),
`FOXGLOVE_EMBED_REGISTRY` (mirror / air-gapped cache). Version and integrity defaults are
kept in sync with `npa.workbench.foxglove` by `npa/tests/docker/test_foxglove_image.py`.

## Run locally

Replace `/path/to/recordings` with a directory containing your MCAP recording.
Keep the terminal open while viewing; Ctrl-C stops and removes the container.

```bash
docker run --rm -p 127.0.0.1:8099:8099 \
  -v /path/to/recordings:/srv/data:ro \
  npa-foxglove-embed:0.58.0

# then open, e.g.
# http://localhost:8099/?src=https://embed.foxglove.dev/&org=my-org&mcap=/data/run.mcap
```

Check `http://localhost:8099/healthz` for `ok: true`, then open the example
viewer URL with your organization and recording filename. A healthy static host
does not establish that Foxglove sign-in or recording access works. If the host
reports an unconfigured viewer, supply `src`; if playback fails, verify that
`/data/run.mcap` resolves and the selected Foxglove deployment can reach it.

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
