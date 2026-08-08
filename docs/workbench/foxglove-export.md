# Export MCAP and open it in Foxglove

The NPA agent can export the active run as MCAP, download it, and open the same
recording in the Foxglove web or desktop application. This is a remote-file
flow: it does not upload the recording to Foxglove Cloud, require a Foxglove API
token, or depend on the paid `@foxglove/embed` feature.

## Agent UI

Open the agent's Viewer panel and use **Export / download MCAP**. If the active
run has no published MCAP yet, the backend invokes the existing run conversion
path first. The browser downloads that MCAP and reveals **Open in Foxglove web**
and **Open in Foxglove desktop** controls.

The links follow Foxglove's documented share-link contract:

```text
https://app.foxglove.dev/~/view?ds=remote-file&ds.url=<encoded-absolute-https-url>
```

The desktop link adds `openIn=desktop`, which Foxglove recommends over relying
only on the `foxglove://` scheme. If the desktop application is not installed,
the HTTPS landing page can offer the browser application instead.

## CLI and SDK

Convert and export a local run directory:

```bash
npa workbench foxglove export-run \
  --input-path <run-directory> \
  --output-path run.mcap \
  --recording-url https://<agent-host>/foxglove/data/<random-name>.mcap
```

Build a link for an MCAP already served by the agent:

```bash
npa workbench foxglove open \
  --recording-url https://<agent-host>/foxglove/data/<random-name>.mcap \
  --target web

npa workbench foxglove open \
  --recording-url https://<agent-host>/foxglove/data/<random-name>.mcap \
  --target desktop --launch
```

SDK callers use `npa.sdk.workbench.foxglove.export_run` and
`foxglove_deep_links`. All link values are URL-encoded by the shared helper.

## Reachability and public recording access

Foxglove fetches remote files directly from the user's browser or desktop app.
The recording URL must therefore be absolute HTTPS and reachable from that
machine. A private agent may require VPN access. For an agent using a self-signed
certificate, the user must trust or accept that certificate before Foxglove can
read the file.

The cross-origin Foxglove application cannot send the agent UI's basic-auth
credentials. Nginx therefore serves `/foxglove/data/` without authentication,
with CORS and byte-range support, under a random unguessable filename. Anyone
who obtains that URL can read the recording until the agent prunes it. Do not
use this path for data whose access policy requires authenticated downloads;
use a suitable signed HTTPS object URL or another protected data path instead.

The inline Foxglove embed remains available when configured, and the default
Lichtblick fallback remains the account-free in-page viewer.

## Optional Foxglove API token

Store an API token only in `~/.npa/credentials.yaml`:

```yaml
tokens:
  FOXGLOVE_API_TOKEN: <your-token>
```

Then secure the file:

```bash
chmod 600 ~/.npa/credentials.yaml
```

The loader also recognizes the `FOXGLOVE_API_TOKEN` environment override. The
token is held in a server-side-only credential field: it is excluded from shared
workbench environments, SSH token forwarding, browser configuration, generated
links, and command output. It is not required by the export/open flow.

References: [Foxglove shareable links](https://docs.foxglove.dev/docs/visualization/shareable-links),
[Foxglove remote files](https://docs.foxglove.dev/docs/visualization/connecting/cloud-data).
