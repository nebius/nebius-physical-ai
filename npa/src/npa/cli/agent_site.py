"""nginx site policy for the agent VM (kept out of the `agent.py` monolith).

Every location block the agent serves lives here — API proxy, Rerun bundle +
recordings, the Lichtblick (OSS MCAP viewer) proxy and its recordings, the
Foxglove SDK assets and their CORS/byte-range data path, and the UI itself — so
the serving rules the embedded viewers depend on are reviewable in one file
instead of buried in the bootstrap f-string.

Serving rules worth knowing:

- ``/lichtblick/recordings/`` is same-origin and deliberately grants **no** CORS:
  the OSS viewer runs in-page.
- ``/foxglove/data/`` is unauthenticated **with** CORS: the official Foxglove app
  runs on a different origin and cannot send the agent's basic-auth credentials.
  It must stay uncompressed and range-capable (MCAP streams by byte range) and
  answer the preflight a Range request triggers.
- SDK + glue assets stay behind basic auth; they are same-origin subresources.
"""

from __future__ import annotations

import json

DEFAULT_LICHTBLICK_PORT = 8081

FOXGLOVE_ASSET_ROOT = "/opt/npa-agent/foxglove"


def foxglove_nginx_locations(*, asset_root: str = FOXGLOVE_ASSET_ROOT) -> str:
    """Return the nginx ``location`` blocks for the agent's Foxglove serving."""
    root = str(asset_root or FOXGLOVE_ASSET_ROOT).rstrip("/")
    return f"""  location /foxglove/data/ {{
    # The Foxglove embedded viewer runs on a different origin and cannot send the
    # agent's basic-auth credentials, so published recordings are readable without
    # auth (same trust model as /rerun/recordings/, but with random file names).
    auth_basic off;
    alias {root}/data/;
    default_type application/octet-stream;
    # Byte ranges carry MCAP playback; a compressed response would break them.
    gzip off;
    # nginx's static module already emits exactly one `Accept-Ranges: bytes`.
    # Adding it here produces `bytes, bytes` in browser clients and can make
    # strict MCAP data-source capability checks reject the recording.
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, If-Range, Content-Type" always;
    add_header Access-Control-Expose-Headers "Accept-Ranges, Content-Range, Content-Length" always;
    add_header Cross-Origin-Resource-Policy "cross-origin" always;
    add_header Cache-Control "no-cache" always;
    # A Range request is not a "simple" CORS request, so the viewer preflights it.
    if ($request_method = OPTIONS) {{
      return 204;
    }}
  }}
  location /foxglove/ {{
    # SDK + glue module: same-origin subresources of the authenticated UI.
    alias {root}/;
    add_header Cache-Control "public, max-age=3600" always;
  }}
"""


# Placeholder token the Lichtblick web bundle ships in its index.html inline
# script: ``LICHTBLICK_SUITE_DEFAULT_LAYOUT = [/*...PLACEHOLDER*/][0];``. Replacing
# the comment with a layout object is the upstream-supported self-hosting hook, so
# the embedded viewer opens with the sim2real point cloud + camera already shown
# (Lichtblick otherwise hides point-cloud topics and picks no image topic).
LICHTBLICK_DEFAULT_LAYOUT_PLACEHOLDER = (
    "/*LICHTBLICK_SUITE_DEFAULT_LAYOUT_PLACEHOLDER*/"
)


def _lichtblick_default_layout_json() -> str:
    """Return the compact JSON for the embedded viewer's default layout.

    A 3D panel exposes either the established ``/heldout/points`` scene or the
    converter's real ``/trajectory`` state-space cloud, alongside two synchronized
    cameras and a real execution-metrics Plot. The JSON is single-quote-free so it
    can be injected as an nginx ``sub_filter`` replacement without escaping.

    ``rgba-fields`` requires the cloud to carry all four of red/green/blue/alpha;
    ``npa.workbench.lichtblick.pack_pointcloud_bytes`` emits the opaque alpha
    channel that makes this mode available (without it the panel falls back to a
    synthetic colormap and the cloud loses its captured colours). ``/camera`` is
    always emitted by the sim2real MCAP writer — from the held-out episode when
    there is one, else mirrored from the first rollout.
    """

    layout = {
        "configById": {
            "3D!npasim2real": {
                "cameraState": {
                    "distance": 7.0,
                    "perspective": True,
                    "phi": 55.0,
                    "target": [0.0, 0.0, 0.0],
                    "targetOffset": [2.3, -1.2, -0.15],
                    "thetaOffset": 45.0,
                    "fovy": 45.0,
                    "near": 0.1,
                    "far": 5000.0,
                },
                "followTf": "world",
                "followMode": "follow-none",
                "scene": {},
                "topics": {
                    "/heldout/points": {
                        "visible": True,
                        "colorMode": "rgba-fields",
                        "pointSize": 4.0,
                    },
                    "/trajectory": {
                        "visible": True,
                        "colorMode": "rgba-fields",
                        "pointSize": 7.0,
                    },
                },
                "layers": {
                    "npa-grid": {
                        "visible": True,
                        "frameLocked": True,
                        "label": "Grid",
                        "instanceId": "npa-grid",
                        "layerId": "foxglove.Grid",
                        "size": 10,
                        "divisions": 10,
                        "lineWidth": 1,
                        "color": "#248eff",
                        "position": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "order": 1,
                    }
                },
            },
            "Image!npacamera": {"imageMode": {"imageTopic": "/camera"}},
            "Image!npaworkspace": {"imageMode": {"imageTopic": "/camera/workspace"}},
            "Plot!npametrics": {
                "paths": [
                    {"value": "/metrics/execution.reward", "enabled": True},
                    {"value": "/metrics/execution.progress", "enabled": True},
                    {"value": "/metrics/execution.state_norm", "enabled": True},
                ],
                "showLegend": True,
                "isSynced": True,
            },
        },
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1.0},
        "layout": {
            "first": {
                "first": "Image!npacamera",
                "second": "Image!npaworkspace",
                "direction": "column",
                "splitPercentage": 50,
            },
            "second": {
                "first": "3D!npasim2real",
                "second": "Plot!npametrics",
                "direction": "column",
                "splitPercentage": 62,
            },
            "direction": "row",
            "splitPercentage": 50,
        },
    }
    return json.dumps(layout, separators=(",", ":"))


def nginx_agent_site_body(
    *,
    backend_port: int,
    rerun_port: int,
    ui_version: str,
    lichtblick_port: int = DEFAULT_LICHTBLICK_PORT,
) -> str:
    """Shared nginx locations for the agent UI (HTTP and HTTPS server blocks)."""
    foxglove_locations = foxglove_nginx_locations()
    lichtblick_default_layout = _lichtblick_default_layout_json()
    lichtblick_layout_placeholder = LICHTBLICK_DEFAULT_LAYOUT_PLACEHOLDER
    return f"""  auth_basic "NPA Agent";
  auth_basic_user_file /etc/nginx/.npa-agent-htpasswd;
  # Describe-this / multimodal chat posts JPEG data-URLs; default 1m rejects them (413 → browser Failed to fetch).
  client_max_body_size 32m;
  location = /healthz {{
    auth_basic off;
    default_type application/json;
    return 200 '{{"ok":true,"service":"npa-agent","welcome":"/welcome","ui":"/","ui_version":"{ui_version}"}}';
  }}
  location = /welcome {{
    auth_basic off;
    alias /opt/npa-agent/welcome.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location = /login-help.html {{
    auth_basic off;
    alias /opt/npa-agent/login-help.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location /api/ {{
    proxy_pass http://127.0.0.1:{backend_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 30s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    client_max_body_size 32m;
  }}
  location /assets/api/ {{
    rewrite ^/assets/api/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{backend_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    client_max_body_size 32m;
  }}
  location ~ "^/rerun/recordings/(cap-[A-Za-z0-9_-]{{43}}\\.rrd)$" {{
    # Rerun WASM cannot attach HTTP Basic credentials to its recording fetch.
    # The backend therefore publishes one random 256-bit, per-load capability
    # filename and deletes the previous capability. Only that unguessable path
    # is anonymously readable; fixed/run-derived recording names remain denied.
    auth_basic off;
    alias /opt/npa-agent/recordings/$1;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
    add_header Cross-Origin-Resource-Policy "same-origin" always;
    # .rrd carries msgpack + metadata that still gzips usefully; the frame
    # payloads are now JPEG-encoded so the win is modest but the transfer is
    # smaller and TTFB unaffected (nginx streams as it compresses).
    gzip on;
    gzip_types application/octet-stream;
    gzip_min_length 1024;
  }}
  location /rerun/recordings/ {{
    return 404;
  }}
{foxglove_locations}  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
    auth_basic off;
    rewrite ^/rerun/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{rerun_port};
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    gzip on;
    gzip_types application/wasm application/javascript text/javascript image/svg+xml;
    gzip_min_length 256;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
  }}
  location /rerun/ {{
    auth_basic off;
    proxy_pass http://127.0.0.1:{rerun_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header Cache-Control "public, max-age=3600" always;
  }}
  location = /lichtblick/recordings/sim2real.mcap {{
    auth_basic off;
    proxy_pass http://127.0.0.1:{backend_port}/lichtblick-recordings/sim2real.mcap;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header Range $http_range;
    proxy_set_header If-Range $http_if_range;
    proxy_force_ranges on;
    gzip off;
    # Keep binary identity intact through browser/test proxies. Transforming the
    # response to Brotli/chunked removes Content-Length and makes Lichtblick abort
    # before its first byte-range request.
    add_header Cache-Control "no-store, no-transform" always;
    # Lichtblick opens the remote file from a browser worker. Chromium applies
    # the CORS-exposed-header filter there even when the viewer is reverse-
    # proxied under this host. Permit only this exact agent origin (never `*`)
    # and expose the size/range headers its BrowserHttpReader requires.
    add_header Access-Control-Allow-Origin "$scheme://$http_host" always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, If-Range" always;
    add_header Access-Control-Expose-Headers "Accept-Ranges, Content-Range, Content-Length, ETag, Last-Modified, X-NPA-File-Size" always;
    # FileResponse emits one Accept-Ranges header, Content-Length on HEAD, and
    # Content-Range for partial GETs. Do not duplicate those headers in nginx.
    #
    # The location is unauthenticated because the worker cannot send basic auth,
    # so never reflect arbitrary Origin values or use wildcard CORS.
    add_header Cross-Origin-Resource-Policy "same-origin" always;
    if ($request_method = OPTIONS) {{
      return 204;
    }}
  }}
  location = /lichtblick/ {{
    # Exact-match the viewer document so we can inject the sim2real default layout
    # into its index.html (the point cloud + camera show without manual topic
    # enabling). Assets keep long caching via the prefix location below.
    auth_basic off;
    proxy_pass http://127.0.0.1:{lichtblick_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    # Force an uncompressed upstream response so sub_filter can rewrite the HTML.
    proxy_set_header Accept-Encoding "";
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    sub_filter_once on;
    sub_filter_types text/html;
    sub_filter '{lichtblick_layout_placeholder}' '{lichtblick_default_layout}';
    add_header Cache-Control "no-store" always;
  }}
  location /lichtblick/ {{
    auth_basic off;
    proxy_pass http://127.0.0.1:{lichtblick_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    # BrowserHttpReader normally requires Content-Length on its initial GET.
    # Some browser automation/network intermediaries expose a truthful chunked
    # response instead. Lichtblick 1.26 has no fallback header, so teach this
    # pinned static bundle to accept the canonical file's X-NPA-File-Size. The
    # replacement does not alter the MCAP transport or its subsequent ranges.
    proxy_set_header Accept-Encoding "";
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    sub_filter_once on;
    sub_filter_types application/javascript text/javascript;
    sub_filter 'const c=i.headers.get("content-length");if(c==null)' 'const c=i.headers.get("content-length")??i.headers.get("x-npa-file-size");if(c==null)';
    add_header Cache-Control "public, max-age=3600" always;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;
  }}"""
