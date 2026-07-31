"""Static assets the agent VM's nginx front end serves.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet): the
nginx location block and the embedded viewer's default layout are large, purely
declarative strings with no deploy logic in them. Both are re-imported into
``npa.cli.agent`` so the bootstrap call sites and tests are unchanged.
"""

from __future__ import annotations

import json


def _lichtblick_default_layout_json() -> str:
    """Return the compact JSON for the embedded viewer's default layout.

    A 3D panel with ``/heldout/points`` made visible (colored by its RGBA fields)
    and framed on the workspace, alongside an Image panel bound to ``/camera``. The
    JSON is single-quote-free so it can be injected as an nginx ``sub_filter``
    replacement without escaping.

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
                    # Orbit around the fixed-camera reconstruction's workspace
                    # centroid so the cloud is framed on load (follow-none).
                    "targetOffset": [2.3, -1.2, -0.15],
                    "thetaOffset": 45.0,
                    "fovy": 45.0,
                    "near": 0.1,
                    "far": 5000.0,
                },
                "followTf": "sim2real",
                "followMode": "follow-none",
                "scene": {},
                "topics": {
                    "/heldout/points": {
                        "visible": True,
                        "colorMode": "rgba-fields",
                        "pointSize": 4.0,
                    }
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
        },
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1.0},
        "layout": {
            "first": "3D!npasim2real",
            "second": "Image!npacamera",
            "direction": "row",
            "splitPercentage": 62,
        },
    }
    return json.dumps(layout, separators=(",", ":"))


def _nginx_agent_site_body(
    *,
    backend_port: int,
    rerun_port: int,
    lichtblick_port: int = 8081,
) -> str:
    """Shared nginx locations for the agent UI (HTTP and HTTPS server blocks)."""
    # Imported lazily: npa.cli.agent imports this module.
    from npa.cli.agent import AGENT_UI_VERSION, LICHTBLICK_DEFAULT_LAYOUT_PLACEHOLDER

    lichtblick_default_layout = _lichtblick_default_layout_json()
    lichtblick_layout_placeholder = LICHTBLICK_DEFAULT_LAYOUT_PLACEHOLDER
    return f"""  auth_basic "NPA Agent";
  auth_basic_user_file /etc/nginx/.npa-agent-htpasswd;
  # Describe-this / multimodal chat posts JPEG data-URLs; default 1m rejects them (413 → browser Failed to fetch).
  client_max_body_size 32m;
  location = /healthz {{
    auth_basic off;
    default_type application/json;
    return 200 '{{"ok":true,"service":"npa-agent","welcome":"/welcome","ui":"/","ui_version":"{AGENT_UI_VERSION}"}}';
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
  location /rerun/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Cross-Origin-Resource-Policy "cross-origin" always;
    # .rrd carries msgpack + metadata that still gzips usefully; the frame
    # payloads are now JPEG-encoded so the win is modest but the transfer is
    # smaller and TTFB unaffected (nginx streams as it compresses).
    gzip on;
    gzip_types application/octet-stream;
    gzip_min_length 1024;
  }}
  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
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
  location /lichtblick/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
    # nginx's static module already emits `Accept-Ranges: bytes`; do NOT add it again
    # (a duplicate makes the browser join it to "bytes, bytes", which fails Lichtblick's
    # `headers.get("accept-ranges") === "bytes"` range-support check).
    #
    # Deliberately NO Access-Control-* headers here. A run's MCAP carries camera
    # frames, VLM critiques and reward signals, and this location is unauthenticated
    # (wasm/worker fetches cannot carry basic auth). Granting `Allow-Origin: *` would
    # let any web page a viewer visits read those recordings off this host. The embed
    # never needs it: the viewer document is proxied from this same origin under
    # /lichtblick/ and the UI pins ds.url to window.location.origin, so the fetch is
    # same-origin — which also makes Accept-Ranges readable without Expose-Headers.
    add_header Cross-Origin-Resource-Policy "same-origin" always;
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
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header Cache-Control "public, max-age=3600" always;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;
  }}"""
