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
    # Adding it here yields `bytes, bytes`, which strict MCAP readers reject.
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


def _lichtblick_learning_layout_json() -> str:
    """Return the replay-first layout for offline policy-learning MCAPs.

    Learning recordings do not contain a reconstructed point cloud. The injected
    page script replaces the placeholder Image topic with the report-validated
    ``npa.camera`` query value. Give that factual held-out camera the full canvas;
    predicted/expert/error series remain available in the Topics sidebar and in
    the companion Rerun blueprint.
    """

    layout = {
        "configById": {
            "Image!npalearningcamera": {
                "imageMode": {"imageTopic": "/camera/__NPA_PRIMARY_CAMERA__"}
            }
        },
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1.0},
        "layout": "Image!npalearningcamera",
    }
    return json.dumps(layout, separators=(",", ":"))


def _lichtblick_default_layout_script() -> str:
    """Select a validated primary-camera layout and size-aware classic worker."""

    learning = _lichtblick_learning_layout_json()
    sim2real = _lichtblick_default_layout_json()
    return (
        "(()=>{const query=new URLSearchParams(window.location.search);"
        'const hintedSize=Number(query.get("npa.size")||0);'
        'if(Number.isSafeInteger(hintedSize)&&hintedSize>0&&typeof window.Worker==="function"){'
        "const NativeWorker=window.Worker;window.Worker=function(scriptUrl,options){"
        'if(options&&options.type==="module")return new NativeWorker(scriptUrl,options);'
        "const absolute=new URL(String(scriptUrl),window.location.href).href;"
        'const wrapped=new URL("/lichtblick/npa-worker.js",window.location.origin);'
        'wrapped.searchParams.set("npa.size",String(hintedSize));'
        'wrapped.searchParams.set("npa.target",absolute);'
        "return new NativeWorker(wrapped.href,options);};window.Worker.prototype=NativeWorker.prototype;}"
        'if(query.get("npa.layout")!=="learning")return '
        f"{sim2real};"
        f"const selected={learning};"
        'const camera=String(query.get("npa.camera")||"");'
        'if(!camera||!camera.split("").every((char)=>'
        '"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-".includes(char)))'
        'throw new Error("invalid primary camera");'
        'selected.configById["Image!npalearningcamera"].imageMode.imageTopic="/camera/"+camera;'
        "return selected;})()"
    )


def _lichtblick_worker_script() -> str:
    """Validate and import one same-origin Lichtblick worker asset.

    Cypress removes ``Content-Length`` while proxying the initial full MCAP GET.
    The server-observed size is therefore restored only for that same-origin
    recording request; byte-range bodies pass through untouched.
    """

    return (
        "(()=>{const params=new URLSearchParams(self.location.search);"
        'const sizeHint=Number(params.get("npa.size")||0);'
        'const rawTarget=params.get("npa.target")||"";'
        'const reject=(message)=>{throw new Error("invalid Lichtblick worker target: "+message);};'
        'const hasUnsafeChar=(value)=>value.split("").some((char)=>'
        "{const code=char.charCodeAt(0);return code===92||code<32||code===127;});"
        'if(!rawTarget||hasUnsafeChar(rawTarget)||rawTarget.startsWith("//")||rawTarget.includes("#"))reject("target");'
        'let target;try{target=new URL(rawTarget,self.location.origin);}catch(_error){reject("url");}'
        'if((target.protocol!=="http:"&&target.protocol!=="https:")||target.origin!==self.location.origin||'
        'target.username||target.password||target.search||target.hash)reject("origin");'
        "let decoded=target.pathname;for(let depth=0;depth<3;depth++){let next;"
        'try{next=decodeURIComponent(decoded);}catch(_error){reject("encoding");}'
        "if(next===decoded)break;decoded=next;}"
        'if(hasUnsafeChar(decoded)||decoded.includes("?")||decoded.includes("#")||'
        'decoded.split("/").some((part)=>part==="."||part==="..")||'
        '!decoded.startsWith("/lichtblick/")||decoded.startsWith("/lichtblick/recordings/")||'
        'decoded==="/lichtblick/npa-worker.js"||!decoded.endsWith(".js"))reject("path");'
        'if(!Number.isSafeInteger(sizeHint)||sizeHint<=0)reject("size");'
        "const nativeFetch=self.fetch.bind(self);self.fetch=async(input,init)=>{"
        "const response=await nativeFetch(input,init);try{"
        'const rawUrl=typeof input==="string"?input:String((input&&input.url)||input||"");'
        "const url=new URL(rawUrl,self.location.href);const headersIn=new Headers((init&&init.headers)||undefined);"
        'if(url.origin===self.location.origin&&url.pathname.startsWith("/lichtblick/recordings/")&&'
        'url.pathname.endsWith(".mcap")&&!headersIn.has("range")&&'
        'response.headers.get("accept-ranges")==="bytes"){const headers=new Headers(response.headers);'
        'headers.set("content-length",String(sizeHint));return new Response(response.body,{'
        "status:response.status,statusText:response.statusText,headers});}}catch(_error){}return response;};"
        "importScripts(target.href);})()"
    )


def nginx_agent_site_body(
    *,
    backend_port: int,
    rerun_port: int,
    ui_version: str = "",
    lichtblick_port: int = DEFAULT_LICHTBLICK_PORT,
) -> str:
    """Shared nginx locations for the agent UI (HTTP and HTTPS server blocks)."""
    if not ui_version:
        from npa.cli.agent import AGENT_UI_VERSION

        ui_version = AGENT_UI_VERSION
    foxglove_locations = foxglove_nginx_locations()
    lichtblick_default_layout = _lichtblick_default_layout_script()
    lichtblick_worker = _lichtblick_worker_script()
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
  location /lichtblick/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    # Remote-file readers require the authoritative Content-Length on their
    # initial GET. Compression changes it to chunked transfer encoding after
    # browser automation proxies decode the body, so keep native MCAP bytes.
    gzip off;
    add_header Cache-Control "no-cache, no-transform" always;
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
  location = /lichtblick/npa-worker.js {{
    auth_basic off;
    default_type application/javascript;
    add_header Cache-Control "no-store" always;
    add_header Cross-Origin-Resource-Policy "same-origin" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Content-Security-Policy "default-src 'none'; script-src 'self'; connect-src 'self'" always;
    return 200 '{lichtblick_worker}';
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
    add_header Content-Security-Policy "default-src 'self' blob: data:; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob:; worker-src 'self' blob:; connect-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'self'" always;
    add_header X-Content-Type-Options "nosniff" always;
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
    add_header Content-Security-Policy "default-src 'self' blob: data:; script-src 'self' 'wasm-unsafe-eval' blob:; worker-src 'self' blob:; connect-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'self'" always;
    add_header X-Content-Type-Options "nosniff" always;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;
  }}"""
