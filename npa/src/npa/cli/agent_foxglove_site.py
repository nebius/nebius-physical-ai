"""nginx serving policy for the agent's Foxglove assets and recordings.

Kept out of ``agent.py`` (and out of the embedded backend, which never renders
nginx config) so the serving rules the embedded viewer depends on live in one
reviewable place:

- ``/foxglove/`` — SDK + glue module, same-origin subresources of the
  authenticated UI, so they stay behind basic auth.
- ``/foxglove/data/`` — published recordings. The Foxglove viewer runs on a
  different origin and cannot send the agent's basic-auth credentials, so this
  path is unauthenticated (same trust model as the existing public
  ``/rerun/recordings/`` path, but with random file names). It must be
  uncompressed and range-capable because MCAP playback streams with HTTP Range
  requests, and it must answer the CORS preflight those requests trigger.
"""

from __future__ import annotations

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
    add_header Accept-Ranges bytes always;
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
