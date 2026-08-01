"""npa.workbench.foxglove - Foxglove embedded-viewer assets and MCAP tooling.

Single source of truth for the pinned `@foxglove/embed` TypeScript SDK
(https://docs.foxglove.dev/docs/embed/typescript-sdk). The SDK is MIT licensed,
has no runtime dependencies, and ships browser-ready ESM in its npm `dist/`
directory, so NPA serves it verbatim instead of vendoring a bundled copy:

- the ``npa-foxglove-embed`` container fetches it at build time,
- the agent VM fetches it at bootstrap time,

both through ``npa/docker/workbench/foxglove-embed/install-sdk.sh`` with the same
pinned version and Subresource-Integrity-style ``sha512`` digest published by the
npm registry.

The embedded *viewer application* itself is hosted by Foxglove
(``https://embed.foxglove.dev/``) or by a self-hosted Foxglove deployment; it is
not redistributed here. Callers must therefore treat "SDK available" and "embed
source configured" as separate conditions (see :func:`sdk_assets_present`).
"""

from __future__ import annotations

from pathlib import Path

# Pinned upstream SDK release. Bump together with the integrity digest, the
# Dockerfile ARG defaults, SUPPORTED_TOOL_VERSIONS, and pyproject's
# [tool.npa.supported-tools] (enforced by npa/tests/docker/test_foxglove_image.py).
FOXGLOVE_EMBED_SDK_VERSION = "0.58.0"

# `dist.integrity` reported by the npm registry for the pinned version. The
# install script recomputes it from the downloaded tarball and refuses to install
# on mismatch.
FOXGLOVE_EMBED_SDK_INTEGRITY = (
    "sha512-hNxqEQWPk2Wm0KmDlNs3Y0TTEl9Wm+4CuppBZcLzK8j8m2EcwbbCVWg43oCsf5HJgwXt7KYorIdoMO7CICQ7Vg=="
)

# npm registry tarball URL template. Registry host is overridable by operators
# through the install script's --registry flag (mirrors / air-gapped caches).
FOXGLOVE_EMBED_SDK_TARBALL_TEMPLATE = (
    "{registry}/@foxglove/embed/-/embed-{version}.tgz"
)
FOXGLOVE_EMBED_DEFAULT_REGISTRY = "https://registry.npmjs.org"

# Default embed application source documented by the Foxglove TypeScript SDK.
# Self-hosted deployments override it (NPA_FOXGLOVE_EMBED_SRC / --foxglove-embed-src).
DEFAULT_FOXGLOVE_EMBED_SRC = "https://embed.foxglove.dev/"

# Container service port for npa-foxglove-embed (static asset + data host).
FOXGLOVE_SERVICE_PORT = 8099

# Agent-VM install root; the bootstrap creates sdk/, app/ and data/ beneath it.
FOXGLOVE_ASSET_ROOT = "/opt/npa-agent/foxglove"

# Files the SDK's browser-ready ESM build is made of (dist/ of the npm package).
SDK_FILES: tuple[str, ...] = (
    "index.js",
    "FoxgloveViewer.js",
    "types.js",
    "layout.generated.js",
)

# Artifact extensions the Foxglove viewer can open directly.
FOXGLOVE_ARTIFACT_EXTENSIONS: tuple[str, ...] = (
    ".mcap",
    ".bag",
    ".db3",
    ".ulg",
    ".ulog",
)

# First 8 bytes of every MCAP file (magic record, MCAP spec).
MCAP_MAGIC = b"\x89MCAP0\r\n"


def sdk_tarball_url(
    version: str = FOXGLOVE_EMBED_SDK_VERSION,
    *,
    registry: str = FOXGLOVE_EMBED_DEFAULT_REGISTRY,
) -> str:
    """Return the npm tarball URL for a pinned ``@foxglove/embed`` release."""
    return FOXGLOVE_EMBED_SDK_TARBALL_TEMPLATE.format(
        registry=str(registry or FOXGLOVE_EMBED_DEFAULT_REGISTRY).rstrip("/"),
        version=str(version or FOXGLOVE_EMBED_SDK_VERSION).strip(),
    )


def sdk_assets_present(root: str | Path) -> tuple[bool, str]:
    """Return ``(ready, reason)`` for an installed SDK asset directory.

    ``root`` is the directory the install script wrote (``.../foxglove/sdk``).
    The reason string is operator-facing and safe to surface in an API payload.
    """
    base = Path(str(root or "")).expanduser()
    if not base.is_dir():
        return False, f"Foxglove SDK assets are not installed at {base}."
    missing = [name for name in SDK_FILES if not (base / name).is_file()]
    if missing:
        return False, (
            "Foxglove SDK assets are incomplete at "
            f"{base} (missing: {', '.join(missing)})."
        )
    return True, ""


__all__ = [
    "DEFAULT_FOXGLOVE_EMBED_SRC",
    "FOXGLOVE_ARTIFACT_EXTENSIONS",
    "FOXGLOVE_ASSET_ROOT",
    "FOXGLOVE_EMBED_DEFAULT_REGISTRY",
    "FOXGLOVE_EMBED_SDK_INTEGRITY",
    "FOXGLOVE_EMBED_SDK_TARBALL_TEMPLATE",
    "FOXGLOVE_EMBED_SDK_VERSION",
    "FOXGLOVE_SERVICE_PORT",
    "MCAP_MAGIC",
    "SDK_FILES",
    "sdk_assets_present",
    "sdk_tarball_url",
]
