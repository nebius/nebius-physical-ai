"""Foxglove deployment options and durable, non-secret viewer settings."""

from __future__ import annotations

import os
from collections.abc import Mapping

import typer


def embed_src_option() -> str:
    return typer.Option(
        "",
        "--foxglove-embed-src",
        help=(
            "Foxglove embed application URL for the viewer pane "
            "(default: $NPA_FOXGLOVE_EMBED_SRC; unset keeps the explicit "
            "self-hosted fallback)."
        ),
    )


def viewer_backend_option() -> str:
    return typer.Option(
        "",
        "--foxglove-viewer-backend",
        help=(
            "Viewer backend: foxglove-sdk or self-hosted "
            "(default: $NPA_FOXGLOVE_VIEWER_BACKEND, then capability-based selection)."
        ),
    )


def org_slug_option() -> str:
    return typer.Option(
        "",
        "--foxglove-org-slug",
        help=(
            "Foxglove organization slug users should sign into "
            "(default: $NPA_FOXGLOVE_ORG_SLUG)."
        ),
    )


def live_url_option() -> str:
    return typer.Option(
        "",
        "--foxglove-live-url",
        help=(
            "Optional live ws:// or wss:// Foxglove/ROS-bridge URL for the viewer pane."
        ),
    )


def resolve_settings(
    *,
    embed_src: str = "",
    viewer_backend: str = "",
    org_slug: str = "",
    live_url: str = "",
    saved: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve CLI > environment > saved settings and validate the backend."""
    previous = saved if isinstance(saved, Mapping) else {}
    env = environ if environ is not None else os.environ

    def choose(value: str, env_name: str, key: str) -> str:
        return str(value or env.get(env_name, "") or previous.get(key, "")).strip()

    settings = {
        "embed_src": choose(embed_src, "NPA_FOXGLOVE_EMBED_SRC", "embed_src"),
        "viewer_backend": choose(
            viewer_backend, "NPA_FOXGLOVE_VIEWER_BACKEND", "viewer_backend"
        ).lower(),
        "org_slug": choose(org_slug, "NPA_FOXGLOVE_ORG_SLUG", "org_slug"),
        "live_url": choose(live_url, "NPA_FOXGLOVE_LIVE_URL", "live_url"),
    }
    backend = settings["viewer_backend"]
    if backend not in {"", "foxglove-sdk", "self-hosted"}:
        raise ValueError(
            "foxglove viewer backend must be 'foxglove-sdk' or 'self-hosted'"
        )
    if backend == "foxglove-sdk" and not settings["embed_src"]:
        raise ValueError(
            "foxglove-sdk requires --foxglove-embed-src or NPA_FOXGLOVE_EMBED_SRC"
        )
    return settings


__all__ = [
    "embed_src_option",
    "live_url_option",
    "org_slug_option",
    "resolve_settings",
    "viewer_backend_option",
]
