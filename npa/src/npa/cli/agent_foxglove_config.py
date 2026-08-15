"""Foxglove deployment options and durable, non-secret viewer settings."""

from __future__ import annotations

import os
from collections.abc import Mapping

import typer

from npa.agent_backend.foxglove_cloud import (
    FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV,
    resolve_cloud_import_timeout_seconds,
)


class FoxgloveSettingsError(ValueError):
    """Expected, operator-correctable Foxglove setting validation failure."""

    def __init__(self, message: str, *, setting: str) -> None:
        super().__init__(message)
        self.setting = setting


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
    cloud_import_timeout_seconds: str = "",
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
        "cloud_import_timeout_seconds": choose(
            cloud_import_timeout_seconds,
            FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV,
            "cloud_import_timeout_seconds",
        ),
    }
    try:
        cloud_timeout = resolve_cloud_import_timeout_seconds(
            settings["cloud_import_timeout_seconds"]
        )
    except ValueError as exc:
        raise FoxgloveSettingsError(
            str(exc), setting="cloud_import_timeout_seconds"
        ) from exc
    settings["cloud_import_timeout_seconds"] = f"{cloud_timeout:g}"
    backend = settings["viewer_backend"]
    if backend not in {"", "foxglove-sdk", "self-hosted"}:
        raise FoxgloveSettingsError(
            "foxglove viewer backend must be 'foxglove-sdk' or 'self-hosted'",
            setting="viewer_backend",
        )
    if backend == "foxglove-sdk" and not settings["embed_src"]:
        raise FoxgloveSettingsError(
            "foxglove-sdk requires --foxglove-embed-src or NPA_FOXGLOVE_EMBED_SRC",
            setting="embed_src",
        )
    return settings


__all__ = [
    "embed_src_option",
    "FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV",
    "FoxgloveSettingsError",
    "live_url_option",
    "org_slug_option",
    "resolve_settings",
    "viewer_backend_option",
]
