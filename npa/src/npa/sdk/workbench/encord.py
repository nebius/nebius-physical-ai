"""SDK surface for the Encord workbench tool.

Every function is a thin, keyword-only wrapper over the implementation in
``npa.workbench.encord``; the CLI calls these same functions.
"""

from __future__ import annotations

from typing import Any

from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    PushReceipt,
)


def push(
    *,
    input_path: str,
    integration: str,
    folder: str,
    output_path: str,
    dataset: str = "",
    media: str = DEFAULT_MEDIA_FILTER,
    transfer: str = DEFAULT_TRANSFER,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
) -> PushReceipt:
    """Register S3 media in place into an Encord folder; write a receipt."""

    from npa.workbench.encord.push import run_push

    return run_push(
        input_path=input_path,
        integration=integration,
        folder=folder,
        output_path=output_path,
        dataset=dataset,
        media=media,
        transfer=transfer,
        poll_timeout_seconds=poll_timeout_seconds,
        workflow_run=workflow_run,
        user_client=user_client,
        storage_client=storage_client,
    )


def cleanup(
    *,
    title_prefix: str,
    dry_run: bool = False,
    user_client: Any = None,
) -> dict[str, Any]:
    """Delete run-scoped Encord folders/collections/presets by title prefix."""

    from npa.workbench.encord.cleanup import run_cleanup

    return run_cleanup(
        title_prefix=title_prefix,
        dry_run=dry_run,
        user_client=user_client,
    )


__all__ = [
    "PushReceipt",
    "cleanup",
    "push",
]
