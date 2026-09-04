"""SDK surface for the Encord workbench tool.

Every function is a thin, keyword-only wrapper over the implementation in
``npa.workbench.encord``; the CLI calls these same functions.
"""

from __future__ import annotations

from typing import Any

from npa.workbench.encord.schemas import (
    DEFAULT_CURATE_POLL_SECONDS,
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
    CurateReceipt,
    PullManifest,
    PushReceipt,
    RoundtripReport,
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


def curate(
    *,
    folder: str,
    filters: list[str],
    collection: str,
    output_path: str,
    workflow_run: str = "",
    poll_seconds: float = DEFAULT_CURATE_POLL_SECONDS,
    user_client: Any = None,
    storage_client: Any = None,
) -> CurateReceipt:
    """Headlessly curate a folder into a Collection; write a receipt."""

    from npa.workbench.encord.curate import run_curate

    return run_curate(
        folder=folder,
        filters=filters,
        collection=collection,
        output_path=output_path,
        workflow_run=workflow_run,
        poll_seconds=poll_seconds,
        user_client=user_client,
        storage_client=storage_client,
    )


def pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    user_client: Any = None,
    storage_client: Any = None,
) -> PullManifest:
    """Materialize a curated Encord source to S3; write a lineage manifest."""

    from npa.workbench.encord.pull import run_pull

    return run_pull(
        source=source,
        source_id=source_id,
        output_path=output_path,
        workflow_run=workflow_run,
        user_client=user_client,
        storage_client=storage_client,
    )


def verify(
    *,
    receipt_uri: str,
    manifest_uri: str,
    output_path: str,
    workflow_run: str = "",
    storage_client: Any = None,
) -> RoundtripReport:
    """Verify a push receipt against a pull manifest; write the report."""

    from npa.workbench.encord.verify import run_verify

    return run_verify(
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        output_path=output_path,
        workflow_run=workflow_run,
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


def system_info() -> dict[str, Any]:
    """Runtime information for the Encord tool; makes no Encord API call."""

    from npa.workbench.encord.system_info import system_info_payload

    return system_info_payload()


__all__ = [
    "CurateReceipt",
    "PullManifest",
    "PushReceipt",
    "RoundtripReport",
    "cleanup",
    "curate",
    "pull",
    "push",
    "system_info",
    "verify",
]
