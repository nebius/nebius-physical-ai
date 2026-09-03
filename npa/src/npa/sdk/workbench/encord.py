"""Thin SDK surface for the Encord transport."""

from __future__ import annotations

from typing import Any, Callable

from npa.workbench.encord.storage import ArtifactStore

from npa.workbench.encord.schemas import (
    DEFAULT_MEDIA_FILTER,
    DEFAULT_POLL_TIMEOUT_SECONDS,
    DEFAULT_TRANSFER,
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
    identity_sidecar_uri: str = "",
    user_client: Any = None,
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    clock: Callable[[], str] | None = None,
    environ: dict[str, str] | None = None,
) -> PushReceipt:
    from npa.workbench.encord import run_push

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
        identity_sidecar_uri=identity_sidecar_uri,
        user_client=user_client,
        storage_client=storage_client,
        artifact_store=artifact_store,
        clock=clock,
        environ=environ,
    )


def pull(
    *,
    source: str,
    source_id: str,
    output_path: str,
    workflow_run: str = "",
    label_export: str = "none",
    user_client: Any = None,
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    downloader: Any = None,
    clock: Callable[[], str] | None = None,
    environ: dict[str, str] | None = None,
) -> PullManifest:
    from npa.workbench.encord import run_pull

    return run_pull(
        source=source,
        source_id=source_id,
        output_path=output_path,
        workflow_run=workflow_run,
        label_export=label_export,
        user_client=user_client,
        storage_client=storage_client,
        artifact_store=artifact_store,
        downloader=downloader,
        clock=clock,
        environ=environ,
    )


def verify_roundtrip(
    *,
    receipt_uri: str,
    manifest_uri: str,
    output_path: str,
    workflow_run: str = "",
    storage_client: Any = None,
    artifact_store: ArtifactStore | None = None,
    clock: Callable[[], str] | None = None,
) -> RoundtripReport:
    from npa.workbench.encord import verify_roundtrip as run_verify

    return run_verify(
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        output_path=output_path,
        workflow_run=workflow_run,
        storage_client=storage_client,
        artifact_store=artifact_store,
        clock=clock,
    )


__all__ = ["PullManifest", "PushReceipt", "RoundtripReport", "pull", "push", "verify_roundtrip"]
