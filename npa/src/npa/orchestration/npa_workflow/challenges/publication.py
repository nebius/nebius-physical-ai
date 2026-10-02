"""Publish immutable evaluation inputs using the selected project's storage scope."""

from npa.clients.config import resolve_project_storage
from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.execution_preflight import resolve_execution_target, verify_execution_target
from npa.orchestration.npa_workflow.submit_credentials import SubmitCredentialContext

from .config import ChallengeSetup


def _storage_for(setup: ChallengeSetup, prefix: str) -> StorageClient:
    storage = resolve_project_storage(
        setup.project, include_shared_credentials=False, include_environment=False
    )
    credentials = SubmitCredentialContext(
        endpoint_url=storage.endpoint_url,
        access_key_id=storage.aws_access_key_id,
        secret_access_key=storage.aws_secret_access_key,
        provenance={"credentials": "project.config"},
    )
    target = resolve_execution_target(
        project=setup.project,
        output_uris=(prefix,),
        credentials=credentials,
        output_kinds={prefix: "directory"},
    )
    verify_execution_target(target, verify_cluster=False)
    return StorageClient(
        endpoint_url=credentials.endpoint_url,
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
    )


def _publish_one(storage: StorageClient, uri: str, payload: bytes) -> None:
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        # An interrupted publication can safely repeat only identical inputs.
        existing = storage.read_bytes_with_etag(uri)
        if existing is None or existing[0] != payload:
            raise ValueError(
                "Input publication conflicts with existing run inputs"
            ) from None
    verified = storage.read_bytes_with_etag(uri)
    if verified is None or verified[0] != payload:
        raise ValueError("Published input readback differs; do not submit this kit")


def publish_inputs(
    setup: ChallengeSetup, prefix: str, payloads: dict[str, bytes]
) -> None:
    """Verify project ownership and conditionally publish recipe and policy bytes.

    Args:
        setup: Selected operator project and run configuration.
        prefix: Exact run input directory in object storage.
        payloads: The recipe and policy runbook to publish.
    Returns:
        None; success requires byte-for-byte remote readback of both inputs.
    Raises:
        ValueError: Scope, ownership or immutable input verification fails.
        RuntimeError: Execution preflight cannot verify the selected project.
        StorageError: Object storage does not satisfy the publication contract.
        BotoCoreError: The storage transport fails.
        ClientError: Object storage rejects an operation.
    """
    storage = _storage_for(setup, prefix)
    for filename in ("recipe.json", "policy.md"):
        _publish_one(storage, f"{prefix}{filename}", payloads[filename])
