"""Terminal roundtrip verifier for Encord transport artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from npa.workbench.encord.integrity import compare_checksums
from npa.workbench.encord.schemas import (
    EncordToolError,
    PullManifest,
    PushReceipt,
    ROUNDTRIP_REPORT_FILENAME,
    RoundtripItem,
    RoundtripReport,
)
from npa.workbench.encord.storage import (
    ArtifactStore,
    ConditionalArtifactStore,
    ObjectStorageGateway,
    S3ObjectStorageGateway,
)


def roundtrip_report_uri_for(output_path: str) -> str:
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{ROUNDTRIP_REPORT_FILENAME}"


def verify_roundtrip(
    *,
    receipt_uri: str,
    manifest_uri: str,
    output_path: str,
    artifact_store: ArtifactStore | None = None,
    storage_client: Any = None,
    workflow_run: str = "",
    clock: Callable[[], str] | None = None,
) -> RoundtripReport:
    """Persist a report and raise when exact identity or integrity does not verify."""

    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    active_artifacts = artifact_store or ConditionalArtifactStore(active_storage)
    object_store = S3ObjectStorageGateway(active_storage)
    report_uri = roundtrip_report_uri_for(output_path)
    try:
        receipt = PushReceipt.model_validate(active_artifacts.read_json(receipt_uri))
        manifest = PullManifest.model_validate(active_artifacts.read_json(manifest_uri))
    except Exception as exc:  # noqa: BLE001 - invalid inputs still produce a report
        failed = RoundtripReport(
            generated_at=(clock or _utc_now)(),
            workflow_run=workflow_run,
            receipt_uri=receipt_uri,
            manifest_uri=manifest_uri,
            report_uri=report_uri,
            status="failed",
            passed=False,
            expected=1,
            matched=0,
            missing=0,
            unexpected=0,
            unresolved=1,
            destination_missing=0,
            size_mismatched=0,
            checksum_mismatched=0,
            checksum_unavailable=0,
            items=[
                RoundtripItem(
                    relation="unresolved",
                    reasons=[f"input artifact validation failed: {type(exc).__name__}"],
                )
            ],
        )
        active_artifacts.create_json(report_uri, failed.model_dump(by_alias=True))
        raise EncordToolError(
            f"Encord roundtrip inputs are invalid; report at {report_uri}"
        ) from exc

    receipt_rows = {item.item_uuid: item for item in receipt.items if item.item_uuid}
    manifest_rows = {item.item_uuid: item for item in manifest.items if item.item_uuid}
    report_items: list[RoundtripItem] = []

    for source in receipt.items:
        if source.outcome != "successful" or not source.item_uuid:
            report_items.append(
                RoundtripItem(
                    source_uri=source.source_uri,
                    record_id=source.record_id,
                    item_uuid=source.item_uuid,
                    expected_size=source.source_size,
                    relation="unresolved",
                    reasons=[f"push row outcome is {source.outcome}"],
                )
            )
            continue
        observed = manifest_rows.get(source.item_uuid)
        if observed is None:
            report_items.append(
                RoundtripItem(
                    source_uri=source.source_uri,
                    record_id=source.record_id,
                    item_uuid=source.item_uuid,
                    expected_size=source.source_size,
                    expected_checksum=source.source_checksum,
                    expected_checksum_kind=source.source_checksum_kind,
                    relation="missing",
                    reasons=["Encord UUID is absent from the pull manifest"],
                )
            )
            continue
        report_items.append(
            _compare_row(source, observed, object_store=object_store)
        )

    for item_uuid in sorted(set(manifest_rows) - set(receipt_rows)):
        observed = manifest_rows[item_uuid]
        report_items.append(
            RoundtripItem(
                source_uri=observed.source_uri,
                record_id=observed.record_id,
                item_uuid=item_uuid,
                destination_uri=observed.destination_uri,
                observed_size=observed.destination_size,
                observed_checksum=observed.destination_checksum,
                observed_checksum_kind=observed.destination_checksum_kind,
                relation="unexpected",
                reasons=["Encord UUID is absent from the push receipt"],
            )
        )

    inputs_complete = (
        receipt.phase == manifest.phase == "final"
        and receipt.status == manifest.status == "completed"
    )
    passed = inputs_complete and all(item.relation == "matched" for item in report_items)
    generated_at = (clock or _utc_now)()
    report = RoundtripReport(
        generated_at=generated_at,
        workflow_run=workflow_run or receipt.workflow_run or manifest.workflow_run,
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        report_uri=report_uri,
        status="completed" if passed else "failed",
        passed=passed,
        expected=len(receipt.items),
        matched=sum(item.relation == "matched" for item in report_items),
        missing=sum(item.relation == "missing" for item in report_items),
        unexpected=sum(item.relation == "unexpected" for item in report_items),
        unresolved=sum(item.relation == "unresolved" for item in report_items),
        destination_missing=sum(
            item.relation == "integrity_failed" and item.integrity_state == "missing"
            for item in report_items
        ),
        size_mismatched=sum(
            item.integrity_state == "size_mismatch" for item in report_items
        ),
        checksum_mismatched=sum(
            item.integrity_state == "checksum_mismatch" for item in report_items
        ),
        checksum_unavailable=sum(
            item.integrity_state == "not_comparable" for item in report_items
        ),
        items=report_items,
    )
    active_artifacts.create_json(report_uri, report.model_dump(by_alias=True))
    if not passed:
        raise EncordToolError(f"Encord roundtrip verification failed; report at {report_uri}")
    return report


def _compare_row(
    source: Any, observed: Any, *, object_store: ObjectStorageGateway
) -> RoundtripItem:
    reasons: list[str] = []
    if observed.outcome != "successful":
        reasons.append(f"pull row outcome is {observed.outcome}")
    if not observed.source_uri:
        reasons.append("pull row omits the stable source URI")
    elif source.source_uri != observed.source_uri:
        reasons.append("source URI differs between receipt and manifest")
    if source.record_id:
        if not observed.record_id:
            reasons.append("pull row omits the required record ID")
        elif source.record_id != observed.record_id:
            reasons.append("record ID differs between receipt and manifest")

    if not observed.destination_uri:
        return RoundtripItem(
            source_uri=source.source_uri,
            record_id=source.record_id,
            item_uuid=source.item_uuid,
            expected_size=source.source_size,
            relation="unresolved",
            reasons=[*reasons, "pull row has no destination URI"],
        )
    metadata = object_store.head(observed.destination_uri)
    expected_size = source.source_size
    observed_size = metadata.size if metadata.exists else 0
    integrity_state = "matched"
    if not metadata.exists:
        integrity_state = "missing"
        reasons.append("destination object is missing")
    elif expected_size and observed_size != expected_size:
        integrity_state = "size_mismatch"
        reasons.append("destination size differs from the source size")

    observed_checksum = metadata.checksum
    observed_kind = metadata.checksum_kind
    source_comparison = compare_checksums(
        source.source_checksum,
        source.source_checksum_kind,
        observed_checksum,
        observed_kind,
    )
    stored_comparison = compare_checksums(
        observed.destination_checksum,
        observed.destination_checksum_kind,
        observed_checksum,
        observed_kind,
    )
    if stored_comparison is False:
        integrity_state = "checksum_mismatch"
        reasons.append("current destination checksum conflicts with the pull manifest")
    if source_comparison is False:
        integrity_state = "checksum_mismatch"
        reasons.append("compatible source and destination checksums differ")
    elif integrity_state == "matched" and source_comparison is None:
        integrity_state = "not_comparable"
        reasons.append(
            "no compatible source-to-destination checksum is available"
        )

    relation = "matched"
    if observed.outcome != "successful":
        relation = "unresolved"
    elif reasons:
        relation = "integrity_failed"
    return RoundtripItem(
        source_uri=source.source_uri,
        record_id=source.record_id,
        item_uuid=source.item_uuid,
        destination_uri=observed.destination_uri,
        expected_size=expected_size,
        observed_size=observed_size,
        expected_checksum=source.source_checksum,
        expected_checksum_kind=source.source_checksum_kind,
        observed_checksum=observed_checksum,
        observed_checksum_kind=observed_kind,
        integrity_state=integrity_state,
        relation=relation,
        reasons=reasons,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
