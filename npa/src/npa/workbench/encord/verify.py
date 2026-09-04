"""Terminal roundtrip verifier: receipt vs manifest by exact identity.

Joins a push receipt to a pull manifest on Encord item uuid and verifies that
every pushed item came back — with matching size and, wherever a content
digest exists on both sides, matching checksum. This is what makes checksum
claims machine-checkable evidence rather than assertions (adopted from
PR #363's verifier).

The verifier fails closed on the receipt itself, too: a receipt that never
reached ``done`` (a write-ahead ``planned`` copy, or a ``failed``/``timeout``
push) or one with nothing attributable to verify is a defect, never a
vacuous pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from npa.workbench.encord.integrity import compare_checksums
from npa.workbench.encord.schemas import (
    ROUNDTRIP_REPORT_FILENAME,
    ChecksumState,
    EncordToolError,
    PullManifest,
    PushReceipt,
    RoundtripItem,
    RoundtripReport,
)
from npa.workbench.encord.storage import (
    artifact_uri_for,
    error_text,
    finalize_artifact,
    read_json,
)


def roundtrip_report_uri_for(output_path: str) -> str:
    """The exact report URI a given --output-path resolves to."""

    return artifact_uri_for(output_path, ROUNDTRIP_REPORT_FILENAME)


def _compare(pushed: Any, pulled: Any) -> RoundtripItem:
    reasons: list[str] = []
    verdict = compare_checksums(
        pushed.source_checksum,
        pushed.source_checksum_kind,
        pulled.checksum,
        pulled.checksum_kind,
    )
    states: dict[bool | None, ChecksumState] = {
        True: "verified",
        False: "mismatched",
        None: "unavailable",
    }
    checksum_state = states[verdict]
    if verdict is False:
        reasons.append("checksum mismatch between source and pulled bytes")
    observed_size = pulled.observed_size or pulled.file_size
    size_comparable = bool(pushed.source_size and observed_size)
    if size_comparable and pushed.source_size != observed_size:
        reasons.append(
            f"size mismatch: pushed {pushed.source_size}, pulled {observed_size}"
        )
    if checksum_state == "unavailable" and not size_comparable:
        # A multipart ETag on both sides and no size from Encord leaves nothing
        # to compare; a green row here would be an unverified claim.
        reasons.append(
            "unverifiable: no comparable checksum and no size on both sides"
        )
    return RoundtripItem(
        source_uri=pushed.source_uri,
        item_uuid=pushed.item_uuid,
        relation="matched",
        expected_size=pushed.source_size,
        observed_size=observed_size,
        expected_checksum=pushed.source_checksum,
        expected_checksum_kind=pushed.source_checksum_kind,
        observed_checksum=pulled.checksum,
        observed_checksum_kind=pulled.checksum_kind,
        checksum_state=checksum_state,
        reasons=reasons,
    )


def run_verify(
    *,
    receipt_uri: str,
    manifest_uri: str,
    output_path: str,
    workflow_run: str = "",
    storage_client: Any = None,
) -> RoundtripReport:
    """Verify the roundtrip and write the report; fail closed on any defect."""

    from npa.clients.storage import StorageClient

    active_storage = storage_client or StorageClient.from_environment()
    report_uri = roundtrip_report_uri_for(output_path)

    items: list[RoundtripItem] = []
    defects: list[str] = []
    run_error: Exception | None = None
    expected = matched = missing = unexpected = 0
    size_mismatched = checksum_verified = checksum_mismatched = checksum_unavailable = 0
    unverifiable = 0
    try:
        receipt = PushReceipt.model_validate(
            read_json(receipt_uri, storage_client=active_storage)
        )
        manifest = PullManifest.model_validate(
            read_json(manifest_uri, storage_client=active_storage)
        )
        if receipt.status != "done":
            defects.append(
                f"push receipt status is {receipt.status!r}, not 'done': the push "
                "never completed, so there is no roundtrip to verify"
            )
        # Push fails closed on any successful item without an exact-identity
        # uuid, so every successful receipt row is attributable by uuid here.
        pushed_ok = [
            item
            for item in receipt.items
            if item.status in ("registered", "uploaded") and item.item_uuid
        ]
        pulled_ok = {
            item.item_uuid: item
            for item in manifest.items
            if item.transfer in ("copy", "download")
        }
        expected = len(pushed_ok)
        if expected == 0:
            defects.append(
                "push receipt has no attributable items; a roundtrip over zero "
                "items proves nothing"
            )
        for pushed in pushed_ok:
            pulled = pulled_ok.get(pushed.item_uuid)
            if pulled is None:
                missing += 1
                items.append(
                    RoundtripItem(
                        source_uri=pushed.source_uri,
                        item_uuid=pushed.item_uuid,
                        relation="missing",
                        expected_size=pushed.source_size,
                        expected_checksum=pushed.source_checksum,
                        expected_checksum_kind=pushed.source_checksum_kind,
                        reasons=["Encord uuid is absent from the pull manifest"],
                    )
                )
                continue
            row = _compare(pushed, pulled)
            items.append(row)
            matched += 1
            if any(reason.startswith("size mismatch") for reason in row.reasons):
                size_mismatched += 1
            if any(reason.startswith("unverifiable") for reason in row.reasons):
                unverifiable += 1
            checksum_verified += row.checksum_state == "verified"
            checksum_mismatched += row.checksum_state == "mismatched"
            checksum_unavailable += row.checksum_state == "unavailable"
        expected_uuids = {item.item_uuid for item in pushed_ok}
        for item_uuid in sorted(set(pulled_ok) - expected_uuids):
            unexpected += 1
            items.append(
                RoundtripItem(
                    source_uri=pulled_ok[item_uuid].source_uri,
                    item_uuid=item_uuid,
                    relation="unexpected",
                    reasons=["pulled item is not in the push receipt"],
                )
            )
    except Exception as exc:  # noqa: BLE001 - recorded in the report, re-raised below
        run_error = exc

    failed = bool(
        defects
        or missing
        or unexpected
        or size_mismatched
        or checksum_mismatched
        or unverifiable
        or expected != matched
    )
    report = RoundtripReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        workflow_run=workflow_run,
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        report_uri=report_uri,
        status="failed" if (failed or run_error is not None) else "passed",
        expected=expected,
        matched=matched,
        missing=missing,
        unexpected=unexpected,
        size_mismatched=size_mismatched,
        checksum_verified=checksum_verified,
        checksum_mismatched=checksum_mismatched,
        checksum_unavailable=checksum_unavailable,
        unverifiable=unverifiable,
        defects=defects,
        error=error_text(run_error),
        items=items,
    )
    finalize_artifact(
        report,
        result_uri=report_uri,
        filename=ROUNDTRIP_REPORT_FILENAME,
        storage_client=active_storage,
        run_error=run_error,
        failure_prefix="Encord verify failed",
        artifact_noun="Report",
    )
    if failed:
        detail = f" {'; '.join(defects)}." if defects else ""
        raise EncordToolError(
            f"Encord roundtrip verification failed: {matched}/{expected} matched, "
            f"{missing} missing, {unexpected} unexpected, {size_mismatched} size "
            f"mismatched, {checksum_mismatched} checksum mismatched, {unverifiable} "
            f"unverifiable.{detail} Report "
            f"written to {report_uri}."
        )
    return report
