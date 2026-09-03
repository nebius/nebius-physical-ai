from __future__ import annotations

import pytest
from pydantic import ValidationError

from npa.workbench.encord.schemas import (
    LabelArtifact,
    OutcomeCounts,
    PullItem,
    PullManifest,
    PushItem,
    PushReceipt,
    RoundtripItem,
    RoundtripReport,
)

NOW = "2026-08-30T00:00:00+00:00"


def successful_push_item(**updates) -> PushItem:
    values = {
        "source_uri": "s3://source-bucket/incoming/clip.mp4",
        "bucket": "source-bucket",
        "object_key": "incoming/clip.mp4",
        "category": "videos",
        "submitted_object_url": "https://storage.test.example/source-bucket/incoming/clip.mp4",
        "source_size": 5,
        "item_uuid": "00000000-0000-0000-0000-000000000061",
        "registration_state": "registered",
        "identity_state": "resolved",
        "outcome": "successful",
    }
    values.update(updates)
    return PushItem(**values)


def push_receipt(item: PushItem, **updates) -> PushReceipt:
    values = {
        "phase": "final",
        "status": "completed",
        "revision": 2,
        "generated_at": NOW,
        "updated_at": NOW,
        "input_uri": "s3://source-bucket/incoming/",
        "encord_domain": "https://api.encord.com",
        "folder_name": "folder",
        "media_filter": "videos-images",
        "counts": OutcomeCounts.from_outcomes([item.outcome]),
        "receipt_uri": "s3://result-bucket/run/push_receipt.json",
        "receipt_store_kind": "s3",
        "items": [item],
    }
    values.update(updates)
    return PushReceipt(**values)


def successful_pull_item(**updates) -> PullItem:
    values = {
        "item_uuid": "00000000-0000-0000-0000-000000000061",
        "source_uri": "s3://source-bucket/incoming/clip.mp4",
        "destination_uri": "s3://result-bucket/run/media/clip.mp4",
        "transfer": "copy",
        "outcome": "successful",
        "destination_exists": True,
        "source_size": 5,
        "destination_size": 5,
        "metadata_uri": "s3://result-bucket/run/items/item.json",
        "metadata_state": "written",
    }
    values.update(updates)
    return PullItem(**values)


def test_completed_push_requires_all_exact_success() -> None:
    receipt = push_receipt(successful_push_item())
    assert receipt.status == "completed"
    with pytest.raises(ValidationError):
        push_receipt(
            PushItem(
                source_uri="s3://source-bucket/incoming/clip.mp4",
                bucket="source-bucket",
                object_key="incoming/clip.mp4",
                category="videos",
            )
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"object_key": "incoming/other.mp4"},
        {
            "submitted_object_url": (
                "https://storage.test.example/source-bucket/incoming/other.mp4"
            )
        },
    ],
)
def test_push_item_reconciles_every_source_identity_field(updates) -> None:
    with pytest.raises(ValidationError):
        successful_push_item(**updates)


def test_successful_pull_requires_stable_source_uri() -> None:
    with pytest.raises(ValidationError):
        successful_pull_item(source_uri="")


@pytest.mark.parametrize(
    "values",
    [
        (1, 0, 1, 0, 0, 0),
        (1, 1, 0, 0, 0, 1),
        (-1, 0, 0, 0, 0, 0),
    ],
)
def test_push_counts_reconcile(values: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        OutcomeCounts(
            discovered=values[0],
            attempted=values[1],
            successful=values[2],
            unresolved=values[3],
            failed=values[4],
            unattempted=values[5],
        )


def test_completed_push_rejects_unlinked_requested_dataset_row() -> None:
    with pytest.raises(ValidationError):
        push_receipt(successful_push_item(link_state="unattempted"), dataset_requested=True)


def test_partial_and_failed_push_states_are_truthful() -> None:
    unresolved = successful_push_item(
        item_uuid="",
        identity_state="unresolved",
        registration_state="unresolved",
        outcome="unresolved",
    )
    receipt = push_receipt(
        unresolved,
        status="partial",
        counts=OutcomeCounts.from_outcomes(["unresolved"]),
    )
    assert receipt.status == "partial"
    with pytest.raises(ValidationError):
        push_receipt(
            unresolved,
            status="failed",
            counts=OutcomeCounts.from_outcomes(["unresolved"]),
        )


def test_pull_counts_reconcile() -> None:
    item = successful_pull_item()
    manifest = PullManifest(
        phase="final",
        status="completed",
        revision=2,
        generated_at=NOW,
        updated_at=NOW,
        encord_domain="https://api.encord.com",
        source_kind="dataset",
        source_id="dataset-id",
        output_uri="s3://result-bucket/run",
        manifest_uri="s3://result-bucket/run/manifest.json",
        counts=OutcomeCounts.from_outcomes(["successful"]),
        label_counts=OutcomeCounts.from_outcomes([]),
        media_copied=1,
        media_downloaded=0,
        media_bytes=5,
        items=[item],
    )
    assert manifest.counts.successful == 1


def test_label_counts_reconcile() -> None:
    artifact = LabelArtifact(data_hash="data-1", outcome="successful", artifact_uri="s3://b/l.json")
    with pytest.raises(ValidationError):
        PullManifest(
            phase="final",
            status="completed",
            revision=1,
            generated_at=NOW,
            updated_at=NOW,
            encord_domain="https://api.encord.com",
            source_kind="project",
            source_id="project-id",
            label_export="initialize",
            label_export_remote_mutation=True,
            output_uri="s3://result-bucket/run",
            manifest_uri="s3://result-bucket/run/manifest.json",
            counts=OutcomeCounts.from_outcomes(["successful"]),
            label_counts=OutcomeCounts.from_outcomes([]),
            media_copied=1,
            media_downloaded=0,
            media_bytes=5,
            items=[successful_pull_item()],
            label_artifacts=[artifact],
        )


def test_roundtrip_report_counts_reconcile() -> None:
    item = RoundtripItem(item_uuid="uuid", relation="matched", integrity_state="matched")
    report = RoundtripReport(
        generated_at=NOW,
        receipt_uri="s3://b/receipt.json",
        manifest_uri="s3://b/manifest.json",
        report_uri="s3://b/report.json",
        status="completed",
        passed=True,
        expected=1,
        matched=1,
        missing=0,
        unexpected=0,
        unresolved=0,
        destination_missing=0,
        size_mismatched=0,
        checksum_mismatched=0,
        checksum_unavailable=0,
        items=[item],
    )
    assert report.passed


def test_serialized_artifacts_validate_on_readback() -> None:
    receipt = push_receipt(successful_push_item())
    payload = receipt.model_dump(by_alias=True)
    assert PushReceipt.model_validate(payload) == receipt
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        PushReceipt.model_validate(payload)
    payload.pop("unknown")
    payload["schema"] = "npa.encord.push_receipt.v0"
    with pytest.raises(ValidationError):
        PushReceipt.model_validate(payload)
