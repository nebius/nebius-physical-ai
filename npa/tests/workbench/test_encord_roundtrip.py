from __future__ import annotations

import hashlib
import sys
import base64
import json
from pathlib import Path

import pytest

from npa.workbench.encord.schemas import (
    EncordToolError,
    OutcomeCounts,
    PullItem,
    PullManifest,
    PushItem,
    PushReceipt,
)
from npa.workbench.encord.verify import roundtrip_report_uri_for, verify_roundtrip

sys.path.insert(0, str(Path(__file__).parents[1]))

from encord_fakes import FakeStorageClient, MemoryArtifactStore  # noqa: E402

NOW = "2026-08-30T00:00:00+00:00"
RECEIPT_URI = "s3://result-bucket/run/push_receipt.json"
MANIFEST_URI = "s3://result-bucket/run/manifest.json"
REPORT_URI = "s3://result-bucket/run/roundtrip_report.json"
DESTINATION = "s3://result-bucket/run/media/uuid-1__clip.mp4"


def artifacts(*, checksum_kind="sha256", checksum=""):
    value = checksum or hashlib.sha256(b"video").hexdigest()
    pushed = PushItem(
        source_uri="s3://source-bucket/incoming/clip.mp4",
        bucket="source-bucket",
        object_key="incoming/clip.mp4",
        category="videos",
        submitted_object_url="https://storage.test.example/source-bucket/incoming/clip.mp4",
        source_size=5,
        source_checksum=value,
        source_checksum_kind=checksum_kind,
        item_uuid="uuid-1",
        registration_state="registered",
        identity_state="resolved",
        outcome="successful",
    )
    receipt = PushReceipt(
        phase="final",
        status="completed",
        revision=2,
        generated_at=NOW,
        updated_at=NOW,
        input_uri="s3://source-bucket/incoming/",
        encord_domain="https://api.encord.com",
        folder_name="folder",
        media_filter="videos-images",
        counts=OutcomeCounts.from_outcomes(["successful"]),
        receipt_uri=RECEIPT_URI,
        receipt_store_kind="s3",
        items=[pushed],
    )
    pulled = PullItem(
        item_uuid="uuid-1",
        source_uri=pushed.source_uri,
        name="clip.mp4",
        source_size=5,
        destination_uri=DESTINATION,
        transfer="download",
        outcome="successful",
        destination_exists=True,
        destination_size=5,
        destination_checksum=value,
        destination_checksum_kind=checksum_kind,
        metadata_uri="s3://result-bucket/run/items/uuid-1.json",
        metadata_state="written",
    )
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
        manifest_uri=MANIFEST_URI,
        counts=OutcomeCounts.from_outcomes(["successful"]),
        label_counts=OutcomeCounts.from_outcomes([]),
        media_copied=0,
        media_downloaded=1,
        media_bytes=5,
        items=[pulled],
    )
    return receipt, manifest


def setup(receipt, manifest):
    store = MemoryArtifactStore()
    store.create_json(RECEIPT_URI, receipt.model_dump(by_alias=True))
    store.create_json(MANIFEST_URI, manifest.model_dump(by_alias=True))
    storage = FakeStorageClient()
    storage.s3.objects[("result-bucket", "run/media/uuid-1__clip.mp4")] = b"video"
    if receipt.items[0].source_checksum_kind in {"sha256", "s3_checksum_sha256"}:
        storage.s3.head_overrides[
            ("result-bucket", "run/media/uuid-1__clip.mp4")
        ] = {
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(b"video").digest()
            ).decode(),
            "ChecksumType": "FULL_OBJECT",
        }
    return store, storage


def test_roundtrip_happy_path_persists_completed_report() -> None:
    receipt, manifest = artifacts()
    store, storage = setup(receipt, manifest)
    report = verify_roundtrip(
        receipt_uri=RECEIPT_URI,
        manifest_uri=MANIFEST_URI,
        output_path=REPORT_URI,
        artifact_store=store,
        storage_client=storage,
    )
    assert report.passed
    assert report.matched == 1
    assert REPORT_URI in store.payloads


def test_roundtrip_report_uri_for_accepts_prefix_or_exact_json() -> None:
    assert roundtrip_report_uri_for("s3://result-bucket/run") == REPORT_URI
    assert roundtrip_report_uri_for(REPORT_URI) == REPORT_URI


@pytest.mark.parametrize("failure", ["missing", "unexpected", "size", "checksum"])
def test_roundtrip_mismatch_matrix_is_nonzero(failure: str) -> None:
    receipt, manifest = artifacts()
    store, storage = setup(receipt, manifest)
    if failure == "missing":
        manifest.items[0].item_uuid = "uuid-other"
        manifest.items[0].destination_uri = (
            "s3://result-bucket/run/media/uuid-other__clip.mp4"
        )
        storage.s3.objects[("result-bucket", "run/media/uuid-other__clip.mp4")] = b"video"
    elif failure == "unexpected":
        extra = manifest.items[0].model_copy(
            update={
                "item_uuid": "uuid-2",
                "destination_uri": "s3://result-bucket/run/media/uuid-2__clip.mp4",
            }
        )
        manifest.items.append(extra)
        manifest.counts = OutcomeCounts.from_outcomes(["successful", "successful"])
        manifest.media_downloaded = 2
        manifest.media_bytes = 10
        storage.s3.objects[("result-bucket", "run/media/uuid-2__clip.mp4")] = b"video"
    elif failure == "size":
        storage.s3.wrong_size_after_write.add(
            ("result-bucket", "run/media/uuid-1__clip.mp4")
        )
    else:
        storage.s3.head_overrides[
            ("result-bucket", "run/media/uuid-1__clip.mp4")
        ] = {
            "ChecksumSHA256": base64.b64encode(
                hashlib.sha256(b"wrong").digest()
            ).decode(),
            "ChecksumType": "FULL_OBJECT",
        }
    store.payloads[MANIFEST_URI] = (
        manifest.model_dump_json(by_alias=True).encode()
    )
    with pytest.raises(EncordToolError):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path=REPORT_URI,
            artifact_store=store,
            storage_client=storage,
        )
    assert REPORT_URI in store.payloads


def test_incompatible_checksum_types_are_not_compared() -> None:
    receipt, manifest = artifacts(checksum_kind="etag_opaque", checksum="opaque-value")
    manifest.items[0].destination_checksum_kind = "sha256"
    manifest.items[0].destination_checksum = hashlib.sha256(b"video").hexdigest()
    store, storage = setup(receipt, manifest)
    with pytest.raises(EncordToolError, match="verification failed"):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path=REPORT_URI,
            artifact_store=store,
            storage_client=storage,
        )
    report = json.loads(store.payloads[REPORT_URI])
    assert report["passed"] is False
    assert report["checksum_unavailable"] == 1
    assert report["items"][0]["integrity_state"] == "not_comparable"
    assert report["items"][0]["relation"] == "integrity_failed"


def test_invalid_input_artifact_still_persists_failed_report() -> None:
    receipt, manifest = artifacts()
    store, storage = setup(receipt, manifest)
    store.payloads[RECEIPT_URI] = b'{"schema":"wrong"}'
    with pytest.raises(EncordToolError, match="inputs are invalid"):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path="s3://result-bucket/run",
            artifact_store=store,
            storage_client=storage,
        )
    assert REPORT_URI in store.payloads


def test_missing_pull_source_uri_is_invalid_and_persists_failed_report() -> None:
    receipt, manifest = artifacts()
    store, storage = setup(receipt, manifest)
    payload = manifest.model_dump(by_alias=True)
    payload["items"][0]["source_uri"] = ""
    store.payloads[MANIFEST_URI] = json.dumps(payload).encode()
    with pytest.raises(EncordToolError, match="inputs are invalid"):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path=REPORT_URI,
            artifact_store=store,
            storage_client=storage,
        )
    report = json.loads(store.payloads[REPORT_URI])
    assert report["passed"] is False


def test_missing_required_record_id_fails_lineage_verification() -> None:
    receipt, manifest = artifacts()
    receipt.items[0].record_id = "record-1"
    store, storage = setup(receipt, manifest)
    with pytest.raises(EncordToolError, match="verification failed"):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path=REPORT_URI,
            artifact_store=store,
            storage_client=storage,
        )
    report = json.loads(store.payloads[REPORT_URI])
    assert report["items"][0]["relation"] == "integrity_failed"
    assert "pull row omits the required record ID" in report["items"][0]["reasons"]


def test_current_destination_head_overrides_conflicting_manifest_evidence() -> None:
    receipt, manifest = artifacts()
    store, storage = setup(receipt, manifest)
    storage.s3.head_overrides[
        ("result-bucket", "run/media/uuid-1__clip.mp4")
    ] = {
        "ChecksumSHA256": base64.b64encode(
            hashlib.sha256(b"wrong").digest()
        ).decode(),
        "ChecksumType": "FULL_OBJECT",
    }
    with pytest.raises(EncordToolError, match="verification failed"):
        verify_roundtrip(
            receipt_uri=RECEIPT_URI,
            manifest_uri=MANIFEST_URI,
            output_path=REPORT_URI,
            artifact_store=store,
            storage_client=storage,
        )
    report = json.loads(store.payloads[REPORT_URI])
    assert report["checksum_mismatched"] == 1
    assert report["items"][0]["observed_checksum"] == hashlib.sha256(
        b"wrong"
    ).hexdigest()
