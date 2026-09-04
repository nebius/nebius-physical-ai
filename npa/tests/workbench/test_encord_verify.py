"""`encord verify`: join receipt to manifest by uuid; size + checksum, fail-closed."""

from __future__ import annotations


import pytest

from encord_fakes import ENDPOINT, REPORT_URI, FakeStorage, fake_uuid
from npa.workbench.encord.schemas import EncordToolError
from npa.workbench.encord.storage import write_json
from npa.workbench.encord.verify import roundtrip_report_uri_for, run_verify


def test_report_uri_helper() -> None:
    assert roundtrip_report_uri_for("s3://b/v/") == "s3://b/v/roundtrip_report.json"


def _verify_fixtures(
    storage: FakeStorage,
    *,
    receipt_overrides=None,
    pulled_overrides=None,
    drop_pulled=False,
    drop_pushed=False,
) -> tuple[str, str]:
    """Write a receipt + manifest pair into the fake object store; return URIs."""

    sha = "a" * 64
    receipt = {
        "schema": "npa.encord.push_receipt.v1",
        "generated_at": "t",
        "input_uri": "s3://bkt/p/",
        "endpoint_url": ENDPOINT,
        "encord_domain": "https://api.encord.com",
        "folder_name": "f",
        "media_filter": "videos-images",
        "status": "done",
        "items": []
        if drop_pushed
        else [
            {
                "key": "p/a.mp4",
                "source_uri": "s3://bkt/p/a.mp4",
                "category": "videos",
                "item_uuid": fake_uuid(90),
                "status": "uploaded",
                "source_size": 6,
                "source_checksum": sha,
                "source_checksum_kind": "sha256",
            }
        ],
    }
    receipt.update(receipt_overrides or {})
    pulled = {
        "item_uuid": fake_uuid(90),
        "name": "p/a.mp4",
        "transfer": "download",
        "observed_size": 6,
        "checksum": sha,
        "checksum_kind": "sha256",
    }
    pulled.update(pulled_overrides or {})
    manifest = {
        "schema": "npa.encord.pull_manifest.v1",
        "generated_at": "t",
        "encord_domain": "https://api.encord.com",
        "source_kind": "dataset",
        "source_id": "d",
        "output_uri": "s3://bkt/out/",
        "items": [] if drop_pulled else [pulled],
    }
    receipt_uri = "s3://bkt/push/push_receipt.json"
    manifest_uri = "s3://bkt/pull/manifest.json"
    write_json(receipt, result_uri=receipt_uri, filename="push_receipt.json", storage_client=storage)
    write_json(manifest, result_uri=manifest_uri, filename="manifest.json", storage_client=storage)
    return receipt_uri, manifest_uri


def _verify(storage: FakeStorage, receipt_uri: str, manifest_uri: str):
    return run_verify(
        receipt_uri=receipt_uri,
        manifest_uri=manifest_uri,
        output_path=REPORT_URI,
        storage_client=storage,
    )


def test_run_verify_passes_on_exact_match() -> None:
    storage = FakeStorage()
    report = _verify(storage, *_verify_fixtures(storage))
    assert report.status == "passed"
    assert report.expected == report.matched == 1
    assert report.checksum_verified == 1 and report.checksum_mismatched == 0
    assert report.defects == []
    payload = storage.written(REPORT_URI)
    assert payload["schema"] == "npa.encord.roundtrip_report.v1"


def test_run_verify_fails_closed_on_checksum_mismatch() -> None:
    storage = FakeStorage()
    uris = _verify_fixtures(storage, pulled_overrides={"checksum": "b" * 64})
    with pytest.raises(EncordToolError, match="1 checksum mismatched"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert payload["items"][0]["checksum_state"] == "mismatched"


def test_run_verify_fails_closed_on_missing_item() -> None:
    storage = FakeStorage()
    uris = _verify_fixtures(storage, drop_pulled=True)
    with pytest.raises(EncordToolError, match="1 missing"):
        _verify(storage, *uris)


@pytest.mark.parametrize("receipt_status", ["planned", "failed", "timeout"])
def test_run_verify_fails_closed_when_the_push_never_completed(receipt_status: str) -> None:
    """A write-ahead or failed receipt is not evidence of a roundtrip."""

    storage = FakeStorage()
    uris = _verify_fixtures(storage, receipt_overrides={"status": receipt_status})
    with pytest.raises(EncordToolError, match=f"status is {receipt_status!r}, not 'done'"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert any("never completed" in defect for defect in payload["defects"])
    # The per-item join still ran and matched: the defect is receipt-level.
    assert payload["matched"] == 1


def test_run_verify_fails_closed_on_zero_attributable_items() -> None:
    """0/0 matched must never read as passed."""

    storage = FakeStorage()
    uris = _verify_fixtures(storage, drop_pushed=True, drop_pulled=True)
    with pytest.raises(EncordToolError, match="no attributable items"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed"
    assert payload["expected"] == payload["matched"] == 0


def test_run_verify_fails_closed_when_an_item_has_no_evidence_at_all() -> None:
    """Multipart ETags on both sides and no size from Encord verify nothing."""

    storage = FakeStorage()
    uris = _verify_fixtures(
        storage,
        receipt_overrides={
            "items": [
                {
                    "key": "p/a.mp4",
                    "source_uri": "s3://bkt/p/a.mp4",
                    "category": "videos",
                    "item_uuid": fake_uuid(90),
                    "status": "registered",
                    "source_size": 6,
                    "source_checksum": "",
                    "source_checksum_kind": "none",
                }
            ]
        },
        pulled_overrides={
            "transfer": "copy",
            "observed_size": 0,
            "file_size": 0,
            "checksum": "",
            "checksum_kind": "none",
        },
    )
    with pytest.raises(EncordToolError, match="1 unverifiable"):
        _verify(storage, *uris)
    payload = storage.written(REPORT_URI)
    assert payload["status"] == "failed" and payload["unverifiable"] == 1
    assert payload["items"][0]["reasons"] == [
        "unverifiable: no comparable checksum and no size on both sides"
    ]


def test_run_verify_incomparable_kinds_are_unavailable_not_failures() -> None:
    # A multipart-source object vs a sha256 download: no comparison exists.
    storage = FakeStorage()
    uris = _verify_fixtures(
        storage, pulled_overrides={"checksum": "c" * 32, "checksum_kind": "md5"}
    )
    report = _verify(storage, *uris)
    assert report.status == "passed"
    assert report.checksum_unavailable == 1


