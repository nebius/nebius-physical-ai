from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from npa.workbench.encord import identity
from npa.workbench.encord.identity import (
    canonical_s3_uri,
    normalize_object_url,
    resolve_exact_identity,
)
from npa.workbench.encord.schemas import EncordToolError, IdentitySidecarRow

SOURCE = "s3://source-bucket/incoming/clip.mp4"
URL = "https://storage.test.example/source-bucket/incoming/clip.mp4"


def item(uuid: str, *, metadata=None, url: str = "", name: str = "clip.mp4"):
    return SimpleNamespace(
        uuid=uuid,
        client_metadata=metadata or {},
        url=url,
        name=name,
    )


def test_exact_returned_object_key_attaches_uuid() -> None:
    candidate = item("uuid-1", metadata={"npa": {"source_uri": SOURCE}})
    result = resolve_exact_identity(
        source_uri=SOURCE, record_id="", submitted_object_url=URL, candidates=[candidate]
    )
    assert result.item_uuid == "uuid-1"


def test_exact_normalized_object_url_attaches_uuid() -> None:
    candidate = item(
        "uuid-1",
        url="HTTPS://STORAGE.TEST.EXAMPLE/source-bucket/incoming/clip%2Emp4?secret=x",
    )
    result = resolve_exact_identity(
        source_uri=SOURCE, record_id="", submitted_object_url=URL, candidates=[candidate]
    )
    assert result.resolved
    assert normalize_object_url(candidate.url) == URL


def test_literal_percent_encoded_key_never_aliases_slash_key() -> None:
    encoded = canonical_s3_uri("source-bucket", "incoming/a%2Fb.mp4")
    slashed = canonical_s3_uri("source-bucket", "incoming/a/b.mp4")
    assert encoded == "s3://source-bucket/incoming/a%252Fb.mp4"
    assert encoded != slashed
    assert normalize_object_url(
        "https://storage.test.example/source-bucket/incoming/a%252Fb.mp4"
    ) != normalize_object_url(
        "https://storage.test.example/source-bucket/incoming/a/b.mp4"
    )
    assert normalize_object_url(
        "https://storage.test.example/source-bucket/incoming/a%2Fb.mp4"
    ) != normalize_object_url(
        "https://storage.test.example/source-bucket/incoming/a/b.mp4"
    )


def test_record_id_metadata_takes_precedence() -> None:
    candidate = item(
        "uuid-1",
        metadata={"npa": {"source_uri": SOURCE, "record_id": "record-1"}},
        name="different-title.mp4",
    )
    result = resolve_exact_identity(
        source_uri=SOURCE,
        record_id="record-1",
        submitted_object_url=URL,
        candidates=[candidate],
    )
    assert result.signal == "record_id_metadata"


def test_conflicting_exact_signals_fail_closed() -> None:
    by_metadata = item("uuid-1", metadata={"npa": {"source_uri": SOURCE}})
    by_url = item("uuid-2", url=URL)
    result = resolve_exact_identity(
        source_uri=SOURCE,
        record_id="",
        submitted_object_url=URL,
        candidates=[by_metadata, by_url],
    )
    assert result.error_code == "identity_conflict"


def test_legacy_basename_only_item_is_unresolved() -> None:
    candidate = item("uuid-1", name="archive/clip.mp4")
    result = resolve_exact_identity(
        source_uri=SOURCE, record_id="", submitted_object_url=URL, candidates=[candidate]
    )
    assert result.error_code == "identity_unresolved"


def test_duplicate_exact_identity_is_unresolved() -> None:
    candidates = [
        item("uuid-1", metadata={"npa": {"source_uri": SOURCE}}),
        item("uuid-2", metadata={"npa": {"source_uri": SOURCE}}),
    ]
    result = resolve_exact_identity(
        source_uri=SOURCE, record_id="", submitted_object_url=URL, candidates=candidates
    )
    assert result.error_code == "identity_conflict"


def test_explicit_sidecar_resolves_only_exact_source() -> None:
    sidecar = IdentitySidecarRow(source_uri=SOURCE, item_uuid="uuid-1")
    result = resolve_exact_identity(
        source_uri=SOURCE,
        record_id="",
        submitted_object_url=URL,
        candidates=[],
        sidecar=sidecar,
    )
    assert result.item_uuid == "uuid-1"


def test_sidecar_uuid_cannot_override_contradictory_inventory_evidence() -> None:
    sidecar = IdentitySidecarRow(source_uri=SOURCE, item_uuid="uuid-1")
    contradictory = item(
        "uuid-1",
        metadata={
            "npa": {"source_uri": "s3://source-bucket/archive/clip.mp4"}
        },
        url="https://storage.test.example/source-bucket/archive/clip.mp4",
    )
    result = resolve_exact_identity(
        source_uri=SOURCE,
        record_id="",
        submitted_object_url=URL,
        candidates=[contradictory],
        sidecar=sidecar,
    )
    assert result.error_code == "identity_conflict"


def test_all_representations_of_resolved_uuid_must_agree() -> None:
    candidates = [
        item("uuid-1", metadata={"npa": {"source_uri": SOURCE}}),
        item(
            "uuid-1",
            metadata={
                "npa": {"source_uri": "s3://source-bucket/archive/clip.mp4"}
            },
        ),
    ]
    result = resolve_exact_identity(
        source_uri=SOURCE,
        record_id="",
        submitted_object_url=URL,
        candidates=candidates,
    )
    assert result.error_code == "identity_conflict"


@pytest.mark.parametrize(
    "bucket,key",
    [("", "a.mp4"), ("bucket", ""), ("bucket", "a/../b.mp4"), ("bucket", "a\\b.mp4")],
)
def test_canonical_s3_uri_rejects_ambiguous_identity(bucket: str, key: str) -> None:
    with pytest.raises(EncordToolError):
        canonical_s3_uri(bucket, key)


def test_identity_module_has_no_basename_fallback() -> None:
    source = inspect.getsource(identity)
    forbidden = ("basename", ".name ==", "rsplit(\"/\", 1)", "split(\"/\")[-1]")
    assert not any(value in source for value in forbidden)
