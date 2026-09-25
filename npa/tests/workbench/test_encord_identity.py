"""Exact identity: canonical S3 URIs, object-URL normalization, fail-closed resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from encord_fakes import ENDPOINT, fake_uuid
from npa.workbench.encord.identity import (
    canonical_s3_uri,
    normalize_object_url,
    resolve_exact_identity,
)
from npa.workbench.encord.schemas import EncordToolError


def test_canonical_s3_uri_never_aliases_keys() -> None:
    # A literal percent triplet in a key must not alias the slash form.
    assert canonical_s3_uri("bkt", "a%2Fb.png") != canonical_s3_uri("bkt", "a/b.png")
    assert canonical_s3_uri("bkt", "runs/a b.png") == "s3://bkt/runs/a%20b.png"
    for bad in ("", "a/", "/a", "a/../b", "a//b"):
        with pytest.raises(EncordToolError):
            canonical_s3_uri("bkt", bad)


def test_normalize_object_url_is_identity_preserving() -> None:
    a = normalize_object_url("HTTPS://Host.example/bkt/p/a%2eмp4")
    b = normalize_object_url("https://host.example/bkt/p/a.мp4")
    assert a == b  # unreserved escapes normalize
    # reserved escapes are preserved: %2F is not a path separator
    assert normalize_object_url("https://h/bkt/a%2Fb") != normalize_object_url(
        "https://h/bkt/a/b"
    )
    with pytest.raises(EncordToolError):
        normalize_object_url("https://user:pw@h/bkt/a")


def test_resolve_exact_identity_prefers_metadata_and_detects_conflicts() -> None:
    meta_item = SimpleNamespace(
        uuid=fake_uuid(81),
        client_metadata={"npa": {"source_uri": "s3://bkt/p/a.mp4"}},
        url="",
    )
    url_item = SimpleNamespace(
        uuid=fake_uuid(82), client_metadata={}, url=f"{ENDPOINT}/bkt/p/a.mp4"
    )
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/a.mp4",
        submitted_object_url=f"{ENDPOINT}/bkt/p/a.mp4",
        candidates=[meta_item, url_item],
    )
    # Two different uuids both claiming the source is a conflict, not a pick.
    assert resolution.error_code == "identity_conflict"
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/a.mp4",
        submitted_object_url=f"{ENDPOINT}/bkt/p/a.mp4",
        candidates=[meta_item],
    )
    assert resolution.resolved and resolution.signal == "metadata"
    resolution = resolve_exact_identity(
        source_uri="s3://bkt/p/other.mp4",
        submitted_object_url="",
        candidates=[meta_item, url_item],
    )
    assert resolution.error_code == "identity_unresolved"

