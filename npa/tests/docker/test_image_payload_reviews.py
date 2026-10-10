"""Reviewed public constants must never hide changed or appended credentials."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
reviews_module = importlib.import_module("image_payload_reviews")
credentials = importlib.import_module("image_payload_credentials")

REFERENCE = b"hf_token = credentials.hf_token\n"


def _review(payload=REFERENCE):
    digest = hashlib.sha256(payload).hexdigest()
    return {
        digest: {
            "sha256": digest,
            "bytes": len(payload),
            "match_counts": {"credential_assignment": 1},
            "reason": "Synthetic operator-variable reference.",
            "source": {"url": "https://example.invalid/source", "revision": "a" * 40},
        }
    }


def test_detector_still_finds_the_reviewed_expression():
    assert (
        credentials.content_credential(io.BytesIO(REFERENCE)) == "credential_assignment"
    )
    kind, receipt = reviews_module.reviewed_content_credential(
        io.BytesIO(REFERENCE), _review()
    )
    assert kind is None
    assert receipt["sha256"] == hashlib.sha256(REFERENCE).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        REFERENCE.replace(b"credentials.hf_token", b"abcdefghijk"),
        REFERENCE + b"AKIA" + b"A" * 16,
        REFERENCE + b"\0" * (credentials.CONTENT_CHUNK + 10) + b"hf_token=abcdefghijk",
        REFERENCE + b"\n",
    ],
)
def test_any_changed_or_appended_bytes_refuse_the_disposition(payload):
    kind, receipt = reviews_module.reviewed_content_credential(
        io.BytesIO(payload), _review()
    )
    assert kind is not None
    assert receipt is None


@pytest.mark.parametrize("mutation", ["size", "kind"])
def test_wrong_review_size_or_finding_kind_fails(mutation):
    rows = _review()
    row = next(iter(rows.values()))
    if mutation == "size":
        row["bytes"] += 1
    else:
        row["match_counts"] = {"private_key_content": 1}
    kind, receipt = reviews_module.reviewed_content_credential(
        io.BytesIO(REFERENCE), rows
    )
    assert kind == "credential_assignment"
    assert receipt is None


def test_unknown_content_is_not_dispositioned():
    kind, receipt = reviews_module.reviewed_content_credential(
        io.BytesIO(b"hf_token=abcdefghijk"), reviews_module.load_reviews()
    )
    assert kind == "credential_assignment"
    assert receipt is None


@pytest.mark.parametrize("mutation", ["schema", "duplicate", "kind", "source"])
def test_invalid_review_catalog_fails_closed(tmp_path, mutation):
    document = {
        "schema_version": "npa.image-payload-content-reviews.v1",
        "content": list(_review().values()),
    }
    if mutation == "schema":
        document["schema_version"] = "unknown"
    elif mutation == "duplicate":
        document["content"] *= 2
    elif mutation == "kind":
        document["content"][0]["match_counts"] = {"unknown": 1}
    else:
        document["content"][0]["source"]["revision"] = "moving-main"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        reviews_module.load_reviews(path)
