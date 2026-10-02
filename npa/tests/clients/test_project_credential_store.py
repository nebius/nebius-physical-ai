"""Exact-project scoped-record retirement for a deleted storage bucket.

Regression coverage for a bug where `npa storage bucket delete` pruned the
legacy top-level `storage` compatibility view and the `storage_setup` journal,
but left `project_credentials.projects[<exact_project>].storage` untouched.
A later `project_credential_record` / `select_project_credentials` call then
rebuilt the top-level view from that stale scoped record, resurrecting the
deleted bucket and its access-key pair.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from npa.clients.project_credential_store import retire_project_bucket_document


def _document(**extra_projects: dict) -> dict:
    projects = {
        "project-a": {
            "project_id": "project-a",
            "storage_selected": True,
            "storage": {
                "bucket": "s3://gone-bucket/",
                "endpoint_url": "https://storage.example",
                "aws_access_key_id": "AK",
                "aws_secret_access_key": "SK",
            },
            "storage_iam": {
                "service_account_id": "sa-1",
                "service_account_managed_by": "npa",
            },
        },
        **extra_projects,
    }
    return {
        "tokens": {"HF_TOKEN": "hf_keep"},
        "project_credentials": {
            "schema_version": "npa.project-credentials.v2",
            "current_project_id": "project-a",
            "projects": projects,
        },
        "storage": deepcopy(projects["project-a"]["storage"]),
        "storage_iam": deepcopy(projects["project-a"]["storage_iam"]),
    }


def test_retire_project_bucket_document_clears_matching_storage_and_selection() -> None:
    document = _document()

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    record = updated["project_credentials"]["projects"]["project-a"]
    assert "storage" not in record
    assert record["storage_selected"] is False
    # IAM ownership evidence is a separate, ownership-gated lifecycle.
    assert record["storage_iam"]["service_account_id"] == "sa-1"
    # The top-level compatibility view is regenerated, not merely left stale.
    assert "storage" not in updated
    assert "storage_iam" not in updated
    # Unrelated top-level state is untouched.
    assert updated["tokens"]["HF_TOKEN"] == "hf_keep"


def test_retire_project_bucket_document_ignores_a_different_bucket_name() -> None:
    """A failed/pending delete for a different bucket must not touch evidence."""
    document = _document()
    before = deepcopy(document)

    updated = retire_project_bucket_document(document, "project-a", "still-here")

    assert updated == before


def test_retire_project_bucket_document_ignores_an_unknown_project() -> None:
    document = _document()
    before = deepcopy(document)

    updated = retire_project_bucket_document(document, "project-missing", "gone-bucket")

    assert updated == before


def test_retire_project_bucket_document_ignores_blank_identifiers() -> None:
    document = _document()
    before = deepcopy(document)

    assert retire_project_bucket_document(document, "", "gone-bucket") == before
    assert retire_project_bucket_document(document, "project-a", "") == before


def test_retire_project_bucket_document_preserves_sibling_project_records() -> None:
    sibling_storage = {
        "bucket": "s3://keep-bucket/",
        "endpoint_url": "https://storage.example",
        "aws_access_key_id": "AK2",
        "aws_secret_access_key": "SK2",
    }
    document = _document(
        **{
            "project-b": {
                "project_id": "project-b",
                "storage_selected": True,
                "storage": deepcopy(sibling_storage),
            }
        }
    )

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    sibling = updated["project_credentials"]["projects"]["project-b"]
    assert sibling["storage"] == sibling_storage
    assert sibling["storage_selected"] is True


def test_retire_project_bucket_document_accepts_the_checkpoint_bucket_alias() -> None:
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["storage"] = {
        "checkpoint_bucket": "s3://gone-bucket/",
        "aws_access_key_id": "AK",
    }

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    assert "storage" not in updated["project_credentials"]["projects"]["project-a"]


def test_retire_project_bucket_document_accepts_the_s3_bucket_alias() -> None:
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["storage"] = {
        "s3_bucket": "s3://gone-bucket/",
        "aws_access_key_id": "AK",
    }

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    assert "storage" not in updated["project_credentials"]["projects"]["project-a"]


@pytest.mark.parametrize(
    "stored_bucket",
    [
        "gone-bucket",
        "s3://gone-bucket/",
        "s3://gone-bucket",
        "  s3://gone-bucket/  ",
        "/gone-bucket/",
        "s3://gone-bucket/prefix/object.bin",
    ],
)
def test_retire_project_bucket_document_normalizes_scheme_slashes_and_whitespace(
    stored_bucket: str,
) -> None:
    """Match `_bucket_name_from_uri`'s normalization so real-world stored forms
    (a full URI, a bare name, a deep object path, stray whitespace) all match
    the bare *bucket_name* the CLI already normalized before calling in.
    """
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["storage"]["bucket"] = (
        stored_bucket
    )

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    assert "storage" not in updated["project_credentials"]["projects"]["project-a"]


@pytest.mark.parametrize(
    "stored_bucket", ["gone-bucket-2", "not-gone-bucket", "gone-buckets"]
)
def test_retire_project_bucket_document_rejects_a_similarly_named_bucket(
    stored_bucket: str,
) -> None:
    """A neighboring bucket name (prefix/suffix collision) must not match."""
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["storage"]["bucket"] = (
        stored_bucket
    )
    before = deepcopy(document)

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    assert updated == before


def test_retire_project_bucket_document_also_clears_matching_terraform_state() -> None:
    """The Terraform remote-state backend for the same bucket is a separate
    scoped record and must be retired in the same atomic rewrite, or it
    resurrects the deleted bucket through `resolve_terraform_state`.
    """
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["terraform_state"] = {
        "bucket": "s3://gone-bucket/",
        "access_key": "TFAK",
        "secret_key": "TFSK",
    }

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    record = updated["project_credentials"]["projects"]["project-a"]
    assert "storage" not in record
    assert "terraform_state" not in record


def test_retire_project_bucket_document_leaves_a_different_terraform_bucket() -> None:
    """A `storage` match must not sweep away an unrelated Terraform backend."""
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["terraform_state"] = {
        "bucket": "s3://other-tf-bucket/",
        "access_key": "TFAK",
    }

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    record = updated["project_credentials"]["projects"]["project-a"]
    assert "storage" not in record
    assert record["terraform_state"]["bucket"] == "s3://other-tf-bucket/"


def test_retire_project_bucket_document_retires_a_terraform_only_record() -> None:
    """A record with no `storage` block (Terraform-only) must still retire.

    `storage_selected` must stay untouched: it describes `storage`, which
    never matched here.
    """
    document = _document()
    record = document["project_credentials"]["projects"]["project-a"]
    del record["storage"]
    record["storage_selected"] = False
    record["terraform_state"] = {"bucket": "s3://gone-bucket/", "access_key": "TFAK"}

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    updated_record = updated["project_credentials"]["projects"]["project-a"]
    assert "terraform_state" not in updated_record
    assert updated_record["storage_selected"] is False


def test_retire_project_bucket_document_preserves_a_different_storage_bucket() -> None:
    """A `terraform_state` match must not sweep away an unrelated storage bucket."""
    document = _document()
    document["project_credentials"]["projects"]["project-a"]["terraform_state"] = {
        "bucket": "s3://gone-bucket/",
        "access_key": "TFAK",
    }
    document["project_credentials"]["projects"]["project-a"]["storage"]["bucket"] = (
        "s3://still-here/"
    )

    updated = retire_project_bucket_document(document, "project-a", "gone-bucket")

    record = updated["project_credentials"]["projects"]["project-a"]
    assert record["storage"]["bucket"] == "s3://still-here/"
    assert record["storage_selected"] is True
    assert "terraform_state" not in record
