"""Hermetic failure-path checks for destructive live-test fixture cleanup."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def cleanup():
    path = Path(__file__).parent / "e2e" / "s3_fixture_cleanup.py"
    spec = importlib.util.spec_from_file_location("s3_fixture_cleanup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prefix", ["", "/", "///", "test-prefix"])
def test_cleanup_rejects_unbounded_prefix_before_contacting_provider(cleanup, prefix):
    client = Mock()
    with pytest.raises(ValueError):
        cleanup.delete_owned_prefix(client, "bucket", prefix)
    client.get_paginator.assert_not_called()
    client.delete_objects.assert_not_called()


def test_cleanup_refuses_provider_keys_outside_the_owned_prefix(cleanup):
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Versions": [{"Key": "other-test/object", "VersionId": "old"}]}
    ]
    with pytest.raises(ValueError, match="outside"):
        cleanup.delete_owned_prefix(client, "bucket", "owned/")
    client.delete_objects.assert_not_called()


def test_cleanup_does_not_accept_http_success_with_per_version_errors(cleanup):
    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Versions": [{"Key": "owned/object", "VersionId": "old"}]}
    ]
    client.delete_objects.return_value = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "Errors": [{"Key": "owned/object", "Code": "AccessDenied"}],
    }
    with pytest.raises(RuntimeError, match="failed to delete"):
        cleanup.delete_owned_prefix(client, "bucket", "owned/")


def test_cleanup_deletes_exact_versions_and_markers_and_verifies_absence(cleanup):
    client = Mock()
    versions = [
        {"Key": "owned/object", "VersionId": "first"},
        {"Key": "owned/object", "VersionId": "second"},
    ]
    marker = {"Key": "owned/object", "VersionId": "marker"}
    client.get_paginator.return_value.paginate.side_effect = [
        [{"Versions": versions, "DeleteMarkers": [marker]}],
        [{}],
    ]
    client.delete_objects.return_value = {}
    client.list_objects_v2.return_value = {}

    cleanup.delete_owned_prefix(client, "bucket", "owned/")

    client.delete_objects.assert_called_once_with(
        Bucket="bucket", Delete={"Objects": [*versions, marker]}
    )
    assert client.get_paginator.return_value.paginate.call_count == 2
    client.list_objects_v2.assert_called_once_with(
        Bucket="bucket", Prefix="owned/", MaxKeys=1
    )


def test_cleanup_rejects_remaining_versions_even_when_current_listing_is_empty(cleanup):
    client = Mock()
    version = {"Key": "owned/object", "VersionId": "retained"}
    client.get_paginator.return_value.paginate.return_value = [{"Versions": [version]}]
    client.delete_objects.return_value = {}
    client.list_objects_v2.return_value = {}

    with pytest.raises(RuntimeError, match="versions or delete markers remain"):
        cleanup.delete_owned_prefix(client, "bucket", "owned/")
