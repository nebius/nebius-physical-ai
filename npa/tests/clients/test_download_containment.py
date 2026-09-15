"""Remote object names cannot choose paths outside a local download tree."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient, StorageError, safe_s3_download_target


@pytest.mark.parametrize("relative", ["../escape", "nested/../../escape", "/escape", "nested\\escape"])
@pytest.mark.parametrize("method", ["download_directory", "download_path"])
def test_tree_download_rejects_remote_traversal(tmp_path, relative, method):
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    client._s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "models/" + relative}]}
    ]

    with pytest.raises(StorageError):
        getattr(client, method)("s3://bucket/models/", str(tmp_path / "cache"))

    client._s3.download_file.assert_not_called()


@pytest.mark.parametrize("method", ["download_directory", "download_path"])
def test_tree_download_rejects_existing_symlink_escape(tmp_path, method):
    cache, outside = tmp_path / "cache", tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (cache / "nested").symlink_to(outside, target_is_directory=True)
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "models/nested/weights.bin"}]}
    ]

    with pytest.raises(StorageError):
        getattr(client, method)("s3://bucket/models/", str(cache))

    client._s3.download_file.assert_not_called()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("method", ["download_directory", "download_path"])
@pytest.mark.parametrize("prefix", ["models/", ""])
def test_tree_download_preserves_nested_file_bytes(tmp_path, method, prefix):
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": prefix + "nested/weights.bin"}]}
    ]
    client._s3.download_file.side_effect = lambda _b, _k, p: Path(p).write_bytes(b"tensor-data")

    getattr(client, method)("s3://bucket/" + prefix, str(tmp_path / "cache"))

    assert (tmp_path / "cache/nested/weights.bin").read_bytes() == b"tensor-data"


def test_download_target_rejects_out_of_prefix_response(tmp_path):
    with pytest.raises(StorageError):
        safe_s3_download_target(tmp_path, "other/config.json", "models/")
