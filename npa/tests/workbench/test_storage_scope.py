from __future__ import annotations

from pathlib import Path

import pytest

from npa.workbench.storage_scope import (
    StorageAuthorizationError,
    StorageScope,
    use_storage_scope,
)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_local_scope_accepts_absolute_and_file_uris(
    tmp_path: Path, operation: str
) -> None:
    scope = StorageScope.from_config(local_roots=[tmp_path])
    target = tmp_path / "nested" / "artifact.json"

    assert scope.authorize(str(target), operation=operation).local_path == target
    assert scope.authorize(target.as_uri(), operation=operation).local_path == target


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize(
    "uri", ["https://example.invalid/a", "ftp://example.invalid/a"]
)
def test_scope_rejects_unknown_schemes(uri: str, operation: str) -> None:
    with pytest.raises(StorageAuthorizationError, match="scheme"):
        StorageScope().authorize(uri, operation=operation)


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize(
    "uri",
    [
        "file://remote.invalid/sandbox/artifact.json",
        "file:///sandbox/artifact.json?version=1",
        "file:///sandbox/artifact.json#fragment",
    ],
)
def test_scope_rejects_noncanonical_file_uris(uri: str, operation: str) -> None:
    with pytest.raises(StorageAuthorizationError, match="file URI"):
        StorageScope.from_config(local_roots=["/sandbox"]).authorize(
            uri, operation=operation
        )


@pytest.mark.parametrize("operation", ["read", "write"])
def test_local_scope_rejects_absolute_and_traversal_escape(
    tmp_path: Path, operation: str
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    scope = StorageScope.from_config(local_roots=[sandbox])

    for candidate in (tmp_path / "outside.json", sandbox / ".." / "outside.json"):
        with pytest.raises(StorageAuthorizationError, match="outside"):
            scope.authorize(str(candidate), operation=operation)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_local_scope_rejects_symlink_escape(tmp_path: Path, operation: str) -> None:
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    (sandbox / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageAuthorizationError, match="outside"):
        StorageScope.from_config(local_roots=[sandbox]).authorize(
            str(sandbox / "escape" / "artifact.json"), operation=operation
        )


@pytest.mark.parametrize("operation", ["read", "write"])
def test_s3_scope_enforces_bucket_and_prefix(operation: str) -> None:
    scope = StorageScope.from_config(s3_roots=["s3://allowed-bucket/team/run"])

    allowed = scope.authorize(
        "s3://allowed-bucket/team/run/artifact.json", operation=operation
    )
    assert (allowed.bucket, allowed.key) == (
        "allowed-bucket",
        "team/run/artifact.json",
    )
    for candidate in (
        "s3://foreign-bucket/team/run/artifact.json",
        "s3://allowed-bucket/team/runner/artifact.json",
        "s3://allowed-bucket/team/run/../secret.json",
        "s3://allowed-bucket/team/run/%2e%2e/secret.json",
    ):
        with pytest.raises(StorageAuthorizationError):
            scope.authorize(candidate, operation=operation)


@pytest.mark.parametrize(
    "module_name",
    ["npa.workbench.dataset.storage", "npa.workbench.insights.storage"],
)
def test_storage_helpers_enforce_s3_read_write_parity(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    from unittest.mock import Mock

    module = importlib.import_module(module_name)
    client = Mock()
    body = Mock()
    body.read.return_value = b"payload"
    client.get_object.return_value = {"Body": body}
    monkeypatch.setattr(module, "_s3_client", lambda: client)
    scope = StorageScope.from_config(s3_roots=["s3://allowed-bucket/team/run"])

    with use_storage_scope(scope):
        module.write_bytes_uri("s3://allowed-bucket/team/run/output.bin", b"payload")
        assert (
            module.read_bytes_uri("s3://allowed-bucket/team/run/output.bin")
            == b"payload"
        )
        with pytest.raises(StorageAuthorizationError):
            module.write_bytes_uri(
                "s3://foreign-bucket/team/run/output.bin", b"payload"
            )
        with pytest.raises(StorageAuthorizationError):
            module.read_bytes_uri("s3://foreign-bucket/team/run/output.bin")

    client.put_object.assert_called_once()
    client.get_object.assert_called_once()
