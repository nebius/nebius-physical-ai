"""Preserve private file handling and exact-object fallback during transfers."""

import io
import os
from pathlib import Path
from threading import Event, Lock, Thread
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from npa.clients.storage import StorageClient, _run_bounded


class _ObservedBody:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.closed = False

    def iter_chunks(self, chunk_size: int):
        for path in self.directory.iterdir():
            assert path.stat().st_mode & 0o077 == 0
        yield b"replacement"

    def close(self) -> None:
        self.closed = True


class _PagedArtifactStore:
    def __init__(self) -> None:
        self.transfer_started = Event()

    def get_paginator(self, operation: str):
        return self

    def paginate(self, **kwargs):
        if kwargs.get("PaginationConfig"):
            yield {"Contents": [{"Key": "models/artifact-000.bin"}]}
            return
        yield {
            "Contents": [
                {"Key": f"models/artifact-{index:03}.bin"} for index in range(16)
            ]
        }
        assert self.transfer_started.is_set(), "Listing outran every download"
        yield {"Contents": [{"Key": "models/tail.bin"}]}

    def download_file(self, bucket: str, key: str, destination: str, **kwargs):
        self.transfer_started.set()
        Path(destination).write_bytes(key.encode())


def test_private_replacement_staging_stays_private(tmp_path: Path) -> None:
    """Keep a private destination's replacement private while bytes arrive.

    Args:
        tmp_path: Isolated local download directory.
    Returns:
        None.
    Raises:
        AssertionError: Replacement bytes have broader access or leak staging.
    """
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"previous complete checkpoint")
    target.chmod(0o600)

    body = _ObservedBody(tmp_path)
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.get_object.return_value = {"Body": body, "ContentLength": 11}

    previous_umask = os.umask(0o022)
    try:
        client.download_file("s3://bucket/checkpoint.bin", str(target))
    finally:
        os.umask(previous_umask)

    assert target.read_bytes() == b"replacement"
    assert target.stat().st_mode & 0o777 == 0o600
    assert body.closed
    assert list(tmp_path.iterdir()) == [target]


def test_replacement_preserves_existing_mode_under_stricter_umask(
    tmp_path: Path,
) -> None:
    """Preserve an existing group-readable destination under a private umask.

    Args:
        tmp_path: Isolated local download directory.
    Returns:
        None.
    Raises:
        AssertionError: Replacement silently changes existing permissions.
    """
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"previous")
    target.chmod(0o640)
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.get_object.return_value = {
        "Body": StreamingBody(io.BytesIO(b"replacement"), 11),
        "ContentLength": 11,
    }

    previous_umask = os.umask(0o077)
    try:
        client.download_file("s3://bucket/checkpoint.bin", str(target))
    finally:
        os.umask(previous_umask)

    assert target.read_bytes() == b"replacement"
    assert target.stat().st_mode & 0o777 == 0o640


def test_head_denied_exact_object_keeps_precedence_over_tree(tmp_path: Path) -> None:
    """Use an exact object found by listing when HEAD cannot authorize it.

    Args:
        tmp_path: Isolated local download directory.
    Returns:
        None.
    Raises:
        AssertionError: A descendant tree displaces the exact listed object.
    """
    target = tmp_path / "download.bin"
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.head_object.side_effect = ClientError(
        {"Error": {"Code": "403"}}, "HeadObject"
    )
    client._s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "checkpoint"}, {"Key": "checkpoint/nested.bin"}]}
    ]
    client._s3.get_object.return_value = {
        "Body": StreamingBody(io.BytesIO(b"checkpoint"), 10),
        "ContentLength": 10,
    }

    def download(bucket: str, key: str, destination: str, **kwargs) -> None:
        Path(destination).write_bytes(key.encode())

    client._s3.download_file.side_effect = download

    result = client.download_path("s3://bucket/checkpoint", str(target))

    assert result == str(target)
    assert target.is_file()
    assert target.read_bytes() == b"checkpoint"


@pytest.mark.parametrize("method", ["download_directory", "download_path"])
def test_download_starts_before_entire_tree_is_listed(
    tmp_path: Path, method: str
) -> None:
    """Begin transfers before requesting every remote listing page.

    Args:
        tmp_path: Isolated local download directory.
        method: Public tree download entry point.
    Returns:
        None.
    Raises:
        AssertionError: Listing is eager or some objects are not downloaded.
    """
    client = StorageClient.__new__(StorageClient)
    client._s3 = _PagedArtifactStore()

    getattr(client, method)("s3://bucket/models/", str(tmp_path))

    assert len(list(tmp_path.iterdir())) == 17
    assert (tmp_path / "tail.bin").read_bytes() == b"models/tail.bin"


def test_upload_starts_before_entire_tree_is_walked(
    tmp_path: Path, monkeypatch
) -> None:
    """Begin uploads before the local directory iterator is exhausted.

    Args:
        tmp_path: Isolated source directory.
        monkeypatch: Scoped directory-walk replacement.
    Returns:
        None.
    Raises:
        AssertionError: Walking is eager or some files are not uploaded.
    """
    transfer_started = Event()

    def walk(directory):
        yield str(directory), [], [f"artifact-{index:03}.bin" for index in range(16)]
        assert transfer_started.is_set(), "Directory walk outran every upload"
        yield str(directory / "nested"), [], ["tail.bin"]

    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.upload_file.side_effect = lambda *args, **kwargs: transfer_started.set()
    monkeypatch.setattr("npa.clients.storage.os.walk", lambda path: walk(Path(path)))

    client.upload_directory(str(tmp_path), "s3://bucket/models/")

    assert client._s3.upload_file.call_count == 17


def test_bounded_runner_does_not_eagerly_consume_blocked_source() -> None:
    """Keep producer consumption at the worker bound while every task blocks."""

    total = 40
    max_workers = 4
    release = Event()
    all_workers_started = Event()
    source_exhausted = Event()
    counter_lock = Lock()
    consumed: list[int] = []
    started = 0
    results: list[int] = []
    errors: list[BaseException] = []

    def tasks():
        nonlocal started
        for index in range(total):
            consumed.append(index)

            def task(value: int = index) -> int:
                nonlocal started
                with counter_lock:
                    started += 1
                    if started == max_workers:
                        all_workers_started.set()
                assert release.wait(timeout=5), "blocked worker was never released"
                return value

            yield task
        source_exhausted.set()

    def consume() -> None:
        try:
            results.extend(_run_bounded(tasks(), max_workers=max_workers))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    consumer = Thread(target=consume)
    consumer.start()
    try:
        assert all_workers_started.wait(timeout=5)
        assert not source_exhausted.wait(timeout=0.2)
        assert len(consumed) == max_workers
    finally:
        release.set()
        consumer.join(timeout=5)

    assert not consumer.is_alive()
    assert errors == []
    assert source_exhausted.is_set()
    assert sorted(results) == list(range(total))


def test_single_file_tree_retains_managed_transfer_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    """Avoid an outer worker pool or multipart override for a lone checkpoint.

    Args:
        tmp_path: Isolated transfer directory.
        monkeypatch: Scoped worker-pool replacement.
    Returns:
        None.
    Raises:
        AssertionError: A single transfer adds a pool or changes SDK defaults.
    """
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"checkpoint")
    client = StorageClient.__new__(StorageClient)
    client._s3 = Mock()
    client._s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "models/checkpoint.bin"}]}
    ]
    pool = Mock(side_effect=AssertionError("A lone file needs no outer pool"))
    monkeypatch.setattr("npa.clients.storage.ThreadPoolExecutor", pool)

    client.upload_directory(str(tmp_path), "s3://bucket/models/")
    client.download_directory("s3://bucket/models/", str(tmp_path))

    client._s3.upload_file.assert_called_once_with(
        str(source), "bucket", "models/checkpoint.bin"
    )
    client._s3.download_file.assert_called_once_with(
        "bucket", "models/checkpoint.bin", str(source)
    )
    pool.assert_not_called()
