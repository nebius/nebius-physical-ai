"""Open pinned DROID video objects through a seekable, per-handle S3 buffer."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

_BLOCK_BYTES = 1024 * 1024
_original_open = None
_settings = None
_sizes = None
_client = None
_client_process = None
_counter_lock = threading.Lock()
_counters = {}


def _count(name, value=1):
    with _counter_lock:
        _counters[name] = _counters.get(name, 0) + value


def _object_sizes(path):
    manifest = json.loads(Path(path).read_text())
    sizes = {}
    for item in manifest["files"]:
        key, size = item["path"], item["size"]
        if not isinstance(key, str) or type(size) is not int or size < 0:
            raise ValueError(
                "manifest requires string paths and nonnegative integer sizes"
            )
        if key.startswith("/") or ".." in Path(key).parts or key in sizes:
            raise ValueError("manifest contains an unsafe or duplicate path")
        sizes[key] = size
    if not any(key.startswith("success/videos/") for key in sizes):
        raise ValueError("manifest contains no success/videos objects")
    return sizes


def _s3_client():
    global _client, _client_process
    if _client_process != os.getpid():
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=_settings.get("aws_profile"))
        _client = session.client(
            "s3",
            endpoint_url=_settings["endpoint_url"],
            region_name=_settings["region"],
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=10,
                read_timeout=45,
                retries={"max_attempts": 0},
                max_pool_connections=16,
            ),
        )
        _client_process = os.getpid()
    return _client


def _get_block(key, start, size):
    end = min(start + _BLOCK_BYTES, size) - 1
    _count("get_attempts")
    try:
        response = _s3_client().get_object(
            Bucket=_settings["bucket"], Key=key, Range=f"bytes={start}-{end}"
        )
        try:
            if (
                response["ResponseMetadata"]["HTTPStatusCode"] != 206
                or response["ContentRange"] != f"bytes {start}-{end}/{size}"
            ):
                raise ValueError("range headers differ from the pinned manifest")
            body = response["Body"].read(end - start + 2)
        finally:
            response["Body"].close()
        if len(body) != end - start + 1:
            raise ValueError("range response differs from the pinned manifest")
    except Exception:
        _count("get_errors")
        raise OSError(
            "DROID range read failed; no sample substitution is permitted"
        ) from None
    _count("http_206")
    _count("s3_response_bytes", len(body))
    return body


class _RangeFile(io.RawIOBase):
    def __init__(self, key):
        super().__init__()
        self._key = key
        self._length = _sizes[key]
        self._position = 0
        self._buffer_start = -1
        self._buffer = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        origins = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self._position,
            io.SEEK_END: self._length,
        }
        if whence not in origins or type(offset) is not int:
            raise ValueError("invalid DROID reader seek")
        position = origins[whence] + offset
        if position < 0:
            raise ValueError("cannot seek before the object")
        self._position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        if size is None or size < 0:
            size = max(0, self._length - self._position)
        remaining = min(size, max(0, self._length - self._position))
        parts = []
        while remaining:
            start = self._position // _BLOCK_BYTES * _BLOCK_BYTES
            if start != self._buffer_start:
                self._buffer = _get_block(self._key, start, self._length)
                self._buffer_start = start
            offset = self._position - start
            count = min(remaining, len(self._buffer) - offset)
            parts.append(self._buffer[offset : offset + count])
            self._position += count
            remaining -= count
        return b"".join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        self._buffer = b""
        super().close()


class _VideoOpen:
    def __init__(self, key):
        self._key = key
        self._handle = None

    def __enter__(self):
        self._handle = _RangeFile(self._key)
        return self._handle

    def __exit__(self, *exception):
        if self._handle is not None:
            self._handle.close()


def _open_video(path, *args, **kwargs):
    prefix = str(Path(_settings["dataset_root"]) / "success/videos") + "/"
    value = str(path)
    if not value.startswith(prefix):
        return _original_open(path, *args, **kwargs)
    if args or set(kwargs) - {"mode"} or kwargs.get("mode", "rb") != "rb":
        raise ValueError("DROID videos support binary reads only")
    suffix = value[len(prefix) :]
    if ".." in Path(suffix).parts or not suffix:
        raise ValueError("invalid DROID video path")
    key = "success/videos/" + suffix
    if key not in _sizes:
        raise ValueError("video is absent from the pinned manifest")
    return _VideoOpen(key)


def install(settings):
    """Route only this dataset's video opens to manifest-constrained S3 reads.

    Args:
        settings: Dataset root, manifest, bucket, HTTPS endpoint, region and optional AWS profile.
    Returns:
        None.
    Raises:
        ValueError: Configuration or manifest is invalid, or an installation conflicts.
    """
    global _settings, _sizes, _original_open
    import fsspec

    if _original_open is not None:
        if settings != _settings:
            raise ValueError("a different DROID adapter is already installed")
        return
    endpoint = urlsplit(settings["endpoint_url"])
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username:
        raise ValueError("an HTTPS S3 endpoint without credentials is required")
    if endpoint.query or endpoint.fragment or endpoint.password:
        raise ValueError("endpoint must not contain credentials or URL parameters")
    sizes = _object_sizes(settings["manifest"])
    _settings, _sizes = dict(settings), sizes
    _original_open = fsspec.open
    fsspec.open = _open_video


def uninstall():
    """Restore the original file opener and release process-local S3 state.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """
    global _original_open, _client, _client_process, _settings, _sizes
    if _original_open is not None:
        import fsspec

        fsspec.open = _original_open
    _original_open = _client = _client_process = _settings = _sizes = None
    _counters.clear()


def counters():
    """Return this process's observed request counts without credential material.

    Args:
        None.
    Returns:
        A copy of process-local request and response-byte counters.
    Raises:
        None.
    """
    with _counter_lock:
        return dict(_counters)
