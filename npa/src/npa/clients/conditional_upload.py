"""Preserve object write conditions when large payloads require multipart upload."""

from __future__ import annotations

import io
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

_MULTIPART_THRESHOLD = 8 * 1024 * 1024
_PART_SIZE = 64 * 1024 * 1024
_MAX_PARTS = 10_000


def _upload_parts(client: Any, body: Any, size: int, upload: dict) -> list[dict]:
    part_size = max(_PART_SIZE, (size + _MAX_PARTS - 1) // _MAX_PARTS)
    parts = []
    while chunk := body.read(part_size):
        number = len(parts) + 1
        response = client.upload_part(**upload, PartNumber=number, Body=chunk)
        parts.append({"PartNumber": number, "ETag": response["ETag"]})
    return parts


def _multipart_put(client: Any, request: dict, body: Any, size: int) -> dict:
    parameters = {key: request[key] for key in ("Bucket", "Key")}
    metadata = {
        key: request[key] for key in ("ContentType", "Metadata") if key in request
    }
    started = client.create_multipart_upload(**parameters, **metadata)
    upload = {**parameters, "UploadId": started["UploadId"]}
    conditions = {
        key: request[key] for key in ("IfMatch", "IfNoneMatch") if key in request
    }
    try:
        parts = _upload_parts(client, body, size, upload)
        return client.complete_multipart_upload(
            **upload, MultipartUpload={"Parts": parts}, **conditions
        )
    except BaseException:
        try:
            client.abort_multipart_upload(**upload)
        except (BotoCoreError, ClientError):
            pass
        raise


def put_object_conditional(client: Any, request: dict) -> dict:
    """Write bytes or a seekable stream with conditions enforced at completion.

    Args:
        client: S3 client supporting conditional multipart completion.
        request: PutObject parameters with exactly one object write condition.
    Returns:
        Provider response containing the completed object's ETag.
    Raises:
        ValueError: Neither or both write conditions were supplied.
        ClientError: Upload or conditional completion failed.
        BotoCoreError: A storage request could not complete.
    """
    if bool(request.get("IfMatch")) == bool(request.get("IfNoneMatch")):
        raise ValueError("choose exactly one conditional object-write guard")
    body = request["Body"]
    if isinstance(body, bytes):
        size = len(body)
        body = io.BytesIO(body)
    else:
        position = body.tell()
        size = body.seek(0, io.SEEK_END) - position
        body.seek(position)
    if size < _MULTIPART_THRESHOLD:
        return client.put_object(**request)
    return _multipart_put(client, request, body, size)
