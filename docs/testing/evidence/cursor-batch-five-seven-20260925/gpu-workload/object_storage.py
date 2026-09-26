"""Read and publish exact S3 objects using environment credentials and readback."""

import os
from urllib.parse import urlsplit

from regression_contract import _sha256


def _client():
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise ValueError(
            "AWS_ENDPOINT_URL must select the preflighted storage endpoint"
        )
    return boto3.client("s3", endpoint_url=endpoint)


def _location(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("an exact s3://bucket/key URI is required")
    if parsed.query or parsed.fragment or parsed.username:
        raise ValueError("S3 URI must not contain credentials, query, or fragment")
    return {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}


def _read(client, uri):
    response = client.get_object(**_location(uri))
    with response["Body"] as body:
        return body.read()


def _publish(client, uri, data):
    client.put_object(**_location(uri), Body=data)
    if _read(client, uri) != data:
        raise ValueError("S3 publication byte readback mismatch")
    return {"sha256": _sha256(data), "size": len(data)}


def _release_exists(client, uri):
    from botocore.exceptions import ClientError

    try:
        return _read(client, uri)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
