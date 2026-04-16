"""
Nebius S3 helper for checkpoint and dataset management.

Zero LeRobot imports — this is pure infrastructure glue.
Reads all configuration from environment variables set by environment.sh.
"""

import argparse
import os
import sys
from pathlib import Path

_UNSET_TEMPLATE_PREFIX = "__TERRAFORM_DEFAULT_"
DEFAULT_S3_ENDPOINT = "__TERRAFORM_DEFAULT_S3_ENDPOINT__"
DEFAULT_REGION = "__TERRAFORM_DEFAULT_REGION__"
DEFAULT_BUCKET = "__TERRAFORM_DEFAULT_BUCKET__"


def _template_default(value):
    if isinstance(value, str) and value.startswith(_UNSET_TEMPLATE_PREFIX):
        return None
    return value

def _endpoint():
    return (
        os.environ.get("NEBIUS_S3_ENDPOINT")
        or _template_default(DEFAULT_S3_ENDPOINT)
        or "https://storage.eu-north1.nebius.cloud"
    )


def _region():
    return os.environ.get("NEBIUS_REGION") or _template_default(DEFAULT_REGION) or "eu-north1"


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=_endpoint(),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=_region(),
    )


def _bucket():
    name = os.environ.get("NEBIUS_S3_BUCKET") or _template_default(DEFAULT_BUCKET)
    if not name:
        sys.exit("NEBIUS_S3_BUCKET is not set. Run: source environment.sh")
    return name


def _iter_objects(client, bucket, prefix):
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            yield obj
        if not response.get("IsTruncated"):
            return
        continuation_token = response.get("NextContinuationToken")


def _key_exists(client, bucket, key):
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_check(_args):
    """Verify S3 connectivity."""
    client = _s3_client()
    bucket = _bucket()
    buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    print(f"Endpoint:  {_endpoint()}")
    print(f"Bucket:    {bucket}")
    print(f"Available: {', '.join(buckets)}")
    if bucket in buckets:
        print("Connection OK")
    else:
        print(f"WARNING: configured bucket '{bucket}' not found in account")


def cmd_upload(args):
    """Upload a local file to S3."""
    local = Path(args.path)
    if not local.exists():
        sys.exit(f"File not found: {local}")
    key = args.key or f"checkpoints/{local.name}"
    bucket = _bucket()
    _s3_client().upload_file(str(local), bucket, key)
    print(f"Uploaded -> s3://{bucket}/{key}")


def cmd_download(args):
    """Download a file or a prefix from S3."""
    bucket = _bucket()
    client = _s3_client()

    if _key_exists(client, bucket, args.key):
        dest = Path(args.dest or Path(args.key).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, args.key, str(dest))
        print(f"Downloaded s3://{bucket}/{args.key} -> {dest}  ({dest.stat().st_size:,} bytes)")
        return

    prefix = args.key.rstrip("/") + "/"
    objects = list(_iter_objects(client, bucket, prefix))
    if not objects:
        sys.exit(f"No S3 object or prefix found: {args.key}")

    dest_root = Path(args.dest or Path(args.key.rstrip("/")).name)
    count = 0
    total_bytes = 0
    for obj in objects:
        relative = obj["Key"][len(prefix):]
        if not relative:
            continue
        local_path = dest_root / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, obj["Key"], str(local_path))
        count += 1
        total_bytes += int(obj.get("Size", 0))

    print(
        f"Downloaded {count} object(s) from s3://{bucket}/{prefix} -> {dest_root}"
        f"  ({total_bytes:,} bytes)"
    )


def cmd_ls(args):
    """List objects under a prefix."""
    client = _s3_client()
    bucket = _bucket()
    prefix = args.prefix or ""
    count = 0
    for obj in _iter_objects(client, bucket, prefix):
        print(f"  {obj['Size']:>12,}  {obj['Key']}")
        count += 1
    if count == 0:
        print(f"(no objects under prefix '{prefix}')")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Nebius S3 helper for LeRobot checkpoints and data")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Verify S3 connectivity")

    p_up = sub.add_parser("upload", help="Upload file to S3")
    p_up.add_argument("path", help="Local file path")
    p_up.add_argument("--key", help="S3 key (default: checkpoints/<filename>)")

    p_dl = sub.add_parser("download", help="Download file or prefix from S3")
    p_dl.add_argument("key", help="S3 object key or prefix")
    p_dl.add_argument("--dest", help="Local destination path")

    p_ls = sub.add_parser("ls", help="List objects in bucket")
    p_ls.add_argument("prefix", nargs="?", help="Key prefix to filter")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"check": cmd_check, "upload": cmd_upload, "download": cmd_download, "ls": cmd_ls}[args.command](args)


if __name__ == "__main__":
    main()
