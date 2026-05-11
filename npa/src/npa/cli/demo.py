"""Demo artifact bootstrap commands."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError
import typer
import yaml

from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import resolve_credentials
from npa.clients.scoped_credentials import (
    SCOPED_CREDENTIAL_ERROR_CODES,
    client_error_code,
    run_with_host_credential_fallback,
)
from npa.errors import ScopedCredentialError

app = typer.Typer(
    name="demo",
    help="Demo artifact bootstrap and verification helpers.",
    no_args_is_help=True,
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "manifests" / "demo-8gpu-h200.yaml"
)
logger = logging.getLogger(__name__)


class DemoManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DemoArtifact:
    name: str
    source_uri: str
    target_path: str
    is_prefix: bool = False
    sha256: str = ""
    size_bytes: int | None = None
    expected_count: int | None = None
    total_size_bytes: int | None = None


@dataclass(frozen=True)
class DemoManifest:
    version: int
    artifacts: list[DemoArtifact]


def load_manifest(path: Path) -> DemoManifest:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise DemoManifestError(f"Manifest must be a mapping: {path}")
    version = int(data.get("version", 0))
    if version != 1:
        raise DemoManifestError(f"Unsupported manifest version: {version}")
    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise DemoManifestError("Manifest must contain a non-empty artifacts list")

    artifacts: list[DemoArtifact] = []
    for idx, item in enumerate(raw_artifacts):
        if not isinstance(item, dict):
            raise DemoManifestError(f"Artifact #{idx} must be a mapping")
        try:
            name = str(item["name"])
            source_uri = str(item["source_uri"])
            target_path = str(item["target_path"])
        except KeyError as exc:
            raise DemoManifestError(
                f"Artifact #{idx} missing required field: {exc}"
            ) from exc
        is_prefix = bool(item.get("is_prefix", False))
        sha256 = str(item.get("sha256", "") or "")
        if not is_prefix and not sha256:
            raise DemoManifestError(f"Artifact {name!r} must define sha256")
        artifacts.append(
            DemoArtifact(
                name=name,
                source_uri=source_uri,
                target_path=target_path,
                is_prefix=is_prefix,
                sha256=sha256,
                size_bytes=_optional_int(item.get("size_bytes")),
                expected_count=_optional_int(item.get("expected_count")),
                total_size_bytes=_optional_int(item.get("total_size_bytes")),
            )
        )
    return DemoManifest(version=version, artifacts=artifacts)


def stage_artifacts(
    *,
    target_bucket: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool = False,
    source_project: str | None = None,
    target_project: str | None = None,
) -> list[dict[str, str]]:
    manifest = load_manifest(manifest_path)
    source_buckets = {
        _parse_s3_uri(artifact.source_uri)[0] for artifact in manifest.artifacts
    }
    target_buckets = {
        _target_bucket_key(target_bucket, artifact.target_path)[0]
        for artifact in manifest.artifacts
    }
    s3 = _stage_s3_client(
        s3_client=s3_client,
        host_s3_client=host_s3_client,
        allow_host_creds=allow_host_creds,
        source_project=source_project,
        target_project=target_project,
        source_buckets=source_buckets,
        target_buckets=target_buckets,
    )
    actions: list[dict[str, str]] = []
    for artifact in manifest.artifacts:
        if artifact.is_prefix:
            action = _stage_prefix(s3, artifact, target_bucket)
        else:
            action = _stage_file(s3, artifact, target_bucket)
        actions.append({"name": artifact.name, "action": action})
    return actions


def verify_artifacts(
    *,
    target_bucket: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    s3_client=None,
) -> list[str]:
    manifest = load_manifest(manifest_path)
    s3 = s3_client or _s3_client()
    issues: list[str] = []
    for artifact in manifest.artifacts:
        bucket, key = _target_bucket_key(target_bucket, artifact.target_path)
        if artifact.is_prefix:
            count, total = _prefix_stats(s3, bucket, _ensure_prefix(key))
            if count == 0:
                issues.append(
                    f"{artifact.name}: missing target prefix s3://{bucket}/{key}"
                )
                continue
            if artifact.expected_count is not None and count != artifact.expected_count:
                issues.append(
                    f"{artifact.name}: expected {artifact.expected_count} objects, found {count}"
                )
            if (
                artifact.total_size_bytes is not None
                and total != artifact.total_size_bytes
            ):
                issues.append(
                    f"{artifact.name}: expected {artifact.total_size_bytes} bytes, found {total}"
                )
            continue

        head = _head_or_none(s3, bucket, key)
        if head is None:
            issues.append(f"{artifact.name}: missing target object s3://{bucket}/{key}")
            continue
        metadata = head.get("Metadata", {}) or {}
        actual_sha = str(metadata.get("sha256", ""))
        if actual_sha != artifact.sha256:
            issues.append(
                f"{artifact.name}: sha256 metadata mismatch "
                f"(expected {artifact.sha256}, found {actual_sha or 'missing'})"
            )
        if (
            artifact.size_bytes is not None
            and int(head.get("ContentLength", -1)) != artifact.size_bytes
        ):
            issues.append(
                f"{artifact.name}: size mismatch "
                f"(expected {artifact.size_bytes}, found {head.get('ContentLength')})"
            )
    return issues


@app.command("stage")
def stage_cmd(
    target_bucket: str = typer.Option(
        ...,
        "--target-bucket",
        help="Target S3 bucket or s3://bucket/prefix receiving staged demo artifacts.",
    ),
    manifest: Path = typer.Option(
        DEFAULT_MANIFEST,
        "--manifest",
        help=(
            "Artifact manifest. The default source bucket must be readable by "
            "the active Nebius S3 principal."
        ),
    ),
    source_project: str = typer.Option(
        "",
        "--source-project",
        help="Project alias whose scoped principal reads manifest source artifacts.",
    ),
    target_project: str = typer.Option(
        "",
        "--target-project",
        help="Project alias whose scoped principal writes staged artifacts.",
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Use --allow-host-creds to fall back to host credentials for source or target S3 operations.",
    ),
) -> None:
    """Stage demo artifacts into an operator-owned bucket."""
    try:
        actions = stage_artifacts(
            target_bucket=target_bucket,
            manifest_path=manifest,
            allow_host_creds=allow_host_creds,
            source_project=source_project or None,
            target_project=target_project or None,
        )
    except (DemoManifestError, ScopedCredentialError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output == "json":
        typer.echo(json.dumps({"status": "ok", "artifacts": actions}, indent=2))
        return
    for item in actions:
        typer.echo(f"{item['action']}: {item['name']}")


@app.command("verify")
def verify_cmd(
    target_bucket: str = typer.Option(
        ...,
        "--target-bucket",
        help="Target S3 bucket or s3://bucket/prefix containing staged demo artifacts.",
    ),
    manifest: Path = typer.Option(
        DEFAULT_MANIFEST, "--manifest", help="Artifact manifest."
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Verify staged demo artifacts without downloading object contents."""
    try:
        issues = verify_artifacts(target_bucket=target_bucket, manifest_path=manifest)
    except (DemoManifestError, ScopedCredentialError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output == "json":
        typer.echo(
            json.dumps(
                {"status": "ok" if not issues else "failed", "issues": issues}, indent=2
            )
        )
    elif issues:
        for issue in issues:
            typer.echo(issue)
    else:
        typer.echo("Demo artifacts verified.")

    if issues:
        raise typer.Exit(1)


def _stage_file(s3, artifact: DemoArtifact, target_bucket: str) -> str:
    source_bucket, source_key = _parse_s3_uri(artifact.source_uri)
    dest_bucket, dest_key = _target_bucket_key(target_bucket, artifact.target_path)
    head = _head_or_none(s3, dest_bucket, dest_key)
    if _head_matches_artifact(head, artifact):
        return "skip"

    data = _read_object(
        s3, source_bucket, source_key, operation=f"read {artifact.name}"
    )
    sha = hashlib.sha256(data).hexdigest()
    if artifact.sha256 and sha != artifact.sha256:
        raise DemoManifestError(
            f"{artifact.name}: source sha256 mismatch "
            f"(expected {artifact.sha256}, found {sha})"
        )
    if artifact.size_bytes is not None and len(data) != artifact.size_bytes:
        raise DemoManifestError(
            f"{artifact.name}: source size mismatch "
            f"(expected {artifact.size_bytes}, found {len(data)})"
        )
    _put_object(
        s3,
        dest_bucket,
        dest_key,
        data,
        metadata={"sha256": sha},
        operation=f"upload {artifact.name}",
    )
    verified = _head_or_none(s3, dest_bucket, dest_key)
    if not _head_matches_artifact(verified, artifact):
        raise DemoManifestError(f"{artifact.name}: upload verification failed")
    return "upload"


def _stage_prefix(s3, artifact: DemoArtifact, target_bucket: str) -> str:
    source_bucket, source_prefix = _parse_s3_uri(artifact.source_uri)
    dest_bucket, dest_prefix = _target_bucket_key(target_bucket, artifact.target_path)
    source_prefix = _ensure_prefix(source_prefix)
    dest_prefix = _ensure_prefix(dest_prefix)
    source_objects = _list_objects(
        s3, source_bucket, source_prefix, operation=f"list {artifact.name}"
    )
    if not source_objects:
        raise DemoManifestError(
            f"{artifact.name}: source prefix is empty: {artifact.source_uri}"
        )

    target_count, target_total = _prefix_stats(s3, dest_bucket, dest_prefix)
    expected_count = artifact.expected_count or len(source_objects)
    expected_total = artifact.total_size_bytes or sum(
        int(obj.get("Size", 0)) for obj in source_objects
    )
    if target_count == expected_count and target_total == expected_total:
        return "skip"

    for obj in source_objects:
        source_key = str(obj["Key"])
        rel = source_key[len(source_prefix) :]
        if not rel:
            continue
        dest_key = dest_prefix + rel
        _copy_object(
            s3,
            source_bucket,
            source_key,
            dest_bucket,
            dest_key,
            operation=f"copy {artifact.name}",
        )
    return "copy"


def _s3_client():
    storage = resolve_project_storage()
    return boto3.client(
        "s3",
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id or None,
        aws_secret_access_key=storage.aws_secret_access_key or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _host_s3_client():
    storage = resolve_project_storage()
    return boto3.client(
        "s3",
        endpoint_url=storage.endpoint_url or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _s3_client_for_project(project: str | None, *, allow_host_creds: bool):
    if not project:
        return _s3_client()
    credentials = resolve_credentials(project, allow_host_creds=allow_host_creds)
    return boto3.client(
        "s3",
        endpoint_url=credentials.endpoint_url,
        aws_access_key_id=credentials.aws_access_key_id or None,
        aws_secret_access_key=credentials.aws_secret_access_key or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _host_s3_client_for_project(project: str | None):
    storage = resolve_project_storage(project)
    return boto3.client(
        "s3",
        endpoint_url=storage.endpoint_url or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _stage_s3_client(
    *,
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool,
    source_project: str | None = None,
    target_project: str | None = None,
    source_buckets: set[str] | None = None,
    target_buckets: set[str] | None = None,
):
    if not source_project and not target_project:
        scoped_s3 = s3_client or _s3_client()
        if not allow_host_creds and host_s3_client is None:
            return scoped_s3
        return _HostFallbackS3(
            scoped_s3,
            host_s3_client or _host_s3_client(),
            allow_host_creds=allow_host_creds,
        )
    if s3_client is not None:
        scoped_s3 = s3_client
        if not allow_host_creds and host_s3_client is None:
            return scoped_s3
        return _HostFallbackS3(
            scoped_s3,
            host_s3_client or _host_s3_client(),
            allow_host_creds=allow_host_creds,
        )
    source_scoped = _s3_client_for_project(
        source_project, allow_host_creds=allow_host_creds
    )
    target_scoped = _s3_client_for_project(
        target_project, allow_host_creds=allow_host_creds
    )
    return _ProjectBoundaryS3(
        source_scoped,
        target_scoped,
        _host_s3_client_for_project(source_project),
        _host_s3_client_for_project(target_project),
        allow_host_creds=allow_host_creds,
        source_project=source_project,
        target_project=target_project,
        source_buckets=source_buckets or set(),
        target_buckets=target_buckets or set(),
    )


class _ProjectBoundaryS3:
    def __init__(
        self,
        source_scoped_s3,
        target_scoped_s3,
        source_host_s3,
        target_host_s3,
        *,
        allow_host_creds: bool,
        source_project: str | None,
        target_project: str | None,
        source_buckets: set[str],
        target_buckets: set[str],
    ) -> None:
        self._source_scoped_s3 = source_scoped_s3
        self._target_scoped_s3 = target_scoped_s3
        self._source_host_s3 = source_host_s3
        self._target_host_s3 = target_host_s3
        self._allow_host_creds = allow_host_creds
        self._source_project = source_project
        self._target_project = target_project
        self._source_buckets = source_buckets
        self._target_buckets = target_buckets

    def head_object(self, *, Bucket: str, Key: str):
        side = self._side_for_bucket(Bucket)
        scoped, host = self._clients(side)
        return self._run(
            side,
            lambda: scoped.head_object(Bucket=Bucket, Key=Key),
            lambda: host.head_object(Bucket=Bucket, Key=Key),
            bucket=Bucket,
            operation=f"head s3://{Bucket}/{Key}",
        )

    def get_object(self, *, Bucket: str, Key: str):
        scoped, host = self._clients("source")
        return self._run(
            "source",
            lambda: scoped.get_object(Bucket=Bucket, Key=Key),
            lambda: host.get_object(Bucket=Bucket, Key=Key),
            bucket=Bucket,
            operation=f"read s3://{Bucket}/{Key}",
        )

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ):
        scoped, host = self._clients("target")
        return self._run(
            "target",
            lambda: scoped.put_object(
                Bucket=Bucket, Key=Key, Body=Body, Metadata=Metadata
            ),
            lambda: host.put_object(
                Bucket=Bucket, Key=Key, Body=Body, Metadata=Metadata
            ),
            bucket=Bucket,
            operation=f"upload s3://{Bucket}/{Key}",
        )

    def copy_object(
        self,
        *,
        Bucket: str,
        Key: str,
        CopySource: dict[str, str],
        MetadataDirective: str,
    ):
        source_bucket = CopySource["Bucket"]
        source_key = CopySource["Key"]
        obj = self.get_object(Bucket=source_bucket, Key=source_key)
        metadata = obj.get("Metadata", {}) or {}
        self.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=obj["Body"].read(),
            Metadata={str(k): str(v) for k, v in metadata.items()},
        )

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, ContinuationToken: str | None = None
    ):
        side = self._side_for_bucket(Bucket)
        scoped, host = self._clients(side)
        kwargs = {"Bucket": Bucket, "Prefix": Prefix}
        if ContinuationToken:
            kwargs["ContinuationToken"] = ContinuationToken
        return self._run(
            side,
            lambda: scoped.list_objects_v2(**kwargs),
            lambda: host.list_objects_v2(**kwargs),
            bucket=Bucket,
            operation=f"list s3://{Bucket}/{Prefix}",
        )

    def _side_for_bucket(self, bucket: str) -> str:
        if bucket in self._source_buckets and bucket not in self._target_buckets:
            return "source"
        return "target"

    def _clients(self, side: str):
        if side == "source":
            return self._source_scoped_s3, self._source_host_s3
        return self._target_scoped_s3, self._target_host_s3

    def _run(
        self, side: str, scoped_operation, host_fallback, *, bucket: str, operation: str
    ):
        project = self._source_project if side == "source" else self._target_project
        if project:
            operation = f"{side} project '{project}' {operation}"
        return run_with_host_credential_fallback(
            scoped_operation,
            host_fallback,
            bucket=bucket,
            operation=operation,
            allow_host_creds=self._allow_host_creds,
            logger=logger,
        )


class _HostFallbackS3:
    def __init__(self, scoped_s3, host_s3, *, allow_host_creds: bool) -> None:
        self._scoped_s3 = scoped_s3
        self._host_s3 = host_s3
        self._allow_host_creds = allow_host_creds

    def head_object(self, *, Bucket: str, Key: str):
        return self._run(
            lambda: self._scoped_s3.head_object(Bucket=Bucket, Key=Key),
            lambda: self._host_s3.head_object(Bucket=Bucket, Key=Key),
            bucket=Bucket,
            operation=f"head s3://{Bucket}/{Key}",
        )

    def get_object(self, *, Bucket: str, Key: str):
        return self._run(
            lambda: self._scoped_s3.get_object(Bucket=Bucket, Key=Key),
            lambda: self._host_s3.get_object(Bucket=Bucket, Key=Key),
            bucket=Bucket,
            operation=f"read s3://{Bucket}/{Key}",
        )

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ):
        return self._run(
            lambda: self._scoped_s3.put_object(
                Bucket=Bucket, Key=Key, Body=Body, Metadata=Metadata
            ),
            lambda: self._host_s3.put_object(
                Bucket=Bucket, Key=Key, Body=Body, Metadata=Metadata
            ),
            bucket=Bucket,
            operation=f"upload s3://{Bucket}/{Key}",
        )

    def copy_object(
        self,
        *,
        Bucket: str,
        Key: str,
        CopySource: dict[str, str],
        MetadataDirective: str,
    ):
        source_bucket = CopySource["Bucket"]
        source_key = CopySource["Key"]
        return self._run(
            lambda: self._scoped_s3.copy_object(
                Bucket=Bucket,
                Key=Key,
                CopySource=CopySource,
                MetadataDirective=MetadataDirective,
            ),
            lambda: self._host_s3.copy_object(
                Bucket=Bucket,
                Key=Key,
                CopySource=CopySource,
                MetadataDirective=MetadataDirective,
            ),
            bucket=source_bucket,
            operation=f"copy s3://{source_bucket}/{source_key} to s3://{Bucket}/{Key}",
        )

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, ContinuationToken: str | None = None
    ):
        kwargs = {"Bucket": Bucket, "Prefix": Prefix}
        if ContinuationToken:
            kwargs["ContinuationToken"] = ContinuationToken
        return self._run(
            lambda: self._scoped_s3.list_objects_v2(**kwargs),
            lambda: self._host_s3.list_objects_v2(**kwargs),
            bucket=Bucket,
            operation=f"list s3://{Bucket}/{Prefix}",
        )

    def _run(self, scoped_operation, host_fallback, *, bucket: str, operation: str):
        return run_with_host_credential_fallback(
            scoped_operation,
            host_fallback,
            bucket=bucket,
            operation=operation,
            allow_host_creds=self._allow_host_creds,
            logger=logger,
        )


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise DemoManifestError(f"Expected s3://bucket/key URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _target_bucket_key(target_bucket: str, target_path: str) -> tuple[str, str]:
    if target_bucket.startswith("s3://"):
        bucket, prefix = _parse_s3_uri(target_bucket.rstrip("/") + "/placeholder")
        base = prefix.rsplit("/", 1)[0].strip("/")
        key = "/".join(part for part in (base, target_path.strip("/")) if part)
    else:
        bucket = target_bucket
        key = target_path.strip("/")
    if not bucket or not key:
        raise DemoManifestError(
            "--target-bucket and target_path must resolve to a bucket and key"
        )
    return bucket, key


def _ensure_prefix(key: str) -> str:
    return key if key.endswith("/") else key + "/"


def _head_matches_artifact(head, artifact: DemoArtifact) -> bool:
    if head is None:
        return False
    metadata = head.get("Metadata", {}) or {}
    if metadata.get("sha256") != artifact.sha256:
        return False
    if (
        artifact.size_bytes is not None
        and int(head.get("ContentLength", -1)) != artifact.size_bytes
    ):
        return False
    return True


def _head_or_none(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = client_error_code(exc)
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        _raise_scoped(exc, bucket, f"head s3://{bucket}/{key}")
    except NoCredentialsError as exc:
        _raise_scoped(exc, bucket, f"head s3://{bucket}/{key}")


def _read_object(s3, bucket: str, key: str, *, operation: str) -> bytes:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except (ClientError, NoCredentialsError) as exc:
        _raise_scoped(exc, bucket, operation)


def _put_object(
    s3,
    bucket: str,
    key: str,
    body: bytes,
    *,
    metadata: dict[str, str],
    operation: str,
) -> None:
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, Metadata=metadata)
    except (ClientError, NoCredentialsError) as exc:
        _raise_scoped(exc, bucket, operation)


def _copy_object(
    s3,
    source_bucket: str,
    source_key: str,
    dest_bucket: str,
    dest_key: str,
    *,
    operation: str,
) -> None:
    try:
        s3.copy_object(
            Bucket=dest_bucket,
            Key=dest_key,
            CopySource={"Bucket": source_bucket, "Key": source_key},
            MetadataDirective="COPY",
        )
    except (ClientError, NoCredentialsError) as exc:
        bucket = source_bucket if isinstance(exc, ClientError) else dest_bucket
        _raise_scoped(exc, bucket, operation)


def _list_objects(s3, bucket: str, prefix: str, *, operation: str) -> list[dict]:
    objects: list[dict] = []
    token = None
    while True:
        try:
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
        except (ClientError, NoCredentialsError) as exc:
            _raise_scoped(exc, bucket, operation)
        objects.extend(
            obj
            for obj in page.get("Contents", [])
            if obj.get("Key") and not str(obj.get("Key")).endswith("/")
        )
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return objects


def _prefix_stats(s3, bucket: str, prefix: str) -> tuple[int, int]:
    objects = _list_objects(
        s3, bucket, prefix, operation=f"list s3://{bucket}/{prefix}"
    )
    return len(objects), sum(int(obj.get("Size", 0)) for obj in objects)


def _raise_scoped(exc: BaseException, bucket: str, operation: str):
    if (
        isinstance(exc, ClientError)
        and client_error_code(exc) not in SCOPED_CREDENTIAL_ERROR_CODES
    ):
        raise exc
    raise ScopedCredentialError(bucket, operation) from exc
