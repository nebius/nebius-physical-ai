"""Search object storage for Studio assets using external project credentials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from typing import Any
from urllib.parse import quote, urlsplit

from botocore.exceptions import BotoCoreError, ClientError

from npa.clients.config import list_projects, resolve_project_storage
from npa.clients.project_credentials import s3_client_for_project
from npa.errors import NpaError
from npa.lifecycle_intent import json_stdout_contract
from npa.workflows.artifacts import render_hint_for_object

_STORAGE_ERRORS = (BotoCoreError, ClientError, NpaError, OSError, ValueError)
_METADATA_FIELDS = {
    "tool": ("npa-tool", "tool", "generator"),
    "tool_evidence": ("npa-tool-evidence", "tool-evidence"),
    "model": ("npa-model", "model"),
    "run_id": ("npa-run-id", "run-id"),
    "gpu": ("npa-gpu", "gpu"),
    "sha256": ("sha256", "npa-sha256"),
}


def _since(value: str) -> datetime:
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "--since requires an ISO timestamp with timezone"
        )
    return date.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search accessible S3 objects; never provision or modify storage."
    )
    projects = parser.add_mutually_exclusive_group()
    projects.add_argument(
        "--project",
        "-p",
        action="append",
        help="Configured project alias; repeat to select several.",
    )
    projects.add_argument(
        "--all-projects",
        action="store_true",
        help="Search every configured project with its own credentials.",
    )
    parser.add_argument(
        "--discover-tenant",
        action="store_true",
        help="Use Nebius inventory to discover the selected tenant's buckets.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Explicit bucket name; otherwise enumerate accessible buckets.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Exact S3 key prefix; empty searches the whole bucket.",
    )
    parser.add_argument(
        "--query", default="", help="Case-insensitive substring of the object key."
    )
    parser.add_argument(
        "--kind",
        choices=("video", "image", "rerun", "mcap", "json", "text", "download"),
    )
    parser.add_argument(
        "--since",
        type=_since,
        help="Only objects modified at or after this ISO timestamp.",
    )
    parser.add_argument(
        "--read-metadata",
        action="store_true",
        help="HEAD matched objects for declared tool/model/run/GPU metadata.",
    )
    parser.add_argument("--output-format", choices=("json", "text"), default="json")
    return parser


def _failure(error: Exception, operation: str) -> dict[str, str]:
    response = getattr(error, "response", {})
    code = response.get("Error", {}).get("Code", type(error).__name__)
    return {"operation": operation, "code": str(code)}


def _bucket_names(
    client: Any, project: str | None, requested: list[str]
) -> tuple[list[str], list[dict]]:
    if requested:
        return sorted(set(requested)), []
    names: set[str] = set()
    errors: list[dict] = []
    configured = resolve_project_storage(
        project, include_shared_credentials=False
    ).checkpoint_bucket
    if configured:
        names.add(
            urlsplit(configured).netloc
            if configured.startswith("s3://")
            else configured
        )
    try:
        pages = client.get_paginator("list_buckets").paginate()
        for page in pages:
            names.update(item["Name"] for item in page.get("Buckets", []))
    except _STORAGE_ERRORS as error:
        errors.append(_failure(error, "list_buckets"))
    return sorted(names), errors


def _matches(obj: dict, args: argparse.Namespace) -> bool:
    if args.query.casefold() not in obj["Key"].casefold():
        return False
    if args.kind and render_hint_for_object(key=obj["Key"]) != args.kind:
        return False
    return args.since is None or obj["LastModified"] >= args.since


def _metadata(client: Any, bucket: str, key: str) -> dict:
    try:
        stored = client.head_object(Bucket=bucket, Key=key).get("Metadata", {})
        metadata = {name.casefold(): value for name, value in stored.items()}
    except _STORAGE_ERRORS as error:
        return {"status": "unavailable", "error": _failure(error, "head_object")}
    declared = {}
    for field, names in _METADATA_FIELDS.items():
        value = next((metadata[name] for name in names if metadata.get(name)), None)
        if value is not None:
            declared[field] = value
    return {
        "status": "declared" if declared else "unknown",
        "basis": "s3_object_metadata",
        "declared": declared,
    }


def _artifact(
    client: Any, project: str | None, bucket: str, obj: dict, metadata: bool
) -> dict:
    key = obj["Key"]
    row = {
        "project": project,
        "endpoint": client.meta.endpoint_url,
        "bucket": bucket,
        "key": key,
        "s3_uri": f"s3://{bucket}/{quote(key, safe='/')}",
        "size_bytes": obj["Size"],
        "last_modified": obj["LastModified"].isoformat(),
        "etag": obj.get("ETag", ""),
        "render_hint": render_hint_for_object(key=key),
        "provenance": {"status": "not_read"},
    }
    if metadata:
        row["provenance"] = _metadata(client, bucket, key)
    return row


def _search_bucket(
    client: Any, project: str | None, bucket: str, args: argparse.Namespace
) -> tuple[dict, list[dict]]:
    source = {
        "project": project,
        "endpoint": client.meta.endpoint_url,
        "bucket": bucket,
        "prefix": args.prefix,
        "complete": False,
        "objects_scanned": 0,
        "matches": 0,
    }
    artifacts = []
    try:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=args.prefix
        )
        for page in pages:
            for obj in page.get("Contents", []):
                source["objects_scanned"] += 1
                if _matches(obj, args):
                    artifacts.append(
                        _artifact(client, project, bucket, obj, args.read_metadata)
                    )
        source["complete"] = True
    except _STORAGE_ERRORS as error:
        source["error"] = _failure(error, "list_objects_v2")
    source["matches"] = len(artifacts)
    return source, artifacts


def _search_project(
    project: str | None, args: argparse.Namespace
) -> tuple[dict, list[dict], list[dict]]:
    discovery = {"project": project, "complete": False, "errors": []}
    try:
        client = s3_client_for_project(project)
        buckets, errors = _bucket_names(client, project, args.bucket)
    except _STORAGE_ERRORS as error:
        discovery["errors"].append(_failure(error, "resolve_project_storage"))
        return discovery, [], []
    discovery.update(complete=not errors, errors=errors, buckets=len(buckets))
    sources, artifacts = [], []
    for bucket in buckets:
        source, found = _search_bucket(client, project, bucket, args)
        sources.append(source)
        artifacts.extend(found)
    return discovery, sources, artifacts


def _search(args: argparse.Namespace) -> dict:
    if args.discover_tenant:
        return _search_tenant(args)
    projects = sorted(list_projects()) if args.all_projects else args.project or [None]
    if not projects:
        raise ValueError(
            "No configured projects; configure storage or select the default credential context"
        )
    discoveries, sources, artifacts = [], [], []
    for project in dict.fromkeys(projects):
        discovery, selected, found = _search_project(project, args)
        discoveries.append(discovery)
        sources.extend(selected)
        artifacts.extend(found)
    complete = all(row["complete"] for row in discoveries + sources)
    return {
        "schema": "npa.studio.artifact-search/v1",
        "complete": complete,
        "scope": "explicit_buckets" if args.bucket else "credential_visible_buckets",
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "discovery": discoveries,
        "sources": sources,
        "artifacts": artifacts,
        "count": len(artifacts),
    }


def _tenant_bucket(target: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    try:
        client = s3_client_for_project(
            target["project"], endpoint_url=target["endpoint"]
        )
        source, found = _search_bucket(
            client, target["project"], target["bucket"], args
        )
    except _STORAGE_ERRORS as error:
        return {
            **target,
            "complete": False,
            "error": _failure(error, "resolve_project_storage"),
        }, []
    source["resource_project_id"] = target["resource_project_id"]
    for row in found:
        row["resource_project_id"] = target["resource_project_id"]
    return source, found


def _search_tenant(args: argparse.Namespace) -> dict:
    from npa.studio_sources import tenant_sources

    if args.all_projects or len(args.project or []) > 1:
        raise ValueError(
            "--discover-tenant selects one project context; omit --all-projects"
        )
    targets, errors = tenant_sources(args.project[0] if args.project else None)
    sources, artifacts = [], []
    for target in targets:
        if args.bucket and target["bucket"] not in args.bucket:
            continue
        source, found = _tenant_bucket(target, args)
        sources.append(source)
        artifacts.extend(found)
    missing = set(args.bucket) - {target["bucket"] for target in targets}
    errors.extend(
        {
            "operation": "locate_requested_bucket",
            "code": "NotDiscovered",
            "bucket": name,
        }
        for name in sorted(missing)
    )
    return {
        "schema": "npa.studio.artifact-search/v1",
        "scope": "selected_tenant_inventory",
        "complete": not errors and all(row["complete"] for row in sources),
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "discovery": [{"complete": not errors, "errors": errors}],
        "sources": sources,
        "artifacts": artifacts,
        "count": len(artifacts),
    }


@json_stdout_contract
def _execute(args: argparse.Namespace, *, output_format: str) -> int:
    result = _search(args)
    if output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{result['count']} objects; coverage {'complete' if result['complete'] else 'partial'}"
        )
        for row in result["artifacts"]:
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in ("project", "s3_uri", "render_hint", "provenance")
                    }
                )
            )
        for row in result["discovery"] + result["sources"]:
            if not row["complete"]:
                print(json.dumps(row))
    return 0 if result["complete"] else 1


def run_search(arguments: list[str]) -> int:
    """Search selected storage contexts and report complete or partial coverage.

    Args:
        arguments: Options following ``npa studio search``.
    Returns:
        Zero for complete listing or one for partial discovery/listing.
    Raises:
        SystemExit: Invalid command arguments or requested help.
        ValueError: No project context can be selected.
    """
    args = _parser().parse_args(arguments)
    return _execute(args, output_format=args.output_format)
