"""Barrier/join stage for ``parallel:`` fan-out workflows.

A ``parallel:`` group's downstream state is a *barrier*: it must only run once
every member finished. :func:`join_shards` makes that contract observable — it
reads each shard's manifest from S3, fails loudly when a shard is missing, and
writes a merged report with per-shard counts.

Used by ``npa/workflows/workbench/npa-workflows/token-factory-parallel-fanout.yaml``
via ``run.shell`` (``npa`` is pip-installed in the rendered task).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

JOIN_REPORT_SCHEMA = "npa.fanout.join_report.v1"
JOIN_REPORT_FILENAME = "join_report.json"


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _list_keys(uri: str) -> list[str]:
    bucket, prefix = _split(uri if uri.endswith("/") else uri + "/")
    s3 = _storage().s3
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in page.get("Contents", []) if item.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _download_json(uri: str) -> dict[str, Any]:
    if not uri.startswith("s3://"):
        return json.loads(Path(uri).read_text(encoding="utf-8"))
    want = uri.rstrip("/").split("/")[-1]
    with tempfile.TemporaryDirectory(prefix="npa-fanout-join-") as tmp:
        local = Path(_storage().download_path(uri, tmp))
        if local.is_dir():
            matches = sorted(local.rglob(want))
            if not matches:
                raise FileNotFoundError(uri)
            local = matches[0]
        return json.loads(local.read_text(encoding="utf-8"))


def _upload_json(payload: dict[str, Any], uri: str) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not uri.startswith("s3://"):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        Path(uri).write_text(body, encoding="utf-8")
        return uri
    with tempfile.TemporaryDirectory(prefix="npa-fanout-join-") as tmp:
        path = Path(tmp) / "out.json"
        path.write_text(body, encoding="utf-8")
        return _storage().upload_file(str(path), uri)


def _shard_names(shards: str | Iterable[str]) -> list[str]:
    if isinstance(shards, str):
        return [part.strip() for part in shards.split(",") if part.strip()]
    return [str(part).strip() for part in shards if str(part).strip()]


def _count_items(payload: Any, items_key: str) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        value = payload.get(items_key)
        if isinstance(value, list):
            return len(value)
        for candidate in ("captions", "records", "items", "results"):
            if isinstance(payload.get(candidate), list):
                return len(payload[candidate])
    return 0


def join_shards(
    *,
    shards_uri: str,
    report_uri: str,
    shards: str | Sequence[str] = "",
    manifest: str = "captions.json",
    items_key: str = "captions",
    run_id: str = "",
) -> dict[str, Any]:
    """Merge every fan-out shard's manifest into one report (the barrier stage).

    ``shards`` is the comma-separated list of expected shard directories under
    ``shards_uri``. A missing shard raises: reaching this stage means the runtime
    barrier said every parallel predecessor succeeded, so a missing artifact is a
    real failure, not something to paper over.
    """

    base = shards_uri.rstrip("/") + "/"
    expected = _shard_names(shards)
    if not expected:
        keys = _list_keys(base)
        _, prefix = _split(base)
        expected = sorted(
            {
                key[len(prefix) :].split("/", 1)[0]
                for key in keys
                if key.startswith(prefix) and "/" in key[len(prefix) :]
            }
        )

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    total = 0
    for shard in expected:
        uri = f"{base}{shard}/{manifest}"
        try:
            payload = _download_json(uri)
        except Exception as exc:  # noqa: BLE001 - reported below as a hard failure
            missing.append(shard)
            entries.append({"shard": shard, "uri": uri, "status": "missing", "error": str(exc)[:200]})
            continue
        count = _count_items(payload, items_key)
        total += count
        entries.append(
            {
                "shard": shard,
                "uri": uri,
                "status": "ok",
                "items": count,
                "model": str(payload.get("model") or "") if isinstance(payload, dict) else "",
            }
        )

    report = {
        "schema": JOIN_REPORT_SCHEMA,
        "run_id": run_id,
        "shards_uri": base,
        "manifest": manifest,
        "shard_count": len(expected),
        "joined_shards": len(expected) - len(missing),
        "missing_shards": missing,
        "total_items": total,
        "shards": entries,
    }
    target = (
        report_uri
        if report_uri.endswith(".json")
        else f"{report_uri.rstrip('/')}/{JOIN_REPORT_FILENAME}"
    )
    report["report_uri"] = _upload_json(report, target)
    print(json.dumps(report))
    if missing:
        raise RuntimeError(
            f"fan-out join incomplete: {len(missing)} shard(s) missing {manifest}: {missing}"
        )
    return report


__all__ = ["JOIN_REPORT_FILENAME", "JOIN_REPORT_SCHEMA", "join_shards"]
