"""Live Nebius S3 e2e for the insights storage-layer fix.

Proves, against real object storage rather than a mock, that:

* ``ingest-run`` pointed at one run prefix (``run-1``) never pulls in a
  lexically-sibling run's artifacts (``run-10``) — the sibling-prefix mixing
  bug in ``list_json_uris``.
* A normal record -> query round trip still works through the append-shard
  layout (``uri_exists`` / ``read_bytes_uri`` fixes did not break real reads).
* Querying a store prefix that was never written returns an empty result
  rather than an error — genuine absence is preserved, not just the failure
  path.

No infrastructure is provisioned by this test. It resolves only the exact,
explicitly named project's already-configured storage credentials — never an
auto-discovered project, never a host-credential fallback — and touches only
a single unique, test-owned prefix that it creates and deletes itself;
nothing outside that prefix is read, written, or deleted. Deletion is
positively verified (inspected ``DeleteObjects`` errors, then follow-up
current-object and version listings) rather than inferred from an HTTP 200.
The selected credential must permit listing and deleting object versions.

Run (env contract):

    NPA_INTEGRATION_E2E=1 \\
    NPA_E2E_PROJECT=<exact configured project alias> \\
    npa/.venv/bin/python -m pytest npa/tests/e2e/test_insights_storage_live_e2e.py -q

Both variables are required; the test skips before touching any cloud
resource if either is unset. ``NPA_E2E_PROJECT`` must name a project with its
own configured object-storage credentials (``~/.npa/credentials.yaml`` /
``~/.npa/config.yaml`` or that project's env-var overrides) — this test never
falls back to ambient host credentials, so it cannot write into an unrelated
configured project by accident. No bucket name or prefix needs to be supplied
directly: the bucket comes from the named project's configured checkpoint
storage, and the prefix is generated fresh per run. This test never prints or
persists the resolved access key/secret; only the derived bucket name and the
prefix it creates appear in assertions/output.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Iterator
from urllib.parse import urlparse

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import (
    s3_client_for_project,
    storage_env_for_project,
)

pytestmark = pytest.mark.e2e

# Ambient session-scoped credentials (e.g. from an assumed role in the shell
# a developer is already using) must never mix with the named project's
# static access-key/secret pair: a stale session token present alongside a
# different static key pair can cause boto3 to sign requests inconsistently
# or reach an unrelated account entirely.
_AMBIENT_SESSION_ENV = ("AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN")


def _required_project() -> str:
    """The exact project to test against; never an auto-discovered default.

    Auto-discovery (the shared ``e2e_project`` fixture's default behavior)
    picks whichever configured project happens to have writable storage,
    which for this test would mean writing fixtures into a project the
    caller did not explicitly choose. Requiring an explicit, nonempty value
    here — checked before any fixture touches S3 — makes that impossible.
    """
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if not project:
        pytest.skip("NPA_E2E_PROJECT must name the exact project to test against")
    return project


def _require_storage(project: str):
    storage = resolve_project_storage(project)
    if not (
        storage.checkpoint_bucket
        and storage.endpoint_url
        and storage.aws_access_key_id
        and storage.aws_secret_access_key
    ):
        pytest.fail(
            f"project {project!r} has no writable, fully-configured object-storage "
            "credentials of its own (this test never falls back to host credentials)"
        )
    return storage


@pytest.fixture
def live_project() -> str:
    return _required_project()


@pytest.fixture(autouse=True)
def insights_embedded_s3_env(
    live_project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the embedded SDK's own ``boto3`` client at the named project's storage.

    The embedded (non-``--service``) path used throughout this file resolves
    its S3 client from process environment only (``AWS_ENDPOINT_URL`` /
    ``NEBIUS_S3_ENDPOINT`` plus boto3's default credential-chain env vars), so
    the project's own credentials must be exported for the duration of the
    test. ``allow_host_creds=False`` means this raises rather than silently
    falling back to ambient host credentials if the project has none of its
    own. This never prints or persists the resolved values.
    """
    _require_storage(live_project)
    for token_var in _AMBIENT_SESSION_ENV:
        monkeypatch.delenv(token_var, raising=False)
    for key, value in storage_env_for_project(
        live_project, allow_host_creds=False
    ).items():
        if value:
            monkeypatch.setenv(key, value)


@pytest.fixture
def insights_bucket_and_prefix(live_project: str) -> tuple[str, str]:
    """Resolve the named project's bucket, deriving a unique sub-prefix under it.

    Uses only the project's already-configured checkpoint bucket; no separate
    bucket/prefix environment variable is required.
    """
    storage = _require_storage(live_project)
    parsed = urlparse(storage.checkpoint_bucket)
    bucket = parsed.netloc if parsed.scheme == "s3" else storage.checkpoint_bucket
    base_prefix = parsed.path.strip("/") if parsed.scheme == "s3" else ""
    owned = f"npa-insights-storage-e2e-{uuid.uuid4().hex[:12]}"
    prefix = "/".join(part for part in (base_prefix, owned) if part)
    return bucket, prefix


@pytest.fixture
def insights_s3(live_project: str):
    return s3_client_for_project(live_project, allow_host_creds=False)


def _list_prefix_versions(s3: Any, bucket: str, prefix: str) -> list[dict[str, str]]:
    """Inventory exact versions, including delete markers, under an owned prefix."""
    paginator = s3.get_paginator("list_object_versions")
    versions: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in [*page.get("Versions", []), *page.get("DeleteMarkers", [])]:
            assert str(item["Key"]).startswith(prefix)
            versions.append({"Key": item["Key"], "VersionId": item["VersionId"]})
    return versions


def _delete_prefix_and_verify(s3: Any, bucket: str, prefix: str) -> None:
    """Delete every object under ``prefix`` and positively confirm it is gone.

    A ``DeleteObjects`` call can return HTTP 200 while its body still lists
    per-key ``Errors`` (a key that was locked, throttled, or otherwise not
    actually removed), and a 200 status alone never proves every key was
    deleted. Versioned buckets also retain data after a key-only deletion.
    Delete exact versions and markers, then verify both inventories are empty.
    """
    versions = _list_prefix_versions(s3, bucket, prefix)
    errors: list[dict[str, Any]] = []
    for start in range(0, len(versions), 1000):
        batch = versions[start : start + 1000]
        if not batch:
            continue
        response = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": batch},
        )
        errors.extend(response.get("Errors") or [])
    assert not errors, (
        f"failed to delete {len(errors)} object(s) under {prefix}: {errors}"
    )

    remaining = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    assert not remaining.get("Contents"), (
        f"objects remain under owned prefix {prefix} after cleanup delete"
    )
    assert not _list_prefix_versions(s3, bucket, prefix), (
        "object versions or delete markers remain after fixture cleanup"
    )


@pytest.fixture
def owned_prefix_cleanup(
    insights_s3: Any, insights_bucket_and_prefix: tuple[str, str]
) -> Iterator[None]:
    """Delete every object under the test-owned prefix, and nothing else."""
    bucket, prefix = insights_bucket_and_prefix
    try:
        yield
    finally:
        _delete_prefix_and_verify(insights_s3, bucket, prefix + "/")


def _put_json(s3: Any, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
    )


def test_cleanup_removes_overwritten_objects_and_delete_markers(
    insights_s3: Any,
    insights_bucket_and_prefix: tuple[str, str],
    owned_prefix_cleanup: None,
) -> None:
    """Exercise cleanup after overwrites and key deletion on real object storage."""
    bucket, prefix = insights_bucket_and_prefix
    key = f"{prefix}/cleanup/probe.json"
    _put_json(insights_s3, bucket, key, {"generation": 1})
    _put_json(insights_s3, bucket, key, {"generation": 2})
    before_delete = _list_prefix_versions(insights_s3, bucket, prefix + "/")
    assert before_delete
    insights_s3.delete_object(Bucket=bucket, Key=key)

    _delete_prefix_and_verify(insights_s3, bucket, prefix + "/")

    assert not _list_prefix_versions(insights_s3, bucket, prefix + "/")


def test_ingest_run_excludes_sibling_run_prefix_on_real_s3(
    insights_s3: Any,
    insights_bucket_and_prefix: tuple[str, str],
    owned_prefix_cleanup: None,
) -> None:
    """``run-1`` must not pull in ``run-10`` on real S3 lexical listing."""
    from npa.sdk.workbench import insights as sdk

    bucket, prefix = insights_bucket_and_prefix
    _put_json(
        insights_s3,
        bucket,
        f"{prefix}/run-1/manifest.json",
        {
            "schema": "npa.dataset.manifest.v1",
            "dataset_id": "keep",
            "version": "v1",
            "record_count": 5,
        },
    )
    _put_json(
        insights_s3,
        bucket,
        f"{prefix}/run-10/manifest.json",
        {
            "schema": "npa.dataset.manifest.v1",
            "dataset_id": "sibling",
            "version": "v1",
            "record_count": 500,
        },
    )

    store_uri = f"s3://{bucket}/{prefix}/store"
    response = sdk.ingest_run(
        input_uri=f"s3://{bucket}/{prefix}/run-1",
        output_uri=store_uri,
    )

    assert response.scanned == 1
    assert [a.uri for a in response.ingested] == [
        f"s3://{bucket}/{prefix}/run-1/manifest.json"
    ]

    queried = sdk.query(input_uri=store_uri, limit=100)
    versions = {row["artifact_version"] for row in queried.records}
    assert versions == {"keep@v1"}


def test_record_and_query_round_trip_on_real_s3(
    insights_bucket_and_prefix: tuple[str, str],
    owned_prefix_cleanup: None,
) -> None:
    """A normal write -> read round trip must still work post-fix."""
    from npa.sdk.workbench import insights as sdk

    bucket, prefix = insights_bucket_and_prefix
    store_uri = f"s3://{bucket}/{prefix}/roundtrip-store"

    recorded = sdk.record(
        output_uri=store_uri,
        records=[
            {
                "run_id": "live-e2e-run",
                "metric_name": "accuracy",
                "value": 0.93,
                "tool": "insights-e2e",
                "stage": "roundtrip",
            }
        ],
    )
    assert recorded.recorded_count == 1

    queried = sdk.query(input_uri=store_uri, run_id="live-e2e-run", limit=10)
    assert queried.count == 1
    assert queried.records[0]["metric_name"] == "accuracy"
    assert queried.records[0]["value"] == pytest.approx(0.93)


def test_query_against_a_never_written_store_is_empty_not_an_error(
    insights_bucket_and_prefix: tuple[str, str],
    owned_prefix_cleanup: None,
) -> None:
    """Genuine absence (never written) must still read back empty, not raise."""
    from npa.sdk.workbench import insights as sdk

    bucket, prefix = insights_bucket_and_prefix
    never_written_store = f"s3://{bucket}/{prefix}/never-written-store"

    queried = sdk.query(input_uri=never_written_store, limit=10)

    assert queried.count == 0
    assert queried.records == []
