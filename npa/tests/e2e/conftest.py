from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from npa.clients.config import list_projects, resolve_project_storage
from npa.clients.project_credentials import s3_client_for_project

E2E_BUCKET_PREFIX = "npa-e2e-test-"
E2E_BUCKET_MAX_AGE_SECONDS = 60 * 60
E2E_BUCKET_MAX_CONCURRENT = 3
E2E_BUCKET_MAX_CREATIONS = 8
E2E_BUCKET_COUNTER = Path("/tmp/npa-e2e-run-bucket-counter.txt")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip e2e tests unless NPA_INTEGRATION_E2E is set."""
    if os.getenv("NPA_INTEGRATION_E2E"):
        return

    skip_marker = pytest.mark.skip(
        reason="e2e tests require NPA_INTEGRATION_E2E=1"
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_marker)


def _bucket_counter_path() -> Path:
    return E2E_BUCKET_COUNTER


def _bucket_count() -> int:
    path = _bucket_counter_path()
    if not path.exists():
        return 0
    text = path.read_text().strip()
    return int(text) if text else 0


def _increment_bucket_counter() -> int:
    new_value = _bucket_count() + 1
    _bucket_counter_path().write_text(f"{new_value}\n")
    return new_value


def _storage_is_configured(project: str | None) -> bool:
    storage = resolve_project_storage(project)
    return bool(
        storage.endpoint_url
        and storage.aws_access_key_id
        and storage.aws_secret_access_key
    )


def _default_e2e_project() -> str | None:
    """Choose the project whose S3 credentials should back e2e test buckets."""
    if "NPA_E2E_PROJECT" in os.environ:
        value = os.environ["NPA_E2E_PROJECT"].strip()
        return value or None

    if _storage_is_configured(None):
        return None

    configured_projects = [
        project for project in list_projects() if _storage_is_configured(project)
    ]
    if "eu-north1" in configured_projects:
        return "eu-north1"
    if configured_projects:
        return configured_projects[0]
    return None


@pytest.fixture(scope="session")
def e2e_project() -> str | None:
    return _default_e2e_project()


@pytest.fixture
def e2e_test_bucket(
    request: pytest.FixtureRequest,
    e2e_project: str | None,
) -> Iterator[str]:
    """Create a real S3 test bucket and tear it down after the test."""
    yield from _test_bucket(request.node.name, e2e_project)


@pytest.fixture(scope="module")
def e2e_module_test_bucket(
    request: pytest.FixtureRequest,
    e2e_project: str | None,
) -> Iterator[str]:
    """Create a real S3 test bucket shared by tests in one module."""
    yield from _test_bucket(request.node.name, e2e_project)


def _test_bucket(test_name: str, e2e_project: str | None) -> Iterator[str]:
    if _bucket_count() >= E2E_BUCKET_MAX_CREATIONS:
        pytest.fail("E2E bucket budget exhausted (8 buckets created this run)")

    client = s3_client_for_project(e2e_project)
    _prune_concurrent_test_buckets(client)

    bucket_name = _bucket_name_for_test(test_name)
    try:
        client.create_bucket(Bucket=bucket_name)
    except Exception:
        time.sleep(30)
        client.create_bucket(Bucket=bucket_name)
    _increment_bucket_counter()

    try:
        yield bucket_name
    finally:
        _force_delete_bucket(client, bucket_name)


def _bucket_name_for_test(test_name: str) -> str:
    purpose = test_name
    if purpose.startswith("test_"):
        purpose = purpose.removeprefix("test_")
    purpose = re.sub(r"[^a-z0-9-]+", "-", purpose.lower().replace("_", "-"))
    timestamp = time.strftime("%Y%m%dt%H%M%Sz", time.gmtime())
    max_purpose_length = 63 - len(E2E_BUCKET_PREFIX) - len(timestamp) - 1
    purpose = purpose.strip("-")[:max_purpose_length].strip("-") or "case"
    return f"{E2E_BUCKET_PREFIX}{purpose}-{timestamp}"


def _list_test_buckets(client: Any) -> list[dict[str, Any]]:
    return [
        bucket
        for bucket in client.list_buckets().get("Buckets", [])
        if bucket["Name"].startswith(E2E_BUCKET_PREFIX)
    ]


def _prune_concurrent_test_buckets(client: Any) -> None:
    buckets = sorted(_list_test_buckets(client), key=lambda bucket: bucket["CreationDate"])
    now = time.time()

    for bucket in buckets:
        age_seconds = now - bucket["CreationDate"].timestamp()
        if age_seconds > E2E_BUCKET_MAX_AGE_SECONDS:
            _force_delete_bucket(client, bucket["Name"])

    buckets = sorted(_list_test_buckets(client), key=lambda bucket: bucket["CreationDate"])
    while len(buckets) >= E2E_BUCKET_MAX_CONCURRENT:
        oldest = buckets.pop(0)
        _force_delete_bucket(client, oldest["Name"])


def _force_delete_bucket(
    client: Any,
    bucket_name: str,
    retries: int = 3,
    backoff_seconds: list[int] | None = None,
) -> None:
    """Empty and delete a bucket; log final failure instead of raising."""
    delays = backoff_seconds or [5, 15, 45]
    for attempt in range(retries):
        try:
            _empty_bucket(client, bucket_name)
            client.delete_bucket(Bucket=bucket_name)
            return
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(delays[attempt])
                continue
            print(f"WARNING: NOVEL_ISSUE_TEARDOWN_FAILED_{bucket_name}: {exc}")


def _empty_bucket(client: Any, bucket_name: str) -> None:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})


@pytest.fixture
def s3_helper(e2e_project: str | None) -> "S3Helper":
    """Provide helpers for verifying real S3 state in e2e tests."""
    return S3Helper(s3_client_for_project(e2e_project))


class S3Helper:
    """Helpers for verifying real S3 state outside the CLI code path."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def head_object(self, bucket: str, key: str) -> dict[str, Any] | None:
        """Return object headers and metadata, or None if the object is missing."""
        try:
            return self.client.head_object(Bucket=bucket, Key=key)
        except self.client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None
            raise

    def get_sha256_metadata(self, bucket: str, key: str) -> str | None:
        """Return x-amz-meta-sha256 metadata, or None if missing."""
        response = self.head_object(bucket, key)
        if response is None:
            return None
        metadata = response.get("Metadata", {})
        for metadata_key, value in metadata.items():
            if metadata_key.lower() == "sha256":
                return value
        return None

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return object keys in bucket under the given prefix."""
        return [obj["Key"] for obj in self.list_object_summaries(bucket, prefix)]

    def list_object_summaries(
        self,
        bucket: str,
        prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Return object summary mappings in bucket under the given prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects.extend(page.get("Contents", []))
        return objects

    def count_objects(self, bucket: str, prefix: str = "") -> int:
        return len(self.list_objects(bucket, prefix))
