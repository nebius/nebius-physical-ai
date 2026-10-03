"""Real project S3 pagination and trigger recovery, without launching a GPU job.

Opt in with NPA_INTEGRATION_E2E=1 and NPA_E2E_PROJECT=<configured project>.
Creates 1,006 tiny objects in one unique prefix, observes the native paginator,
and verifies exact discovery and an idle second poll. The launcher is a local
recorder: this proves storage discovery and watermark behavior, not training.
All fixture versions and delete markers are removed and absence is verified.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pytest

from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import s3_client_for_project
from npa.workflows.sim_to_real_trigger import (
    LocalWatermarkStore,
    PipelineLaunch,
    TriggerConfig,
    TriggerObject,
    run_once,
)

from .s3_fixture_cleanup import delete_owned_prefix

pytestmark = pytest.mark.e2e


@pytest.fixture
def owned_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Any, TriggerConfig]]:
    project = os.environ.get("NPA_E2E_PROJECT", "").strip()
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not project:
        pytest.skip("explicit live opt-in and exact project are required")
    for name in ("AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    storage = resolve_project_storage(project)
    assert all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    )
    parsed = urlparse(storage.checkpoint_bucket)
    bucket = parsed.netloc if parsed.scheme == "s3" else storage.checkpoint_bucket
    parent = parsed.path.strip("/") if parsed.scheme == "s3" else ""
    prefix = (
        "/".join(filter(None, (parent, f"npa-trigger-e2e-{uuid.uuid4().hex}"))) + "/"
    )
    client = s3_client_for_project(project, allow_host_creds=False)
    config = TriggerConfig(
        s3_endpoint=storage.endpoint_url,
        s3_bucket=bucket,
        s3_prefix=prefix,
        pipeline_s3_prefix="unused/{run_id}",
    )
    try:
        yield client, config
    finally:
        delete_owned_prefix(client, bucket, prefix)


class RecordingLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[TriggerObject, ...]] = []

    def launch(
        self, config: TriggerConfig, objects: tuple[TriggerObject, ...]
    ) -> PipelineLaunch:
        self.calls.append(objects)
        return PipelineLaunch(
            run_id="pagination-proof",
            status="recorded",
            input_data_uri=config.input_data_uri,
        )


def test_native_pagination_discovers_every_object_before_one_trigger(
    owned_storage: tuple[Any, TriggerConfig], tmp_path: Path, record_property: Any
) -> None:
    client, config = owned_storage
    expected = {
        f"{config.s3_prefix}data/chunk-000/episode_{index:06d}.parquet"
        for index in range(1005)
    }
    keys = sorted(expected | {f"{config.s3_prefix}notes.txt"})

    def put(key: str) -> None:
        client.put_object(Bucket=config.s3_bucket, Key=key, Body=b"pagination fixture")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(put, keys))
    pages: list[tuple[int, object, bool]] = []

    def observe(parsed: dict[str, Any], **_kwargs: Any) -> None:
        pages.append(
            (
                len(parsed.get("Contents", [])),
                parsed.get("IsTruncated"),
                bool(parsed.get("NextContinuationToken")),
            )
        )

    client.meta.events.register("after-call.s3.ListObjectsV2", observe)
    launcher = RecordingLauncher()
    store = LocalWatermarkStore(tmp_path / "watermark.json")
    first = run_once(config, s3_client=client, watermark_store=store, launcher=launcher)
    assert first.status == "triggered" and first.new_object_count == len(expected)
    assert len(launcher.calls) == 1
    assert {obj.key for obj in launcher.calls[0]} == expected
    assert list(launcher.calls[0]) == sorted(
        launcher.calls[0], key=lambda obj: (obj.last_modified, obj.key)
    )
    assert len(pages) >= 2 and sum(page[0] for page in pages) == len(keys)
    assert all(type(page[1]) is bool for page in pages)
    assert all(page[1] is True and page[2] for page in pages[:-1])
    assert pages[-1][1] is False
    record_property("provider_pages", repr(pages))
    second = run_once(
        config, s3_client=client, watermark_store=store, launcher=launcher
    )
    assert second.status == "idle" and second.new_object_count == 0
    assert len(launcher.calls) == 1
    client.meta.events.unregister("after-call.s3.ListObjectsV2", observe)
