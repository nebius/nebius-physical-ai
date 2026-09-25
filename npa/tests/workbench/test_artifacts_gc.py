"""Unit tests for the thin S3 artifact garbage collector.

Covers the safety contract from issue #525: dry-run is the default and never
writes; only provably terminal, expired, unpinned runs are deleted; every
ambiguous case (missing/unreadable manifest, live or unknown status, unknown
age, pin marker) is kept; apply mode re-verifies liveness and pins before
deleting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError

from npa.workbench.artifacts_gc import (
    DEFAULT_RETENTION_DAYS,
    MANIFEST_SUFFIX,
    PIN_MARKER,
    GcError,
    RetentionPolicy,
    RunInfo,
    apply_plan,
    decide_run,
    delete_run_prefix,
    describe_run,
    discover_run_prefixes,
    is_terminal_status,
    parse_manifest_status,
    plan_gc,
    plan_summary,
    run_age_days,
)

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def _missing(operation: str = "HeadObject") -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "missing"}}, operation)


class FakePaginator:
    def __init__(self, fake: "FakeS3", operation: str) -> None:
        self._fake = fake
        self._operation = operation

    def paginate(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        yield self._fake._paginate(self._operation, **kwargs)


class FakeS3:
    """Minimal in-memory boto3 surface: head/get, paginated list, delete."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.delete_calls: list[list[str]] = []

    def add(
        self,
        bucket: str,
        key: str,
        body: bytes = b"x",
        last_modified: datetime | None = None,
    ) -> None:
        self.objects[(bucket, key)] = {
            "Body": body,
            "Size": len(body),
            "LastModified": last_modified or NOW,
        }

    def add_manifest(
        self,
        bucket: str,
        run_prefix: str,
        status: str,
        updated_at: datetime,
    ) -> None:
        import json

        self.add(
            bucket,
            f"{run_prefix}/{MANIFEST_SUFFIX}",
            json.dumps(
                {"status": status, "updated_at": updated_at.isoformat()}
            ).encode(),
        )

    # -- boto3 surface ----------------------------------------------------
    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise _missing("HeadObject")
        return {"ContentLength": item["Size"]}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise _missing("GetObject")
        return {"Body": BytesIO(item["Body"])}

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self, name)

    def delete_objects(self, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        keys = [entry["Key"] for entry in Delete["Objects"]]
        self.delete_calls.append(list(keys))
        deleted = []
        for key in keys:
            if (Bucket, key) in self.objects:
                del self.objects[(Bucket, key)]
                deleted.append({"Key": key})
        return {"Deleted": deleted, "Errors": []}

    # -- paginator backend -------------------------------------------------
    def _paginate(self, operation: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        bucket = kwargs["Bucket"]
        if operation == "list_objects_v2" and "Delimiter" in kwargs:
            prefix = kwargs.get("Prefix", "")
            seen: set[str] = set()
            for bkt, key in self.objects:
                if bkt != bucket or not key.startswith(prefix):
                    continue
                rest = key[len(prefix) :]
                if "/" in rest:
                    seen.add(prefix + rest.split("/", 1)[0] + "/")
            return {"CommonPrefixes": [{"Prefix": p} for p in sorted(seen)]}
        prefix = kwargs.get("Prefix", "")
        contents = [
            {"Key": key, "Size": item["Size"], "LastModified": item["LastModified"]}
            for (bkt, key), item in sorted(self.objects.items())
            if bkt == bucket and key.startswith(prefix)
        ]
        return {"Contents": contents}


def _run(
    prefix: str = "runs/r1",
    status: str | None = "SUCCEEDED",
    age_days: float | None = 100.0,
    pinned: bool = False,
    size_bytes: int = 10,
    object_count: int = 2,
) -> RunInfo:
    updated_at = NOW - timedelta(days=age_days) if age_days is not None else None
    newest = NOW - timedelta(days=age_days) if age_days is not None else None
    return RunInfo(
        prefix=prefix,
        manifest_status=status,
        updated_at=updated_at,
        newest_object_mtime=newest,
        size_bytes=size_bytes,
        object_count=object_count,
        pinned=pinned,
    )


POLICY = RetentionPolicy(retention_days=90)


# -- status helpers --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("SUCCEEDED", True),
        ("succeeded", True),
        ("FAILED", True),
        ("failed_quota", True),
        ("FAILED_STARTUP", True),
        ("CANCELLED", True),
        ("cancelled", True),
        ("RUNNING", False),
        ("PLANNED", False),
        ("SUBMITTED", False),
        ("", False),
        (None, False),
    ],
)
def test_is_terminal_status(status: object, expected: bool) -> None:
    assert is_terminal_status(status) is expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b'{"status": "SUCCEEDED"}', "SUCCEEDED"),
        (b'{"status": "  "}', None),
        (b'{"other": 1}', None),
        (b"not json", None),
        (b"[1, 2]", None),
    ],
)
def test_parse_manifest_status(payload: bytes, expected: str | None) -> None:
    assert parse_manifest_status(payload) == expected


# -- decide_run: the fail-closed contract ----------------------------------


def test_live_run_is_kept() -> None:
    decision = decide_run(_run(status="RUNNING"), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "live")


def test_missing_manifest_is_kept() -> None:
    decision = decide_run(_run(status=None), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "no-manifest")


def test_unknown_status_is_kept() -> None:
    decision = decide_run(_run(status="MYSTERIOUS"), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "live")


def test_pinned_terminal_run_is_kept() -> None:
    decision = decide_run(_run(status="SUCCEEDED", pinned=True), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "pinned")


def test_young_terminal_run_is_kept() -> None:
    decision = decide_run(_run(status="SUCCEEDED", age_days=10.0), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "young")


def test_unknown_age_is_kept() -> None:
    decision = decide_run(_run(status="SUCCEEDED", age_days=None), POLICY, NOW)
    assert (decision.delete, decision.reason) == (False, "unknown-age")


def test_expired_terminal_run_is_deleted() -> None:
    decision = decide_run(_run(status="FAILED", age_days=100.0), POLICY, NOW)
    assert (decision.delete, decision.reason) == (True, "expired")


def test_retention_boundary_is_exclusive() -> None:
    just_under = decide_run(_run(age_days=89.9), POLICY, NOW)
    just_over = decide_run(_run(age_days=90.1), POLICY, NOW)
    assert just_under.delete is False
    assert just_over.delete is True


def test_plan_summary_aggregates() -> None:
    decisions = plan_gc(
        [
            _run("runs/old", status="SUCCEEDED", age_days=200.0, size_bytes=100),
            _run("runs/live", status="RUNNING", age_days=200.0),
            _run("runs/young", status="SUCCEEDED", age_days=5.0),
        ],
        POLICY,
        NOW,
    )
    summary = plan_summary(decisions)
    assert summary["runs_scanned"] == 3
    assert summary["runs_to_delete"] == 1
    assert summary["bytes_to_delete"] == 100
    assert summary["kept_by_reason"] == {"live": 1, "young": 1}


def test_run_age_prefers_manifest_then_newest_object() -> None:
    manifest_anchored = _run(age_days=50.0)
    manifest_anchored.newest_object_mtime = NOW - timedelta(days=5)
    assert run_age_days(manifest_anchored, NOW) == pytest.approx(50.0)

    manifest_anchored.updated_at = None
    assert run_age_days(manifest_anchored, NOW) == pytest.approx(5.0)


# -- S3 discovery / description --------------------------------------------


def test_discover_run_prefixes_finds_manifests() -> None:
    s3 = FakeS3()
    s3.add_manifest("bkt", "wf/run-a", "SUCCEEDED", NOW - timedelta(days=100))
    s3.add_manifest("bkt", "wf/nested/run-b", "FAILED", NOW - timedelta(days=100))
    s3.add("bkt", "wf/orphan/data.bin")  # no manifest: invisible to the collector

    assert discover_run_prefixes(s3, "bkt", "wf") == ["wf/nested/run-b", "wf/run-a"]


def test_discover_run_prefixes_accepts_a_direct_run_prefix() -> None:
    s3 = FakeS3()
    s3.add_manifest("bkt", "wf/run-a", "SUCCEEDED", NOW - timedelta(days=100))
    assert discover_run_prefixes(s3, "bkt", "wf/run-a") == ["wf/run-a"]


def test_describe_run_missing_manifest_fails_closed() -> None:
    s3 = FakeS3()
    s3.add("bkt", "wf/run-a/data.bin", b"data")
    info = describe_run(s3, "bkt", "wf/run-a")
    assert info.manifest_status is None
    assert info.object_count == 1
    assert decide_run(info, POLICY, NOW).delete is False


def test_describe_run_reads_pin_marker() -> None:
    s3 = FakeS3()
    s3.add_manifest("bkt", "wf/run-a", "SUCCEEDED", NOW - timedelta(days=200))
    s3.add("bkt", f"wf/run-a/{PIN_MARKER}", b"keep this")
    info = describe_run(s3, "bkt", "wf/run-a")
    assert info.pinned is True
    assert decide_run(info, POLICY, NOW).reason == "pinned"


# -- dry-run vs apply -------------------------------------------------------


def _seed_bucket(s3: FakeS3) -> None:
    s3.add_manifest("bkt", "wf/old-ok", "SUCCEEDED", NOW - timedelta(days=200))
    s3.add("bkt", "wf/old-ok/ckpt.bin", b"0" * 16)
    s3.add_manifest("bkt", "wf/live", "RUNNING", NOW - timedelta(days=200))
    s3.add("bkt", "wf/live/ckpt.bin", b"0" * 16)


def test_dry_run_plan_never_writes() -> None:
    from npa.cli.workbench.artifacts_gc import build_gc_plan

    s3 = FakeS3()
    _seed_bucket(s3)
    decisions, _ = build_gc_plan(s3, "bkt", "wf", POLICY, max_depth=3)
    assert [d.run.prefix for d in decisions if d.delete] == ["wf/old-ok"]
    assert s3.delete_calls == []


def test_apply_deletes_only_expired_terminal_runs() -> None:
    from npa.cli.workbench.artifacts_gc import build_gc_plan

    s3 = FakeS3()
    _seed_bucket(s3)
    decisions, _ = build_gc_plan(s3, "bkt", "wf", POLICY, max_depth=3)
    result = apply_plan(s3, "bkt", decisions)
    assert result.deleted_runs == ["wf/old-ok"]
    assert result.deleted_objects == 2  # manifest + checkpoint
    assert ("bkt", "wf/live/ckpt.bin") in s3.objects  # live run untouched
    assert s3.delete_calls, "apply must issue real deletes"


def test_apply_rechecks_liveness_before_deleting() -> None:
    from npa.cli.workbench.artifacts_gc import build_gc_plan

    s3 = FakeS3()
    _seed_bucket(s3)
    decisions, _ = build_gc_plan(s3, "bkt", "wf", POLICY, max_depth=3)
    # The run restarts between planning and applying.
    s3.add_manifest("bkt", "wf/old-ok", "RUNNING", NOW - timedelta(days=200))
    result = apply_plan(s3, "bkt", decisions)
    assert result.deleted_runs == []
    assert result.skipped_runs == [("wf/old-ok", "became-live")]
    assert ("bkt", "wf/old-ok/ckpt.bin") in s3.objects


@pytest.mark.parametrize("pin_marker", [PIN_MARKER, ".custom-retain"])
def test_apply_rechecks_configured_pin_before_deleting(pin_marker: str) -> None:
    from npa.cli.workbench.artifacts_gc import build_gc_plan

    s3 = FakeS3()
    _seed_bucket(s3)
    s3.add("bkt", "wf/unrelated/data.bin", b"unrelated")
    policy = RetentionPolicy(retention_days=90, pin_marker=pin_marker)
    decisions, _ = build_gc_plan(s3, "bkt", "wf", policy, max_depth=3)

    # The run is pinned after planning but before apply.
    s3.add("bkt", f"wf/old-ok/{pin_marker}", b"retain")
    result = apply_plan(s3, "bkt", decisions, pin_marker)

    assert result.deleted_runs == []
    assert result.skipped_runs == [("wf/old-ok", "pinned")]
    assert ("bkt", "wf/old-ok/ckpt.bin") in s3.objects
    assert ("bkt", f"wf/old-ok/{pin_marker}") in s3.objects
    assert ("bkt", "wf/unrelated/data.bin") in s3.objects
    assert s3.delete_calls == []


@pytest.mark.parametrize("error_code", ["AccessDenied", "NoSuchBucket"])
def test_apply_pin_recheck_failure_prevents_deletion(error_code: str) -> None:
    from npa.cli.workbench.artifacts_gc import build_gc_plan

    s3 = FakeS3()
    _seed_bucket(s3)
    decisions, _ = build_gc_plan(s3, "bkt", "wf", POLICY, max_depth=3)
    original_head_object = s3.head_object

    def deny_pin_check(Bucket: str, Key: str) -> dict[str, Any]:
        if Key == f"wf/old-ok/{PIN_MARKER}":
            raise ClientError(
                {"Error": {"Code": error_code, "Message": "pin check failed"}},
                "HeadObject",
            )
        return original_head_object(Bucket=Bucket, Key=Key)

    s3.head_object = deny_pin_check  # type: ignore[method-assign]
    with pytest.raises(ClientError) as error:
        apply_plan(s3, "bkt", decisions)

    assert error.value.response["Error"]["Code"] == error_code
    assert ("bkt", "wf/old-ok/ckpt.bin") in s3.objects
    assert s3.delete_calls == []


def test_delete_refuses_keys_outside_the_prefix() -> None:
    s3 = FakeS3()
    s3.add("bkt", "wf/run-a/a.bin", b"a")
    s3.objects[("bkt", "elsewhere/evil.bin")] = {
        "Body": b"e",
        "Size": 1,
        "LastModified": NOW,
    }
    # Corrupt the listing order so the foreign key appears under the prefix
    # scan: the guard must still refuse.
    original = s3._paginate

    def hostile(operation: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        page = original(operation, **kwargs)
        if operation == "list_objects_v2" and "Delimiter" not in kwargs:
            page["Contents"].append(
                {"Key": "elsewhere/evil.bin", "Size": 1, "LastModified": NOW}
            )
        return page

    s3._paginate = hostile  # type: ignore[method-assign]
    with pytest.raises(GcError):
        delete_run_prefix(s3, "bkt", "wf/run-a")
    assert ("bkt", "elsewhere/evil.bin") in s3.objects


def test_retention_policy_default_matches_documented_window() -> None:
    assert DEFAULT_RETENTION_DAYS == 90
    assert RetentionPolicy().retention_days == DEFAULT_RETENTION_DAYS
