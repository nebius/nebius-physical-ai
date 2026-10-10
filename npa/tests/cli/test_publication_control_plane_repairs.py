"""CPU controls for the independent candidate review's publication findings."""

import asyncio
import hashlib
import io
import json
from types import SimpleNamespace
from dataclasses import replace

import pytest
from fastapi import HTTPException, Request
from typer.testing import CliRunner

from npa.agent_backend.publication_reader import PublicationConflict
from npa.cli import agent_stage_runtime as runtime
from npa.cli.workbench import sim2real as cli
from npa.workflows.sim2real import monitor, publication

from .test_publication_verified_cache import _VersionedStore, _response_bytes
from .test_stage_report_read_identity import (
    _BUCKET,
    _install_runtime_dependencies,
    _report_inventory,
)
from .test_stage14_twentieth_agent_controls import _import_rendered_backend
from .test_verified_viewer_downloads import _viewer_context


def test_empty_inventory_has_no_invented_publication_authority():
    result = runtime._committed_publication_artifacts(
        object(), "unit", "run", [], include_snapshot=True
    )
    assert result == ([], None, None, [], None)
    runtime._assert_legacy_publication_snapshot_still_unjournaled(
        object(), "unit", result[-1]
    )


@pytest.mark.parametrize(
    "command, service",
    [("regen", "regen_sim2real_rrd"), ("heldout-only", "rerun_heldout_eval_only")],
)
def test_cli_publication_conflicts_are_typed_and_sanitized(
    monkeypatch, command, service
):
    monkeypatch.setattr(cli, "build_config_from_env", lambda **_: object())

    def conflict(*_args, **_kwargs):
        raise PublicationConflict("sensitive provider context")

    monkeypatch.setattr(cli, service, conflict)
    result = CliRunner().invoke(cli.app, ["rerun", command, "--run-id", "fixture"])
    assert result.exit_code == 1
    assert "publication conflict" in result.output
    assert "sensitive provider context" not in result.output
    assert "Traceback" not in result.output


def test_local_only_diagnostic_does_not_claim_published_uri(monkeypatch):
    monkeypatch.setattr(cli, "build_config_from_env", lambda **_: object())
    monkeypatch.setattr(
        cli,
        "rerun_heldout_eval_only",
        lambda *_a, **_k: {"report_uri": "/fixture/report.json"},
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "rerun",
            "heldout-only",
            "--run-id",
            "fixture",
            "--no-publish",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["published_report_uri"] is None
    assert payload["published_renders_uri"] is None
    assert payload["local_report_path"] == "/fixture/report.json"


@pytest.mark.parametrize("failure", ["publishing", "malformed", "denied"])
def test_status_and_one_shot_watch_report_publication_unavailable(
    monkeypatch, failure, capsys
):
    from botocore.exceptions import ClientError
    from workflows.test_sim2real_monitor import _mock_s3_client

    _, targets, _ = _report_inventory(b"{}")
    journal = publication._journal_bytes(
        transaction_id="a" * 64,
        attempt_id="b" * 32,
        state="publishing",
        objects=targets,
    )
    if failure == "malformed":
        journal = b"invalid journal"
    client = _mock_s3_client(
        {
            "runs/run-a/reports/.sim2real-publication.json": journal.decode(),
            "runs/run-a/state/workflow_state.json": '{"status":"completed"}',
        }
    )
    if failure == "denied":
        original = client._s3.head_object.side_effect

        def head(**kwargs):
            if kwargs["Key"].endswith(".sim2real-publication.json"):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "private detail"}},
                    "HeadObject",
                )
            return original(**kwargs)

        client._s3.head_object.side_effect = head
    monkeypatch.setattr(monitor.StorageClient, "from_environment", lambda **_: client)
    monkeypatch.setattr(
        monitor,
        "load_operator_config",
        lambda: monitor.OperatorConfig("unit", "https://storage.example", "", ""),
    )
    result = monitor.watch_sim2real_status(
        "run-a", watch=False, json_output=True, s3_bucket=_BUCKET, s3_prefix="runs"
    )
    assert result["publication_state"] == (
        "publishing" if failure == "publishing" else "unavailable"
    )
    assert result["publication_error"] == (
        "storage_unavailable" if failure == "denied" else "publication_conflict"
    )
    assert result["stages"]["report"]["state"] != "SUCCEEDED"
    assert "private detail" not in capsys.readouterr().out


def test_large_listing_still_checks_bytes_when_body_cannot_be_retained(monkeypatch):
    data, targets, inventory = _report_inventory(b"{}")
    journal = publication._journal_bytes(
        transaction_id="a" * 64, attempt_id="b" * 32, state="committed", objects=targets
    )
    store = _VersionedStore(SimpleNamespace(objects=data, journal=journal))
    _install_runtime_dependencies(monkeypatch, store, inventory)
    monkeypatch.setattr(runtime, "_PUBLICATION_CACHE_MAX_BYTES", 1)
    runtime._clear_verified_publication_cache()
    for _ in range(3):
        runtime._committed_publication_artifacts(store, _BUCKET, "run-a", inventory)
    assert store.recording_reads == 3
    assert not runtime._PUBLICATION_CACHE_ENTRIES
    key = next(key for key in data if key.endswith(".rrd"))
    data[key] = b"X" * len(data[key])
    with pytest.raises(PublicationConflict, match="bytes"):
        runtime._committed_publication_artifacts(store, _BUCKET, "run-a", inventory)
    runtime._clear_verified_publication_cache()


@pytest.mark.parametrize("journal_race", [False, True])
@pytest.mark.parametrize("identity", ["etag", "version"])
def test_legacy_range_reads_only_requested_bytes_and_fences_journal(
    monkeypatch, tmp_path, journal_race, identity
):
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="npa_repair_range"
    )
    context = _viewer_context(module, tmp_path, False)
    if identity == "version":
        context.artifact = replace(
            context.artifact, source_etag="", source_version_id="fixture-version"
        )
    ranges, transferred = [], []

    class Store:
        def get_object(self, *, Bucket, Key, **conditions):
            if Key.endswith(".sim2real-publication.json"):
                if journal_race and ranges:
                    return {"Body": io.BytesIO(context.journal)}
                raise KeyError(Key)
            if identity == "version":
                assert conditions["VersionId"] == "fixture-version"
            else:
                assert conditions["IfMatch"] == hashlib.sha256(context.good).hexdigest()
            value = conditions["Range"]
            ranges.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            data = context.good[start : end + 1]
            transferred.append(len(data))
            return {
                "Body": io.BytesIO(data),
                "ContentLength": len(data),
                "ContentRange": f"bytes {start}-{end}/{len(context.good)}",
            }

    store = Store()
    for value, expected in [
        ("bytes=0-3", context.good[:4]),
        ("bytes=4-7", context.good[4:8]),
    ]:
        request = Request({"type": "http", "headers": [(b"range", value.encode())]})
        metadata = module._artifact_content_response_context(
            "run-a", "", context.artifact, False
        )
        if journal_race:
            with pytest.raises(HTTPException) as error:
                module._artifact_stream_response(
                    store, request, "demo-bucket", context.artifact, metadata
                )
            assert error.value.status_code == 409
            break
        response = module._artifact_stream_response(
            store, request, "demo-bucket", context.artifact, metadata
        )
        assert asyncio.run(_response_bytes(response)) == expected
    assert transferred == ([4] if journal_race else [4, 4])


@pytest.mark.parametrize("fault", ["short", "extra", "extent", "total", "missing"])
def test_legacy_range_rejects_bad_extent_or_body_before_response(
    monkeypatch, tmp_path, fault
):
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="npa_repair_bad_range"
    )
    context = _viewer_context(module, tmp_path, False)
    bodies = []

    class Store:
        def get_object(self, *, Bucket, Key, **conditions):
            if Key.endswith(".sim2real-publication.json"):
                raise KeyError(Key)
            assert conditions["Range"] == "bytes=4-7"
            data = context.good[4:8]
            if fault == "short":
                data = data[:-1]
            elif fault == "extra":
                data += b"x"
            body = io.BytesIO(data)
            bodies.append(body)
            content_range = f"bytes 4-7/{len(context.good)}"
            if fault == "extent":
                content_range = f"bytes 0-3/{len(context.good)}"
            elif fault == "total":
                content_range = "bytes 4-7/999"
            elif fault == "missing":
                content_range = ""
            return {"Body": body, "ContentLength": 4, "ContentRange": content_range}

    metadata = module._artifact_content_response_context(
        "run-a", "", context.artifact, False
    )
    request = Request({"type": "http", "headers": [(b"range", b"bytes=4-7")]})
    with pytest.raises(HTTPException) as error:
        module._artifact_stream_response(
            Store(), request, "demo-bucket", context.artifact, metadata
        )
    assert error.value.status_code in {409, 502}
    assert bodies and all(body.closed for body in bodies)
