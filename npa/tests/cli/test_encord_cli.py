"""CLI tests for `npa workbench encord` (SaaS and S3 never touched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import npa.clients.credentials as credentials_module
from npa.cli.main import app
from npa.workbench.encord.schemas import (
    CurateReceipt,
    EncordToolError,
    PullManifest,
    PushReceipt,
)

runner = CliRunner()

PUSH_ARGS = [
    "workbench", "encord", "push",
    "--input-path", "s3://bkt/p/",
    "--integration", "i",
    "--folder", "f",
    "--output-path", "s3://bkt/out/",
]


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # ENCORD_* env vars are scrubbed by the root conftest's ambient-credential list.
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "missing.yaml")


def _receipt(**overrides) -> PushReceipt:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        input_uri="s3://bkt/p/",
        endpoint_url="https://storage.test",
        encord_domain="https://api.encord.com",
        integration_id="00000000-0000-0000-0000-000000000003",
        integration_title="nebius-s3",
        folder_uuid="00000000-0000-0000-0000-000000000001",
        folder_name="fresh",
        media_filter="videos-images",
        status="done",
        files_discovered=2,
        units_done=2,
        receipt_uri="s3://bkt/out/push_receipt.json",
    )
    payload.update(overrides)
    return PushReceipt(**payload)


def _manifest(**overrides) -> PullManifest:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        encord_domain="https://api.encord.com",
        source_kind="collection",
        source_id="00000000-0000-0000-0000-000000000009",
        source_name="keepers",
        output_uri="s3://bkt/pull",
        manifest_uri="s3://bkt/pull/manifest.json",
        items_total=1,
        media_copied=1,
    )
    payload.update(overrides)
    return PullManifest(**payload)


def _curate_receipt(**overrides) -> CurateReceipt:
    payload = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        encord_domain="https://api.encord.com",
        folder_uuid="00000000-0000-0000-0000-000000000001",
        folder_name="src",
        collection_uuid="00000000-0000-0000-0000-0000000000c8",
        collection_name="keepers",
        collection_created=True,
        preset_uuid="00000000-0000-0000-0000-00000000012c",
        preset_name="npa-curate-run-1",
        items_total=3,
        items_selected=2,
        status="done",
        receipt_uri="s3://bkt/curate/curate_receipt.json",
    )
    payload.update(overrides)
    return CurateReceipt(**payload)


def _cleanup_summary(**overrides) -> dict:
    summary = {
        "folders_deleted": ["npa-e2e-run1"],
        "items_deleted": 3,
        "collections_deleted": [],
        "presets_deleted": [],
        "datasets_undeletable": ["npa-e2e-run1"],
    }
    summary.update(overrides)
    return summary


def test_encord_group_registered() -> None:
    result = runner.invoke(app, ["workbench", "encord", "--help"])
    assert result.exit_code == 0
    assert "push" in result.output and "pull" in result.output
    assert "curate" in result.output and "cleanup" in result.output
    assert "system-info" in result.output


def test_push_help_contains_contract_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "push", "--help"])
    assert result.exit_code == 0
    for option in ("input-path", "integration", "folder", "output-path", "media"):
        assert option in result.output


def test_push_happy_path_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return _receipt()

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "s3://bkt/p/",
            "--integration", "nebius-s3",
            "--folder", "fresh",
            "--dataset", "ds-a",
            "--output-path", "s3://bkt/out/",
            "--workflow-run", "run-1",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "npa.encord.push_receipt.v1"
    assert captured["media"] == "videos-images"
    assert captured["transfer"] == "register"
    assert captured["dataset"] == "ds-a"
    assert captured["workflow_run"] == "run-1"


def test_push_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("npa.sdk.workbench.encord.push", lambda **kwargs: _receipt())
    result = runner.invoke(app, [*PUSH_ARGS, "--output", "text"])
    assert result.exit_code == 0, result.output
    assert "pushed 2/2 item(s) to Encord folder 'fresh'" in result.output


def test_push_rejects_local_input_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "/tmp/media",
            "--integration", "i",
            "--folder", "f",
            "--output-path", "s3://bkt/out/",
        ],
    )
    assert result.exit_code == 1
    assert "S3" in result.output


def test_push_rejects_local_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "push",
            "--input-path", "s3://bkt/p/",
            "--integration", "i",
            "--folder", "f",
            "--output-path", "/tmp/out",
        ],
    )
    assert result.exit_code == 1


def test_push_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_push(**kwargs):
        raise EncordToolError("Encord push failed: 1 unit error(s)")

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(app, PUSH_ARGS)
    assert result.exit_code == 1
    assert "unit error" in result.output


def test_push_missing_credential_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # No monkeypatched SDK: run_push resolves the endpoint from environ then
    # fails on auth; both are acceptable fail-closed messages, never a traceback.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.test")
    result = runner.invoke(app, PUSH_ARGS)
    assert result.exit_code == 1
    assert "ENCORD_SSH_KEY" in result.output or "credential" in result.output.lower()


def test_pull_help_contains_source_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "pull", "--help"])
    assert result.exit_code == 0
    for option in ("source", "source-id", "output-path"):
        assert option in result.output


def test_pull_happy_path_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return _manifest()

    monkeypatch.setattr("npa.sdk.workbench.encord.pull", fake_pull)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "keepers",
            "--output-path", "s3://bkt/pull/",
            "--output", "text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pulled 1 item(s)" in result.output
    assert captured["source"] == "collection"
    assert captured["source_id"] == "keepers"


def test_pull_happy_path_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("npa.sdk.workbench.encord.pull", lambda **kwargs: _manifest())
    result = runner.invoke(
        app,
        ["workbench", "encord", "pull", "--source", "dataset", "--source-id", "ds",
         "--output-path", "s3://bkt/pull/", "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema"] == "npa.encord.pull_manifest.v1"


def test_pull_rejects_local_output_and_bad_source() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "x",
            "--output-path", "/tmp/out",
        ],
    )
    assert result.exit_code == 1
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "everything",
            "--source-id", "x",
            "--output-path", "s3://bkt/pull/",
        ],
    )
    assert result.exit_code != 0


def test_pull_missing_credential_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # No monkeypatched SDK: run_pull resolves the endpoint from environ then
    # fails on auth; both are acceptable fail-closed messages, never a traceback.
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.test")
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "pull",
            "--source", "collection",
            "--source-id", "00000000-0000-0000-0000-000000000009",
            "--output-path", "s3://bkt/pull/",
        ],
    )
    assert result.exit_code == 1
    assert "ENCORD_SSH_KEY" in result.output or "credential" in result.output.lower()


def test_pull_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pull(**kwargs):
        raise EncordToolError("Encord pull failed: 1 item error(s)")

    monkeypatch.setattr("npa.sdk.workbench.encord.pull", fake_pull)
    result = runner.invoke(
        app,
        ["workbench", "encord", "pull", "--source", "collection", "--source-id", "x",
         "--output-path", "s3://bkt/pull/"],
    )
    assert result.exit_code == 1 and "item error" in result.output


def test_curate_help_contains_contract_options() -> None:
    result = runner.invoke(app, ["workbench", "encord", "curate", "--help"])
    assert result.exit_code == 0
    for option in ("folder", "filter", "collection", "output-path", "poll-seconds"):
        assert option in result.output


def test_curate_happy_path_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_curate(**kwargs):
        captured.update(kwargs)
        return _curate_receipt()

    monkeypatch.setattr("npa.sdk.workbench.encord.curate", fake_curate)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "brightness:0.2:0.8,sharpness:0.3:1",
            "--filter", "width:32:4096",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
            "--workflow-run", "run-1",
            "--output", "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "npa.encord.curate_receipt.v1"
    assert captured["folder"] == "src"
    assert captured["filters"] == ["brightness:0.2:0.8,sharpness:0.3:1", "width:32:4096"]
    assert captured["collection"] == "keepers"
    assert captured["workflow_run"] == "run-1"


def test_curate_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.sdk.workbench.encord.curate", lambda **kwargs: _curate_receipt()
    )
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "width:1:100000",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
            "--output", "text",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "curated 2 of 3 item(s)" in result.output


def test_curate_rejects_local_output_path() -> None:
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "width:1:100000",
            "--collection", "keepers",
            "--output-path", "/tmp/out",
        ],
    )
    assert result.exit_code == 1


def test_curate_tool_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_curate(**kwargs):
        raise EncordToolError("Encord curate selected 0 items.")

    monkeypatch.setattr("npa.sdk.workbench.encord.curate", fake_curate)
    result = runner.invoke(
        app,
        [
            "workbench", "encord", "curate",
            "--folder", "src",
            "--filter", "brightness:0.2:0.8",
            "--collection", "keepers",
            "--output-path", "s3://bkt/curate/",
        ],
    )
    assert result.exit_code == 1
    assert "selected 0 items" in result.output



def test_system_info_reports_setup_without_touching_encord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENCORD_SSH_KEY_FILE", "/keys/encord.pem")
    monkeypatch.setenv("ENCORD_DOMAIN", "https://api.us.encord.com")
    result = runner.invoke(app, ["workbench", "encord", "system-info", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool"] == "encord" and payload["status"] == "ok"
    assert payload["encord_domain"] == "https://api.us.encord.com"
    # Names only — never the credential value.
    assert payload["credential_transports"] == ["ENCORD_SSH_KEY_FILE"]
    assert "/keys/encord.pem" not in result.output
    assert payload["schemas"]["push"] == "npa.encord.push_receipt.v1"
    assert "width" in payload["curate_metrics"]
    result = runner.invoke(app, ["workbench", "encord", "system-info"])
    assert result.exit_code == 0
    assert "credential_transports: ['ENCORD_SSH_KEY_FILE']" in result.output



def test_cleanup_help_and_dry_run_text(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["workbench", "encord", "cleanup", "--help"])
    assert result.exit_code == 0
    assert "title-prefix" in result.output and "dry-run" in result.output

    captured: dict = {}

    def fake_cleanup(**kwargs):
        captured.update(kwargs)
        return _cleanup_summary()

    monkeypatch.setattr("npa.sdk.workbench.encord.cleanup", fake_cleanup)
    result = runner.invoke(
        app,
        ["workbench", "encord", "cleanup", "--title-prefix", "npa-e2e-", "--dry-run",
         "--output", "text"],
    )
    assert result.exit_code == 0, result.output
    assert "would delete 1 folder(s) (3 item(s))" in result.output
    assert "1 dataset(s) need app-side removal" in result.output
    assert captured == {"title_prefix": "npa-e2e-", "dry_run": True}


def test_cleanup_json_and_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("npa.sdk.workbench.encord.cleanup", lambda **kwargs: _cleanup_summary())
    result = runner.invoke(
        app, ["workbench", "encord", "cleanup", "--title-prefix", "npa-e2e-", "--output", "json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["folders_deleted"] == ["npa-e2e-run1"]

    def short_prefix(**kwargs):
        raise EncordToolError("title prefix must be at least 4 characters")

    monkeypatch.setattr("npa.sdk.workbench.encord.cleanup", short_prefix)
    result = runner.invoke(app, ["workbench", "encord", "cleanup", "--title-prefix", "np"])
    assert result.exit_code == 1 and "at least 4" in result.output


def test_unexpected_exceptions_are_not_rewritten_as_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only EncordToolError is exit 1; a bug propagates (exit 2 via app_entry)."""

    def broken_push(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("npa.sdk.workbench.encord.push", broken_push)
    result = runner.invoke(app, PUSH_ARGS)
    assert isinstance(result.exception, RuntimeError)
    assert "encord push failed: boom" not in result.output


def test_json_failure_still_emits_exactly_one_json_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_push(**kwargs):
        raise EncordToolError("Encord push failed: 1 unit error(s)")

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(app, [*PUSH_ARGS, "--output", "json"])
    assert result.exit_code == 1
    document = json.loads(result.stdout)
    assert document["result"] == "error" and document["mutated"] is False
    assert "unit error" in result.stderr
