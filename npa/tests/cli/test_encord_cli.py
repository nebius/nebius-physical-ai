"""CLI contract tests for the stateless Encord client."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.encord.schemas import EncordToolError

runner = CliRunner()


class Result(SimpleNamespace):
    def model_dump(self, *, by_alias: bool = False):
        del by_alias
        return dict(self.payload)


def test_encord_group_exposes_only_transport_verbs() -> None:
    result = runner.invoke(app, ["workbench", "encord", "--help"])
    assert result.exit_code == 0, result.output
    assert "push" in result.output
    assert "pull" in result.output
    assert "verify-roundtrip" in result.output
    assert "seed-demo" not in result.output


def test_push_defaults_to_register_and_forwards_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return Result(
            payload={"schema": "npa.encord.push_receipt.v1"},
            status="completed",
            counts=SimpleNamespace(successful=1, discovered=1),
            receipt_uri="s3://bucket/out/push_receipt.json",
        )

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(
        app,
        [
            "workbench",
            "encord",
            "push",
            "--input-path",
            "s3://bucket/input/",
            "--integration",
            "integration",
            "--folder",
            "folder",
            "--identity-sidecar",
            "s3://bucket/identity.json",
            "--output-path",
            "s3://bucket/out/push_receipt.json",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema"] == "npa.encord.push_receipt.v1"
    assert captured["transfer"] == "register"
    assert captured["identity_sidecar_uri"] == "s3://bucket/identity.json"


def test_upload_requires_explicit_cli_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_push(**kwargs):
        captured.update(kwargs)
        return Result(
            payload={"schema": "npa.encord.push_receipt.v1"},
            status="completed",
            counts=SimpleNamespace(successful=1, discovered=1),
            receipt_uri="s3://bucket/out/push_receipt.json",
        )

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fake_push)
    result = runner.invoke(
        app,
        [
            "workbench",
            "encord",
            "push",
            "--input-path",
            "s3://bucket/input/",
            "--folder",
            "folder",
            "--transfer",
            "upload",
            "--output-path",
            "s3://bucket/out/",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["transfer"] == "upload"
    assert captured["integration"] == ""


def test_pull_defaults_to_no_label_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_pull(**kwargs):
        captured.update(kwargs)
        return Result(
            payload={"schema": "npa.encord.pull_manifest.v1"},
            status="completed",
            counts=SimpleNamespace(successful=1, discovered=1),
            manifest_uri="s3://bucket/out/manifest.json",
        )

    monkeypatch.setattr("npa.sdk.workbench.encord.pull", fake_pull)
    result = runner.invoke(
        app,
        [
            "workbench",
            "encord",
            "pull",
            "--source",
            "dataset",
            "--source-id",
            "dataset-id",
            "--output-path",
            "s3://bucket/out",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["label_export"] == "none"


def test_pull_help_discloses_initialize_mutation() -> None:
    result = runner.invoke(app, ["workbench", "encord", "pull", "--help"])
    assert result.exit_code == 0, result.output
    assert "label-export" in result.output
    assert "remote" in result.output
    assert "label-row" in result.output


def test_verify_forwards_both_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return Result(
            payload={"schema": "npa.encord.roundtrip_report.v1", "passed": True},
            status="completed",
            matched=1,
            expected=1,
            report_uri="s3://bucket/report.json",
        )

    monkeypatch.setattr("npa.sdk.workbench.encord.verify_roundtrip", fake_verify)
    result = runner.invoke(
        app,
        [
            "workbench",
            "encord",
            "verify-roundtrip",
            "--receipt-uri",
            "s3://bucket/push_receipt.json",
            "--manifest-uri",
            "s3://bucket/manifest.json",
            "--output-path",
            "s3://bucket/report.json",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["receipt_uri"] == "s3://bucket/push_receipt.json"
    assert captured["manifest_uri"] == "s3://bucket/manifest.json"


def test_contract_error_exits_one_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs):
        del kwargs
        raise EncordToolError("exact identity is unresolved")

    monkeypatch.setattr("npa.sdk.workbench.encord.push", fail)
    result = runner.invoke(
        app,
        [
            "workbench",
            "encord",
            "push",
            "--input-path",
            "s3://bucket/input/",
            "--integration",
            "integration",
            "--folder",
            "folder",
            "--output-path",
            "s3://bucket/out/",
        ],
    )
    assert result.exit_code == 1
    assert "exact identity is unresolved" in result.output
    assert "Traceback" not in result.output
