"""Exercise project-selected health checks with real configuration parsing."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import health
from npa.clients import config, credentials


@pytest.fixture
def project_files():
    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CONFIG_PATH.write_text(
        yaml.safe_dump(
            {
                "default_project": "other",
                "projects": {
                    "target": {"project_id": "project-target"},
                    "other": {"project_id": "project-other"},
                },
            }
        )
    )
    data = {
        "storage": {
            "bucket": "s3://host-bucket",
            "endpoint": "https://host.invalid",
            "access_key_id": "host-access",
            "secret_access_key": "host-secret",
        },
        "project_credentials": {
            "schema_version": "npa.project-credentials.v2",
            "projects": {
                "project-target": {
                    "storage": {
                        "bucket": "target-bucket/results",
                        "endpoint_url": "https://target.invalid",
                        "aws_access_key_id": "target-access",
                        "aws_secret_access_key": "target-secret",
                    }
                }
            },
        },
    }
    credentials.CREDENTIALS_PATH.write_text(yaml.safe_dump(data))
    credentials.CREDENTIALS_PATH.chmod(0o600)
    return data


def _invoke(*options):
    return CliRunner().invoke(
        app,
        [
            "workbench",
            "health",
            "preflight",
            "--checks",
            "s3",
            "--json",
            *options,
        ],
    )


@pytest.fixture
def host_storage_environment(monkeypatch):
    for name, value in {
        "AWS_ACCESS_KEY_ID": "unrelated-access",
        "AWS_SECRET_ACCESS_KEY": "unrelated-secret",
        "AWS_ENDPOINT_URL": "https://unrelated.invalid",
        "NPA_CHECKPOINT_BUCKET": "s3://unrelated-bucket",
    }.items():
        monkeypatch.setenv(name, value)


def test_project_probe_uses_exact_record_and_preserves_files(
    project_files, monkeypatch
):
    before = [
        path.read_bytes() for path in (config.CONFIG_PATH, credentials.CREDENTIALS_PATH)
    ]
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unrelated-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unrelated-secret")
    monkeypatch.setenv("NEBIUS_S3_BUCKET", "s3://unrelated-bucket")
    result = _invoke("--project", "target")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    factory.assert_called_once_with(
        endpoint_url="https://target.invalid",
        aws_access_key_id="target-access",
        aws_secret_access_key="target-secret",
    )
    client.probe_list_access.assert_called_once_with("s3://target-bucket/results")
    client.list_checkpoints.assert_not_called()
    assert before == [
        path.read_bytes() for path in (config.CONFIG_PATH, credentials.CREDENTIALS_PATH)
    ]
    assert "target-secret" not in result.output


@pytest.mark.parametrize(
    "missing", ["bucket", "endpoint_url", "aws_access_key_id", "aws_secret_access_key"]
)
def test_partial_project_storage_does_not_use_host_credentials(
    project_files, monkeypatch, missing, host_storage_environment
):
    project_files["project_credentials"]["projects"]["project-target"]["storage"].pop(
        missing
    )
    credentials.CREDENTIALS_PATH.write_text(yaml.safe_dump(project_files))
    factory = Mock(side_effect=AssertionError("must not reach storage"))
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    result = _invoke("-p", "target")
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["checks"][0]["status"] == "FAIL"
    factory.assert_not_called()


def test_missing_project_record_does_not_use_environment_credentials(
    project_files, monkeypatch, host_storage_environment
):
    factory = Mock()
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    result = _invoke("--project", "other")
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["checks"][0]["status"] == "FAIL"
    factory.assert_not_called()


def test_invalid_project_store_preserves_json_and_other_checks(project_files):
    project_files["project_credentials"]["schema_version"] = "unsupported-schema"
    credentials.CREDENTIALS_PATH.write_text(yaml.safe_dump(project_files))
    before = credentials.CREDENTIALS_PATH.read_bytes()
    result = _invoke("--project", "target", "--checks", "s3,nebius", "--offline")
    assert result.exit_code == 1
    assert [(item["name"], item["status"]) for item in json.loads(result.stdout)["checks"]] == [
        ("s3", "FAIL"), ("nebius", "SKIP"),
    ]
    assert credentials.CREDENTIALS_PATH.read_bytes() == before


def test_unknown_project_returns_json_failure_without_a_probe(
    project_files, monkeypatch
):
    factory = Mock(side_effect=AssertionError("must not reach storage"))
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    result = _invoke("--project", "missing")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "Unknown project alias" in payload["checks"][0]["remedy"]
    factory.assert_not_called()


def test_explicitly_deselected_storage_fails(project_files):
    project_files["project_credentials"]["projects"]["project-target"][
        "storage_selected"
    ] = False
    credentials.CREDENTIALS_PATH.write_text(yaml.safe_dump(project_files))
    result = _invoke("--project", "target", "--offline")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_project_offline_never_constructs_storage_client(project_files, monkeypatch):
    factory = Mock(side_effect=AssertionError("offline must not reach storage"))
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    result = _invoke("--project", "target", "--offline")
    assert result.exit_code == 0
    assert "not probed" in json.loads(result.stdout)["checks"][0]["summary"]
    factory.assert_not_called()


def test_warn_only_retains_failure_in_json(project_files):
    result = _invoke("--project", "missing", "--warn-only")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["ok"] is False


def test_project_storage_failure_preserves_other_requested_checks(project_files):
    result = _invoke("--project", "missing", "--checks", "hf,s3,nebius", "--offline")
    assert result.exit_code == 1
    checks = json.loads(result.stdout)["checks"]
    assert [(check["name"], check["status"]) for check in checks] == [
        ("hf", "WARN"), ("s3", "FAIL"), ("nebius", "SKIP"),
    ]


def test_no_project_retains_host_storage_selection(project_files, monkeypatch):
    client = Mock()
    monkeypatch.setattr(
        health.StorageClient, "from_environment", Mock(return_value=client)
    )
    result = _invoke()
    assert result.exit_code == 0, result.output
    client.probe_list_access.assert_called_once_with("s3://host-bucket")


def test_project_preserves_non_storage_credentials(project_files):
    original = replace(
        credentials.load_credentials(), tokens={"HF_TOKEN": "synthetic-hf"}
    )
    resolved = health._project_credentials("target", original)
    assert resolved.hf_token == original.hf_token
    assert resolved.s3_bucket != original.s3_bucket


@pytest.mark.parametrize("section", ["storage", "object_storage", "object-storage", "terraform_state"])
def test_inline_project_storage_stays_read_only_without_legacy_migration(
    project_files, monkeypatch, host_storage_environment, section
):
    del project_files["project_credentials"]
    credentials.CREDENTIALS_PATH.write_text(yaml.safe_dump(project_files))
    document = yaml.safe_load(config.CONFIG_PATH.read_text())
    document["projects"]["target"][section] = {
        "bucket": "inline-bucket/prefix",
        "endpoint": "https://inline.invalid",
        "access_key": "inline-access",
        "secret_key": "inline-secret",
    }
    config.CONFIG_PATH.write_text(yaml.safe_dump(document))
    paths = (config.CONFIG_PATH, credentials.CREDENTIALS_PATH)
    before = [path.read_bytes() for path in paths]
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(health.StorageClient, "from_environment", factory)
    result = _invoke("--project", "target")
    assert result.exit_code == 0, result.output
    factory.assert_called_once_with(
        endpoint_url="https://inline.invalid",
        aws_access_key_id="inline-access",
        aws_secret_access_key="inline-secret",
    )
    client.probe_list_access.assert_called_once_with("s3://inline-bucket/prefix")
    assert [path.read_bytes() for path in paths] == before
