"""Verify private setup, canonical submission, recovery, and export boundaries."""

import json
import os
from types import SimpleNamespace

import pytest

from npa.workflows.video_sweep import operator


@pytest.fixture
def configuration(tmp_path):
    path = tmp_path / "sweep.json"
    operator._initialize(path)
    config = json.loads(path.read_text())
    config.update(
        project="example",
        infra="k8s/example",
        bucket="example-bucket",
        sources=["s3://example-bucket/source.mp4"],
    )
    path.write_text(json.dumps(config))
    return path, config


def test_initialization_is_private_and_never_overwrites(configuration):
    path, config = configuration
    assert path.stat().st_mode & 0o777 == 0o600
    assert operator._load(path) == config
    with pytest.raises(FileExistsError):
        operator._initialize(path)


@pytest.mark.parametrize(
    "change",
    [
        {"threshold": float("nan")},
        {"samples": 1},
        {"run_id": "../escape"},
        {"HF_TOKEN": "secret"},
        {"infra": "<context>"},
        {"sources": ["https://example.com/video.mp4"]},
    ],
)
def test_invalid_configuration_stops_before_submission(configuration, change):
    path, config = configuration
    config.update(change)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        operator._load(path)


def test_submit_uses_canonical_runtime_and_secret_names(configuration, monkeypatch):
    _, config = configuration
    monkeypatch.setenv("HF_TOKEN", "private-token-value")
    command = operator._submit_command(config)
    assert command[command.index("--runtime") + 1 :][:2] == ["--max-wait-seconds", "0"]
    assert command[command.index("submit") + 1].endswith(
        "testing/video-variant-sweep.yaml"
    )
    assert "HF_TOKEN" in command and "private-token-value" not in command
    assert "--run-id" in command and "--resume-run" not in command
    resumed = operator._submit_command(config, resume=True)
    assert "--resume-run" in resumed and "--run-id" not in resumed


def test_submission_failure_never_exports(configuration, monkeypatch):
    path, _ = configuration
    called = []
    monkeypatch.setattr(operator, "_credentials", lambda _: None)
    monkeypatch.setattr(operator, "_preflight", lambda *_: called.append("preflight"))
    monkeypatch.setattr(operator, "_stage_inputs", lambda _: called.append("inputs"))

    def fail(*_):
        called.append("submit")
        raise RuntimeError("private-service-details")

    monkeypatch.setattr(operator, "_invoke", fail)
    assert operator.main(["run", "--config", str(path)]) == 1
    assert called == ["preflight", "inputs", "submit"]
    assert not list(path.parent.glob("*-demo"))


def test_export_requires_only_storage_and_never_submits(
    configuration, monkeypatch, capsys
):
    from npa.workflows.video_sweep import demo

    path, _ = configuration
    monkeypatch.setattr(operator, "_credentials", lambda _: None)
    monkeypatch.setattr(
        demo, "export_demo", lambda *_: {"candidates": [1, 2], "accepted": 1}
    )
    monkeypatch.setattr(
        operator, "_invoke", lambda *_: pytest.fail("Export submitted compute")
    )
    assert operator.main(["export", "--config", str(path)]) == 0
    assert "2 variants, 1 accepted" in capsys.readouterr().out


def test_credentials_use_exact_project_storage(configuration, monkeypatch):
    from npa.clients import config as client_config
    from npa.clients import credentials

    _, config = configuration
    monkeypatch.setattr(credentials, "load_credentials", lambda: None)
    monkeypatch.setattr(credentials, "shared_credential_env", lambda _: {})

    def storage(project, **kwargs):
        assert project == "example"
        assert kwargs == {
            "include_shared_credentials": False,
            "include_environment": False,
        }
        return SimpleNamespace(
            aws_access_key_id="key",
            aws_secret_access_key="secret",
            endpoint_url="https://storage.example",
        )

    monkeypatch.setattr(client_config, "resolve_project_storage", storage)
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(key, raising=False)
    operator._credentials(config)
    assert os.environ["AWS_ACCESS_KEY_ID"] == "key"
    assert os.environ["AWS_ENDPOINT_URL"] == "https://storage.example"
