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
    assert "--stage-src" in command
    assert command[command.index("--max-wait-seconds") + 1] == "0"
    assert command[command.index("submit") + 1].endswith(
        "testing/video-variant-sweep-cosmos3.yaml"
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


@pytest.mark.parametrize(
    "accepted,tracked,exports",
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_failed_run_exports_only_fully_tracked_rejections(
    configuration, monkeypatch, capsys, accepted, tracked, exports
):
    import subprocess
    from npa.workflows.video_sweep import demo

    path, _ = configuration
    monkeypatch.setattr(operator, "_credentials", lambda _: None)
    monkeypatch.setattr(operator, "_preflight", lambda *_: None)
    monkeypatch.setattr(operator, "_stage_inputs", lambda _: None)
    monkeypatch.setattr(
        operator.artifacts,
        "exists",
        lambda uri: tracked or not uri.endswith("lineage.json"),
    )
    monkeypatch.setattr(
        operator.artifacts, "read_json", lambda _: {"items": [{"accepted": accepted}]}
    )
    calls = []
    monkeypatch.setattr(demo, "export_demo", lambda *args: calls.append(args))

    def fail(*_):
        raise subprocess.CalledProcessError(1, ["workflow", "submit"])

    monkeypatch.setattr(operator, "_invoke", fail)
    assert operator.main(["run", "--config", str(path)]) == 1
    assert bool(calls) is exports
    assert ("workflow failure is retained" in capsys.readouterr().out) is exports


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


def test_selected_gpu_reaches_both_workers_without_changing_cpu_stages(configuration):
    from pathlib import Path
    from npa.orchestration.npa_workflow import build_plan
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit

    _, config = configuration
    config["accelerators"] = "RTXPRO6000:1"
    command = operator._submit_command(config)
    assert "accelerators=RTXPRO6000:1" in command
    assert "--accelerators" not in command
    spec = load_spec_for_submit(
        Path(operator._spec()), config_overrides=operator._variables(config)
    )
    steps = build_plan(spec, run_id="test").steps
    workers = [step for step in steps if step.state.startswith("worker-")]
    assert len(workers) == 2
    assert all(
        step.resources_profile["accelerators"] == "RTXPRO6000:1" for step in workers
    )
    assert all(
        "accelerators" not in step.resources_profile
        for step in steps
        if step not in workers
    )


def test_pinned_images_and_target_match_preflight_and_submit(configuration):
    path, config = configuration
    tool = "workflow.video_sweep.generate_cosmos3"
    image = "registry.example/cosmos3@sha256:" + "a" * 64
    config["image_overrides"] = {tool: image}
    path.write_text(json.dumps(config))
    assert operator._load(path) == config
    for command in (
        operator._image_preflight_command(config),
        operator._submit_command(config),
        operator._submit_command(config, resume=True),
    ):
        assert command[command.index("--image-override") + 1] == f"{tool}={image}"
        assert command[command.index("--infra") + 1] == config["infra"]
        assert command[command.index("--project") + 1] == config["project"]
        assert "accelerators=B200:1" in command


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        {"tool": "image:mutable"},
        {"bad=tool": "image@sha256:" + "a" * 64},
        {"tool": "image@sha256:short"},
        {"tool": None},
    ],
)
def test_invalid_image_pins_fail_before_provider_calls(configuration, overrides):
    path, config = configuration
    config["image_overrides"] = overrides
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        operator._load(path, offline=True)


def test_run_cannot_change_or_remove_image_pins(configuration, monkeypatch):
    _, config = configuration
    tool = "workflow.video_sweep.generate_cosmos3"
    config["image_overrides"] = {tool: "image@sha256:" + "a" * 64}
    stored = {}
    monkeypatch.setattr(operator.artifacts, "exists", lambda uri: uri in stored)
    monkeypatch.setattr(operator.artifacts, "read_json", stored.__getitem__)
    monkeypatch.setattr(operator.artifacts, "write_json", stored.__setitem__)
    operator._stage_inputs(config)
    operator._stage_inputs(config)
    changed = dict(config, image_overrides={tool: "image@sha256:" + "b" * 64})
    with pytest.raises(ValueError, match="new run ID"):
        operator._stage_inputs(changed)
    with pytest.raises(ValueError, match="new run ID"):
        operator._stage_inputs(dict(config, image_overrides={}))


def test_existing_unpinned_run_cannot_add_image_pins(configuration, monkeypatch):
    _, config = configuration
    stored = {}
    monkeypatch.setattr(operator.artifacts, "exists", lambda uri: uri in stored)
    monkeypatch.setattr(operator.artifacts, "read_json", stored.__getitem__)
    monkeypatch.setattr(operator.artifacts, "write_json", stored.__setitem__)
    operator._stage_inputs(config)
    before = dict(stored)
    config["image_overrides"] = {"tool": "image@sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="new run ID"):
        operator._stage_inputs(config)
    assert stored == before


def test_partial_staging_can_resume_with_the_same_image_pins(
    configuration, monkeypatch
):
    _, config = configuration
    config["image_overrides"] = {"tool": "image@sha256:" + "a" * 64}
    stored = {}
    monkeypatch.setattr(operator.artifacts, "exists", lambda uri: uri in stored)
    monkeypatch.setattr(operator.artifacts, "read_json", stored.__getitem__)

    def interrupted_write(uri, value):
        if uri.endswith("variants.json"):
            raise ConnectionError("Interrupted input staging")
        stored[uri] = value

    monkeypatch.setattr(operator.artifacts, "write_json", interrupted_write)
    with pytest.raises(ConnectionError):
        operator._stage_inputs(config)
    monkeypatch.setattr(operator.artifacts, "write_json", stored.__setitem__)
    operator._stage_inputs(config)
    assert len(stored) == 3
