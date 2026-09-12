"""Exercise prebuilt BYOF commands inside the standard workflow worker context."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import byof as byof_cli
from npa.workflows.byof import worker


REPO_URL = "https://github.com/example/capability.git"
REPO_REF = "1" * 40
IMAGE = "registry.example/solutions/capability@sha256:" + "a" * 64


def _load_runner():
    path = Path(__file__).resolve().parents[3] / "npa/scripts/run_byof_repo.py"
    spec = importlib.util.spec_from_file_location("byof_worker_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture_upload(context, directory, prefix):
    context.uploads.append(prefix)
    for path in Path(directory).rglob("*"):
        if path.is_file():
            context.files[path.relative_to(directory).as_posix()] = path.read_bytes()
    return prefix


@pytest.fixture
def allocated_worker(tmp_path, monkeypatch):
    runner = _load_runner()
    root = tmp_path / "baked-source"
    root.mkdir()
    (root / "npa_source_metadata.json").write_text(
        json.dumps(
            {
                "source": "oss-byof",
                "repo": REPO_URL,
                "ref": REPO_REF,
            }
        )
    )
    context = SimpleNamespace(runner=runner, root=root, files={}, uploads=[])
    monkeypatch.setattr(runner, "BYOF_REPO_MOUNT", str(root))
    monkeypatch.setattr(
        runner, "resolve_container_registry", lambda *_: "registry.example/solutions"
    )
    monkeypatch.setattr(
        runner, "_run", lambda *_args, **_kwargs: pytest.fail("nested host launch")
    )
    monkeypatch.setattr(byof_cli, "_load_runner", lambda: runner)
    monkeypatch.setenv("NPA_WORKFLOW_RUN_ID", "outer-run-capability-attempt")
    monkeypatch.setenv("NPA_WORKFLOW_STATE", "capability")
    monkeypatch.setenv("NPA_TASK_IMAGE", IMAGE)
    storage = SimpleNamespace(
        upload_directory=lambda directory, prefix: _capture_upload(
            context, directory, prefix
        )
    )
    monkeypatch.setattr(worker.StorageClient, "from_environment", lambda: storage)
    return context


def _arguments(context, source, **overrides):
    script = context.root / "capability.py"
    script.write_text(source)
    options = {
        "repo_url": REPO_URL,
        "repo_ref": REPO_REF,
        "base_profile": "prebuilt",
        "base_image": IMAGE,
        "workload": "solution-smoke",
        "solution_name": "example",
        "capability_name": "real_capability",
        "smoke_artifact_name": "result.json",
        "smoke_command": f"{shlex.quote(sys.executable)} capability.py",
        "run_id": "outer-run",
        "output_root": "s3://test-bucket/results",
        **overrides,
    }
    return ["workbench", "byof", "run", *byof_cli.build_byof_argv(**options)]


CAPABILITY = """import json, os
from pathlib import Path
output = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"])
(output / os.environ["BYOF_SMOKE_ARTIFACT_NAME"]).write_text(json.dumps({
    "result": 6 * 7, "source_dir": str(Path.cwd()),
    "image": os.environ["BYOF_IMAGE"], "run_id": os.environ["NPA_BYOF_RUN_ID"],
}))
print("capability executed")
"""


def test_cli_worker_executes_and_uploads_without_nested_launch(allocated_worker):
    context = allocated_worker
    result = CliRunner().invoke(app, _arguments(context, CAPABILITY))
    assert result.exit_code == 0, result.output
    output = json.loads(context.files["result.json"])
    assert output == {
        "result": 42,
        "source_dir": str(context.root),
        "image": IMAGE,
        "run_id": "outer-run",
    }
    assert context.uploads == ["s3://test-bucket/results/outer-run/"]
    assert context.files["solution_smoke_stdout.log"] == b"capability executed\n"
    summary = json.loads(context.files["npa_byof_summary.json"])
    assert summary["status"] == "success"
    assert summary["workflow_worker_run_id"] == "outer-run-capability-attempt"
    assert summary["run_id"] == "outer-run"
    artifact = next(
        item for item in summary["artifacts"] if item["path"] == "result.json"
    )
    assert (
        artifact["sha256"] == hashlib.sha256(context.files["result.json"]).hexdigest()
    )


@pytest.mark.parametrize(
    "source,exit_code", [("print('missing artifact')", 0), ("raise SystemExit(7)", 7)]
)
def test_failed_capability_uploads_diagnostics_and_fails(
    allocated_worker, source, exit_code
):
    context = allocated_worker
    result = CliRunner().invoke(app, _arguments(context, source))
    assert result.exit_code != 0
    summary = json.loads(context.files["npa_byof_summary.json"])
    assert summary["status"] == "failed"
    assert summary["smoke_exit_code"] == exit_code
    assert summary["smoke_artifact_present"] is False
    assert "diagnostics uploaded" in result.output


@pytest.mark.parametrize(
    "image",
    ["registry.example/solutions/capability:latest", IMAGE.replace("a" * 64, "b" * 64)],
)
def test_image_mismatch_fails_before_command_or_upload(allocated_worker, image):
    context = allocated_worker
    result = CliRunner().invoke(app, _arguments(context, CAPABILITY, base_image=image))
    assert result.exit_code != 0
    assert context.uploads == []
    assert context.files == {}


@pytest.mark.parametrize(
    "option,value",
    [
        ("repo_ref", "2" * 40),
        ("smoke_artifact_name", "../escape.json"),
        ("workload", "container-verify"),
        ("base_profile", "ubuntu"),
    ],
)
def test_invalid_worker_request_fails_before_execution(allocated_worker, option, value):
    context = allocated_worker
    result = CliRunner().invoke(app, _arguments(context, CAPABILITY, **{option: value}))
    assert result.exit_code != 0
    assert context.files == {}


@pytest.mark.parametrize("postprocess_result", [{"status": "verified"}, None])
def test_wan_worker_requires_postprocess_after_actual_upload(
    allocated_worker, monkeypatch, postprocess_result
):
    context = allocated_worker
    image = (
        "registry.example/solutions/npa-wan2-2@"
        + context.runner.wan_accepted_image_manifest()["oci_digest"]
    )
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    calls = []

    def postprocess(key, evidence):
        assert "wan2_2_ti2v_5b_text_to_video.json" in context.files
        calls.append((key, evidence.run_prefix_uri))
        return postprocess_result

    monkeypatch.setattr(context.runner, "run_registered_postprocess", postprocess)
    args = _arguments(
        context,
        CAPABILITY,
        base_image=image,
        solution_name="wan2.2",
        capability_name="wan2.2_ti2v_5b_text_to_video",
        smoke_artifact_name="wan2_2_ti2v_5b_text_to_video.json",
    )
    result = CliRunner().invoke(app, args)
    assert (result.exit_code == 0) == (postprocess_result is not None), result.output
    assert calls == [("wan2.2", "s3://test-bucket/results/outer-run/")]


def test_upload_failure_prevents_success(allocated_worker, monkeypatch):
    context = allocated_worker

    def upload_failed(*_args):
        raise RuntimeError("upload unavailable")

    monkeypatch.setattr(
        worker.StorageClient,
        "from_environment",
        lambda: SimpleNamespace(upload_directory=upload_failed),
    )
    result = CliRunner().invoke(app, _arguments(context, CAPABILITY))
    assert result.exit_code != 0
    assert "upload unavailable" in result.output
