"""Verify turnkey submission, early opt-out, and checksum-checked offline fetching."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ISAAC_IMAGE = "registry.example.invalid/npa-isaac-lab@sha256:" + "a" * 64


@pytest.fixture
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_physical_augmentation_demo.py"
    )
    spec = importlib.util.spec_from_file_location("physical_demo_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_real_callback_arguments_and_rtx_profile(
    runner, monkeypatch, tmp_path
):
    import inspect

    calls = []

    def capture(callback, **kwargs):
        inspect.signature(callback).bind_partial(**kwargs)
        calls.append((callback.__name__, kwargs))

    monkeypatch.setattr(runner, "call_cli_callback", capture)
    monkeypatch.setattr(
        runner,
        "resolve_project_storage",
        lambda *args, **kwargs: NS(checkpoint_bucket="s3://example-bucket"),
    )
    monkeypatch.setattr(runner, "_fetch", lambda *args: None)
    runner.main(
        [
            "--project",
            "test",
            "--output-path",
            str(tmp_path),
            "--isaac-image",
            ISAAC_IMAGE,
            "--provision",
            "--kubeconfig",
            str(tmp_path / "kubeconfig"),
        ]
    )
    assert [name for name, _ in calls] == [
        "preflight_command",
        "provision_if_absent_cmd",
        "submit_cmd",
    ]
    assert calls[1][1]["gpu_workload_profile"] == "rtx-rendering"
    assert calls[1][1]["sky_smoke"] is True
    assert calls[2][1]["stage_src"] is True
    assert calls[2][1]["runtime"] is True
    assert calls[2][1]["yaml_path"].is_file()
    assert calls[2][1]["var"] == [
        "bucket=example-bucket",
        f"isaac_image={ISAAC_IMAGE}",
    ]


def test_opt_out_and_invalid_id_stop_before_any_mutation(runner, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("A rejected run must not provision or submit")

    monkeypatch.setattr(runner, "call_cli_callback", forbidden)
    base = ["--project", "test", "--output-path", str(tmp_path)]
    with pytest.raises(ValueError, match="opted out"):
        runner.main(base + ["--provision", "--no-accept-eula"])
    with pytest.raises(ValueError, match="Run ID"):
        runner.main(base + ["--fetch-run", "../../other-run"])
    with pytest.raises(ValueError, match="--isaac-image"):
        runner.main(base)


@pytest.mark.parametrize("corrupt", [False, True])
def test_fetch_does_not_allocate_and_rejects_corruption(
    runner, monkeypatch, tmp_path, corrupt
):
    files = {
        name: b"fixture"
        for name in (
            "demo.html",
            "demo.mp4",
            "demo-validation.json",
            "report.json",
            "demonstrations.rrd",
        )
    }
    checksums = {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items()
    }
    files["checksums.json"] = json.dumps(checksums).encode()
    if corrupt:
        files["demo.mp4"] = b"corrupted"
        (tmp_path / "demo.html").write_bytes(b"previous verified demo")

    def download(uri, path):
        assert uri.startswith(
            "s3://example-bucket/physical-augmentation/test-run/reports/"
        )
        Path(path).write_bytes(files[uri.rsplit("/", 1)[-1]])

    monkeypatch.setattr(
        runner,
        "resolve_project_storage",
        lambda *args, **kwargs: NS(checkpoint_bucket="s3://example-bucket"),
    )
    monkeypatch.setattr(
        runner, "storage_client_for_project", lambda project: NS(download_file=download)
    )
    monkeypatch.setattr(
        runner, "_submit", lambda *args: pytest.fail("Fetch must not submit")
    )
    args = [
        "--project",
        "test",
        "--output-path",
        str(tmp_path),
        "--fetch-run",
        "test-run",
    ]
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            runner.main(args)
        assert (tmp_path / "demo.html").read_bytes() == b"previous verified demo"
        assert not (tmp_path / "demo.mp4").exists()
    else:
        assert runner.main(args) == 0
        assert (tmp_path / "demo.html").read_bytes() == b"fixture"
