"""Narrow NCore closure and atomic runtime dependency delivery."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import threading

import pytest


PACKAGING = Path(__file__).parents[2] / "docker/workbench/ncore"


def runtime():
    return importlib.import_module("npa.workflows.ncore_runtime")


def test_image_does_not_bake_application_distributions_or_whole_source_annex():
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    assert "pip install" not in dockerfile
    assert "COPY src/npa /" not in dockerfile
    assert "whole-sources" not in dockerfile
    assert "native-builder" not in dockerfile
    assert "base-source-lock.json" in dockerfile
    assert "--image-only" in dockerfile


def test_runtime_lock_is_exact_and_excludes_unrelated_npa_dependencies():
    module = runtime()
    lock = module.read_lock(PACKAGING / "runtime-lock.json")
    names = {item["name"] for item in lock["artifacts"]}
    assert {
        "numpy",
        "scipy",
        "pillow",
        "boto3",
        "pydantic",
        "typer",
        "debugpy",
    } <= names
    assert not names & {
        "rerun-sdk",
        "lancedb",
        "pyarrow",
        "torch",
        "av",
        "matplotlib",
        "paramiko",
    }
    assert (
        next(a["version"] for a in lock["artifacts"] if a["name"] == "numpy")
        == "1.26.4"
    )
    assert len(names) == len(lock["artifacts"])


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    module = runtime()
    lock = {"schema": 1, "artifacts": []}
    pin = tmp_path / "lock.json"
    pin.write_text(json.dumps(lock))
    monkeypatch.setattr(
        module, "RUNTIME_LOCK_SHA256", hashlib.sha256(pin.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(module, "SOURCE_ROOTS", (str(tmp_path / "source"),))
    monkeypatch.setattr(module, "source_identity", lambda: "source-a")
    monkeypatch.setattr(module, "verify_platform", lambda: None)
    calls = []

    def install(destination, _lock):
        calls.append(destination)
        (destination / "bin").mkdir()
        (destination / "bin/python").write_bytes(b"interpreter")
        (destination / "package.py").write_text("value = 1\n")

    monkeypatch.setattr(module, "install_runtime", install)
    monkeypatch.setattr(module, "verify_runtime", lambda _: None)
    return module, pin, tmp_path / "cache", calls


def test_runtime_reuses_only_complete_verified_cache(fake_runtime):
    module, pin, cache, calls = fake_runtime
    first = module.ensure_runtime(lock_path=pin, cache_root=cache)
    assert first == module.ensure_runtime(lock_path=pin, cache_root=cache)
    assert first.parent == cache
    assert not calls[0].exists()  # temporary generation was atomically renamed
    assert len(calls) == 1
    assert (first / module.READY_MARKER).is_file()


def test_concurrent_initializers_publish_one_complete_runtime(fake_runtime):
    module, pin, cache, calls = fake_runtime
    barrier = threading.Barrier(4)

    def prepare(_):
        barrier.wait()
        return module.ensure_runtime(lock_path=pin, cache_root=cache)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(prepare, range(4)))
    assert len(set(paths)) == len(calls) == 1
    assert not list(cache.glob("*.partial*"))


def test_install_failure_never_publishes_ready_or_raw_diagnostics(
    fake_runtime, monkeypatch
):
    module, pin, cache, _ = fake_runtime

    def fail(path, _lock):
        (path / "partial").write_text("incomplete")
        raise RuntimeError("sensitive upstream diagnostic")

    monkeypatch.setattr(module, "install_runtime", fail)
    with pytest.raises(module.NcoreRuntimeError, match="preparation failed") as caught:
        module.ensure_runtime(lock_path=pin, cache_root=cache)
    assert "sensitive" not in str(caught.value)
    assert not list(cache.rglob(module.READY_MARKER))
    assert not list(cache.glob("*.partial*"))


def test_corrupted_cache_is_rejected_before_execution(fake_runtime):
    module, pin, cache, calls = fake_runtime
    ready = module.ensure_runtime(lock_path=pin, cache_root=cache)
    (ready / "package.py").write_text("tampered")
    with pytest.raises(module.NcoreRuntimeError, match="cache integrity"):
        module.ensure_runtime(lock_path=pin, cache_root=cache)
    assert len(calls) == 1


def test_source_identity_separates_runtime_caches(fake_runtime, monkeypatch):
    module, pin, cache, calls = fake_runtime
    old = module.ensure_runtime(lock_path=pin, cache_root=cache)
    monkeypatch.setattr(module, "source_identity", lambda: "source-b")
    assert module.ensure_runtime(lock_path=pin, cache_root=cache) != old
    assert len(calls) == 2


def test_unreviewed_lock_fails_before_any_install(fake_runtime):
    module, pin, cache, calls = fake_runtime
    pin.write_text(pin.read_text() + "\n")
    with pytest.raises(module.NcoreRuntimeError, match="lock SHA256"):
        module.ensure_runtime(lock_path=pin, cache_root=cache)
    assert not calls and not cache.exists()


def test_runtime_cache_follows_operator_storage(monkeypatch, tmp_path):
    module = runtime()
    monkeypatch.delenv("NPA_NCORE_RUNTIME_CACHE", raising=False)
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    assert module.cache_directory() == tmp_path / "model-cache/ncore/runtime"
    monkeypatch.setenv("NPA_NCORE_RUNTIME_CACHE", str(tmp_path / "override"))
    assert module.cache_directory() == tmp_path / "override"


def test_download_hash_mismatch_never_reaches_installer(tmp_path, monkeypatch):
    module = runtime()
    monkeypatch.setattr(
        module, "download_public_https", lambda _, output, **__: output.write(b"wrong")
    )
    item = {
        "filename": "sample.whl",
        "url": "https://files.pythonhosted.org/sample.whl",
        "sha256": "0" * 64,
    }
    with pytest.raises(module.NcoreRuntimeError, match="artifact SHA256"):
        module.download_artifact(item, tmp_path / "sample.whl")
    assert not (tmp_path / "sample.whl").exists()


def test_adapter_reuses_public_callback_defaults_and_json(monkeypatch):
    from typer.testing import CliRunner
    from npa.workbench.nurec import colmap
    from npa.workflows.ncore import application

    requests = []

    def convert(request):
        requests.append(request)
        print("internal diagnostic")
        return {"status": "ok", "counts": {"images": 2}}

    monkeypatch.setattr(colmap, "convert_colmap", convert)
    result = CliRunner().invoke(
        application(),
        [
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            "s3://fixture/input/",
            "--output-path",
            "s3://fixture/output/",
            "--output-format",
            "json",
            "--no-include-downsampled-images",
            "--rig-mode",
            "preserve",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "ok", "counts": {"images": 2}}
    assert requests[0].dataset_root == "."
    assert requests[0].rig_mode == "preserve"
    assert requests[0].include_downsampled_images is False


def test_adapter_keeps_public_path_failure_redaction():
    from typer.testing import CliRunner
    from npa.workflows.ncore import application

    result = CliRunner().invoke(
        application(),
        [
            "workbench",
            "nurec",
            "convert-colmap",
            "--input-path",
            "/private-capture",
            "--output-path",
            "s3://fixture/output/",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "error": "Invalid conversion options: input_path",
    }
    assert "/private-capture" not in result.output


def test_runtime_requirements_mirror_all_locked_artifacts():
    module = runtime()
    lock = module.read_lock(PACKAGING / "runtime-lock.json")
    for phase in ("runtime", "build"):
        actual = {
            line
            for line in (PACKAGING / f"{phase}-requirements.lock")
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        }
        assert actual == {
            f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}"
            for a in lock["artifacts"]
            if a["phase"] == phase
        }


def test_runtime_exec_preserves_arguments_signals_and_credentials_without_python_overlay(
    monkeypatch, tmp_path
):
    module = runtime()
    monkeypatch.setattr(module, "ensure_runtime", lambda: tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/untrusted-overlay")
    monkeypatch.setenv("PYTHONHOME", "/untrusted-home")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-runtime-only-value")
    captured = []
    monkeypatch.setattr(module.os, "execve", lambda *args: captured.append(args))
    module.exec_runtime(
        "tools.data_converter.colmap.converter",
        ["--root-dir", "a directory", "$(literal)"],
    )
    executable, argv, env = captured[0]
    assert executable == str(tmp_path / "bin/python")
    assert argv == [
        executable,
        "-I",
        "-B",
        "-m",
        "tools.data_converter.colmap.converter",
        "--root-dir",
        "a directory",
        "$(literal)",
    ]
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["AWS_SESSION_TOKEN"] == "synthetic-runtime-only-value"


def test_source_selection_excludes_unused_sensor_simulation_and_test_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "staging", PACKAGING / "stage_upstream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path in (
        "ncore/sensors/__init__.py",
        "ncore/impl/sensors/camera.py",
        "ncore/data/v4/test.py",
    ):
        assert not module.selected(path, "ncore")


def test_unlocked_architecture_fails_before_download(monkeypatch):
    module = runtime()
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    with pytest.raises(module.NcoreRuntimeError, match="CPython 3.12 on Linux x86_64"):
        module.verify_platform()


def test_adapter_runtime_failure_keeps_json_contract_and_hides_diagnostics(
    monkeypatch, capsys
):
    from npa.workflows import ncore

    module = runtime()

    def fail():
        raise module.NcoreRuntimeError("sensitive upstream diagnostic")

    monkeypatch.setattr(module, "ensure_runtime", fail)
    monkeypatch.setattr(
        ncore.sys,
        "argv",
        ["npa", "workbench", "nurec", "convert-colmap", "--output-format", "json"],
    )
    with pytest.raises(SystemExit) as caught:
        ncore.main()
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "failed",
        "error": "NCore runtime preparation or integrity check failed",
    }
    assert not output.err and "sensitive" not in output.out


def test_atomic_relocation_keeps_console_script_record_hashes_valid(
    fake_runtime, monkeypatch
):
    import base64
    import csv
    import io

    module, pin, cache, _ = fake_runtime
    install = module.install_runtime

    def with_entrypoint(destination, lock):
        install(destination, lock)
        raw = f"#!{destination}/bin/python\nprint('entrypoint')\n".encode()
        (destination / "bin/sample").write_bytes(raw)
        record = destination / "lib/python3.12/site-packages/sample.dist-info/RECORD"
        record.parent.mkdir(parents=True)
        record.write_text(
            f"../../../bin/sample,sha256={base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')},{len(raw)}\n"
        )

    monkeypatch.setattr(module, "install_runtime", with_entrypoint)
    ready = module.ensure_runtime(lock_path=pin, cache_root=cache)
    raw = (ready / "bin/sample").read_bytes()
    assert raw.startswith(f"#!{ready}/bin/python\n".encode())
    record = ready / "lib/python3.12/site-packages/sample.dist-info/RECORD"
    [row] = list(csv.reader(io.StringIO(record.read_text())))
    assert row[1] == "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(raw).digest()
    ).decode().rstrip("=")
    assert row[2] == str(len(raw))
