"""B300 assembler integrity and inference-process routing regressions."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock
from urllib.parse import urlparse
import zipfile

import pytest

from npa.workbench.flex_pi import compiler
from npa.workbench.flex_pi.runtime import FlexPiRequest, run_inference


def _wheel(monkeypatch, *, binary=b"assembler"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(compiler.PTXAS_MEMBER, binary)
        archive.writestr(compiler.LICENSE_MEMBER, b"vendor terms")
        archive.writestr("../outside", b"must not be extracted")
    data = buffer.getvalue()
    monkeypatch.setattr(compiler, "WHEEL_BYTES", len(data))
    monkeypatch.setattr(compiler, "WHEEL_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(compiler, "PTXAS_SHA256", hashlib.sha256(binary).hexdigest())
    response = io.BytesIO(data)
    response.status = 200
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(
        compiler.http.client, "HTTPSConnection", Mock(return_value=connection)
    )
    return data


def test_compiler_checks_bytes_and_extracts_only_exact_members(tmp_path, monkeypatch):
    _wheel(monkeypatch)
    directory = tmp_path / "compiler"
    environment, receipt = compiler.prepare_b300_compiler(directory)
    assert environment == {
        "TRITON_PTXAS-BLACKWELL_PATH": str(directory / "ptxas"),
        "TRITON_PTXAS_PATH": str(directory / "ptxas"),
    }
    assert (directory / "ptxas").read_bytes() == b"assembler"
    assert (directory / "ptxas").stat().st_mode & 0o777 == 0o700
    assert (directory / "License.txt").read_bytes() == b"vendor terms"
    assert not (tmp_path / "outside").exists()
    assert not (directory / "compiler.whl").exists()
    assert receipt["ptxas_sha256"] == hashlib.sha256(b"assembler").hexdigest()
    compiler.http.client.HTTPSConnection.assert_called_once_with(
        "files.pythonhosted.org", timeout=60
    )
    connection = compiler.http.client.HTTPSConnection.return_value
    connection.request.assert_called_once_with("GET", urlparse(compiler.WHEEL_URL).path)
    connection.close.assert_called_once_with()


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/compiler.whl",
        "http://files.pythonhosted.org/compiler.whl",
        "https://example.org/compiler.whl",
        "https://user@files.pythonhosted.org/compiler.whl",
        "https://files.pythonhosted.org/compiler.whl?download=1",
        "https://files.pythonhosted.org/compiler.whl#fragment",
    ],
)
def test_compiler_rejects_other_download_origins(tmp_path, monkeypatch, url):
    _wheel(monkeypatch)
    monkeypatch.setattr(compiler, "WHEEL_URL", url)
    with pytest.raises(ValueError, match="pinned HTTPS origin"):
        compiler.prepare_b300_compiler(tmp_path / "compiler")
    compiler.http.client.HTTPSConnection.assert_not_called()


@pytest.mark.parametrize("status", [301, 302, 307, 404, 500])
def test_compiler_rejects_redirects_and_http_errors(tmp_path, monkeypatch, status):
    _wheel(monkeypatch)
    connection = compiler.http.client.HTTPSConnection.return_value
    connection.getresponse.return_value.status = status
    with pytest.raises(OSError, match=f"HTTP {status}"):
        compiler.prepare_b300_compiler(tmp_path / "compiler")
    connection.request.assert_called_once()
    connection.close.assert_called_once_with()
    assert not (tmp_path / "compiler/ptxas").exists()


@pytest.mark.parametrize("mismatch", ["size", "wheel_hash", "binary_hash"])
def test_compiler_rejects_unverified_bytes_before_execution(
    tmp_path, monkeypatch, mismatch
):
    data = _wheel(monkeypatch)
    if mismatch == "size":
        monkeypatch.setattr(compiler, "WHEEL_BYTES", len(data) - 1)
    elif mismatch == "wheel_hash":
        monkeypatch.setattr(compiler, "WHEEL_SHA256", "0" * 64)
    else:
        monkeypatch.setattr(compiler, "PTXAS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="pinned size|verification failed"):
        compiler.prepare_b300_compiler(tmp_path / "compiler")
    assert not (tmp_path / "compiler/ptxas").exists()


@pytest.mark.parametrize(
    ("gpu", "compiled", "dry_run", "fetches"),
    [
        ("B300", True, False, 1),
        ("B300", False, False, 0),
        ("B200", True, False, 0),
        ("B300", True, True, 0),
    ],
)
def test_only_real_compiled_b300_inference_fetches_and_records_compiler(
    tmp_path, monkeypatch, gpu, compiled, dry_run, fetches
):
    calls = []
    receipt = {"version": "12.9.86", "runtime_fetch": True}

    def prepare(directory, *, base_python):
        calls.append(directory)
        return (
            "/run-owned/python",
            {"TRITON_PTXAS-BLACKWELL_PATH": "/run-owned/ptxas"},
            receipt,
        )

    monkeypatch.setattr(compiler, "prepare_b300_runtime", prepare)
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "npa.flex_pi.public_observation.v1",
                "dataset": {
                    "id": "flex-pi/robotwin_3d",
                    "revision": "bee164afe94041d8c3d7dd1203725c41163fc3f4",
                },
                "observation": {"episode": 0, "rgb": {"a": {}, "b": {}, "c": {}}},
            }
        )
    )

    def runner(argv, **kwargs):
        if fetches:
            assert argv[0] == "/run-owned/python"
        assert kwargs["env"].get("TRITON_PTXAS-BLACKWELL_PATH") == (
            "/run-owned/ptxas" if fetches else None
        )
        path = Path(argv[argv.index("--output-json") + 1])
        path.write_text(
            json.dumps(
                {
                    "regime": "action-only",
                    "actions": [[0.0] * 14 for _ in range(32)],
                    "metrics": {"inference_seconds": 0.1},
                    "runtime": {"cuda": True, "gpu_name": "NVIDIA " + gpu},
                }
            )
        )
        return subprocess.CompletedProcess(
            argv, 0, stdout="FLEX_PI_REAL_INFERENCE_PASSED"
        )

    result = run_inference(
        FlexPiRequest(
            input_path=str(manifest),
            output_path=str(tmp_path / "output"),
            expected_gpu=gpu,
            torch_compile=compiled,
            dry_run=dry_run,
        ),
        runner=runner,
    )
    assert len(calls) == fetches
    assert result.get("compiler") == (receipt if fetches else None)


def test_private_runtime_is_hash_locked_and_probed_before_return(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_FLEX_PI_TOKEN", "private-service-token")
    monkeypatch.setattr(
        compiler,
        "prepare_b300_compiler",
        lambda directory: ({"TRITON_PTXAS_PATH": "/private/ptxas"}, {}),
    )
    calls = []

    def run(argv, **kwargs):
        assert "NPA_FLEX_PI_TOKEN" not in kwargs["env"]
        assert kwargs["env"]["TRITON_PTXAS_PATH"] == "/private/ptxas"
        calls.append(argv)
        output = (
            json.dumps({"compiled_kernel_preflight": "passed"})
            if len(calls) == 4
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout=output)

    monkeypatch.setattr(compiler.subprocess, "run", run)
    python, environment, receipt = compiler.prepare_b300_runtime(
        tmp_path / "runtime", base_python="/vendor/python"
    )
    assert python == str(tmp_path / "runtime/vendor/bin/python")
    assert calls[1][:4] == ["/vendor/python", "-m", "venv", "--system-site-packages"]
    assert {"--no-deps", "--no-index", "--ignore-installed", "--require-hashes"} <= set(
        calls[2]
    )
    assert calls[3] == [python, "-c", compiler._COMPILED_KERNEL_PREFLIGHT]
    assert environment["TRITON_PTXAS_PATH"] == "/private/ptxas"
    assert Path(environment["TRITON_CACHE_DIR"]).is_relative_to(tmp_path / "runtime")
    assert Path(environment["TORCHINDUCTOR_CACHE_DIR"]).is_relative_to(
        tmp_path / "runtime"
    )
    lock = (
        Path(compiler.__file__).with_name("b300-runtime-requirements.txt").read_bytes()
    )
    assert (
        receipt["vendor_runtime"]["requirements_sha256"]
        == hashlib.sha256(lock).hexdigest()
    )


def test_private_runtime_stops_at_unverified_install(tmp_path, monkeypatch):
    monkeypatch.setattr(compiler, "prepare_b300_compiler", lambda directory: ({}, {}))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        failed = "install" in argv
        return subprocess.CompletedProcess(
            argv, int(failed), stdout="hash mismatch" if failed else ""
        )

    monkeypatch.setattr(compiler.subprocess, "run", run)
    with pytest.raises(ValueError, match="hash mismatch"):
        compiler.prepare_b300_runtime(
            tmp_path / "runtime", base_python="/vendor/python"
        )
    assert len(calls) == 3


def test_b300_runtime_lock_has_only_hash_pinned_official_wheels():
    lock = (
        Path(compiler.__file__).with_name("b300-runtime-requirements.txt").read_text()
    )
    lines = [line for line in lock.splitlines() if line and not line.startswith("#")]
    assert len(lines) == 19
    for line in lines:
        name, separator, artifact = line.partition(" @ ")
        assert name and separator
        url, separator, digest = artifact.partition(" --hash=sha256:")
        assert (
            separator
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest)
        )
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname in {"download.pytorch.org", "files.pythonhosted.org"}
        assert parsed.path.endswith(".whl") and not parsed.query
