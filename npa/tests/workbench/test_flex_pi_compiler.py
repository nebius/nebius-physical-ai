"""B300 assembler integrity and inference-process routing regressions."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
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
    monkeypatch.setattr(
        compiler.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data)
    )
    return data


def test_compiler_checks_bytes_and_extracts_only_exact_members(tmp_path, monkeypatch):
    _wheel(monkeypatch)
    directory = tmp_path / "compiler"
    environment, receipt = compiler.prepare_b300_compiler(directory)
    assert environment == {"TRITON_PTXAS-BLACKWELL_PATH": str(directory / "ptxas")}
    assert (directory / "ptxas").read_bytes() == b"assembler"
    assert (directory / "ptxas").stat().st_mode & 0o777 == 0o700
    assert (directory / "License.txt").read_bytes() == b"vendor terms"
    assert not (tmp_path / "outside").exists()
    assert not (directory / "compiler.whl").exists()
    assert receipt["ptxas_sha256"] == hashlib.sha256(b"assembler").hexdigest()


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

    def prepare(directory):
        calls.append(directory)
        return {"TRITON_PTXAS-BLACKWELL_PATH": "/run-owned/ptxas"}, receipt

    monkeypatch.setattr(compiler, "prepare_b300_compiler", prepare)
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
