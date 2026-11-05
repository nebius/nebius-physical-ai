"""Protect the Lyra bootstrap's runtime-only model and simulator boundary."""

from pathlib import Path
import subprocess

import pytest

from npa.workflows import lyra_stage

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/lyra2"


def test_entrypoint_preserves_argument_boundaries():
    result = subprocess.run(
        ["bash", str(IMAGE / "entrypoint.sh"), "/usr/bin/printf", "%s", "a b"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "a b"


def test_dedicated_interpreter_does_not_require_isaac(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("NPA_LYRA_BASE_PYTHON", "/usr/bin/python3.12")
    monkeypatch.setattr(
        lyra_stage.subprocess, "run", lambda argv, **kw: calls.append(argv)
    )
    result = lyra_stage._torch_environment(tmp_path)
    assert calls[0][0] == "/usr/bin/python3.12"
    assert result == str(tmp_path / "environment/bin/python")
    assert not any("isaac" in item for argv in calls for item in argv)


def test_checkpoint_mismatch_is_rejected_before_inference(monkeypatch, tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, _size):
            yield b"not-the-native-checkpoint"

    monkeypatch.setattr(lyra_stage.requests, "get", lambda *args, **kw: Response())
    with pytest.raises(ValueError, match="upstream LFS SHA-256"):
        lyra_stage._fetch_checkpoint(tmp_path / "model.pt")


def test_image_repairs_the_header_and_every_shipped_python_runtime():
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    bootstrap_lock = (IMAGE / "bootstrap-packages.lock").read_text(encoding="utf-8")

    assert (
        "fec8e4a308c03cd98dace67370a8d5a8031c36741796065c9f6e3f9548ab0cd2 "
        "https://security.ubuntu.com/ubuntu/pool/main/l/linux/"
        "linux-libc-dev_6.8.0-147.147_amd64.deb"
    ) in bootstrap_lock
    assert "python3.12 -m pip install --no-cache-dir --no-deps --upgrade" in dockerfile
    assert "certifi-2026.10.23-py3-none-any.whl" in dockerfile
    assert "version('certifi') == '2026.10.23'" in dockerfile
