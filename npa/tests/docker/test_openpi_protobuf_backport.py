"""Refuse unreviewed vendor bytes and preserve truthful installed-file metadata."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2] / "docker/workbench/openpi"


def module():
    spec = importlib.util.spec_from_file_location(
        "protobuf_backport", ROOT / "backport_protobuf_json.py"
    )
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_unreviewed_source_is_unchanged(tmp_path: Path) -> None:
    patcher = module()
    source = tmp_path / "json_format.py"
    source.write_bytes(patcher.ORIGINAL)
    with pytest.raises(RuntimeError, match="reviewed upstream bytes"):
        patcher.apply_backport(source)
    assert source.read_bytes() == patcher.ORIGINAL


def test_patch_refreshes_source_bytecode_and_record_without_changing_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = module()
    source = tmp_path / "google/protobuf/json_format.py"
    source.parent.mkdir(parents=True)
    original = (
        b"def convert(self, value, sub_message, path):\n    if True:\n"
        + patcher.ORIGINAL
        + b"\n"
    )
    patched = original.replace(patcher.ORIGINAL, patcher.PATCHED)
    source.write_bytes(original)
    monkeypatch.setattr(
        patcher, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest()
    )
    monkeypatch.setattr(patcher, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    record_path = Path("protobuf-4.25.8.dist-info/RECORD")
    record = tmp_path / record_path
    record.parent.mkdir()
    license_file = record.parent / "LICENSE"
    license_file.write_text("synthetic license fixture\n")
    license_bytes = license_file.read_bytes()
    record.write_text(
        "google/protobuf/json_format.py,,\nprotobuf-4.25.8.dist-info/LICENSE,,\nprotobuf-4.25.8.dist-info/RECORD,,\n"
    )
    distribution = SimpleNamespace(
        version="4.25.8", files=[record_path], locate_file=lambda path: tmp_path / path
    )
    monkeypatch.setattr(patcher.metadata, "distribution", lambda _: distribution)
    patcher.main()
    assert source.read_bytes() == patched
    assert license_file.read_bytes() == license_bytes
    entries = list(csv.reader(record.read_text().splitlines()))
    tracked = {row[0]: row[1:] for row in entries}
    compiled = next(source.parent.glob("__pycache__/json_format.*.pyc"))
    assert int.from_bytes(compiled.read_bytes()[4:8], "little") == 3
    for path in (source, compiled):
        data = path.read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        assert tracked[path.relative_to(tmp_path).as_posix()] == [
            "sha256=" + digest,
            str(len(data)),
        ]
    assert tracked[record_path.as_posix()] == ["", ""]
    previous = record.read_bytes()
    patcher.main()
    assert record.read_bytes() == previous


def test_docker_applies_and_exercises_backport_before_model_setup() -> None:
    docker = (ROOT / "Dockerfile").read_text()
    patch = docker.index("/opt/venv/bin/python /opt/npa-openpi-backport-protobuf.py")
    verify = docker.index("/opt/venv/bin/python /opt/npa-openpi-verify-protobuf.py")
    assert docker.index("uv sync") < patch < verify < docker.index("'jax==0.6.2'")
    patcher = module()
    assert (
        patcher.ORIGINAL_SHA256
        == "01795eef8361486af4a29f0df8eace5e82f42d0fc286c2e4c6249bc31405a339"
    )
    assert (
        patcher.PATCHED_SHA256
        == "07213c2aa14ca29ce3a8bc31c0aff9f38855ce794f03dc387b56600911ff1c38"
    )
