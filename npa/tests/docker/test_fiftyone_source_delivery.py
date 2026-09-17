"""Test MongoDB source integrity, executable identity, and recipient delivery."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/fiftyone"
SPEC = importlib.util.spec_from_file_location("fiftyone_source_delivery", IMAGE / "verify_source.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
SOURCE_FILES = (
    "LICENSE-Community.txt",
    "README.md",
    "SConstruct",
    "buildscripts/scons.py",
    "docs/building.md",
    "etc/pip/compile-requirements.txt",
    "src/mongo/db/mongod_main.cpp",
    "src/third_party/wiredtiger/SConscript",
)


def _archive(path: Path, manifest: dict, *, omitted: str = "") -> None:
    prefix = f"mongo-{manifest['source']['git_revision']}/"
    with tarfile.open(path, "w:gz") as archive:
        for name in SOURCE_FILES:
            if name == omitted:
                continue
            data = b"source fixture\n"
            entry = tarfile.TarInfo(prefix + name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    manifest["source"]["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["source"]["archive_bytes"] = path.stat().st_size


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    manifest = json.loads((IMAGE / "mongodb-source.json").read_text())
    notices = tmp_path / "notices"
    notices.mkdir()
    mongod = tmp_path / "bin/mongod"
    mongod.parent.mkdir()
    mongod.write_bytes(b"executable fixture")
    manifest["binary"]["file_sha256"] = hashlib.sha256(mongod.read_bytes()).hexdigest()
    manifest["binary"]["file_bytes"] = mongod.stat().st_size
    _archive(notices / "mongodb-source.tar.gz", manifest)
    (notices / "source.json").write_text(json.dumps(manifest))
    (notices / "version.json").write_bytes((IMAGE / "mongodb-source-version.json").read_bytes())
    for name in VERIFY._NOTICES:
        (notices / name).write_text("recipient notice\n")
    (mongod.parent / "MONGODB_SOURCE.md").write_bytes((notices / "SOURCE.md").read_bytes())
    build = {
        "version": manifest["version"],
        "gitVersion": manifest["binary"]["git_revision"],
        "modules": [],
        "environment": {"distmod": "ubuntu2204", "distarch": "x86_64"},
    }

    def version_command(argv, *, text):
        assert argv == [str(mongod), "--version"] and text
        return "db version v7.0.40\nBuild Info: " + json.dumps(build)

    monkeypatch.setattr(VERIFY.subprocess, "check_output", version_command)
    return notices, mongod, manifest, build


def test_complete_delivery_checks_executable_archive_and_readable_directions(delivery, monkeypatch, capsys):
    notices, mongod, _, _ = delivery
    monkeypatch.setattr("sys.argv", ["verify_source", "--mongod", str(mongod), "--notices-dir", str(notices)])
    assert VERIFY.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_delivery"] == "verified"
    assert report["source_members"] == len(SOURCE_FILES)


@pytest.mark.parametrize("kind", ["size", "checksum"])
def test_source_archive_tampering_fails_before_archive_read(delivery, kind):
    notices, _, manifest, _ = delivery
    source = copy.deepcopy(manifest["source"])
    if kind == "size":
        source["archive_bytes"] += 1
    else:
        source["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=kind):
        VERIFY._verify_archive(notices / "mongodb-source.tar.gz", source)


@pytest.mark.parametrize("required", SOURCE_FILES)
def test_missing_build_or_license_file_is_not_complete_source(delivery, required):
    notices, _, manifest, _ = delivery
    path = notices / "mongodb-source.tar.gz"
    _archive(path, manifest, omitted=required)
    with pytest.raises(ValueError, match="lacks required file"):
        VERIFY._verify_archive(path, manifest["source"])


@pytest.mark.parametrize("field,value,error", [
    ("version", "0.0.0", "version differs"),
    ("gitVersion", "0" * 40, "revision differs"),
    ("modules", ["unexpected"], "undeclared modules"),
    ("environment", {"distmod": "other", "distarch": "x86_64"}, "platform differs"),
])
def test_binary_identity_must_match_delivered_source(delivery, field, value, error):
    _, mongod, manifest, build = delivery
    build[field] = value
    with pytest.raises(ValueError, match=error):
        VERIFY._verify_binary(mongod, manifest)


def test_public_origin_must_match_binary_even_when_binary_manifest_matches(delivery):
    _, mongod, manifest, _ = delivery
    manifest["source"]["git_origin_revision"] = "0" * 40
    with pytest.raises(ValueError, match="public source origin differs"):
        VERIFY._verify_binary(mongod, manifest)


def test_changed_executable_is_rejected_before_execution(delivery, monkeypatch):
    _, mongod, manifest, _ = delivery
    mongod.write_bytes(b"a changed executable")
    calls = []
    monkeypatch.setattr(VERIFY.subprocess, "check_output", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="executable checksum differs"):
        VERIFY._verify_binary(mongod, manifest)
    assert calls == []


@pytest.mark.parametrize("name", VERIFY._NOTICES)
def test_missing_recipient_notice_fails(delivery, name):
    notices, mongod, manifest, _ = delivery
    (notices / name).unlink()
    with pytest.raises(FileNotFoundError):
        VERIFY._verify_delivery(notices, mongod, manifest)


def test_source_directions_must_travel_beside_the_executable(delivery):
    notices, mongod, manifest, _ = delivery
    (mongod.parent / "MONGODB_SOURCE.md").write_text("stale directions")
    with pytest.raises(ValueError, match="directions are missing beside"):
        VERIFY._verify_delivery(notices, mongod, manifest)


def test_source_build_metadata_identifies_public_commit(delivery):
    notices, mongod, manifest, _ = delivery
    (notices / "version.json").write_text(json.dumps({"version": "7.0.40", "githash": "0" * 40}))
    with pytest.raises(ValueError, match="source-build version metadata is inconsistent"):
        VERIFY._verify_delivery(notices, mongod, manifest)


def test_dockerfile_delivers_source_in_same_image_and_keeps_binary_pin():
    text = (IMAGE / "Dockerfile").read_text()
    manifest = json.loads((IMAGE / "mongodb-source.json").read_text())
    assert f"ARG MONGODB_VERSION={manifest['version']}" in text
    assert f"ARG MONGODB_SHA256={manifest['binary']['archive_sha256']}" in text
    for filename in ("mongodb-source.json", "mongodb-source-version.json", "SOURCE.md", "verify_source.py"):
        assert f"docker/workbench/fiftyone/{filename}" in text
    assert '-o /opt/fiftyone/mongodb-notices/mongodb-source.tar.gz "$SOURCE_URL"' in text
    assert 'cp /opt/fiftyone/mongodb-notices/SOURCE.md "$DBBIN/MONGODB_SOURCE.md"' in text
    assert '--mongod "$DBBIN/mongod" --notices-dir /opt/fiftyone/mongodb-notices' in text
    assert manifest["source"]["url"].endswith(manifest["source"]["git_revision"])
    assert manifest["source"]["git_origin_revision"] == manifest["binary"]["git_revision"]
