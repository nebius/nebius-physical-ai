"""Fail-closed source integrity and native-runtime packaging contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest


PACKAGING = Path(__file__).resolve().parents[2] / "docker/workbench/ncore"


def assembler():
    spec = importlib.util.spec_from_file_location(
        "assemble_native_sources", PACKAGING / "assemble_native_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_lock(tmp_path, *, filename="source.tar.gz", payload=b"source"):
    path = tmp_path / "lock.json"
    path.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "artifacts": [
                            {
                                "filename": filename,
                                "url": "https://example.invalid/source.tar.gz",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ]
                    }
                ]
            }
        )
    )
    return path


def test_annex_reuses_verified_readonly_cache_without_network(tmp_path, monkeypatch):
    module = assembler()
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *_: pytest.fail("network")
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    artifact = cache / "source.tar.gz"
    artifact.write_bytes(b"source")
    artifact.chmod(0o444)
    lock = source_lock(tmp_path)
    output = tmp_path / "annex"
    module.assemble(lock, output, cache)
    module.assemble(lock, output, cache)
    assert (output / artifact.name).read_bytes() == artifact.read_bytes() == b"source"
    assert (output / "native-source-lock.json").read_bytes() == lock.read_bytes()


@pytest.mark.parametrize("corrupt_existing", [False, True])
def test_bad_source_bytes_are_never_blessed(tmp_path, corrupt_existing):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "source.tar.gz").write_bytes(b"source" if corrupt_existing else b"bad")
    out = tmp_path / "out"
    out.mkdir()
    if corrupt_existing:
        (out / "source.tar.gz").write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assembler().assemble(source_lock(tmp_path), out, cache)
    assert not (out / "native-source-lock.json").exists()
    assert not list(out.glob("*.partial"))


@pytest.mark.parametrize("name", ["../escape", "/escape", "a/b", ".."])
def test_annex_rejects_paths_outside_destination(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe artifact name"):
        assembler().assemble(source_lock(tmp_path, filename=name), tmp_path / "out")


def test_runtime_source_pins_match_both_unchanged_requirement_closures():
    lock = json.loads((PACKAGING / "native-source-lock.json").read_text())
    sources = {
        (c["name"], c["version"]) for c in lock["components"] if c["kind"] == "sdist"
    }
    required = set()
    for name in ("npa-requirements.lock", "converter-requirements.lock"):
        required.update(
            re.findall(
                r"^(numpy|scipy|av)==(\S+)", (PACKAGING / name).read_text(), re.M
            )
        )
    assert sources == required
    (ffmpeg,) = [c for c in lock["components"] if c["name"] == "ffmpeg"]
    assert (
        ffmpeg["artifacts"][0]["url"]
        == "https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz"
    )
    assert (
        ffmpeg["artifacts"][0]["sha256"]
        == "b6863adde98898f42602017462871b5f6333e65aec803fdd7a6308639c52edf3"
    )
    assert all(
        c["name"] not in {"x264", "x265", "pyav-ffmpeg"} for c in lock["components"]
    )


def test_every_native_debian_binary_has_exact_corresponding_source():
    source = json.loads((PACKAGING / "native-source-lock.json").read_text())
    sources = {
        (c["name"], c["version"])
        for c in source["components"]
        if c["kind"] == "debian-source"
    }
    binary = json.loads((PACKAGING / "debian-native-lock.json").read_text())
    for package in binary["packages"]:
        name, _, version = package["source"].partition(" (")
        assert (name, version.rstrip(")") or package["version"]) in sources
        assert package["libraries"]
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in package["libraries"].values()
        )


def test_native_build_has_no_opaque_wheel_fallback_and_delivers_recipes():
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    builder = (PACKAGING / "build_native.py").read_text()
    ffmpeg_builder = (PACKAGING / "build_ffmpeg.py").read_text()
    for switch in (
        "--disable-gpl",
        "--disable-nonfree",
        "--disable-autodetect",
        "--enable-zlib",
    ):
        assert switch in ffmpeg_builder
    assert "--no-binary=numpy,scipy,av" in dockerfile
    assert "--no-build-isolation" in builder and '"--no-index"' in builder
    assert "gpgv --keyring" in dockerfile
    assert "COPY --from=native-builder /opt/ncore /opt/ncore" in dockerfile
    assert "COPY --from=assembled /opt/ncore /opt/ncore" in dockerfile
    assert "verify_native.py" in dockerfile
    assert "20260906T183022Z" in dockerfile
    assert "pillow==12.3.0" in (PACKAGING / "converter-requirements.in").read_text()
