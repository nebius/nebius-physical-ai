"""Exercise the immutable, allowlisted Habitat-Sim source projection."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
MANIFEST = json.loads((PACKAGE / "source-manifest.json").read_text())
SPEC = importlib.util.spec_from_file_location(
    "habitat_source_preparer", PACKAGE / "prepare_source.py"
)
assert SPEC is not None and SPEC.loader is not None
PREPARER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARER)


def _archive(path: Path, entries: list[tuple[str, bytes, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, kind, link in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = link
            member.size = len(payload) if kind == tarfile.REGTYPE else 0
            bundle.addfile(member, io.BytesIO(payload) if member.isfile() else None)


def test_source_manifest_pins_every_official_archive_and_projection() -> None:
    source = MANIFEST["source"]
    assert source["repository"] == "https://github.com/facebookresearch/habitat-sim"
    assert source["revision"] == "57ee4941dc4765240f0f91f70b2c97a919bf9038"
    rows = [source, *MANIFEST["dependencies"]]
    assert len(rows) == 14
    for row in rows:
        assert row["archive_url"].startswith("https://codeload.github.com/")
        assert row["archive_bytes"] > 0
        assert len(row["archive_sha256"]) == 64
        assert row["license"] and len(row["license_sha256"]) == 64
    assert MANIFEST["expected_projection"] == {
        "file_count": 7909,
        "inventory_sha256": (
            "00b3daad92208771e8a42f520e8340483ef2928bb46dbd2a13a428905200ba8f"
        ),
    }


def test_source_preparer_refuses_redirect_and_removes_partial(
    tmp_path, monkeypatch
) -> None:
    payload = b"immutable source bytes"
    item = {
        "name": "fixture",
        "archive_url": "https://codeload.github.com/example/project/tar.gz/" + "a" * 40,
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }

    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "https://example.invalid/redirect"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    with pytest.raises(PREPARER.SourceError, match="redirected"):
        PREPARER._download(item, tmp_path, opener=lambda *_a, **_k: Response(payload))
    assert list(tmp_path.iterdir()) == []


def test_source_download_refuses_non_https_before_opening(tmp_path) -> None:
    item = {
        "archive_url": "http://codeload.github.com/example/project/tar.gz/" + "a" * 40,
        "archive_bytes": 0,
        "archive_sha256": hashlib.sha256(b"").hexdigest(),
    }
    opened = False

    def opener(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("insecure URL must be rejected before opening")

    with pytest.raises(PREPARER.SourceError, match="official GitHub codeload HTTPS"):
        PREPARER._download(item, tmp_path, opener=opener)
    assert opened is False


def test_source_extraction_skips_only_exact_declared_links(tmp_path) -> None:
    archive = tmp_path / "source.tar.gz"
    entries = [
        ("source-root", b"", tarfile.DIRTYPE, ""),
        ("source-root/LICENSE", b"license", tarfile.REGTYPE, ""),
        ("source-root/ignored-link", b"", tarfile.SYMTYPE, "target"),
    ]
    _archive(archive, entries)
    PREPARER._extract(archive, tmp_path / "accepted", ["ignored-link"])
    assert (tmp_path / "accepted/LICENSE").read_bytes() == b"license"
    assert not (tmp_path / "accepted/ignored-link").exists()
    with pytest.raises(PREPARER.SourceError, match="undeclared source archive link"):
        PREPARER._extract(archive, tmp_path / "rejected")


def test_internal_vendored_closure_refuses_changed_or_extra_blob(tmp_path) -> None:
    root = tmp_path / "source"
    selected = root / "src/deps/internal"
    selected.mkdir(parents=True)
    blob = selected / "exact.h"
    blob.write_bytes(b"exact")
    component = {
        "name": "internal",
        "root": "src/deps/internal",
        "files": [{"path": "exact.h", "sha256": hashlib.sha256(b"exact").hexdigest()}],
    }
    PREPARER._verify_internal_vendored(root, [component])
    (selected / "undeclared.h").write_bytes(b"extra")
    with pytest.raises(PREPARER.SourceError, match="vendored source closure changed"):
        PREPARER._verify_internal_vendored(root, [component])


def test_excluded_noncommercial_and_gui_gitlinks_are_never_fetched() -> None:
    excluded = {row["name"] for row in MANIFEST["excluded_gitlinks"]}
    selected = {row["name"] for row in MANIFEST["dependencies"]}
    script = (PACKAGE / "prepare_source.py").read_text()
    assert excluded == {"glfw", "rlr-audio-propagation"}
    assert not excluded & selected
    assert "git submodule" not in script and "--recursive" not in script
    assert "data/scene_datasets" in MANIFEST["forbidden_paths"]


def test_final_source_projection_preserves_the_manifest_directory_layout() -> None:
    dockerfile = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
    projection_copy = dockerfile.split("cp -a /opt/habitat-sim/src/CMakeLists.txt", 1)[
        1
    ].split("cp -a /opt/habitat-sim/src_python", 1)[0]
    assert "/opt/habitat-sim/src/esp" in projection_copy
    assert "/opt/habitat-sim/src/shaders" in projection_copy
    assert "/opt/habitat-sim/src/utils" in projection_copy
    assert "/opt/source-projection/src/" in projection_copy
    assert "/opt/habitat-sim/src/deps /opt/source-projection/src/" in dockerfile
    assert "/opt/habitat-sim/src_python /opt/source-projection/" in dockerfile
    assert "/opt/habitat-sim/src/esp /opt/habitat-sim/src_python" not in dockerfile
    assert set(MANIFEST["source"]["final_projection"]) == {
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "data/default.physics_config.json",
        "src/CMakeLists.txt",
        "src/cmake",
        "src/deps",
        "src/esp",
        "src/shaders",
        "src/utils",
        "src_python",
    }
