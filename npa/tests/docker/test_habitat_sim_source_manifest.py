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
PBR_CONFIG = b"""group = pbr-images

[file]
filename = bluts/brdflut_ldr_512x512.png

[file]
filename = env_maps/anniversary_lounge_1k.hdr

[file]
filename = env_maps/autoshop_01_1k.hdr

[file]
filename = env_maps/brown_photostudio_02_1k.hdr

[file]
filename = env_maps/lythwood_room_1k.hdr

[file]
filename = env_maps/blue_photo_studio_1k.hdr
"""


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
    assert len(rows) == 15
    for row in rows:
        assert row["archive_url"].startswith("https://codeload.github.com/")
        assert row["archive_bytes"] > 0
        assert len(row["archive_sha256"]) == 64
        assert row["license"] and len(row["license_sha256"]) == 64
    assert MANIFEST["expected_projection"] == {
        "file_count": 8135,
        "inventory_sha256": (
            "ea722cdf72a64b0d4cc0b329007dd5a17daabb7565605c07d9550b715ebd0501"
        ),
    }


def test_required_pbr_configuration_is_bound_to_the_official_source() -> None:
    assert len(PBR_CONFIG) == 327
    assert hashlib.sha256(PBR_CONFIG).hexdigest() == (
        "0fedbc71e140aca2fb286b0582beb57106990f997b7a5647690a85d4bd106fce"
    )
    assert MANIFEST["source"]["required_projection_files"] == [
        {
            "path": "data/pbr/PbrImages.conf",
            "bytes": 327,
            "sha256": (
                "0fedbc71e140aca2fb286b0582beb57106990f997b7a5647690a85d4bd106fce"
            ),
        }
    ]


def test_required_pbr_configuration_refuses_omission_or_wrong_content(
    tmp_path: Path,
) -> None:
    required = MANIFEST["source"]["required_projection_files"]
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(PREPARER.SourceError, match="PbrImages.conf"):
        PREPARER._verify_required_source_files(root, required)

    config = root / "data/pbr/PbrImages.conf"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"x" * len(PBR_CONFIG))
    with pytest.raises(PREPARER.SourceError, match="PbrImages.conf"):
        PREPARER._verify_required_source_files(root, required)

    config.write_bytes(PBR_CONFIG)
    PREPARER._verify_required_source_files(root, required)


def test_parent_projection_keeps_only_required_pbr_configuration(tmp_path: Path) -> None:
    root = tmp_path / "source"
    for name in ("LICENSE", "MANIFEST.in", "README.md", "pyproject.toml", "setup.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"fixture")
    for name in ("src/deps/basis-universal/transcoder", "src_python"):
        (root / name).mkdir(parents=True)
    data = root / "data"
    (data / "pbr/env_maps").mkdir(parents=True)
    (data / "default.physics_config.json").write_bytes(b"{}")
    (data / "pbr/PbrImages.conf").write_bytes(PBR_CONFIG)
    (data / "pbr/env_maps/forbidden.hdr").write_bytes(b"not projected")
    (data / "scene_datasets").mkdir()

    PREPARER._prune_parent(root)

    assert (data / "pbr/PbrImages.conf").read_bytes() == PBR_CONFIG
    assert not (data / "pbr/env_maps").exists()
    assert not (data / "scene_datasets").exists()


def test_openexr_imath_fetchcontent_source_is_exact_and_projected() -> None:
    imath = next(row for row in MANIFEST["dependencies"] if row["name"] == "imath")
    assert imath == {
        "name": "imath",
        "repository": "https://github.com/AcademySoftwareFoundation/Imath",
        "revision": "d690a3fcff4e877ead5ae56c7e964595ade8a35e",
        "upstream_tag": "v3.1.9",
        "archive_url": (
            "https://codeload.github.com/AcademySoftwareFoundation/Imath/tar.gz/"
            "d690a3fcff4e877ead5ae56c7e964595ade8a35e"
        ),
        "archive_bytes": 598874,
        "archive_sha256": (
            "8655ebd702791fd7dec8b4e2aa829a0cbddbec48c84a970f09eef21d9ced8e18"
        ),
        "target": "src/deps/imath",
        "license": "BSD-3-Clause",
        "license_path": "LICENSE.md",
        "license_bytes": 1497,
        "license_sha256": (
            "c20236d3b39fd20eba8e3d1fb3b892a5483df2e7d8d61bf43f165d3fac22f601"
        ),
    }
    assert MANIFEST["dependency_projection"]["imath"] == [
        "CMakeLists.txt",
        "LICENSE.md",
        "cmake",
        "config",
        "src",
    ]


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
        "data/pbr/PbrImages.conf",
        "src/CMakeLists.txt",
        "src/cmake",
        "src/deps",
        "src/esp",
        "src/shaders",
        "src/utils",
        "src_python",
    }
    assert "mkdir -p /opt/source-projection/data/pbr" in dockerfile
    assert (
        "cp /opt/habitat-sim/data/pbr/PbrImages.conf "
        "/opt/source-projection/data/pbr/"
    ) in dockerfile
