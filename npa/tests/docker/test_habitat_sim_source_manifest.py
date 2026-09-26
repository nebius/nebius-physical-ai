"""Exercise the immutable, allowlisted Habitat-Sim source projection."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

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


def _staging_fixture(monkeypatch, output: Path) -> dict:
    payload = b"verified inert source\n"
    inventory = {
        "schema_version": "npa.habitat-sim.source-projection.v1",
        "file_count": 1,
        "files": [
            {
                "path": "source.txt",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    encoded = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()

    def extract(_archive, staged, _excluded):
        assert not os.path.lexists(output)
        staged.mkdir()
        (staged / "source.txt").write_bytes(payload)

    monkeypatch.setattr(PREPARER, "_download", lambda _item, temp: temp / "inert")
    monkeypatch.setattr(PREPARER, "_extract", extract)
    for name in (
        "_verify_license",
        "_apply_metadata_patches",
        "_verify_required_source_files",
        "_prune_parent",
        "_verify_internal_vendored",
    ):
        monkeypatch.setattr(PREPARER, name, lambda *_args: None)
    return _staging_manifest(encoded)


def _staging_manifest(encoded: bytes) -> dict:
    return {
        "schema_version": "npa.habitat-sim.source-manifest.v1",
        "source": {
            "license_path": "LICENSE",
            "metadata_patches": [],
            "required_projection_files": [],
            "final_projection": ["source.txt"],
        },
        "internal_vendored": [],
        "dependencies": [],
        "dependency_projection": {},
        "forbidden_paths": [],
        "expected_projection": {
            "file_count": 1,
            "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        },
    }


def _forbid_source_effects(monkeypatch):
    effects = []
    for owner, names in (
        (PREPARER, ("_owned_destination", "_download", "_extract")),
        (PREPARER.tempfile, ("mkdtemp",)),
        (PREPARER.shutil, ("rmtree",)),
        (PREPARER.os, ("open",)),
    ):
        for name in names:
            effect = Mock(side_effect=AssertionError("unexpected source side effect"))
            monkeypatch.setattr(owner, name, effect)
            effects.append(effect)
    return effects


def _replace_record(manifest, location, value):
    record = manifest
    for key in location[:-1]:
        record = record[key]
    record[location[-1]] = value


_PATH_LOCATIONS = [
    ("source", "license_path"),
    ("source", "metadata_patches", 0, "path"),
    ("source", "required_projection_files", 0, "path"),
    ("source", "final_projection", 0),
    ("dependencies", 0, "name"),
    ("dependencies", 0, "target"),
    ("dependencies", 0, "license_path"),
    ("dependency_projection", "assimp", 0),
    ("internal_vendored", 0, "root"),
    ("internal_vendored", 0, "files", 0, "path"),
    ("forbidden_paths", 0),
]


@pytest.mark.parametrize("location", _PATH_LOCATIONS)
@pytest.mark.parametrize(
    "value",
    [
        "../outside",
        "/outside",
        "a/../b",
        "a//b",
        "./a",
        "a/",
        "a\\b",
        "",
        "a\x00b",
        None,
    ],
)
def test_manifest_paths_refuse_before_any_effect(monkeypatch, location, value):
    manifest = copy.deepcopy(MANIFEST)
    _replace_record(manifest, location, value)
    effects = _forbid_source_effects(monkeypatch)
    with pytest.raises(PREPARER.SourceError):
        PREPARER.stage(manifest, Path("unused-output"))
    assert all(effect.call_count == 0 for effect in effects)


@pytest.mark.parametrize("section", ["source", "dependencies"])
def test_excluded_link_paths_refuse_before_any_effect(monkeypatch, section):
    manifest = copy.deepcopy(MANIFEST)
    item = manifest[section][0] if section == "dependencies" else manifest[section]
    item["excluded_archive_links"] = ["../outside"]
    effects = _forbid_source_effects(monkeypatch)
    with pytest.raises(PREPARER.SourceError):
        PREPARER.stage(manifest, Path("unused-output"))
    assert all(effect.call_count == 0 for effect in effects)


@pytest.mark.parametrize("target", ["src", "src/deps", "src/other/item", "source/item"])
def test_dependency_target_requires_strict_deps_descendant(monkeypatch, target):
    manifest = copy.deepcopy(MANIFEST)
    manifest["dependencies"][0]["target"] = target
    effects = _forbid_source_effects(monkeypatch)
    with pytest.raises(PREPARER.SourceError, match="src/deps descendant"):
        PREPARER.stage(manifest, Path("unused-output"))
    assert all(effect.call_count == 0 for effect in effects)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "src/deps/assimp"),
        ("target", "src/deps/assimp/nested"),
        ("name", "assimp"),
        ("name", "habitat-sim"),
    ],
)
def test_dependency_aliases_refuse_before_any_effect(monkeypatch, field, value):
    manifest = copy.deepcopy(MANIFEST)
    manifest["dependencies"][1][field] = value
    effects = _forbid_source_effects(monkeypatch)
    with pytest.raises(PREPARER.SourceError, match="must not overlap"):
        PREPARER.stage(manifest, Path("unused-output"))
    assert all(effect.call_count == 0 for effect in effects)


def test_checked_in_manifest_paths_are_canonical_without_side_effects(monkeypatch):
    before = copy.deepcopy(MANIFEST)
    effects = _forbid_source_effects(monkeypatch)
    PREPARER._validate_manifest_paths(MANIFEST)
    assert MANIFEST == before
    assert all(effect.call_count == 0 for effect in effects)


@pytest.mark.parametrize("boundary", ["symlink", "foreign-owner", "shared-write"])
def test_dependency_containment_refuses_mocked_boundary_before_effects(
    monkeypatch, boundary
):
    root = Path("inert-staging")
    uid = os.getuid()

    def lstat(path):
        mode, owner = stat.S_IFDIR | 0o700, uid
        if path == root / "src":
            if boundary == "symlink":
                mode = stat.S_IFLNK | 0o700
            elif boundary == "foreign-owner":
                owner += 1
            else:
                mode |= 0o020
        return SimpleNamespace(st_mode=mode, st_uid=owner)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    effects = _forbid_source_effects(monkeypatch)
    with pytest.raises(PREPARER.SourceError, match="owned and not shared or linked"):
        PREPARER._materialize_dependency(
            MANIFEST["dependencies"][0], root, Path("unused"), []
        )
    assert all(effect.call_count == 0 for effect in effects)


def _mock_staging_metadata(monkeypatch, events):
    def lstat(path):
        events.append(("lstat", path))
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=os.getuid())

    def exists(path):
        events.append(("exists", path))
        return True

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    monkeypatch.setattr(Path, "exists", exists)


def _mock_dependency_effects(monkeypatch, events, target):
    def download(*_args):
        events.append(("download", target))
        return Path("inert-archive")

    monkeypatch.setattr(PREPARER, "_download", download)
    monkeypatch.setattr(
        PREPARER.shutil, "rmtree", lambda path: events.append(("remove", path))
    )
    monkeypatch.setattr(
        PREPARER,
        "_extract",
        lambda _archive, path, _links: events.append(("extract", path)),
    )
    monkeypatch.setattr(PREPARER, "_verify_license", lambda *_args: None)
    monkeypatch.setattr(PREPARER, "_prune_dependency", lambda *_args: None)


def test_dependency_checks_containment_before_mocked_remove_and_extract(monkeypatch):
    events = []
    root = Path("inert-staging")
    target = root / "src/deps/assimp"
    _mock_staging_metadata(monkeypatch, events)
    _mock_dependency_effects(monkeypatch, events, target)
    PREPARER._materialize_dependency(
        MANIFEST["dependencies"][0], root, Path("unused"), []
    )
    assert events == [
        ("lstat", root),
        ("lstat", root / "src"),
        ("lstat", root / "src/deps"),
        ("lstat", target),
        ("download", target),
        ("exists", target),
        ("remove", target),
        ("extract", target),
    ]


def test_source_preparer_helpers_and_exports_follow_contribution_contract():
    module = ast.parse((PACKAGE / "prepare_source.py").read_text())
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    assert all(node.end_lineno - node.lineno + 1 < 40 for node in functions)
    for node in module.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef)
        ) and not node.name.startswith("_"):
            documentation = ast.get_docstring(node)
            assert documentation and all(
                section in documentation for section in ("Args:", "Returns:", "Raises:")
            )


def test_stage_exposes_only_complete_verified_projection(tmp_path, monkeypatch):
    output, inventory = tmp_path / "source", tmp_path / "inventory.json"
    manifest = _staging_fixture(monkeypatch, output)
    original = PREPARER._rename_noreplace

    def publish(staged, destination):
        assert not output.exists()
        assert (staged / "source.txt").read_bytes() == b"verified inert source\n"
        assert inventory.is_file()
        original(staged, destination)

    monkeypatch.setattr(PREPARER, "_rename_noreplace", publish)
    PREPARER.stage(manifest, output, inventory)
    assert json.loads(inventory.read_bytes()) == PREPARER._inventory(output)
    assert not list(tmp_path.glob(".npa-habitat-source-*"))


@pytest.mark.parametrize(
    "boundary",
    [
        "_download",
        "_extract",
        "_verify_license",
        "_apply_metadata_patches",
        "_verify_required_source_files",
        "_prune_parent",
        "_verify_internal_vendored",
        "_assert_forbidden",
        "_inventory",
    ],
)
def test_stage_failure_removes_only_owned_partial_staging(
    tmp_path, monkeypatch, boundary
):
    output = tmp_path / "source"
    manifest = _staging_fixture(monkeypatch, output)
    unrelated = tmp_path / "caller-data"
    unrelated.write_bytes(b"keep")

    def fail(*_args):
        raise PREPARER.SourceError("inert verification failure")

    monkeypatch.setattr(PREPARER, boundary, fail)
    with pytest.raises(PREPARER.SourceError, match="inert verification"):
        PREPARER.stage(manifest, output)
    assert not output.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".npa-habitat-source-*"))


def test_stage_inventory_mismatch_never_publishes(tmp_path, monkeypatch):
    output = tmp_path / "source"
    manifest = _staging_fixture(monkeypatch, output)
    manifest["expected_projection"]["inventory_sha256"] = "0" * 64
    with pytest.raises(PREPARER.SourceError, match="projection changed"):
        PREPARER.stage(manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".npa-habitat-source-*"))


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_stage_refuses_existing_or_concurrent_output(tmp_path, monkeypatch, kind):
    output, inventory = tmp_path / "source", tmp_path / "inventory.json"
    manifest = _staging_fixture(monkeypatch, output)
    original = PREPARER._rename_noreplace

    def publish(staged, destination):
        if kind == "directory":
            output.mkdir()
        elif kind == "file":
            output.write_bytes(b"caller")
        else:
            output.symlink_to(tmp_path / "absent")
        original(staged, destination)

    monkeypatch.setattr(PREPARER, "_rename_noreplace", publish)
    with pytest.raises(FileExistsError):
        PREPARER.stage(manifest, output, inventory)
    assert os.path.lexists(output)
    assert not inventory.exists()
    assert not list(tmp_path.glob(".npa-habitat-source-*"))
    with pytest.raises(PREPARER.SourceError, match="already exists"):
        PREPARER.stage(manifest, output, inventory)


@pytest.mark.parametrize("verification_failure", [True, False])
def test_stage_cleanup_failure_preserves_original_result(
    tmp_path, monkeypatch, capsys, verification_failure
):
    output = tmp_path / "source"
    manifest = _staging_fixture(monkeypatch, output)
    if verification_failure:
        manifest["expected_projection"]["file_count"] = 2

    def fail_cleanup(_path):
        raise OSError("inert cleanup refusal")

    monkeypatch.setattr(PREPARER.shutil, "rmtree", fail_cleanup)
    if verification_failure:
        with pytest.raises(PREPARER.SourceError, match="projection changed"):
            PREPARER.stage(manifest, output)
    else:
        PREPARER.stage(manifest, output)
        assert output.is_dir()
    assert "staging cleanup failed" in capsys.readouterr().err


def test_stage_preserves_existing_inventory_without_fetch(tmp_path, monkeypatch):
    output, inventory = tmp_path / "source", tmp_path / "inventory.json"
    manifest = _staging_fixture(monkeypatch, output)
    inventory.write_bytes(b"caller inventory")
    with pytest.raises(PREPARER.SourceError, match="already exists"):
        PREPARER.stage(manifest, output, inventory)
    assert inventory.read_bytes() == b"caller inventory"
    assert not output.exists()


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
PBR_PROJECTION = {
    "data/pbr/PbrImages.conf": (
        327,
        "0fedbc71e140aca2fb286b0582beb57106990f997b7a5647690a85d4bd106fce",
    ),
    "data/pbr/anniversary_lounge.pbr_config.json": (
        716,
        "8dda877423f42922fb279d30e5143df41bdb22e836df88326773d444b03d0c06",
    ),
    "data/pbr/autoshop_01.pbr_config.json": (
        709,
        "7f0f1b94d23ac6c7b6938993d0f7cc4c045a104171e46937c8b0dfecb2e97e5c",
    ),
    "data/pbr/blue_photo_studio.pbr_config.json": (
        715,
        "e85b29608a1d610e95969f3b7266fe46a85419f0d04a57ff899a55044480c074",
    ),
    "data/pbr/bluts/brdflut_ldr_512x512.png": (
        92377,
        "a67189439f81536426e6bdccf1819852ccf76dbec6a0a8309d8f3f986861de94",
    ),
    "data/pbr/brown_photostudio.pbr_config.json": (
        718,
        "d3fdf77171b51707f08bada04c9d03fbaaef9aa132137ca0c359ea1766cc5ae3",
    ),
    "data/pbr/env_maps/anniversary_lounge_1k.hdr": (
        1612616,
        "7ca7917180c19864e7c4dba97a2c4660eb91c22307e4afed6f044759bc285721",
    ),
    "data/pbr/env_maps/autoshop_01_1k.hdr": (
        1581959,
        "8bdadcf34de814dee528c4b5010a74769a9ec1cc5ad5652a97d7fc8c703ede01",
    ),
    "data/pbr/env_maps/blue_photo_studio_1k.hdr": (
        1657733,
        "65f086b4e9b64f4e2d01e3268a30fd982572be81d88c0477349e3a704032294f",
    ),
    "data/pbr/env_maps/brown_photostudio_02_1k.hdr": (
        1648273,
        "77a64d58dd57475d2f0def31b1d4a8749dd7f223b5cb8875d67bb03972c61a04",
    ),
    "data/pbr/env_maps/lythwood_room_1k.hdr": (
        1399983,
        "e60da9023ebeb6f933b8e2611803113032bd848038f7a92b747875f243865361",
    ),
    "data/pbr/license.txt": (
        1416,
        "63c9792282c792c05c4b39704b52329b59cace1957b06db4c731b5bd58428238",
    ),
    "data/pbr/lythwood_room.pbr_config.json": (
        711,
        "e3285233b9b185d4c1e05011602132720779532d8acd3023b0818a623c2f2f36",
    ),
}


def _archive(path: Path, entries: list[tuple[str, bytes, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, kind, link in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = link
            member.size = len(payload) if kind == tarfile.REGTYPE else 0
            bundle.addfile(member, io.BytesIO(payload) if member.isfile() else None)


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = headers or {}
        self.read_calls = 0
        self.bytes_read = 0

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


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
        "file_count": 8147,
        "inventory_sha256": (
            "db14891b7403f0f0e0554c847a1fcfff404d68abf3fdc878b08c3e01e851edb2"
        ),
    }


def test_pillow_metadata_patch_is_exact_and_matches_runtime_closure() -> None:
    patch = MANIFEST["source"]["metadata_patches"]
    assert patch == [
        {
            "path": "pyproject.toml",
            "upstream_file_bytes": 4829,
            "upstream_file_sha256": (
                "43e4dbeea6f9e5294469471c58da6dba497f496b2b38d7913655728d50a22b08"
            ),
            "preimage_utf8": (
                '    "attrs>=19.1.0",\n'
                '    "gitpython",\n'
                '    "imageio",\n'
                '    "imageio-ffmpeg",\n'
                '    "matplotlib",\n'
                '    "numba>=0.60.0",\n'
                '    "numpy>=2.0.0,<2.4",\n'
                '    "numpy-quaternion>=2024.0.0",\n'
                '    "pillow==10.4.0",\n'
                '    "scipy>=1.13.0",\n'
                '    "tqdm",\n'
            ),
            "preimage_sha256": (
                "280e033f7e47f04fd61db02a59b05c6e7bb8a31dec2ff6dc1c85711c76522e01"
            ),
            "postimage_utf8": (
                '    "attrs>=19.1.0",\n'
                '    "gitpython",\n'
                '    "imageio",\n'
                '    "imageio-ffmpeg",\n'
                '    "numpy>=2.0.0,<2.4",\n'
                '    "pillow==12.3.0",\n'
                '    "tqdm",\n'
            ),
            "postimage_sha256": (
                "a21250ec068ba2c8150372d2f93c4a385ce6accb02d8cfc78348ebe628725b90"
            ),
            "expected_match_count": 1,
            "patched_file_bytes": 4735,
            "patched_file_sha256": (
                "c5a6d3d39e89158e3bc35b696481ebfa12357b8cd1c941c2be38d2ef0dbdcf93"
            ),
            "reason": (
                "Pin Pillow and keep the optional numpy-quaternion/numba/scipy/matplotlib stack "
                "runtime-fetched to keep the public bootstrap ELF closure bounded"
            ),
        }
    ]
    assert '"pillow==12.3.0",' in patch[0]["postimage_utf8"]
    for omitted in ("matplotlib", "numba", "numpy-quaternion", "scipy"):
        assert f'"{omitted}' not in patch[0]["postimage_utf8"]
    runtime = (PACKAGE / "requirements-runtime.lock").read_text(encoding="utf-8")
    licenses = json.loads((PACKAGE / "licenses.json").read_text(encoding="utf-8"))
    pillow = next(row for row in licenses["python_wheels"] if row["name"] == "pillow")
    assert "pillow==12.3.0 --hash=sha256:" in runtime
    assert pillow["version"] == "12.3.0"


def test_pillow_metadata_patch_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    before = b'[project]\ndependencies = [\n    "pillow==10.4.0",\n]\n'
    after = b'[project]\ndependencies = [\n    "pillow==12.3.0",\n]\n'
    patch = {
        "path": "pyproject.toml",
        "upstream_file_bytes": len(before),
        "upstream_file_sha256": hashlib.sha256(before).hexdigest(),
        "preimage_utf8": '    "pillow==10.4.0",\n',
        "preimage_sha256": hashlib.sha256(b'    "pillow==10.4.0",\n').hexdigest(),
        "postimage_utf8": '    "pillow==12.3.0",\n',
        "postimage_sha256": hashlib.sha256(b'    "pillow==12.3.0",\n').hexdigest(),
        "expected_match_count": 1,
        "patched_file_bytes": len(after),
        "patched_file_sha256": hashlib.sha256(after).hexdigest(),
    }
    root = tmp_path / "source"
    root.mkdir()
    target = root / "pyproject.toml"
    target.write_bytes(before)
    PREPARER._apply_metadata_patches(root, [patch])
    assert target.read_bytes() == after
    with pytest.raises(PREPARER.SourceError, match="patch input changed"):
        PREPARER._apply_metadata_patches(root, [patch])

    target.write_bytes(before)
    wrong_preimage = dict(patch)
    wrong_preimage["preimage_utf8"] = '    "pillow==10.3.0",\n'
    wrong_preimage["preimage_sha256"] = hashlib.sha256(
        wrong_preimage["preimage_utf8"].encode()
    ).hexdigest()
    with pytest.raises(PREPARER.SourceError, match="patch preimage changed"):
        PREPARER._apply_metadata_patches(root, [wrong_preimage])
    assert target.read_bytes() == before


@pytest.mark.parametrize("path", ["../pyproject.toml", "/pyproject.toml"])
def test_metadata_patch_refuses_unsafe_target(path: str, tmp_path: Path) -> None:
    patch = {"path": path}
    with pytest.raises(PREPARER.SourceError, match="patch path is unsafe"):
        PREPARER._apply_metadata_patches(tmp_path, [patch])


def test_required_pbr_configuration_is_bound_to_the_official_source() -> None:
    assert len(PBR_CONFIG) == 327
    assert hashlib.sha256(PBR_CONFIG).hexdigest() == (
        "0fedbc71e140aca2fb286b0582beb57106990f997b7a5647690a85d4bd106fce"
    )
    observed = {
        row["path"]: (row["bytes"], row["sha256"])
        for row in MANIFEST["source"]["required_projection_files"]
    }
    assert observed == PBR_PROJECTION


def test_pbr_configuration_references_the_complete_render_resource_set() -> None:
    referenced = {
        "data/pbr/" + line.split("=", 1)[1].strip()
        for line in PBR_CONFIG.decode().splitlines()
        if line.startswith("filename = ")
    }
    render_resources = {
        path for path in PBR_PROJECTION if path.endswith((".png", ".hdr"))
    }
    assert referenced == render_resources


def test_required_pbr_configuration_refuses_omission_or_wrong_content(
    tmp_path: Path,
) -> None:
    required = [MANIFEST["source"]["required_projection_files"][0]]
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


def test_required_pbr_projection_refuses_each_omission_or_wrong_content(
    tmp_path: Path,
) -> None:
    payloads = {path: f"fixture:{path}".encode() for path in PBR_PROJECTION}
    required = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in payloads.items()
    ]
    root = tmp_path / "source"
    for path, payload in payloads.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    PREPARER._verify_required_source_files(root, required)
    for row in required:
        target = root / row["path"]
        expected = target.read_bytes()
        target.unlink()
        with pytest.raises(PREPARER.SourceError, match="required source projection"):
            PREPARER._verify_required_source_files(root, required)
        target.write_bytes(b"x" * len(expected))
        with pytest.raises(PREPARER.SourceError, match="required source projection"):
            PREPARER._verify_required_source_files(root, required)
        target.write_bytes(expected)


@pytest.mark.parametrize("path", ["../escape", "data/pbr/../escape", "/data/pbr/file"])
def test_pbr_projection_refuses_traversal(path: str, tmp_path: Path) -> None:
    row = {"path": path, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    with pytest.raises(PREPARER.SourceError):
        PREPARER._required_pbr_paths([row])


def test_parent_projection_keeps_complete_required_pbr_projection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    for name in ("LICENSE", "MANIFEST.in", "README.md", "pyproject.toml", "setup.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"fixture")
    for name in ("src/deps/basis-universal/transcoder", "src_python"):
        (root / name).mkdir(parents=True)
    data = root / "data"
    data.mkdir()
    (data / "default.physics_config.json").write_bytes(b"{}")
    for row in MANIFEST["source"]["required_projection_files"]:
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"projected fixture")
    (data / "pbr/env_maps/forbidden.hdr").write_bytes(b"not projected")
    (data / "scene_datasets").mkdir()

    PREPARER._prune_parent(root, MANIFEST["source"]["required_projection_files"])

    observed = {
        path.relative_to(root).as_posix()
        for path in (data / "pbr").rglob("*")
        if path.is_file()
    }
    assert observed == set(PBR_PROJECTION)
    assert not (data / "pbr/env_maps/forbidden.hdr").exists()
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


def _download_item(payload: bytes) -> dict[str, object]:
    return {
        "name": "fixture",
        "archive_url": (
            "https://codeload.github.com/example/project/tar.gz/" + "a" * 40
        ),
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _mock_download_target(monkeypatch, payload=b"inert source bytes"):
    """Replace path effects and upstream delivery with in-memory observations."""
    item = _download_item(payload)
    stream = io.BytesIO()
    target = Mock(spec=Path)
    target.open.return_value = nullcontext(stream)
    contained = Mock(return_value=target)
    monkeypatch.setattr(PREPARER, "_contained_source_path", contained)
    response = _Response(payload, str(item["archive_url"]))
    opener = Mock(return_value=response)
    directory = Mock(spec=Path)
    return SimpleNamespace(
        item=item,
        stream=stream,
        target=target,
        contained=contained,
        response=response,
        opener=opener,
        directory=directory,
    )


def test_mock_download_success_retains_exact_bytes_without_cleanup(monkeypatch):
    """Successful in-memory verification retains the operation's target."""
    fixture = _mock_download_target(monkeypatch)

    result = PREPARER._download(fixture.item, fixture.directory, opener=fixture.opener)

    assert result is fixture.target
    assert fixture.stream.getvalue() == b"inert source bytes"
    fixture.target.open.assert_called_once_with("xb")
    fixture.target.unlink.assert_not_called()
    fixture.contained.assert_called_once_with(
        fixture.directory, "fixture.tar.gz", "source archive"
    )


class _NoAddNoteError(OSError):
    """Simulate only the absence of add_note, not Python 3.10 execution."""

    def __getattribute__(self, name):
        if name == "add_note":
            raise AttributeError("add_note unavailable in this inert fixture")
        return super().__getattribute__(name)


def _inject_mock_read_error(fixture, primary):
    """Retain the exact original raising frame for independently checked identity."""
    original_tracebacks = []

    def fail_read(_size):
        try:
            raise primary
        except OSError:
            original_tracebacks.append(primary.__traceback__)
            raise

    fixture.response.read = fail_read
    return original_tracebacks


def _assert_primary_traceback(observed, primary, original):
    """Check object and original traceback identity without filesystem effects."""
    assert observed.value is primary
    trace = observed.value.__traceback__
    while trace is not None:
        if trace is original:
            return
        trace = trace.tb_next
    pytest.fail("original initiating traceback missing")


@pytest.mark.parametrize(
    ("cleanup_fails", "no_add_note", "report_fails"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_mock_download_preserves_primary_identity_and_traceback(
    monkeypatch, cleanup_fails, no_add_note, report_fails
):
    """Cleanup/reporting failures preserve the initiating error even without notes."""
    fixture = _mock_download_target(monkeypatch)
    error_type = _NoAddNoteError if no_add_note else OSError
    primary = error_type("inert stream failure")
    original = _inject_mock_read_error(fixture, primary)
    reporter = Mock(
        side_effect=OSError("inert stderr refusal") if report_fails else None
    )
    monkeypatch.setattr(PREPARER, "print", reporter, raising=False)
    if no_add_note:
        assert not hasattr(primary, "add_note")
    if cleanup_fails:
        fixture.target.unlink.side_effect = PermissionError("inert removal refusal")
    with pytest.raises(OSError) as observed:
        PREPARER._download(fixture.item, fixture.directory, opener=fixture.opener)
    _assert_primary_traceback(observed, primary, original[0])
    fixture.target.unlink.assert_called_once_with(missing_ok=True)
    if cleanup_fails:
        reporter.assert_called_once_with(
            "source archive cleanup failed: PermissionError: inert removal refusal",
            file=PREPARER.sys.stderr,
        )
    else:
        reporter.assert_not_called()
    assert not getattr(primary, "__notes__", [])


@pytest.mark.parametrize("failure_stage", ["response", "open"])
def test_mock_download_never_removes_uncreated_target(monkeypatch, failure_stage):
    """Only successful exclusive creation establishes this operation's ownership."""
    fixture = _mock_download_target(monkeypatch)
    primary = FileExistsError("inert caller-owned target")
    if failure_stage == "response":
        fixture.opener.side_effect = primary
    else:
        fixture.target.open.side_effect = primary

    with pytest.raises(FileExistsError) as observed:
        PREPARER._download(fixture.item, fixture.directory, opener=fixture.opener)

    assert observed.value is primary
    fixture.target.unlink.assert_not_called()
    assert not getattr(primary, "__notes__", [])


def _mock_primary_failure(primary, original):
    """Inject an ordinary failure and retain its exact initial traceback."""

    def fail(*_args, **_kwargs):
        try:
            raise primary
        except OSError:
            original.append(primary.__traceback__)
            raise

    return fail


def _mock_projection_publication(monkeypatch):
    """Observe links, identity checks and removal without filesystem operations."""
    staged = MagicMock(spec=Path)
    output, inventory, source = (Mock(spec=Path) for _ in range(3))
    staged.parent.__truediv__.return_value = source
    owned = SimpleNamespace(st_dev=11, st_ino=22)
    inventory.lstat.return_value = source.lstat.return_value = owned
    link, rename, reporter, events = (Mock() for _ in range(4))
    monkeypatch.setattr(PREPARER.os, "link", link)
    monkeypatch.setattr(PREPARER, "_rename_noreplace", rename)
    monkeypatch.setattr(PREPARER, "print", reporter, raising=False)
    for name, observation in [
        ("link", link),
        ("rename", rename),
        ("inventory_stat", inventory.lstat),
        ("source_stat", source.lstat),
        ("unlink", inventory.unlink),
        ("report", reporter),
    ]:
        events.attach_mock(observation, name)
    return SimpleNamespace(
        staged=staged,
        output=output,
        inventory=inventory,
        source=source,
        link=link,
        rename=rename,
        reporter=reporter,
        events=events,
    )


@pytest.mark.parametrize("has_inventory", [False, True])
def test_mock_projection_publication_success_keeps_inventory(
    monkeypatch, has_inventory
):
    """Successful publication retains inventory without cleanup or warning."""
    fixture = _mock_projection_publication(monkeypatch)
    inventory = fixture.inventory if has_inventory else None
    PREPARER._publish_projection(fixture.staged, fixture.output, inventory)
    expected = (
        [call.link(fixture.source, inventory, follow_symlinks=False)]
        if has_inventory
        else []
    )
    expected.append(call.rename(fixture.staged, fixture.output))
    assert fixture.events.mock_calls == expected


@pytest.mark.parametrize("cleanup_failure", ["unlink", "stat"])
@pytest.mark.parametrize("report_fails", [False, True])
def test_mock_projection_cleanup_preserves_primary(
    monkeypatch, cleanup_failure, report_fails
):
    """Inventory cleanup and reporting errors cannot replace publication failure."""
    fixture = _mock_projection_publication(monkeypatch)
    primary, original = OSError("inert publication failure"), []
    fixture.rename.side_effect = _mock_primary_failure(primary, original)
    failing = (
        fixture.inventory.unlink
        if cleanup_failure == "unlink"
        else fixture.inventory.lstat
    )
    failing.side_effect = PermissionError("inert inventory cleanup refusal")
    if report_fails:
        fixture.reporter.side_effect = OSError("inert stderr refusal")
    with pytest.raises(OSError) as observed:
        PREPARER._publish_projection(fixture.staged, fixture.output, fixture.inventory)
    _assert_primary_traceback(observed, primary, original[0])
    expected = [
        call.link(fixture.source, fixture.inventory, follow_symlinks=False),
        call.rename(fixture.staged, fixture.output),
        call.inventory_stat(),
    ]
    if cleanup_failure == "unlink":
        expected.extend([call.source_stat(), call.unlink()])
    expected.append(
        call.report(
            "warning: source inventory cleanup failed", file=PREPARER.sys.stderr
        )
    )
    assert fixture.events.mock_calls == expected


@pytest.mark.parametrize("ownership", ["mismatch", "uncreated"])
def test_mock_projection_cleanup_refuses_unowned_inventory(monkeypatch, ownership):
    """Mismatched identity or unsuccessful link creation never permits removal."""
    fixture = _mock_projection_publication(monkeypatch)
    primary, original = OSError("inert publication refusal"), []
    failure = fixture.link if ownership == "uncreated" else fixture.rename
    failure.side_effect = _mock_primary_failure(primary, original)
    fixture.inventory.lstat.return_value = SimpleNamespace(st_dev=11, st_ino=23)
    with pytest.raises(OSError) as observed:
        PREPARER._publish_projection(fixture.staged, fixture.output, fixture.inventory)
    _assert_primary_traceback(observed, primary, original[0])
    fixture.inventory.unlink.assert_not_called()
    fixture.reporter.assert_not_called()
    if ownership == "uncreated":
        fixture.rename.assert_not_called()
        fixture.inventory.lstat.assert_not_called()
        fixture.source.lstat.assert_not_called()
    else:
        fixture.inventory.lstat.assert_called_once_with()
        fixture.source.lstat.assert_called_once_with()


def _mock_staging_paths():
    """Model the three owned children without creating filesystem entries."""
    temp = MagicMock(spec=Path)
    staged, archives, inventory, output = (Mock(spec=Path) for _ in range(4))
    temp.__truediv__.side_effect = {
        "projection": staged,
        "archives": archives,
        "inventory.json": inventory,
    }.__getitem__
    return SimpleNamespace(
        temp=temp, staged=staged, archives=archives, inventory=inventory, output=output
    )


def _record_staging_observations(fixture, factory):
    """Capture ordered observations from already mocked lifecycle operations."""
    for name, observation in [
        ("create", factory),
        ("mkdir", fixture.archives.mkdir),
        ("materialize", fixture.materialize),
        ("write", fixture.inventory.write_bytes),
        ("publish", fixture.publish),
        ("remove", fixture.remove),
        ("report", fixture.reporter),
    ]:
        fixture.events.attach_mock(observation, name)


def _mock_owned_staging(monkeypatch):
    """Replace every stage path/creation/write/publication/removal with mocks."""
    fixture = _mock_staging_paths()
    factory = Mock(return_value="inert-staging-token")
    fixture.materialize = Mock(return_value=b"verified inert inventory\n")
    fixture.publish, fixture.remove, fixture.reporter, fixture.events = (
        Mock() for _ in range(4)
    )
    monkeypatch.setattr(PREPARER, "Path", Mock(return_value=fixture.temp))
    monkeypatch.setattr(PREPARER.tempfile, "mkdtemp", factory)
    monkeypatch.setattr(PREPARER, "_materialize", fixture.materialize)
    monkeypatch.setattr(PREPARER, "_publish_projection", fixture.publish)
    monkeypatch.setattr(PREPARER.shutil, "rmtree", fixture.remove)
    monkeypatch.setattr(PREPARER, "print", fixture.reporter, raising=False)
    _record_staging_observations(fixture, factory)
    return fixture


def _expected_mock_staging(fixture, failure=None):
    """Specify the intended lifecycle independently of candidate observations."""
    expected = [
        call.create(prefix=".npa-habitat-source-", dir=fixture.output.parent),
        call.mkdir(mode=0o700),
        call.materialize({}, fixture.staged, fixture.archives),
    ]
    if failure != "materialize":
        expected.extend(
            [
                call.write(b"verified inert inventory\n"),
                call.publish(fixture.staged, fixture.output, fixture.inventory),
            ]
        )
    expected.append(call.remove(fixture.temp))
    return expected


@pytest.mark.parametrize(
    ("cleanup_fails", "report_fails"), [(False, False), (True, False), (True, True)]
)
def test_mock_staging_success_preserves_publication_order(
    monkeypatch, cleanup_fails, report_fails
):
    """Completed publication stays successful even if its cleanup warning fails."""
    fixture = _mock_owned_staging(monkeypatch)
    if cleanup_fails:
        fixture.remove.side_effect = PermissionError("inert staging cleanup refusal")
    if report_fails:
        fixture.reporter.side_effect = OSError("inert stderr refusal")
    assert (
        PREPARER._stage_owned_projection({}, fixture.output, fixture.inventory) is None
    )
    expected = _expected_mock_staging(fixture)
    if cleanup_fails:
        expected.append(
            call.report(
                "warning: temporary source staging cleanup failed",
                file=PREPARER.sys.stderr,
            )
        )
    assert fixture.events.mock_calls == expected


@pytest.mark.parametrize("failure", ["materialize", "publish"])
@pytest.mark.parametrize("report_fails", [False, True])
def test_mock_staging_failure_preserves_primary(monkeypatch, failure, report_fails):
    """Staging cleanup/reporting preserves the primary exception and traceback."""
    fixture = _mock_owned_staging(monkeypatch)
    primary, original = OSError("inert source staging failure"), []
    getattr(fixture, failure).side_effect = _mock_primary_failure(primary, original)
    fixture.remove.side_effect = PermissionError("inert staging cleanup refusal")
    if report_fails:
        fixture.reporter.side_effect = OSError("inert stderr refusal")
    with pytest.raises(OSError) as observed:
        PREPARER._stage_owned_projection({}, fixture.output, fixture.inventory)
    _assert_primary_traceback(observed, primary, original[0])
    expected = _expected_mock_staging(fixture, failure)
    expected.append(
        call.report(
            "warning: temporary source staging cleanup failed", file=PREPARER.sys.stderr
        )
    )
    assert fixture.events.mock_calls == expected


@pytest.mark.parametrize("declared_length", ["invalid", "-1", "23"])
def test_source_download_refuses_declared_length_before_body_consumption(
    tmp_path: Path, declared_length: str
) -> None:
    payload = b"immutable source bytes"
    item = _download_item(payload)
    response = _Response(
        payload,
        str(item["archive_url"]),
        {"Content-Length": declared_length},
    )

    with pytest.raises(PREPARER.SourceError, match="Content-Length"):
        PREPARER._download(item, tmp_path, opener=lambda *_a, **_k: response)

    assert response.read_calls == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("expected_bytes", [None, -1, True, "22"])
def test_source_download_refuses_invalid_pin_before_opening(
    tmp_path: Path, expected_bytes: object
) -> None:
    item = _download_item(b"immutable source bytes")
    item["archive_bytes"] = expected_bytes
    opened = False

    def opener(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("invalid pin must be rejected before opening")

    with pytest.raises(PREPARER.SourceError, match="positive integer"):
        PREPARER._download(item, tmp_path, opener=opener)

    assert opened is False


def test_source_download_supports_absent_declared_length(tmp_path: Path) -> None:
    payload = b"immutable source bytes"
    item = _download_item(payload)
    response = _Response(payload, str(item["archive_url"]))

    target = PREPARER._download(item, tmp_path, opener=lambda *_a, **_k: response)

    assert target.read_bytes() == payload


def test_source_download_stops_oversized_stream_before_exceeding_pin(
    tmp_path: Path,
) -> None:
    expected = b"pin"
    response = _Response(
        expected + b"extra", str(_download_item(expected)["archive_url"])
    )

    with pytest.raises(PREPARER.SourceError, match="exceeds pinned byte count"):
        PREPARER._download(
            _download_item(expected),
            tmp_path,
            opener=lambda *_a, **_k: response,
        )

    assert response.bytes_read == len(expected) + 1
    assert list(tmp_path.iterdir()) == []


def test_source_download_accepts_exact_declared_bytes_and_hash(tmp_path: Path) -> None:
    payload = b"immutable source bytes"
    item = _download_item(payload)
    response = _Response(
        payload,
        str(item["archive_url"]),
        {"Content-Length": str(len(payload))},
    )

    target = PREPARER._download(item, tmp_path, opener=lambda *_a, **_k: response)

    assert target.read_bytes() == payload


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
        *PBR_PROJECTION,
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
        "cp -a /opt/habitat-sim/data/pbr/. /opt/source-projection/data/pbr/"
        in dockerfile
    )
