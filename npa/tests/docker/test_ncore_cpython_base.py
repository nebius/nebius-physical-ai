"""Bind the fixed CPython base and its bundled notices to the reviewed byte profile."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PACKAGING = Path(__file__).resolve().parents[2] / "docker/workbench/ncore"
INDEX = "782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
ELFS = "fd9c96ddcf94b6e8cc74d04e0d555bf6964c47f0117514def0cc8d7ac180e698"
NOTICE_HASHES = {
    "LICENSE.third-party": "341832873fd316a37927e79385093fbbfd40a467428480835fe435a80cadf4e5",
    "LICENSE.expat": "31b15de82aa19a845156169a17a5488bf597e561b2c318d159ed583139b25e87",
    "LICENSE.mpdecimal": "b07528d8b1dbf1e2d2741052996f0876e23342ce2d30d0effa39c5457716c25a",
    "LICENSE.hacl": "998ce04fb8ad9dedb0bc1b44938f8c3dcf1089780fa105ce1c8c30fb5554d78c",
    "LICENSE.hacl-krml": "6b7cb4bd5441f287eeb8b745073c11bc5d2974252bec3b6d8c101ff67f918a26",
    "LICENSE.hacl-fstar": "54079ca9757dbaa2a54053fbd9be60f77c5e06695fa997191fc141cd20d41d59",
    "LICENSE.blake2": "a27d7169aa4b7db350c5e46da87184c4b7a8ac8e4d86dce38845fdffd789d7b5",
    "LICENSE.blake2-python": "74baf34496c290dd3b44efca41d040893117c9c1a901c2831b629c9cf235ceb1",
}


@pytest.fixture
def sources():
    spec = importlib.util.spec_from_file_location("ncore_base_sources", PACKAGING / "base_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_recipe_native_lock_and_cpython_profile_agree():
    lock = json.loads((PACKAGING / "base-source-lock.json").read_bytes())
    native = json.loads((PACKAGING / "native-bootstrap-lock.json").read_bytes())
    expected = "python:3.12.14-slim-bookworm@sha256:" + INDEX
    assert lock["base_image"] == native["provenance"]["base_image"] == expected
    assert f"FROM {expected} AS runtime-base" in (PACKAGING / "Dockerfile").read_text()
    assert "Base: " + expected in (PACKAGING / "CPYTHON-BASE-RECIPE.txt").read_text()
    profile = lock["cpython_provenance"]
    assert profile["source_commit"] == "2abcf904b8dac8c999d2b3aac76681abb333798a"
    assert profile["source_archive"]["sha256"] == (
        "5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a"
    )
    assert profile["embedded_components"]["expat"] == "2.8.3"
    assert len(lock["cpython_files"]) == 665 and len(lock["cpython_elf_files"]) == 61
    raw = json.dumps(lock["cpython_elf_files"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == ELFS
    paths = {item["path"] for item in lock["cpython_files"]}
    grammars = {Path(path).name for path in paths if path.endswith(".pickle")}
    assert grammars == {"Grammar3.12.14.final.0.pickle", "PatternGrammar3.12.14.final.0.pickle"}
    assert not lock["python_distributions"] and not lock["python_wheels"]
    assert len(lock["debian_binaries"]) == 99
    assert {p["name"] for p in native["debian_binaries"]} == {"libgnutls30", "libssh2-1"}


@pytest.mark.parametrize("name,expected", NOTICE_HASHES.items())
def test_exact_bundled_notice_is_selected_and_copied_before_assembly(name, expected):
    lock = json.loads((PACKAGING / "base-source-lock.json").read_bytes())
    path = "usr/share/doc/npa-ncore/cpython/" + name
    selected = next(item for item in lock["notices"] if item["path"] == path)
    source = next(item for item in lock["cpython_provenance"]["notices"] if item["path"] == path)
    assert selected["sha256"] == source["sha256"] == expected
    assert hashlib.sha256((PACKAGING / "notices/cpython" / name).read_bytes()).hexdigest() == expected
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    copy = "COPY --chmod=0644 docker/workbench/ncore/notices/cpython/ /usr/share/doc/npa-ncore/cpython/"
    assert dockerfile.index(copy) < dockerfile.index("base_sources.py assemble-root")
    terms = {item["path"] for item in lock["notices"]}
    assert set(lock["cpython_provenance"]["full_license_paths"]) <= terms


@pytest.mark.parametrize("damage", ["changed", "missing", "symlink"])
def test_notice_assembly_refuses_changed_missing_or_escaping_bytes(sources, tmp_path, damage):
    lock = json.loads((PACKAGING / "base-source-lock.json").read_bytes())
    item = next(row for row in lock["notices"] if row["path"].endswith("cpython/LICENSE.expat"))
    root = tmp_path / "root"
    target = root / item["path"]
    target.parent.mkdir(parents=True)
    if damage == "changed":
        target.write_bytes(b"changed notice")
    elif damage == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes((PACKAGING / "notices/cpython/LICENSE.expat").read_bytes())
        target.symlink_to(outside)
    with pytest.raises(ValueError, match="retained file (SHA256 changed|escapes root)"):
        sources.copy_locked_file(root, tmp_path / "output", item)


@pytest.mark.parametrize("module,old_hash", [
    ("_bz2", "b6497e0f8eec8ce883214ce21565193b18d0f6d16053efe350bcdb445451b0ce"),
    ("_lzma", "f7a7a64489be0a6c63f6c5a2278c34cf2181d9e6a327bdb5bf9b276dc4d891ec"),
    ("zlib", "45107dd2a941be1b24140f6b05662d486f8660d30d5ec2fb7ec31d631d993177"),
])
def test_old_decompressor_bytes_are_refused_by_actual_source_lock(sources, tmp_path, module, old_hash):
    lock = json.loads((PACKAGING / "base-source-lock.json").read_bytes())
    path = f"usr/local/lib/python3.12/lib-dynload/{module}.cpython-312-x86_64-linux-gnu.so"
    inventory = {"layers": [{"files": {path: {"elf": True, "sha256": old_hash}},
                            "debian_packages": [], "python_packages": []}]}
    with pytest.raises(ValueError, match="unmapped ELF: " + path):
        sources.verify_coverage(lock, inventory, tmp_path)
