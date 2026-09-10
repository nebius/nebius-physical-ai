"""Exercise NCore's documentation merge into the selected public filesystem."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


PACKAGING = Path(__file__).resolve().parents[2] / "docker/workbench/ncore"
DOCS = Path("usr/share/doc/npa-ncore")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder_docs(root):
    docs = root / DOCS
    (docs / "recipes").mkdir(parents=True)
    for name in (
        "runtime-lock.json",
        "source-lock.json",
        "REDISTRIBUTION.md",
        "BASE-SOURCES.md",
        "build-requirements.lock",
        "runtime-requirements.lock",
    ):
        shutil.copy2(PACKAGING / name, docs / name)
    for name in ("Dockerfile", "stage_upstream.py"):
        shutil.copy2(PACKAGING / name, docs / "recipes" / name)
    shutil.copy2(PACKAGING.parents[2] / "src/npa/_public_https.py", docs / "recipes")
    shutil.copytree(PACKAGING / "notices/cpython", docs / "cpython")
    (docs / "notices").mkdir()
    for name in ("NPA-LICENSE", "NOTICE-NVIDIA-NCORE-COLMAP", "NOTICE-NVIDIA-SKILLS"):
        shutil.copy2(PACKAGING / "notices" / name, docs / "notices" / name)
    (docs / "LICENSE-APACHE-2.0").write_text("synthetic upstream license fixture\n")
    (docs / "npa-source-sha").write_text("a" * 40 + "\n")
    (docs / ".delivery-index").write_text("synthetic hidden documentation\n")
    (docs / "notice-link").symlink_to("cpython/LICENSE.expat")
    os.link(docs / ".delivery-index", docs / "delivery-index")
    (docs / "recipes/stage_upstream.py").chmod(0o751)
    return docs


def _copy_docs_from_dockerfile(builder, public):
    # Execute the recipe's actual argv, preserving /.; Path would normalize it away.
    lines = (PACKAGING / "Dockerfile").read_text().splitlines()
    commands = [
        shlex.split(line.strip().removesuffix("\\").strip())[1:]
        for line in lines
        if line.strip().startswith("&& cp ") and "/usr/share/doc/npa-ncore" in line
    ]
    (command,) = commands
    relocated = []
    for argument in command:
        if argument.startswith("/public-root/"):
            argument = argument.replace("/public-root/", f"{public}/", 1)
        elif argument.startswith("/"):
            argument = f"{builder}{argument}"
        relocated.append(argument)
    subprocess.run(relocated, check=True, capture_output=True)


@pytest.fixture
def assembled_docs(tmp_path):
    builder, public = tmp_path / "builder", tmp_path / "public-root"
    docs = _builder_docs(builder)
    sources = _load_script("base_sources")
    lock = json.loads((PACKAGING / "base-source-lock.json").read_bytes())
    selected = [
        item
        for item in sources.retained_files(lock)
        if item["path"].startswith(DOCS.as_posix() + "/")
    ]
    assert selected
    for item in selected:
        sources.copy_locked_file(builder, public, item)
    assert (public / DOCS / "cpython/LICENSE.expat").is_file()
    assert not (public / DOCS / "runtime-lock.json").exists()
    _copy_docs_from_dockerfile(builder, public)
    return docs, public


def test_documentation_merge_preserves_paths_bytes_and_archive_metadata(assembled_docs):
    docs, public = assembled_docs
    target = public / DOCS
    assert not (target / "npa-ncore").exists()
    assert {p.relative_to(target) for p in target.rglob("*")} == {
        p.relative_to(docs) for p in docs.rglob("*")
    }
    for source in [docs, *docs.rglob("*")]:
        copied = target / source.relative_to(docs)
        before, after = source.lstat(), copied.lstat()
        assert (after.st_mode, after.st_uid, after.st_gid, after.st_mtime_ns) == (
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_mtime_ns,
        )
        if source.is_symlink():
            assert copied.is_symlink() and copied.readlink() == source.readlink()
        elif source.is_file():
            assert copied.read_bytes() == source.read_bytes()
    assert (target / ".delivery-index").stat().st_ino == (
        target / "delivery-index"
    ).stat().st_ino


@pytest.fixture
def packaging_verifier(assembled_docs, monkeypatch):
    _, public = assembled_docs
    sources = public / "opt/ncore/src"
    sources.mkdir(parents=True)
    (sources / "converter.py").write_bytes(b"# synthetic converter fixture\n")
    base_sources = _load_script("base_sources")
    inventory = {
        "files": {"converter.py": base_sources.file_digest(sources / "converter.py")},
        "sources": json.loads((PACKAGING / "source-lock.json").read_bytes()),
    }
    (sources / "source-inventory.json").write_text(json.dumps(inventory))
    verifier = _load_script("verify-packaging")
    monkeypatch.setattr(verifier, "RUNTIME_LOCK", public / DOCS / "runtime-lock.json")
    monkeypatch.setattr(verifier, "Path", lambda path: public / path.lstrip("/"))
    # Isolate container-only prerequisites; inventory parsing/hashes stay real.
    monkeypatch.setattr(verifier, "shutil", SimpleNamespace(which=lambda name: name))
    monkeypatch.setattr(
        verifier,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
    )
    monkeypatch.setattr(sys, "argv", ["verify-packaging.py", "--image-only"])
    return verifier, public


def test_merged_documentation_reaches_offline_packaging_proof(
    packaging_verifier, capsys
):
    verifier, _ = packaging_verifier
    verifier.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "packaging-ready"
    assert report["validation"] == "image-boundary-only"


@pytest.mark.parametrize(
    "missing",
    [
        DOCS / "runtime-lock.json",
        Path("opt/ncore/src/source-inventory.json"),
        DOCS / "npa-source-sha",
    ],
)
def test_packaging_proof_rejects_missing_inventory_paths(packaging_verifier, missing):
    verifier, public = packaging_verifier
    verifier.main()
    (public / missing).unlink()
    with pytest.raises(FileNotFoundError) as error:
        verifier.main()
    assert error.value.filename == str(public / missing)
