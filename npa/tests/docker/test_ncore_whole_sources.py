"""Whole-image source coverage must include overwritten and deleted bytes."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


SCRIPT = Path(__file__).parents[2] / "docker/workbench/ncore/whole_image_sources.py"


@pytest.fixture
def source_module():
    spec = importlib.util.spec_from_file_location("whole_image_sources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def archive_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return output.getvalue()


def test_source_identity_preserves_epoch_and_binary_rebuild(source_module):
    raw = b"Package: sample\nStatus: install ok installed\nArchitecture: amd64\nVersion: 2:3.0-1+b2\nSource: origin (2:3.0-1)\n\n"
    assert source_module.debian_packages(raw) == [
        {
            "name": "sample",
            "version": "2:3.0-1+b2",
            "architecture": "amd64",
            "source": "origin",
            "source_version": "2:3.0-1",
        }
    ]


def test_inventory_keeps_superseded_packages_and_whiteouted_bytes(
    source_module, tmp_path
):
    status = b"Package: sample\nStatus: install ok installed\nArchitecture: amd64\nVersion: 1\n\n"
    layers = [
        archive_bytes({"var/lib/dpkg/status": status, "usr/bin/old": b"\x7fELFold"}),
        archive_bytes(
            {
                "var/lib/dpkg/status": status.replace(b"Version: 1", b"Version: 2"),
                "usr/bin/.wh.old": b"",
                "usr/bin/new": b"\x7fELFnew",
            }
        ),
    ]
    config = {
        "rootfs": {"diff_ids": ["sha256:" + source_module.digest(x) for x in layers]}
    }
    files = {f"layer{i}.tar": value for i, value in enumerate(layers)}
    files["config.json"] = json.dumps(config).encode()
    files["manifest.json"] = json.dumps(
        [{"Config": "config.json", "Layers": list(files)[:-1]}]
    ).encode()
    target = tmp_path / "image.tar"
    target.write_bytes(archive_bytes(files))
    result = source_module.inventory(target)
    assert {
        p["version"] for layer in result["layers"] for p in layer["debian_packages"]
    } == {"1", "2"}
    assert "usr/bin/old" not in result["final_files"]
    assert result["layers"][0]["files"]["usr/bin/old"]["elf"]
    assert result["final_files"]["usr/bin/new"]["elf"]


def test_annex_reuses_native_archive_and_refuses_corruption(source_module, tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    (native / "native.tar").write_bytes(b"native source")
    lock = {
        "schema": 1,
        "artifacts": [
            {
                "path": "native/native.tar",
                "delivery": "native",
                "filename": "native.tar",
                "sha256": source_module.digest(b"native source"),
                "url": "https://example.org/native.tar",
            }
        ],
    }
    annex = tmp_path / "annex"
    source_module.assemble(lock, annex, native, [], offline=True)
    assert not (annex / "native/native.tar").exists()
    assert source_module.verify_artifacts(lock, annex, native)["artifacts"] == 1
    (native / "native.tar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        source_module.verify_artifacts(lock, annex, native)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../../escape"])
def test_unsafe_archive_paths_fail(source_module, path):
    with pytest.raises(ValueError, match="unsafe"):
        source_module.safe_path(path)


def test_offline_assembly_never_fetches(source_module, tmp_path, monkeypatch):
    monkeypatch.setattr(
        source_module.urllib.request, "urlopen", lambda *_: pytest.fail("network")
    )
    lock = {
        "artifacts": [
            {
                "path": "sources/missing.tar",
                "filename": "missing.tar",
                "sha256": "0" * 64,
                "url": "https://example.org/missing.tar",
            }
        ]
    }
    with pytest.raises(ValueError, match="offline"):
        source_module.assemble(
            lock, tmp_path / "out", tmp_path / "native", [], offline=True
        )


def coverage_fixture():
    package = {
        "name": "sample",
        "version": "1:2-1+b1",
        "architecture": "amd64",
        "source": "origin",
        "source_version": "1:2-1",
    }
    lock = {
        "base_diff_ids": [],
        "debian_binaries": [
            {
                "name": "sample",
                "version": "1:2-1+b1",
                "architecture": "amd64",
                "source": "debian:origin@1:2-1",
                "elf_files": [{"path": "usr/bin/sample", "sha256": "a" * 64}],
            }
        ],
        "python_distributions": [],
        "python_wheels": [],
        "cpython_elf_files": [],
    }
    inv = {
        "layers": [
            {
                "diff_id": "sha256:" + "b" * 64,
                "debian_packages": [package],
                "python_packages": [],
                "files": {"usr/bin/sample": {"elf": True, "sha256": "a" * 64}},
            }
        ]
    }
    return lock, inv


def test_inventory_refuses_binary_bytes_even_with_matching_version(
    source_module, tmp_path
):
    lock, inv = coverage_fixture()
    assert source_module.verify_coverage(lock, inv, tmp_path)["elf_occurrences"] == 1
    inv["layers"][0]["files"]["usr/bin/sample"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="unmapped ELF"):
        source_module.verify_coverage(lock, inv, tmp_path)


def test_inventory_refuses_uncovered_ancestor_source_version(source_module, tmp_path):
    lock, inv = coverage_fixture()
    inv["layers"][0]["debian_packages"][0]["source_version"] = "1:2-0"
    with pytest.raises(ValueError, match="source identity"):
        source_module.verify_coverage(lock, inv, tmp_path)


def test_preferred_form_source_must_match_archive(source_module, tmp_path):
    raw = b"# source\r\nvalue = 1\r\n"
    (tmp_path / "source.tar").write_bytes(archive_bytes({"pkg/src/pkg/code.py": raw}))
    lock = {
        "artifacts": [
            {
                "path": "source.tar",
                "filename": "source.tar",
                "sha256": source_module.digest((tmp_path / "source.tar").read_bytes()),
            }
        ],
        "preferred_source": [
            {
                "artifact": source_module.digest(
                    (tmp_path / "source.tar").read_bytes()
                ),
                "prefix": "opt/env/site-packages/pkg/",
            }
        ],
    }
    files = {
        "opt/env/site-packages/pkg/code.py": {
            "sha256": source_module.digest(raw.replace(b"\r\n", b"\n"))
        }
    }
    assert (
        source_module.verify_preferred_source(lock, files, tmp_path, tmp_path)[
            "preferred_source_files"
        ]
        == 1
    )
    files["opt/env/site-packages/pkg/code.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preferred source"):
        source_module.verify_preferred_source(lock, files, tmp_path, tmp_path)


def test_unknown_elf_cannot_be_hidden_by_duplicate_tar_path(source_module, tmp_path):
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as archive:
        for raw in (b"\x7fELFhidden", b"plain"):
            member = tarfile.TarInfo("usr/bin/sample")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    raw = layer.getvalue()
    config = json.dumps(
        {"rootfs": {"diff_ids": ["sha256:" + source_module.digest(raw)]}}
    ).encode()
    saved = tmp_path / "image.tar"
    saved.write_bytes(
        archive_bytes(
            {
                "layer.tar": raw,
                "config.json": config,
                "manifest.json": json.dumps(
                    [{"Config": "config.json", "Layers": ["layer.tar"]}]
                ).encode(),
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate layer path"):
        source_module.inventory(saved)


def test_actual_lock_reuses_all_native_identities_and_has_no_private_paths():
    packaging = SCRIPT.parent
    whole = json.loads((packaging / "whole-source-lock.json").read_text())
    native = json.loads((packaging / "native-source-lock.json").read_text())
    refs = {a["sha256"]: a for a in whole["artifacts"]}
    for component in native["components"]:
        for artifact in component["artifacts"]:
            assert refs[artifact["sha256"]]["delivery"] == "native"
            assert refs[artifact["sha256"]]["filename"] == artifact["filename"]
    assert len(refs) == len(whole["artifacts"])
    assert not any("local_file" in wheel for wheel in whole["python_wheels"])
    assert "/home/ubuntu/" not in (packaging / "whole-source-lock.json").read_text()


def test_metadata_rejects_replaced_trust_anchor(source_module, tmp_path, monkeypatch):
    keyring = tmp_path / "keys.gpg"
    keyring.write_bytes(b"unexpected keyring")
    lock = {"debian_keyring_sha256": "0" * 64}
    monkeypatch.setattr(
        source_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("must check keyring before gpgv"),
    )
    with pytest.raises(ValueError, match="keyring SHA256"):
        source_module.verify_debian_metadata(lock, tmp_path, tmp_path, keyring)
