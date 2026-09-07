"""Whole-image source coverage must include overwritten and deleted bytes."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


SCRIPT = Path(__file__).parents[2] / "docker/workbench/ncore/base_sources.py"


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


def test_base_lock_delivers_only_retained_base_sources_and_has_no_private_paths():
    packaging = SCRIPT.parent
    lock = json.loads((packaging / "base-source-lock.json").read_text())
    artifacts = {a["sha256"]: a for a in lock["artifacts"]}
    assert len(artifacts) == len(lock["artifacts"])
    assert all(a.get("delivery") != "native" for a in artifacts.values())
    assert lock["python_wheels"] == []
    assert lock["python_distributions"] == []
    assert {c["id"] for c in lock["components"]} == {
        *(p["source"] for p in lock["debian_binaries"]),
        "cpython:3.12.12",
    }
    assert "/home/ubuntu/" not in (packaging / "base-source-lock.json").read_text()


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


def test_removed_native_receipt_cannot_authorize_new_image_bytes(
    source_module, tmp_path
):
    lock, inv = coverage_fixture()
    inv["layers"][0]["files"]["opt/ncore/absent.so"] = {"elf": True, "sha256": "f" * 64}
    (tmp_path / "native-reader.json").write_text(
        json.dumps({"libraries": {"/opt/ncore/absent.so": {"sha256": "f" * 64}}})
    )
    with pytest.raises(ValueError, match="unmapped ELF"):
        source_module.verify_coverage(lock, inv, tmp_path)


def test_published_filesystem_has_no_inherited_layers_or_base_pip():
    dockerfile = (SCRIPT.parent / "Dockerfile").read_text()
    final = dockerfile.rsplit("FROM scratch AS public-image", 1)
    assert len(final) == 2
    assert "COPY --from=assembled /public-root/ /" in final[1]
    assert "assemble-root" in final[0]
    lock = json.loads((SCRIPT.parent / "base-source-lock.json").read_text())
    assert lock["base_diff_ids"] == []
    assert lock["python_distributions"] == []
    assert len({p["name"] for p in lock["debian_binaries"]}) == len(
        lock["debian_binaries"]
    )
    assert not {"passwd", "gnupg2"} & {
        p["name"] for p in lock["debian_binaries"]
    }


def test_notice_only_components_do_not_deliver_full_sources():
    lock = json.loads((SCRIPT.parent / "base-source-lock.json").read_text())
    components = {c["name"]: c for c in lock["components"]}
    for name in ("openssl", "openssh", "libzstd", "lz4", "xz-utils", "libselinux"):
        assert components[name]["delivery"] == "notice"
        assert components[name]["artifacts"] == []
        assert components[name]["license_reason"]
    for name in ("glibc", "gcc-12", "bash", "coreutils"):
        assert components[name]["delivery"] == "source"
        assert components[name]["artifacts"]


def test_root_copy_checks_non_elf_source_and_refuses_drift(source_module, tmp_path):
    root, output = tmp_path / "root", tmp_path / "output"
    (root / "usr/bin").mkdir(parents=True)
    (root / "usr/bin/service").write_bytes(b"#!/bin/sh\nexit 0\n")
    row = {
        "path": "usr/bin/service",
        "sha256": source_module.digest(b"#!/bin/sh\nexit 0\n"),
    }
    source_module.copy_locked_file(root, output, row)
    assert (output / "usr/bin/service").read_bytes() == (
        root / "usr/bin/service"
    ).read_bytes()
    (root / "usr/bin/service").write_bytes(b"#!/bin/sh\nexit 9\n")
    with pytest.raises(ValueError, match="retained file SHA256"):
        source_module.copy_locked_file(root, tmp_path / "bad", row)


def test_root_copy_does_not_follow_unreviewed_links(source_module, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to("/outside")
    with pytest.raises(ValueError, match="retained link"):
        source_module.copy_locked_file(
            root, tmp_path / "out", {"path": "link", "link": "/expected"}
        )


def test_source_transformation_requires_reviewed_input_and_output(
    source_module, tmp_path
):
    original = tmp_path / "upstream.tar"
    original.write_bytes(
        archive_bytes(
            {
                "source/build.c": b"int main() {}",
                "source/tests/key": b"synthetic fixture",
            }
        )
    )
    recipe = {
        "input_sha256": source_module.file_digest(original),
        "remove": ["source/tests"],
        "output_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="transformed source SHA256"):
        source_module.transform_archive(original, tmp_path / "out.tar.xz", recipe)
    recipe["input_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="transformation input SHA256"):
        source_module.transform_archive(original, tmp_path / "out.tar.xz", recipe)


def test_source_selection_keeps_configure_inputs_and_all_production_bytes(
    source_module, tmp_path, monkeypatch
):
    original, transformed = tmp_path / "input.tar", tmp_path / "source.tar.xz"
    original.write_bytes(
        archive_bytes(
            {
                "pkg/src/library.c": b"int exported(void) { return 42; }\n",
                "pkg/COPYING": b"fixture license text",
                "pkg/tests/Makefile.in": b"# configure input\n",
                "pkg/tests/private.key": b"unneeded upstream fixture",
            }
        )
    )
    recipe = {
        "input_sha256": source_module.file_digest(original),
        "remove": ["pkg/tests"],
        "keep": ["pkg/tests/Makefile.in"],
        "output_sha256": "0" * 64,
    }
    real_digest = source_module.file_digest
    monkeypatch.setattr(
        source_module,
        "file_digest",
        lambda p: "0" * 64 if p == transformed else real_digest(p),
    )
    source_module.transform_archive(original, transformed, recipe)
    recipe["output_sha256"] = real_digest(transformed)
    monkeypatch.setattr(source_module, "file_digest", real_digest)
    repeated = tmp_path / "repeat.tar.xz"
    source_module.transform_archive(original, repeated, recipe)
    assert repeated.read_bytes() == transformed.read_bytes()
    with tarfile.open(transformed) as archive:
        assert "pkg/tests/private.key" not in archive.getnames()
        assert (
            archive.extractfile("pkg/src/library.c").read()
            == b"int exported(void) { return 42; }\n"
        )
        assert archive.extractfile("pkg/COPYING").read() == b"fixture license text"
        assert (
            archive.extractfile("pkg/tests/Makefile.in").read()
            == b"# configure input\n"
        )


def test_test_patch_filter_preserves_production_hunks_and_kept_build_scripts(
    source_module,
):
    production = b"--- a/src/library.c\n+++ b/src/library.c\n@@ -1 +1 @@\n-old\n+new\n"
    test = (
        b"--- a/src/tests/vector.c\n+++ b/src/tests/vector.c\n@@ -1 +1 @@\n-old\n+new\n"
    )
    makefile = test.replace(b"vector.c", b"Makefile.in")
    assert (
        source_module.filtered_patch(
            production + test + makefile, {"src/tests"}, {"src/tests/Makefile.in"}
        )
        == production + makefile
    )


def test_required_source_cannot_be_replaced_with_a_url(source_module):
    lock = {
        "schema": 2,
        "artifacts": [],
        "components": [
            {
                "delivery": "source",
                "artifacts": [],
                "license_reason": "GPL-3.0-or-later",
                "url": "https://example.org/source",
            }
        ],
    }
    with pytest.raises(ValueError, match="corresponding source"):
        source_module.validate_delivery(lock)


def test_source_transform_without_buildability_proof_is_rejected(source_module):
    lock = {
        "schema": 2,
        "components": [],
        "artifacts": [
            {
                "sha256": "b" * 64,
                "transformation": {
                    "input_sha256": "a" * 64,
                    "output_sha256": "b" * 64,
                    "reason": "source-only tests",
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="buildability proof"):
        source_module.validate_delivery(lock)


def test_real_source_transformations_keep_required_build_inputs(source_module):
    lock = json.loads((SCRIPT.parent / "base-source-lock.json").read_text())
    source_module.validate_delivery(lock)
    transformed = {
        a["filename"].split("_")[0]: a["transformation"]
        for a in lock["artifacts"]
        if "transformation" in a
    }
    assert set(transformed) == {"gcc-12", "libgcrypt20", "systemd"}
    gcc = next(iter(transformed["gcc-12"]["nested"].values()))
    assert any("gcc/testsuite/selftests/" in p for p in gcc["keep"])
    assert any(p.endswith("testsuite/Makefile.in") for p in gcc["keep"])
    for recipe in transformed.values():
        proof = recipe["build_proof"]
        assert proof["status"] == "built-and-libraries-installed"
        assert proof["output_elf_sha256"]
        assert proof["delivered_source_sha256"] == recipe["output_sha256"]


def test_embedded_original_notice_is_delivered_and_hash_checked(
    source_module, tmp_path
):
    import base64

    raw = b"Original upstream copyright and permission text.\n"
    item = {
        "path": "usr/share/doc/library/NOTICE",
        "content_base64": base64.b64encode(raw).decode(),
        "sha256": source_module.digest(raw),
    }
    source_module.copy_locked_file(tmp_path / "unused", tmp_path / "out", item)
    assert (tmp_path / "out" / item["path"]).read_bytes() == raw
    item["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="notice SHA256"):
        source_module.copy_locked_file(tmp_path / "unused", tmp_path / "bad", item)
