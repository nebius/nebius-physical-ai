"""Separate authenticated build inputs from source verified in final image bytes."""

import copy
import importlib.util
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile

import pytest


PACKAGING = Path(__file__).parents[2] / "docker/workbench/ncore"


@pytest.fixture
def sources():
    spec = importlib.util.spec_from_file_location(
        "base_sources", PACKAGING / "base_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tar(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return stream.getvalue()


def _artifact(sources, path, raw, **fields):
    return {
        "path": path,
        "filename": Path(path).name,
        "sha256": sources.digest(raw),
        "url": "https://snapshot.debian.org/" + path,
        **fields,
    }


def _repository(sources, source_artifacts):
    packages = (
        "Package: sample\nVersion: 2:3.0-1+b1\nArchitecture: amd64\n"
        "Source: origin (2:3.0-1)\nFilename: pool/sample.deb\n"
        f"SHA256: {sources.digest(b'binary')}\n\n"
    ).encode()
    checksums = "".join(f" {a['sha256']} 1 {a['filename']}\n" for a in source_artifacts)
    source_index = (
        "Package: origin\nVersion: 2:3.0-1\nChecksums-Sha256:\n" + checksums + "\n"
    ).encode()
    raw = {
        "Packages.xz": lzma.compress(packages),
        "Sources.xz": lzma.compress(source_index),
    }
    release = "SHA256:\n" + "".join(
        f" {sources.digest(value)} {len(value)} main/{name}\n"
        for name, value in raw.items()
    )
    # The gpgv boundary is controlled separately; parsing/decompression/hashes are real.
    raw["InRelease"] = b"signed release fixture"
    artifacts = [
        _artifact(sources, "metadata/snapshot/" + name, value, delivery="build-only")
        for name, value in raw.items()
    ]
    by_name = {a["filename"]: a for a in artifacts}
    repo = {
        "id": "snapshot",
        "inrelease": by_name["InRelease"]["sha256"],
        "indexes": {
            name: {"artifact": by_name[name]["sha256"], "release_path": "main/" + name}
            for name in ("Packages.xz", "Sources.xz")
        },
    }
    return repo, artifacts, raw, release.encode()


def _selection(sources, repo, artifacts):
    source_id = "debian:origin@2:3.0-1"
    return {
        "components": [
            {
                "id": source_id,
                "kind": "debian-source",
                "name": "origin",
                "delivery": "source",
                "license_reason": "GPL source required",
                "artifacts": [a["sha256"] for a in artifacts],
                "source_index": repo["indexes"]["Sources.xz"]["artifact"],
            }
        ],
        "debian_binaries": [
            {
                "name": "sample",
                "version": "2:3.0-1+b1",
                "architecture": "amd64",
                "source": source_id,
                "sha256": sources.digest(b"binary"),
                "url": "https://snapshot.debian.org/pool/sample.deb",
                "package_index": repo["indexes"]["Packages.xz"]["artifact"],
                "files": [],
                "elf_files": [],
            }
        ],
    }


def _lock(sources, repo, artifacts, metadata_artifacts):
    return {
        "schema": 2,
        "base_diff_ids": [],
        "artifacts": artifacts + metadata_artifacts,
        "debian_repositories": [repo],
        "debian_keyring_sha256": sources.digest(b"public keys"),
        **_selection(sources, repo, artifacts),
        "notices": [
            {
                "path": "usr/share/doc/sample/copyright",
                "sha256": sources.digest(b"notice"),
            }
        ],
        "cpython_files": [],
        "cpython_elf_files": [],
        "python_wheels": [],
        "python_distributions": [],
    }


@pytest.fixture
def delivery(sources, tmp_path, monkeypatch):
    source_bytes = {
        "sources/origin.orig.tar.xz": _tar(
            {"origin/library.c": b"int exported(void) { return 42; }\n"}
        ),
        "sources/origin.dsc": b"Original source description\n",
    }
    artifacts = [_artifact(sources, path, raw) for path, raw in source_bytes.items()]
    repo, metadata_artifacts, metadata_bytes, release = _repository(sources, artifacts)
    lock = _lock(sources, repo, artifacts, metadata_artifacts)
    cache, annex, metadata = (
        tmp_path / name for name in ("cache", "annex", "build-inputs")
    )
    cache.mkdir()
    for path, raw in {**source_bytes, **metadata_bytes}.items():
        (cache / Path(path).name).write_bytes(raw)
    keyring = tmp_path / "keyring.gpg"
    keyring.write_bytes(b"public keys")

    def gpgv(argv, **kwargs):
        assert argv == [
            "gpgv",
            "--keyring",
            str(keyring.resolve()),
            "--output",
            "-",
            str(metadata / "metadata/snapshot/InRelease"),
        ]
        assert kwargs == {"capture_output": True, "check": True}
        return subprocess.CompletedProcess(argv, 0, stdout=release)

    monkeypatch.setattr(sources.subprocess, "run", gpgv)
    sources.assemble(lock, annex, annex, [cache], metadata=metadata, offline=True)
    return lock, annex, metadata, keyring


def _inventory(sources, tmp_path, files, extra_layers=()):
    layers = [_tar(files), *extra_layers]
    entries = {f"layer{i}.tar": raw for i, raw in enumerate(layers)}
    entries["config.json"] = json.dumps(
        {
            "rootfs": {
                "diff_ids": ["sha256:" + sources.digest(raw) for raw in layers],
            }
        }
    ).encode()
    entries["manifest.json"] = json.dumps(
        [
            {
                "Config": "config.json",
                "Layers": [f"layer{i}.tar" for i in range(len(layers))],
            }
        ]
    ).encode()
    image = tmp_path / "image.tar"
    image.write_bytes(_tar(entries))
    return sources.inventory(image)


def _published_files(sources, lock, annex):
    files = {"var/lib/dpkg/status": b"", "usr/share/doc/sample/copyright": b"notice"}
    for item in lock["artifacts"]:
        if item.get("delivery") != "build-only":
            files[f"{sources.SOURCE_ANNEX}/{item['path']}"] = (
                annex / item["path"]
            ).read_bytes()
    return files


def test_signed_inputs_are_separate_and_authenticate_binary_source_and_archives(
    sources, delivery
):
    lock, annex, metadata, keyring = delivery
    assert not (annex / "metadata").exists()
    assert len(list(metadata.rglob("*.xz"))) == 2
    assert sources.verify_debian_metadata(lock, annex, annex, keyring, metadata) == {
        "authenticated_binary_versions": 1,
        "authenticated_source_versions": 1,
    }


@pytest.mark.parametrize("name", ["InRelease", "Packages.xz", "Sources.xz"])
@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_metadata_missing_or_tampered_fails_before_authentication(
    sources, delivery, name, change
):
    lock, annex, metadata, keyring = delivery
    path = metadata / "metadata/snapshot" / name
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed metadata")
    with pytest.raises(ValueError, match="SHA256 mismatch or missing artifact"):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


def test_signature_failure_is_fatal(sources, delivery, monkeypatch):
    lock, annex, metadata, keyring = delivery

    def reject(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr=b"BAD signature")

    monkeypatch.setattr(sources.subprocess, "run", reject)
    with pytest.raises(subprocess.CalledProcessError):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


def test_missing_trust_anchor_cannot_reach_signature_verification(
    sources, delivery, monkeypatch
):
    lock, annex, metadata, keyring = delivery
    keyring.unlink()
    monkeypatch.setattr(
        sources.subprocess, "run", lambda *a, **k: pytest.fail("gpgv reached")
    )
    with pytest.raises(FileNotFoundError):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


@pytest.mark.parametrize("name", ["Packages.xz", "Sources.xz"])
def test_rehashed_index_still_requires_its_signed_release_hash(sources, delivery, name):
    lock, annex, metadata, keyring = delivery
    item = next(a for a in lock["artifacts"] if a["filename"] == name)
    path = metadata / item["path"]
    path.write_bytes(lzma.compress(lzma.decompress(path.read_bytes()), preset=0))
    old_sha, item["sha256"] = item["sha256"], sources.file_digest(path)
    assert old_sha != item["sha256"]
    lock["debian_repositories"][0]["indexes"][name]["artifact"] = item["sha256"]
    if name == "Sources.xz":
        lock["components"][0]["source_index"] = item["sha256"]
    else:
        lock["debian_binaries"][0]["package_index"] = item["sha256"]
    with pytest.raises(ValueError, match="signed index SHA256/size mismatch"):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


@pytest.mark.parametrize("field", ["sha256", "source", "url", "package_index"])
def test_authenticated_binary_mapping_cannot_be_rewritten(sources, delivery, field):
    lock, annex, metadata, keyring = delivery
    lock["debian_binaries"][0][field] = "changed"
    with pytest.raises(ValueError, match="binary/source"):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


def test_authenticated_source_archive_set_is_complete(sources, delivery):
    lock, annex, metadata, keyring = delivery
    lock["components"][0]["artifacts"].pop()
    with pytest.raises(ValueError, match="incomplete authenticated source"):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


@pytest.mark.parametrize("field", ["filename", "source_index"])
def test_authenticated_source_identity_cannot_change(sources, delivery, field):
    lock, annex, metadata, keyring = delivery
    if field == "filename":
        lock["artifacts"][0]["filename"] = "substituted.tar.xz"
    else:
        lock["components"][0][field] = "changed"
    with pytest.raises(ValueError, match="source filename mismatch|unverified Debian"):
        sources.verify_debian_metadata(lock, annex, annex, keyring, metadata)


@pytest.mark.parametrize("mode", ["missing", "inside", "parent", "equal", "symlink"])
def test_metadata_directory_cannot_overlap_source_annex(
    sources, delivery, tmp_path, mode
):
    lock, annex, metadata, _ = delivery
    if mode == "symlink":
        metadata = tmp_path / "alias"
        metadata.symlink_to(annex, target_is_directory=True)
    else:
        metadata = {
            "missing": None,
            "inside": annex / "build",
            "parent": annex.parent,
            "equal": annex,
        }.get(mode, metadata)
    with pytest.raises(ValueError, match="metadata directory"):
        sources.verify_artifacts(lock, annex, annex, metadata)


@pytest.mark.parametrize(
    "mutation", ["missing-class", "extra-input", "missing-index", "source-as-metadata"]
)
def test_build_only_classification_exactly_matches_signed_population(
    sources, delivery, mutation
):
    lock = copy.deepcopy(delivery[0])
    if mutation == "missing-class":
        del lock["artifacts"][-1]["delivery"]
    elif mutation == "extra-input":
        lock["artifacts"].append(
            _artifact(sources, "metadata/extra", b"unneeded", delivery="build-only")
        )
    elif mutation == "missing-index":
        del lock["debian_repositories"][0]["indexes"]["Sources.xz"]
    else:
        lock["artifacts"][0]["delivery"] = "build-only"
        lock["debian_repositories"][0]["inrelease"] = lock["artifacts"][0]["sha256"]
    with pytest.raises(ValueError, match="signed metadata|corresponding source"):
        sources.validate_delivery(lock)


@pytest.mark.parametrize(
    "change", ["missing", "changed", "renamed", "original-transform"]
)
def test_valid_external_annex_cannot_replace_final_image_source(
    sources, delivery, tmp_path, change
):
    lock, annex, metadata, _ = delivery
    files = _published_files(sources, lock, annex)
    path = sources.SOURCE_ANNEX + "/" + lock["artifacts"][0]["path"]
    assert sources.verify_coverage(lock, _inventory(sources, tmp_path, files), annex)
    if change == "missing":
        del files[path]
    elif change == "changed":
        files[path] = b"altered source"
    elif change == "renamed":
        files[path + ".renamed"] = files.pop(path)
    else:
        original = _tar(
            {
                "origin/library.c": b"int exported(void) { return 42; }\n",
                "origin/tests/fixture": b"optional source fixture",
            }
        )
        artifact = lock["artifacts"][0]
        artifact["transformation"] = {
            "input_sha256": sources.digest(original),
            "output_sha256": artifact["sha256"],
            "reason": "remove optional fixtures",
            "build_proof": {"status": "fixture"},
        }
        lock["components"][0]["artifacts"][0] = sources.digest(original)
        files[path] = original
    assert sources.verify_artifacts(lock, annex, annex, metadata)
    with pytest.raises(ValueError, match="missing or changed delivered source"):
        sources.verify_coverage(lock, _inventory(sources, tmp_path, files), annex)


@pytest.mark.parametrize(
    "location", ["annex", "elsewhere", "renamed", "whiteouted", "build-directory"]
)
def test_published_metadata_is_rejected_by_path_and_hash_in_all_layers(
    sources, delivery, tmp_path, location
):
    lock, annex, metadata, _ = delivery
    files = _published_files(sources, lock, annex)
    item = lock["artifacts"][-1]
    path = item["path"]
    raw = b"changed bytes at a metadata path"
    extra = ()
    if location == "annex":
        path = sources.SOURCE_ANNEX + "/" + path
    elif location == "elsewhere":
        path = "elsewhere/" + path
    elif location in {"renamed", "whiteouted"}:
        path = "unrelated/renamed.bin"
        raw = (metadata / item["path"]).read_bytes()
        if location == "whiteouted":
            extra = (_tar({"unrelated/.wh.renamed.bin": b""}),)
    elif location == "build-directory":
        path = "build/base-metadata/accidental-copy"
    files[path] = raw
    inv = _inventory(sources, tmp_path, files, extra)
    if location == "whiteouted":
        assert path not in inv["final_files"]
    with pytest.raises(ValueError, match="published build-only metadata"):
        sources.verify_coverage(lock, inv, annex)


def test_final_root_requires_sources_and_original_notices(sources, delivery, tmp_path):
    lock, annex, _, _ = delivery
    root = tmp_path / "root"
    for path, raw in _published_files(sources, lock, annex).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    assert sources.verify_root(lock, root)
    notice = root / lock["notices"][0]["path"]
    notice.write_bytes(b"changed notice")
    with pytest.raises(ValueError, match="retained sha256"):
        sources.verify_root(lock, root)
    notice.write_bytes(b"notice")
    (root / sources.SOURCE_ANNEX / lock["artifacts"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="delivered source"):
        sources.verify_root(lock, root)
    target = root / sources.SOURCE_ANNEX / lock["artifacts"][0]["path"]
    target.symlink_to(annex / lock["artifacts"][0]["path"])
    with pytest.raises(ValueError, match="delivered source"):
        sources.verify_root(lock, root)


def test_extra_layer_cannot_hide_delivery_changes(sources, delivery, tmp_path):
    lock, annex, _, _ = delivery
    files = _published_files(sources, lock, annex)
    inv = _inventory(sources, tmp_path, files, (_tar({"unrelated": b"extra layer"}),))
    with pytest.raises(ValueError, match="exactly one assembled layer"):
        sources.verify_coverage(lock, inv, annex)


def test_prepare_source_checks_delivered_archives_without_build_inputs(
    sources, delivery, tmp_path
):
    lock, annex, metadata, _ = delivery
    raw = _tar(
        {"debian/patches/series": b"", "debian/rules": b"# original build rules\n"}
    )
    artifact = _artifact(sources, "sources/origin.debian.tar.xz", raw)
    lock["artifacts"].append(artifact)
    lock["components"][0]["artifacts"].append(artifact["sha256"])
    (annex / artifact["path"]).write_bytes(raw)
    for item in lock["artifacts"]:
        if item.get("delivery") == "build-only":
            (metadata / item["path"]).unlink()
    root = sources.prepare_source(
        lock, annex, lock["components"][0]["id"], tmp_path / "prepared"
    )
    assert (root / "library.c").read_bytes() == b"int exported(void) { return 42; }\n"
    assert (root / "debian/rules").read_bytes() == b"# original build rules\n"
    (annex / artifact["path"]).write_bytes(b"corrupt source")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        sources.prepare_source(
            lock, annex, lock["components"][0]["id"], tmp_path / "bad"
        )


def test_current_lock_keeps_all_required_source_outside_six_build_inputs(sources):
    lock = json.loads((PACKAGING / "base-source-lock.json").read_text())
    sources.validate_delivery(lock)
    build_only = [a for a in lock["artifacts"] if a.get("delivery") == "build-only"]
    assert len(build_only) == 6
    assert {a["filename"] for a in build_only} == {
        "InRelease",
        "Packages.xz",
        "Sources.xz",
    }
    artifacts = {
        a.get("transformation", {}).get("input_sha256", a["sha256"]): a
        for a in lock["artifacts"]
    }
    for component in lock["components"]:
        for sha in component["artifacts"]:
            assert artifacts[sha].get("delivery", "source") == "source"
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    assert dockerfile.count("--metadata /build/base-metadata") == 3
    assert (
        "from=base-source-delivery,source=/build/base-metadata,target=/build/base-metadata"
        in dockerfile
    )
    assert "COPY --from=base-source-delivery /build" not in dockerfile
