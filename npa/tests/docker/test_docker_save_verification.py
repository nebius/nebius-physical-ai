"""Exercise generic Docker-save identity before the complete-byte scanner."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import core  # noqa: E402
from image_byte_scan import docker_save_verification as VERIFIER  # noqa: E402
from image_byte_scan import prepare  # noqa: E402


@pytest.fixture(autouse=True)
def authorized_roots(tmp_path: Path):
    tmp_path.chmod(0o700)
    with core.authorized_roots(tmp_path, ROOT):
        yield


def _tar(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _archive(
    tmp_path: Path,
    *,
    compressed: bool = False,
    oci_layout: bool = False,
    layer_entries: list[tuple[str, bytes]] | None = None,
    layer_payload: bytes | None = None,
    repeat: int = 1,
) -> tuple[Path, str]:
    raw_layer = _tar(layer_entries or [("opt/result.txt", b"physical-ai")])
    stored_layer = (
        layer_payload
        if layer_payload is not None
        else gzip.compress(raw_layer, mtime=0)
        if compressed
        else raw_layer
    )
    config = {
        "rootfs": {
            "type": "layers",
            "diff_ids": ["sha256:" + hashlib.sha256(raw_layer).hexdigest()] * repeat,
        }
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    stored_layer_digest = hashlib.sha256(stored_layer).hexdigest()
    config_name = (
        "blobs/sha256/" + config_digest if oci_layout else config_digest + ".json"
    )
    layer_name = (
        "blobs/sha256/" + stored_layer_digest if oci_layout else "layer/layer.tar"
    )
    manifest = [
        {
            "Config": config_name,
            "Layers": [layer_name] * repeat,
            "RepoTags": ["npa-robocasa:test"],
        }
    ]
    members = [
        ("manifest.json", json.dumps(manifest).encode()),
        (config_name, config_bytes),
        (layer_name, stored_layer),
    ]
    image_id = "sha256:" + config_digest
    if oci_layout:
        manifest_document = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": "sha256:" + config_digest,
                "size": len(config_bytes),
            },
            "layers": [
                {
                    "mediaType": (
                        "application/vnd.oci.image.layer.v1.tar+gzip"
                        if compressed
                        else "application/vnd.oci.image.layer.v1.tar"
                    ),
                    "digest": "sha256:" + stored_layer_digest,
                    "size": len(stored_layer),
                }
            ]
            * repeat,
        }
        manifest_bytes = json.dumps(manifest_document, sort_keys=True).encode()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        index = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": manifest_document["mediaType"],
                    "digest": "sha256:" + manifest_digest,
                    "size": len(manifest_bytes),
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
        members.extend(
            [
                ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                ("index.json", json.dumps(index).encode()),
                ("blobs/sha256/" + manifest_digest, manifest_bytes),
            ]
        )
        image_id = "sha256:" + manifest_digest
    archive = tmp_path / "image.tar"
    archive.write_bytes(_tar(members))
    archive.chmod(0o600)
    return archive, image_id


@pytest.mark.parametrize(
    ("compressed", "oci_layout"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_verifier_binds_complete_layer_and_config(
    tmp_path: Path, compressed: bool, oci_layout: bool
) -> None:
    archive, image_id = _archive(tmp_path, compressed=compressed, oci_layout=oci_layout)

    report = VERIFIER.verify(archive, image_id)

    assert report["schema_version"] == VERIFIER.SCHEMA
    assert report["valid"] is True
    assert report["expected_image_id"] == image_id
    if oci_layout:
        assert report["image_manifest_digest"] == image_id
    else:
        assert report["image_config_digest"] == image_id
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 1
    assert report["content_bytes_read"] == len(b"physical-ai")
    assert (
        report["docker_save_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert core.verification_archive_digest(report) == report["docker_save_sha256"]
    fd = os.open(archive, os.O_RDONLY)
    try:
        layers = core.graph(fd, os.fstat(fd).st_size, report, image_id)
    finally:
        os.close(fd)
    assert len(layers) == 1


def test_repeated_layer_member_is_physically_verified_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True, repeat=3)
    calls = []
    original = VERIFIER._verify_layer

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(VERIFIER, "_verify_layer", counted)
    report = VERIFIER.verify(archive, image_id)

    assert report["layer_count"] == 3
    assert report["regular_files_read"] == 3
    assert calls == [True]


def test_decoded_layer_limit_applies_during_first_diff_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_layer = _tar([("opt/result.txt", b"physical-ai")])
    archive, image_id = _archive(tmp_path, compressed=True)
    monkeypatch.setattr(core, "DOCKER_SAVE_DECODED_LAYER_LIMIT", len(raw_layer) - 1)
    population_calls = []
    monkeypatch.setattr(
        VERIFIER,
        "_regular_population",
        lambda *_args: population_calls.append(True),
    )

    with pytest.raises(core.ScanError, match="docker_save_decoded_layer_limit"):
        VERIFIER.verify(archive, image_id)

    assert population_calls == []


def test_verifier_rejects_concatenated_gzip_before_later_optional_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_layer = _tar([("opt/result.txt", b"physical-ai")])
    first_member = gzip.compress(raw_layer)
    oversized_later_header = b"\x1f\x8b\x08\x08" + b"\0" * 4 + b"\0\xff" + b"A" * 32
    archive, image_id = _archive(
        tmp_path,
        compressed=True,
        layer_payload=first_member + oversized_later_header,
        layer_entries=[("opt/result.txt", b"physical-ai")],
    )
    monkeypatch.setattr(core, "GZIP_HEADER_LIMIT", 16)
    population_calls = []
    monkeypatch.setattr(
        VERIFIER,
        "_regular_population",
        lambda *_args: population_calls.append(True),
    )

    with pytest.raises(core.ScanError, match="gzip_trailing_member_or_bytes"):
        VERIFIER.verify(archive, image_id)

    assert population_calls == []


def test_verifier_regular_file_limit_precedes_population_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path, layer_entries=[("opt/result.txt", b"12345")])
    monkeypatch.setattr(core, "TAR_REGULAR_FILE_LIMIT", 4)

    with pytest.raises(core.ScanError, match="tar_regular_file_limit"):
        VERIFIER.verify(archive, image_id)


def test_verifier_zero_run_limit_is_incremental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "ZERO_RECORD_LIMIT", 512)
    sink = VERIFIER._PopulationSink()
    sink.zeros(b"\0" * 512, {})

    with pytest.raises(core.ScanError, match="docker_save_zero_record_limit"):
        sink.zeros(b"\0" * 512, {})


def test_repeated_layer_member_limit_fails_before_layer_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(
        tmp_path,
        oci_layout=True,
        repeat=core.DOCKER_SAVE_LAYER_MEMBER_REPEAT_LIMIT + 1,
    )
    calls = []
    monkeypatch.setattr(
        VERIFIER,
        "_verify_layer",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(core.ScanError, match="docker_save_layer_member_repeat_limit"):
        VERIFIER.verify(archive, image_id)
    assert calls == []


def test_verifier_opens_once_and_hashes_parses_rehashes_held_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    opened = []
    digested = []
    duplicated = []
    original_open = core.open_private_fd
    original_digest = core.descriptor_digest
    original_dup = os.dup
    original_path_open = Path.open

    def open_once(path, **kwargs):
        result = original_open(path, **kwargs)
        if Path(path) == archive:
            opened.append(result[1])
        return result

    def digest_descriptor(fd):
        digested.append(fd)
        return original_digest(fd)

    def duplicate_descriptor(fd):
        duplicated.append(fd)
        return original_dup(fd)

    def reject_archive_reopen(path, *args, **kwargs):
        if path == archive:
            raise AssertionError("archive reopened by path")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(core, "open_private_fd", open_once)
    monkeypatch.setattr(core, "descriptor_digest", digest_descriptor)
    monkeypatch.setattr(os, "dup", duplicate_descriptor)
    monkeypatch.setattr(Path, "open", reject_archive_reopen)

    report = VERIFIER.verify(archive, image_id)

    assert report["valid"]
    assert len(opened) == 1
    assert digested == [opened[0], opened[0]]
    assert duplicated == [opened[0]]


@pytest.mark.parametrize("replacement", ["permissions", "symlink"])
def test_verifier_uses_private_nofollow_archive_contract(
    tmp_path: Path, replacement: str
) -> None:
    archive, image_id = _archive(tmp_path)
    candidate = archive
    expected = "input_permissions"
    if replacement == "permissions":
        archive.chmod(0o640)
    else:
        candidate = tmp_path / "archive-link.tar"
        candidate.symlink_to(archive)
        expected = "input_symlink"

    with pytest.raises(core.ScanError, match=expected):
        VERIFIER.verify(candidate, image_id)


def test_outer_extension_is_rejected_before_tarfile_can_allocate_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = tarfile.TarInfo("pax")
    member.type = tarfile.XHDTYPE
    member.size = 2**30
    archive = tmp_path / "image.tar"
    archive.write_bytes(member.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024)
    archive.chmod(0o600)

    monkeypatch.setattr(
        VERIFIER.tarfile,
        "open",
        lambda *args, **kwargs: pytest.fail("tarfile opened before outer preflight"),
    )
    with pytest.raises(core.ScanError, match="docker_save_outer_extension_unsupported"):
        VERIFIER.verify(archive, "sha256:" + "f" * 64)
    fd = os.open(archive, os.O_RDONLY)
    try:
        with pytest.raises(
            core.ScanError, match="docker_save_outer_extension_unsupported"
        ):
            core.graph(fd, os.fstat(fd).st_size, {}, "sha256:" + "f" * 64)
    finally:
        os.close(fd)


def test_outer_metadata_is_bounded_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path)
    monkeypatch.setattr(core, "DOCKER_SAVE_METADATA_LIMIT", 64)

    with pytest.raises(
        core.ScanError, match="docker_save_metadata_regular_or_too_large"
    ):
        VERIFIER.verify(archive, image_id)


def test_verifier_rehash_detects_same_inode_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path)
    original = core.descriptor_digest
    calls = 0

    def mutate_before_rehash(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            writer = os.open(archive, os.O_WRONLY | os.O_CLOEXEC)
            try:
                os.pwrite(writer, b"X", 0)
            finally:
                os.close(writer)
        return original(fd)

    monkeypatch.setattr(core, "descriptor_digest", mutate_before_rehash)

    with pytest.raises(core.ScanError, match="docker_save_archive_changed"):
        VERIFIER.verify(archive, image_id)
    assert calls == 2


def test_verifier_rejects_path_replacement_while_parsing_held_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path)
    original_bytes = archive.read_bytes()
    original_graph = VERIFIER._graph

    def replace_path(*args, **kwargs):
        result = original_graph(*args, **kwargs)
        replacement = tmp_path / "replacement.tar"
        replacement.write_bytes(original_bytes)
        replacement.chmod(0o600)
        os.replace(replacement, archive)
        return result

    monkeypatch.setattr(VERIFIER, "_graph", replace_path)

    with pytest.raises(core.ScanError, match="docker_save_archive_changed"):
        VERIFIER.verify(archive, image_id)


def test_verifier_rejects_wrong_inspected_image_id(tmp_path: Path) -> None:
    archive, _image_id = _archive(tmp_path)

    with pytest.raises(core.ScanError, match="docker_save_expected_image"):
        VERIFIER.verify(archive, "sha256:" + "f" * 64)


def test_verifier_rejects_layer_that_disagrees_with_diff_id(tmp_path: Path) -> None:
    archive, image_id = _archive(tmp_path, layer_payload=_tar([("changed", b"bytes")]))

    with pytest.raises(core.ScanError, match="docker_save_layer_diff_id"):
        VERIFIER.verify(archive, image_id)


def test_verifier_rejects_unsafe_inner_path(tmp_path: Path) -> None:
    archive, image_id = _archive(
        tmp_path, layer_entries=[("../escape", b"not extracted")]
    )

    with pytest.raises(core.ScanError, match="tar_path_escape"):
        VERIFIER.verify(archive, image_id)


def test_preparation_accepts_bound_generic_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(VERIFIER.verify(archive, image_id)))
    inventory = tmp_path / "literals.json"
    inventory.write_text(json.dumps({"literals": ["absent-private-marker"]}))
    tools = tmp_path / "tools.json"
    native = tmp_path / "native.json"
    tools.write_text("{}")
    native.write_text("{}")
    for path in (archive, verification, inventory, tools, native):
        path.chmod(0o600)

    class FakeDetector:
        def __init__(self, *_args: object) -> None:
            self.joined = False

        def finish(self) -> dict[str, object]:
            self.joined = True
            return {}

        def abort(self) -> None:
            self.joined = True

    monkeypatch.setattr(
        prepare,
        "tools_bindings",
        lambda _path: (
            {"path": str(tools), "sha256": "synthetic"},
            {"path": str(ROOT / ".gitleaks.toml"), "sha256": "synthetic"},
        ),
    )
    monkeypatch.setattr(
        prepare,
        "native_engine",
        lambda _path: {"kind": "synthetic-unexecuted-binding"},
    )
    monkeypatch.setattr(core, "Detector", FakeDetector)
    monkeypatch.setattr(core, "input_snapshots", lambda _authorization: [])
    args = SimpleNamespace(
        tools_receipt=tools,
        native_receipt=native,
        archive=archive,
        verification_report=verification,
        expected_image_id=image_id,
        policy_mode="exact-literals",
        literal_inventory=inventory,
        literal_matching_policy="exact-substring-v1",
    )
    output = tmp_path / "authorization"
    output.mkdir(mode=0o700)
    with core.authorized_roots(tmp_path, ROOT):
        result = prepare.authorize(args, output)
        assert (
            core.bound_json(result["verification_report"])["schema_version"]
            == VERIFIER.SCHEMA
        )


def test_cli_writes_owner_only_bound_report(tmp_path: Path) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    output = tmp_path / "cli-output"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "npa/scripts/image_byte_scan/docker_save_verification.py"),
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(ROOT),
            "--archive",
            str(archive),
            "--expected-image-id",
            image_id,
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result
    report = json.loads((output / "verification.json").read_text())
    assert report["expected_image_id"] == image_id
    assert (output / "verification.json").stat().st_mode & 0o777 == 0o600


def test_cli_rejects_output_directory_replaced_before_held_fd_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    output = tmp_path / "cli-output"
    original_publish = core.publish_staged_private_json

    def replace_after_publish(held_fd, name, result, staged):
        identity = original_publish(held_fd, name, result, staged)
        output.rename(tmp_path / "displaced-output")
        output.mkdir(mode=0o700)
        return identity

    monkeypatch.setattr(core, "publish_staged_private_json", replace_after_publish)
    result = VERIFIER.main(
        [
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(ROOT),
            "--archive",
            str(archive),
            "--expected-image-id",
            image_id,
            "--output-dir",
            str(output),
        ]
    )

    assert result == 1
    assert capsys.readouterr().out == "Docker-save graph verification failed\n"


def test_cli_does_not_publish_valid_report_before_archive_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    output = tmp_path / "cli-output"

    def reject_finalize(_self):
        raise core.ScanError("synthetic_final_archive_change")

    monkeypatch.setattr(VERIFIER._ArchiveVerification, "finalize", reject_finalize)
    result = VERIFIER.main(
        [
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(ROOT),
            "--archive",
            str(archive),
            "--expected-image-id",
            image_id,
            "--output-dir",
            str(output),
        ]
    )

    assert result == 1
    assert capsys.readouterr().out == "Docker-save graph verification failed\n"
    assert output.exists()
    assert not (output / "verification.json").exists()
    assert not (output / "verification.json.pending").exists()
