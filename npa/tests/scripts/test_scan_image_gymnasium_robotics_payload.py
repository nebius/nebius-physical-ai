from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/scripts/scan_image_gymnasium_robotics_payload.py"
SPEC = importlib.util.spec_from_file_location("gymnasium_payload_scan", SCRIPT)
assert SPEC and SPEC.loader
SCAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCAN
SPEC.loader.exec_module(SCAN)
VERIFIER_SCRIPT = ROOT / "npa/docker/workbench/gymnasium-robotics/verify_image.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "gymnasium_image_verifier", VERIFIER_SCRIPT
)
assert VERIFIER_SPEC and VERIFIER_SPEC.loader
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
sys.modules[VERIFIER_SPEC.name] = VERIFIER
VERIFIER_SPEC.loader.exec_module(VERIFIER)
BOOTSTRAP_SCRIPT = (
    ROOT / "npa/docker/workbench/gymnasium-robotics/runtime-bootstrap.py"
)
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "gymnasium_runtime_bootstrap_for_scan_tests", BOOTSTRAP_SCRIPT
)
assert BOOTSTRAP_SPEC and BOOTSTRAP_SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
sys.modules[BOOTSTRAP_SPEC.name] = BOOTSTRAP
BOOTSTRAP_SPEC.loader.exec_module(BOOTSTRAP)


class _UnseekableBytesIO(io.BytesIO):
    """Force zipfile to emit a data descriptor instead of backpatching."""

    def seekable(self) -> bool:
        return False

    def seek(self, *args: object, **kwargs: object) -> int:
        raise io.UnsupportedOperation("fixture is intentionally unseekable")


def _descriptor_zip(*, signed: bool) -> bytes:
    stream = _UnseekableBytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("neutral.txt", b"neutral archive member")
    content = stream.getvalue()
    descriptor_start, central_offset, eocd_offset = _descriptor_offsets(content)
    assert content[descriptor_start : descriptor_start + 4] == b"PK\x07\x08"
    if signed:
        return content
    raw = bytearray(content)
    del raw[descriptor_start : descriptor_start + 4]
    struct.pack_into("<L", raw, eocd_offset - 4 + 16, central_offset - 4)
    return bytes(raw)


def _descriptor_offsets(content: bytes) -> tuple[int, int, int]:
    eocd_offset = content.rfind(b"PK\x05\x06")
    central_offset = struct.unpack_from("<L", content, eocd_offset + 16)[0]
    signed = content[central_offset - 16 : central_offset - 12] == b"PK\x07\x08"
    return central_offset - (16 if signed else 12), central_offset, eocd_offset


def _replace_descriptor(content: bytes, descriptor: bytes) -> bytes:
    start, central_offset, eocd_offset = _descriptor_offsets(content)
    raw = bytearray(content)
    raw[start:central_offset] = descriptor
    delta = len(descriptor) - (central_offset - start)
    struct.pack_into("<L", raw, eocd_offset + delta + 16, central_offset + delta)
    return bytes(raw)


def _replace_local_header_field(
    content: bytes, *, offset: int, replacement: bytes
) -> bytes:
    """Replace bytes in the first local header without changing central metadata."""

    raw = bytearray(content)
    raw[offset : offset + len(replacement)] = replacement
    return bytes(raw)


def _zip_with_member(name: str, *, extra: bytes = b"") -> bytes:
    stream = io.BytesIO()
    info = zipfile.ZipInfo(name)
    info.extra = extra
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(info, b"" if name.endswith("/") else b"neutral")
    return stream.getvalue()


def _tar_bytes(
    files: dict[str, bytes],
    *,
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
    suffix: bytes = b"",
    gname: str = "",
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.gname = gname
            archive.addfile(info, io.BytesIO(raw))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
        for name, target in (hardlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = target
            archive.addfile(info)
    return output.getvalue() + suffix


def _required() -> dict[str, bytes]:
    return {name: f"neutral:{name}".encode() for name in SCAN.REQUIRED}


def _docker_save(
    path: Path,
    files: dict[str, bytes],
    *,
    user: str = "ubuntu",
    env: list[str] | None = None,
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
    layer_suffix: bytes = b"",
    config_name: str | None = None,
    configured_diff_ids: list[str] | None = None,
    structural_gname: str = "",
) -> tuple[str, list[str]]:
    base = _tar_bytes({"etc/neutral-base": b"base"}, gname=structural_gname)
    app = _tar_bytes(
        files,
        symlinks=symlinks,
        hardlinks=hardlinks,
        suffix=layer_suffix,
        gname=structural_gname,
    )
    diff_ids = [
        "sha256:" + hashlib.sha256(base).hexdigest(),
        "sha256:" + hashlib.sha256(app).hexdigest(),
    ]
    config = json.dumps(
        {
            "config": {"User": user, "Env": env or []},
            "rootfs": {
                "type": "layers",
                "diff_ids": configured_diff_ids or diff_ids,
            },
            "history": [{"created_by": "base"}, {"created_by": "neutral"}],
        },
        separators=(",", ":"),
    ).encode()
    actual_name = config_name or hashlib.sha256(config).hexdigest() + ".json"
    manifest = json.dumps(
        [
            {
                "Config": actual_name,
                "RepoTags": ["neutral:test"],
                "Layers": ["base/layer.tar", "app/layer.tar"],
            }
        ],
        separators=(",", ":"),
    ).encode()
    outer = _tar_bytes(
        {
            "manifest.json": manifest,
            actual_name: config,
            "base/layer.tar": base,
            "app/layer.tar": app,
        },
        gname=structural_gname,
    )
    path.write_bytes(outer)
    return hashlib.sha256(config).hexdigest(), diff_ids


@pytest.fixture
def structural_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SCAN, "_reviewed_image_graph", lambda *_: None)
    monkeypatch.setattr(SCAN, "_neutral_candidate", lambda *_: None)


def test_structural_scan_covers_every_layer_and_rootfs_byte(
    tmp_path: Path, structural_scan: None
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, _required())
    result = SCAN.scan(image)
    assert result["status"] == "passed"
    assert result["layer_count"] == 2
    assert result["unresolved_findings"] == 0
    assert result["upstream_runtime_payload_count"] == 0
    assert result["shadow_asset_count"] == 0
    assert result["runtime_cache_entry_count"] == 0
    assert result["accepted_manifest_present"] is False
    assert result["release_authorized"] is False


def test_structural_container_text_cannot_create_unlocalized_false_positive(
    tmp_path: Path, structural_scan: None
) -> None:
    image = tmp_path / "structural-text.tar"
    structural_marker = "password" + "=" + ("x" * 16)
    _docker_save(image, _required(), structural_gname=structural_marker)
    assert SCAN.SECRET_TEXT.search(image.read_bytes()) is not None
    assert SCAN.scan(image)["status"] == "passed"


def test_source_reviewed_trust_roots_are_pinned_but_built_graph_is_withheld() -> None:
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in SCAN.EXPECTED_NEUTRAL_FILE_SHA256.values()
    )
    assert SCAN.EXPECTED_BASE["uncompressed_layer_digest"] == (
        "sha256:6078cde548a521a729def2ee7875e9f65513c18f0d4bac4db817417617d7006a"
    )
    assert SCAN.EXPECTED_IMAGE_CONFIG_SHA256 is None
    assert SCAN.EXPECTED_ORDERED_LAYER_DIFF_IDS is None
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in VERIFIER.EXPECTED_NEUTRAL_FILE_SHA256.values()
    )
    assert SCAN.EXPECTED_SYSTEM_WHEEL_FILES == VERIFIER.EXPECTED_SYSTEM_WHEEL_FILES
    apt = json.loads(
        (
            ROOT / "npa/docker/workbench/gymnasium-robotics/apt-runtime.lock.json"
        ).read_text()
    )
    locked = {
        item["package"]: item for item in apt["resolved_binary_packages"]
    }
    for record in SCAN.EXPECTED_SYSTEM_WHEEL_FILES.values():
        assert locked[record["package"]]["sha256"] == record["package_sha256"]


@pytest.mark.parametrize("mutation", ["entrypoint", "notice", "extra", "partial"])
def test_reviewed_image_graph_closes_every_candidate_added_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    accepted = tmp_path / "accepted.tar"
    config_digest, layer_diff_ids = _docker_save(accepted, _required())
    monkeypatch.setattr(SCAN, "EXPECTED_IMAGE_CONFIG_SHA256", config_digest)
    monkeypatch.setattr(
        SCAN, "EXPECTED_ORDERED_LAYER_DIFF_IDS", tuple(layer_diff_ids)
    )
    monkeypatch.setattr(SCAN, "_neutral_candidate", lambda *_: None)
    assert SCAN.scan(accepted)["status"] == "passed"

    changed = _required()
    if mutation == "entrypoint":
        changed["usr/local/bin/npa-gymnasium-entrypoint"] += b"\nchanged"
    elif mutation == "notice":
        changed[
            "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md"
        ] += b"\nchanged"
    elif mutation == "extra":
        changed["opt/innocent-extra.bin"] = b"arbitrary renamed payload"
    else:
        changed["opt/innocent-fragment.bin"] = b"truncated-or-encoded-payload-fragment"
    candidate = tmp_path / f"{mutation}.tar"
    _docker_save(candidate, changed)
    with pytest.raises(ValueError, match="reviewed neutral image config bytes changed"):
        SCAN.scan(candidate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"user": "root"}, "non-root ubuntu user"),
        ({"config_name": "unbound.json"}, "config filename"),
        ({"configured_diff_ids": ["sha256:" + "0" * 64]}, "diff IDs"),
        ({"layer_suffix": b"not-padding"}, "unaccounted tar bytes"),
    ],
)
def test_docker_save_integrity_refusals(
    tmp_path: Path,
    structural_scan: None,
    kwargs: dict[str, object],
    message: str,
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, _required(), **kwargs)
    with pytest.raises(ValueError, match=message):
        SCAN.scan(image)


@pytest.mark.parametrize(
    "path",
    [
        "opt/venv/bin/python",
        "workspace/.cache/npa/runtime/file",
        "opt/runtime-cache/current/receipt.json",
        "opt/nvidia/lib/libcuda.so",
        "usr/local/cuda/version.json",
        "tmp/gymnasium_robotics/envs/assets/hand.xml",
        "tmp/mujoco-3.12.0.whl",
        "root/.docker/config.json",
        "workspace/byof-runs/output.json",
    ],
)
def test_forbidden_payload_or_state_path_refuses(
    tmp_path: Path, structural_scan: None, path: str
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), path: b"payload"})
    with pytest.raises(ValueError, match="forbidden image path"):
        SCAN.scan(image)


def test_only_exact_locked_system_bootstrap_wheel_path_and_bytes_are_allowed(
    tmp_path: Path,
    structural_scan: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "usr/share/python-wheels/pip-24.0-py3-none-any.whl"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w") as archive:
        archive.writestr("pip/__init__.py", b"exact reviewed bootstrap")
    content = stream.getvalue()
    record = {**SCAN.EXPECTED_SYSTEM_WHEEL_FILES[path]}
    record["sha256"] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(SCAN, "EXPECTED_SYSTEM_WHEEL_FILES", {path: record})

    accepted = tmp_path / "accepted-system-wheel.tar"
    _docker_save(accepted, {**_required(), path: content})
    assert SCAN.scan(accepted)["status"] == "passed"

    mismatched = tmp_path / "mismatched-system-wheel.tar"
    _docker_save(mismatched, {**_required(), path: content + b"-changed"})
    with pytest.raises(ValueError, match="reviewed system bootstrap wheel changed"):
        SCAN.scan(mismatched)

    renamed = tmp_path / "renamed-system-wheel.tar"
    _docker_save(renamed, {**_required(), "opt/innocent.bin": content})
    with pytest.raises(ValueError, match="system bootstrap wheel at unauthorized"):
        SCAN.scan(renamed)


def test_in_pod_verifier_ignores_unreadable_locked_base_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_source = ROOT / "npa/docker/workbench/gymnasium-robotics"
    lock_root = tmp_path / "opt/npa/gymnasium-robotics"
    lock_root.mkdir(parents=True)
    for name in VERIFIER.EXPECTED_LOCK_FILENAMES:
        (lock_root / name).write_bytes((image_source / name).read_bytes())
    for relative in VERIFIER.EXPECTED_FIXED_FILE_SHA256:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_name = (
            "build.sh"
            if target.name == "npa-gymnasium-entrypoint"
            else target.name
        )
        target.write_bytes((image_source / source_name).read_bytes())
    system_wheels = {}
    for relative, record in VERIFIER.EXPECTED_SYSTEM_WHEEL_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = f"fixture:{relative}".encode()
        target.write_bytes(content)
        system_wheels[relative] = {
            **record,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    monkeypatch.setattr(VERIFIER, "EXPECTED_SYSTEM_WHEEL_FILES", system_wheels)

    shadow = tmp_path / "etc/shadow"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("root:locked", encoding="utf-8")
    shadow.chmod(0o000)
    assert VERIFIER.verify(tmp_path)["status"] == "passed"


def test_live_non_root_verifier_defers_only_root_private_paths() -> None:
    visible = VERIFIER._forbidden_roots_visible_to_verifier(Path("/"))
    assert not set(visible) & set(
        VERIFIER.PRIVILEGED_ROOTS_DEFERRED_TO_COMPLETE_BYTE_SCAN
    )
    assert VERIFIER._forbidden_roots_visible_to_verifier(Path("/offline")) == (
        VERIFIER.FORBIDDEN_ROOTS
    )
    assert all(
        str(path).startswith("/root/")
        for path in VERIFIER.PRIVILEGED_ROOTS_DEFERRED_TO_COMPLETE_BYTE_SCAN
    )


def test_exact_shadow_asset_byte_refuses_at_an_innocent_path(
    tmp_path: Path, structural_scan: None
) -> None:
    asset = ROOT / "npa/docker/workbench/gymnasium-robotics/asset-lock.json"
    lock = json.loads(asset.read_text())
    forbidden_digest = next(iter(lock["directly_loaded_xml"].values()))
    forbidden = next(
        value
        for value in SCAN.KNOWN_FORBIDDEN_CONTENT_SHA256
        if value == forbidden_digest
    )
    content = b"fixture-byte"
    monkeypatch_digest = hashlib.sha256(content).hexdigest()
    original = SCAN.KNOWN_FORBIDDEN_CONTENT_SHA256
    SCAN.KNOWN_FORBIDDEN_CONTENT_SHA256 = frozenset({*original, monkeypatch_digest})
    try:
        image = tmp_path / "image.tar"
        _docker_save(image, {**_required(), "opt/innocent.bin": content})
        with pytest.raises(ValueError, match="forbidden upstream/runtime byte"):
            SCAN.scan(image)
    finally:
        SCAN.KNOWN_FORBIDDEN_CONTENT_SHA256 = original
    assert forbidden == forbidden_digest


def test_forbidden_content_hash_refuses_at_raw_container_boundary(
    tmp_path: Path,
    structural_scan: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, _required())
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    monkeypatch.setattr(SCAN, "KNOWN_FORBIDDEN_CONTENT_SHA256", frozenset({digest}))
    with pytest.raises(
        ValueError,
        match="forbidden upstream/runtime byte: complete Docker-save archive",
    ):
        SCAN.scan(image)


def test_forbidden_content_hash_refuses_at_decoded_member_boundary(
    tmp_path: Path,
    structural_scan: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"decoded forbidden fixture"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(SCAN, "KNOWN_FORBIDDEN_CONTENT_SHA256", frozenset({digest}))
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), "opt/decoded.bin": content})
    with pytest.raises(
        ValueError,
        match=r"forbidden upstream/runtime byte: decoded member: opt/decoded\.bin",
    ):
        SCAN.scan(image)


@pytest.mark.parametrize(
    "content",
    [
        b"-----BEGIN PRIVATE KEY-----\nsecret",
        b"password=correct-horse-battery-staple",
        b"nvcr.io/vendor/image",
        b"accept_eula=true",
    ],
)
def test_secret_or_vendor_signature_refuses(
    tmp_path: Path, structural_scan: None, content: bytes
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), "opt/innocent.txt": content})
    with pytest.raises(ValueError, match="forbidden") as captured:
        SCAN.scan(image)
    message = str(captured.value)
    assert "decoded member: opt/innocent.txt" in message
    assert content.decode("utf-8") not in message


@pytest.mark.parametrize("kind", ["zip", "gzip"])
def test_decoded_secret_in_nested_content_refuses_with_member_label_only(
    tmp_path: Path, structural_scan: None, kind: str
) -> None:
    content = b"api" + b"_key=" + (b"x" * 16)
    if kind == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("credential.txt", content)
        nested = stream.getvalue()
        name = "opt/nested.zip"
        label = "decoded member: opt/nested.zip:credential.txt"
    else:
        nested = gzip.compress(content, mtime=0)
        name = "opt/nested.gz"
        label = "decoded member: opt/nested.gz:expanded-gzip"
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), name: nested})
    with pytest.raises(ValueError, match="forbidden secret signature") as captured:
        SCAN.scan(image)
    message = str(captured.value)
    assert label in message
    assert content.decode("utf-8") not in message


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("opt/malformed.tar", b"not-a-tar", "declared tar does not match bytes"),
        (
            "opt/ambiguous.gz",
            gzip.compress(b"first", mtime=0) + gzip.compress(b"second", mtime=0),
            "ambiguous compressed stream",
        ),
    ],
)
def test_malformed_or_ambiguous_nested_archive_refuses(
    tmp_path: Path,
    structural_scan: None,
    name: str,
    content: bytes,
    message: str,
) -> None:
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), name: content})
    with pytest.raises(ValueError, match=message):
        SCAN.scan(image)


@pytest.mark.parametrize("kind", ["zip", "tar-gz"])
def test_nested_upstream_path_refuses(
    tmp_path: Path, structural_scan: None, kind: str
) -> None:
    if kind == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("gymnasium_robotics/envs/assets/hand.xml", "payload")
        nested = stream.getvalue()
        name = "opt/nested.zip"
    else:
        nested = gzip.compress(
            _tar_bytes({"gymnasium_robotics/envs/assets/hand.xml": b"payload"}),
            mtime=0,
        )
        name = "opt/nested.tar.gz"
    image = tmp_path / "image.tar"
    _docker_save(image, {**_required(), name: nested})
    with pytest.raises(ValueError, match="forbidden nested archive member"):
        SCAN.scan(image)


@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
def test_zip_data_descriptor_is_bound_to_central_directory(signed: bool) -> None:
    content = _descriptor_zip(signed=signed)

    infos = SCAN._validated_zip_infos("nested.zip", content)

    assert [info.filename for info in infos] == ["neutral.txt"]


@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
@pytest.mark.parametrize("field_index", [0, 1, 2], ids=["crc", "compressed", "size"])
def test_zip_data_descriptor_field_mismatch_refuses(
    signed: bool, field_index: int
) -> None:
    content = _descriptor_zip(signed=signed)
    start, central_offset, _eocd_offset = _descriptor_offsets(content)
    descriptor = bytearray(content[start:central_offset])
    field_offset = (4 if signed else 0) + (field_index * 4)
    value = struct.unpack_from("<L", descriptor, field_offset)[0]
    struct.pack_into("<L", descriptor, field_offset, value ^ 1)

    with pytest.raises(
        ValueError, match="zip data descriptor does not match central directory"
    ):
        SCAN._validated_zip_infos(
            "nested.zip", _replace_descriptor(content, bytes(descriptor))
        )


@pytest.mark.parametrize(
    "field_offset", [14, 18, 22], ids=["crc", "compressed-size", "file-size"]
)
def test_zip_descriptor_local_size_or_crc_mismatch_refuses(field_offset: int) -> None:
    content = _descriptor_zip(signed=True)
    mutated = _replace_local_header_field(
        content,
        offset=field_offset,
        replacement=struct.pack("<L", 1),
    )

    with pytest.raises(ValueError, match="zip local descriptor metadata"):
        SCAN._validated_zip_infos("nested.zip", mutated)


@pytest.mark.parametrize("name", ["neutral.txt", "directory/"])
def test_zip_local_filename_mismatch_refuses(name: str) -> None:
    content = (
        _descriptor_zip(signed=True)
        if name == "neutral.txt"
        else _zip_with_member(name)
    )
    replacement = b"x" if name[0] != "x" else b"y"
    mutated = _replace_local_header_field(content, offset=30, replacement=replacement)

    with pytest.raises(ValueError, match="zip local filename does not match"):
        SCAN._validated_zip_infos("nested.zip", mutated)


def test_zip_local_filename_encoding_disagreement_refuses() -> None:
    content = _descriptor_zip(signed=True)
    flags = struct.unpack_from("<H", content, 6)[0]
    mutated = _replace_local_header_field(
        content,
        offset=6,
        replacement=struct.pack("<H", flags ^ 0x800),
    )

    with pytest.raises(ValueError, match="unsupported zip local header"):
        SCAN._validated_zip_infos("nested.zip", mutated)


def test_zip_local_extra_field_ambiguity_refuses() -> None:
    content = _zip_with_member("neutral.txt", extra=struct.pack("<HH", 0xCAFE, 0))

    with pytest.raises(ValueError, match="unsupported zip extra field"):
        SCAN._validated_zip_infos("nested.zip", content)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("malformed-signature", "unsupported zip data descriptor"),
        ("truncated", "unsupported zip data descriptor"),
        ("gap", "unsupported zip data descriptor"),
        ("ambiguous", "unsupported zip data descriptor"),
    ],
)
def test_malformed_zip_data_descriptor_refuses(mutation: str, message: str) -> None:
    content = _descriptor_zip(signed=True)
    start, central_offset, _eocd_offset = _descriptor_offsets(content)
    descriptor = content[start:central_offset]
    replacements = {
        "malformed-signature": b"BAD!" + descriptor[4:],
        "truncated": descriptor[:-1],
        "gap": descriptor + b"\0",
        "ambiguous": b"PK\x07\x08" + descriptor,
    }

    with pytest.raises(ValueError, match=message):
        SCAN._validated_zip_infos(
            "nested.zip", _replace_descriptor(content, replacements[mutation])
        )


def test_zip_member_count_limit_accepts_exact_boundary() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(SCAN.MAX_NESTED_ARCHIVE_MEMBERS):
            archive.writestr(f"{index:05d}", b"")

    infos = SCAN._validated_zip_infos("boundary.zip", stream.getvalue())

    assert len(infos) == SCAN.MAX_NESTED_ARCHIVE_MEMBERS


def test_zip_member_count_limit_refuses_before_infolist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = SCAN.MAX_NESTED_ARCHIVE_MEMBERS + 1
    content = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, 0, 0, 0)

    def forbidden_zipfile(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ZipFile must not be constructed for an oversized archive")

    monkeypatch.setattr(SCAN.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="zip archive member count exceeds limit"):
        SCAN._validated_zip_infos("oversized.zip", content)


def test_link_to_forbidden_cache_refuses(tmp_path: Path, structural_scan: None) -> None:
    image = tmp_path / "image.tar"
    _docker_save(
        image,
        _required(),
        symlinks={"opt/neutral-link": "/workspace/.cache/npa/runtime"},
    )
    with pytest.raises(ValueError, match="forbidden image link target"):
        SCAN.scan(image)


def test_missing_required_neutral_file_refuses(
    tmp_path: Path, structural_scan: None
) -> None:
    files = _required()
    files.pop("opt/npa/gymnasium-robotics/runtime-bootstrap.py")
    image = tmp_path / "image.tar"
    _docker_save(image, files)
    with pytest.raises(ValueError, match="required image files absent"):
        SCAN.scan(image)


def _complete_neutral_rootfs() -> tuple[dict[str, bytes], list[str]]:
    requirements = (
        "# status: complete\n"
        + "\n".join(
            f"{name}=={version} --hash=sha256:" + "a" * 64
            for name, version in sorted(SCAN.EXPECTED_PYTHON_DISTRIBUTIONS.items())
        )
        + "\n"
    )
    artifacts = [
        {
            "name": "gymnasium-robotics-source",
            "role": "solution-source",
            "sha256": SCAN.EXPECTED_SOURCE_FIELDS["farama_gymnasium_robotics"][
                "archive_sha256"
            ],
        }
    ]
    for index, name in enumerate(sorted(SCAN.EXPECTED_PYTHON_DISTRIBUTIONS)):
        artifacts.append(
            {
                "name": "mujoco-3.12.0-cp312-linux-x86_64"
                if name == "mujoco"
                else f"wheel-{index}",
                "role": "python-wheel",
                "sha256": (
                    SCAN.EXPECTED_SOURCE_FIELDS["mujoco"]["wheel_sha256"]
                    if name == "mujoco"
                    else hashlib.sha256(name.encode()).hexdigest()
                ),
            }
        )
    source = {
        "schema": "npa.gymnasium-robotics.runtime-fetch-lock.v2",
        "status": "complete",
        "source_commit": SCAN.EXPECTED_SOURCE,
        "mujoco_version": "3.12.0",
        "requirements_lock_sha256": hashlib.sha256(requirements.encode()).hexdigest(),
        "delivery": {
            "source": "operator-owned-runtime-cache",
            "baked_runtime": "neutral-bootstrap-only",
            "weights": "none",
            "data_assets": "runtime-cache-only",
            "runtime_cache": "operator-owned-and-external",
            "outputs": "operator-owned-run-artifacts",
        },
        "expected_python_distribution_count": 19,
        "resolved_python_artifact_count": 19,
        "artifacts": artifacts,
        "components": {
            name: dict(fields) for name, fields in SCAN.EXPECTED_SOURCE_FIELDS.items()
        },
    }
    apt = {
        "schema": "npa.gymnasium-robotics.neutral-bootstrap-apt-lock.v2",
        "status": "complete",
        "base": {
            **SCAN.EXPECTED_BASE,
            "uncompressed_layer_digest": "sha256:" + "b" * 64,
        },
        "resolved_binary_packages": [
            {"package": "python3", "sha256": "f" * 64},
            *(
                {
                    "package": record["package"],
                    "sha256": record["package_sha256"],
                }
                for record in SCAN.EXPECTED_SYSTEM_WHEEL_FILES.values()
            ),
        ],
        "resolved_source_packages": [{"name": "python3.12", "version": "exact"}],
    }
    corresponding = {
        "schema": "npa.gymnasium-robotics.baked-corresponding-source-lock.v2",
        "status": "complete",
        "scope": "candidate-image-layers-only",
        "deliveries": [
            {
                "binary_component": "ubuntu-neutral-bootstrap-closure",
                "artifacts": [{"sha256": "c" * 64}],
            }
        ],
    }
    rootfs = _required()
    rootfs.update(
        {
            "opt/npa/gymnasium-robotics/source-lock.json": json.dumps(source).encode(),
            "opt/npa/gymnasium-robotics/apt-runtime.lock.json": json.dumps(
                apt
            ).encode(),
            "opt/npa/gymnasium-robotics/corresponding-source.lock.json": json.dumps(
                corresponding
            ).encode(),
            "opt/npa/gymnasium-robotics/requirements.lock": requirements.encode(),
            "opt/npa/gymnasium-robotics/asset-lock.json": (
                ROOT / "npa/docker/workbench/gymnasium-robotics/asset-lock.json"
            ).read_bytes(),
        }
    )
    rootfs.update(
        {
            path: f"fixture:{path}".encode()
            for path in SCAN.EXPECTED_SYSTEM_WHEEL_FILES
        }
    )
    diff_ids = ["sha256:" + "b" * 64]
    return rootfs, diff_ids


def test_scanner_runtime_distribution_baseline_matches_fetch_inputs() -> None:
    requirements_in = (
        ROOT / "npa/docker/workbench/gymnasium-robotics/requirements.in"
    ).read_text()
    declared: dict[str, str] = {}
    for source_line in requirements_in.splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?"
            r"==(?P<version>[^\s]+)",
            line,
        )
        assert match is not None
        declared[SCAN._normalize_distribution(match.group("name"))] = match.group(
            "version"
        )

    source_lock = json.loads(
        (
            ROOT / "npa/docker/workbench/gymnasium-robotics/source-lock.json"
        ).read_text()
    )
    assert declared == SCAN.EXPECTED_PYTHON_DISTRIBUTIONS
    assert len(declared) == BOOTSTRAP.EXPECTED_WHEEL_COUNT
    assert source_lock["expected_python_distribution_count"] == len(declared)


def test_neutral_semantic_gate_can_accept_only_a_complete_reviewed_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rootfs, diff_ids = _complete_neutral_rootfs()
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_BASE",
        {**SCAN.EXPECTED_BASE, "uncompressed_layer_digest": diff_ids[0]},
    )
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_NEUTRAL_FILE_SHA256",
        {
            name: hashlib.sha256(
                rootfs[f"opt/npa/gymnasium-robotics/{name}"]
            ).hexdigest()
            for name in SCAN.EXPECTED_NEUTRAL_FILE_SHA256
        },
    )
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_SYSTEM_WHEEL_FILES",
        {
            path: {
                **record,
                "sha256": hashlib.sha256(rootfs[path]).hexdigest(),
            }
            for path, record in SCAN.EXPECTED_SYSTEM_WHEEL_FILES.items()
        },
    )
    SCAN._neutral_candidate(rootfs, diff_ids)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("source-lock.json", "reviewed neutral image file changed"),
        ("requirements.lock", "reviewed neutral image file changed"),
        ("runtime-bootstrap.py", "reviewed neutral image file changed"),
    ],
)
def test_neutral_semantic_gate_rejects_any_trusted_file_drift(
    monkeypatch: pytest.MonkeyPatch, path: str, message: str
) -> None:
    rootfs, diff_ids = _complete_neutral_rootfs()
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_BASE",
        {**SCAN.EXPECTED_BASE, "uncompressed_layer_digest": diff_ids[0]},
    )
    expected = {
        name: hashlib.sha256(rootfs[f"opt/npa/gymnasium-robotics/{name}"]).hexdigest()
        for name in SCAN.EXPECTED_NEUTRAL_FILE_SHA256
    }
    monkeypatch.setattr(SCAN, "EXPECTED_NEUTRAL_FILE_SHA256", expected)
    rootfs[f"opt/npa/gymnasium-robotics/{path}"] += b"drift"
    with pytest.raises(ValueError, match=message):
        SCAN._neutral_candidate(rootfs, diff_ids)
