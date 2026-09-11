from __future__ import annotations

import bz2
import copy
import gzip
import importlib.util
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/scripts/scan_image_gymnasium_robotics_payload.py"
SPEC = importlib.util.spec_from_file_location("gymnasium_payload_scan", SCRIPT)
assert SPEC and SPEC.loader
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)
VERIFIER_SCRIPT = ROOT / "npa/docker/workbench/gymnasium-robotics/verify_image.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "gymnasium_image_verifier", VERIFIER_SCRIPT
)
assert VERIFIER_SPEC and VERIFIER_SPEC.loader
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER)
PRODUCTION_EXPECTED_ASSET_LOCK = SCAN.EXPECTED_ASSET_LOCK
PRODUCTION_EXPECTED_SOURCE_FIELDS = SCAN.EXPECTED_SOURCE_FIELDS
PRODUCTION_EXPECTED_BASE = dict(SCAN.EXPECTED_BASE)
PRODUCTION_EXPECTED_COMPLETE_LOCK_SHA256 = dict(SCAN.EXPECTED_COMPLETE_LOCK_SHA256)


def _tar_bytes(
    files: dict[str, bytes],
    *,
    file_modes: dict[str, int] | None = None,
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
    fifos: tuple[str, ...] = (),
    devices: tuple[str, ...] = (),
    suffix: bytes = b"",
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            if file_modes and name in file_modes:
                info.mode = file_modes[name]
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
        for name in fifos:
            info = tarfile.TarInfo(name)
            info.type = tarfile.FIFOTYPE
            archive.addfile(info)
        for name in devices:
            info = tarfile.TarInfo(name)
            info.type = tarfile.CHRTYPE
            info.devmajor = 0
            info.devminor = 0
            archive.addfile(info)
    return output.getvalue() + suffix


def _regular_record(raw: bytes, path: str = "") -> dict[str, object]:
    return {
        "kind": "regular",
        "mode": 0o644,
        "uid": 0,
        "gid": 0,
        "mtime": 0,
        "uname": "",
        "gname": "",
        "pax_headers": {"path": path} if len(path.encode()) > 100 else {},
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


BASE_FILES = {"etc/npa-synthetic-base": b"reviewed-base-bytes"}
BASE_LAYER = _tar_bytes(BASE_FILES)
TEST_EXPECTED_BASE = {
    **PRODUCTION_EXPECTED_BASE,
    "uncompressed_layer_digest": "sha256:" + hashlib.sha256(BASE_LAYER).hexdigest(),
}

XML = {f"hand/test-{index}.xml": f"xml-{index}".encode() for index in range(5)}
MATERIAL = {
    **{f"stls/hand/test-{index}.stl": f"mesh-{index}".encode() for index in range(12)},
    "textures/test.png": b"texture-a",
    "textures/test-hidden.png": b"texture-b",
}
NOTICE = b"Shadow notice"
ANNEX = b"preferred source"
ANNEX_SHA256 = hashlib.sha256(ANNEX).hexdigest()
TEST_SOURCE_FIELDS = {
    name: dict(fields) for name, fields in PRODUCTION_EXPECTED_SOURCE_FIELDS.items()
}
TEST_SOURCE_FIELDS["farama_gymnasium_robotics"]["archive_sha256"] = ANNEX_SHA256
SOURCE_LOCK = {
    "status": "complete",
    "components": {name: dict(fields) for name, fields in TEST_SOURCE_FIELDS.items()},
}
SOURCE_LOCK["components"]["shadow_sr_common"].update(
    {
        "preferred_form_sha256": ANNEX_SHA256,
        "transformation_manifest_sha256": ANNEX_SHA256,
    }
)
APT_LOCK = {
    "status": "complete",
    "base": TEST_EXPECTED_BASE,
    "resolved_binary_packages": [{"name": "libegl1", "version": "exact"}],
    "resolved_source_packages": [{"name": "libglvnd", "version": "exact"}],
}
CORRESPONDING_LOCK = {
    "status": "complete",
    "deliveries": [
        {
            "binary_component": component,
            **{field: ANNEX_SHA256 for field in SCAN.DELIVERY_DIGEST_ROLES[component]},
            **(
                {
                    "license": "MIT",
                    "repository": SCAN.EXPECTED_SOURCE_FIELDS[
                        "farama_gymnasium_robotics"
                    ]["repository"],
                    "source_commit": SCAN.EXPECTED_SOURCE,
                }
                if component == "farama-gymnasium-robotics"
                else {
                    "license": "GPL-2.0-only AND Apache-2.0",
                    "source_commit": SCAN.EXPECTED_SHADOW_COMMIT,
                }
                if component == "shadow-hand-xml-mesh-texture-assets"
                else {}
            ),
            "required_artifact_roles": sorted(
                SCAN.DELIVERY_DIGEST_ROLES[component].values()
            ),
            "artifacts": [
                {
                    "role": role,
                    "path": f"{component}/{role}.bin",
                    "sha256": ANNEX_SHA256,
                }
                for role in sorted(SCAN.DELIVERY_DIGEST_ROLES[component].values())
            ],
        }
        for component in (
            "farama-gymnasium-robotics",
            "shadow-hand-xml-mesh-texture-assets",
            "ubuntu-runtime-closure",
        )
    ],
}
ASSET_LOCK = {
    "source_commit": SCAN.EXPECTED_SOURCE,
    "asset_notice_sha256": hashlib.sha256(NOTICE).hexdigest(),
    "directly_loaded_xml": {
        name: hashlib.sha256(raw).hexdigest() for name, raw in XML.items()
    },
    "directly_loaded_mesh_texture": {
        name: hashlib.sha256(raw).hexdigest() for name, raw in MATERIAL.items()
    },
}
ASSET_PREFIX = "opt/venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/"
REQUIREMENTS = (
    "# status: complete\n"
    + "\n".join(
        f"{name}=={version} --hash=sha256:" + "a" * 64
        for name, version in sorted(SCAN.EXPECTED_PYTHON_DISTRIBUTIONS.items())
    )
    + "\n"
)
COMMON_FILES = {
    "opt/npa/gymnasium-robotics/source-lock.json": json.dumps(SOURCE_LOCK).encode(),
    "opt/npa/gymnasium-robotics/apt-runtime.lock.json": json.dumps(APT_LOCK).encode(),
    "opt/npa/gymnasium-robotics/asset-lock.json": json.dumps(ASSET_LOCK).encode(),
    "opt/npa/gymnasium-robotics/requirements.lock": REQUIREMENTS.encode(),
    "opt/npa/gymnasium-robotics/capability_smoke.py": b"print('smoke')\n",
    "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md": b"notices\n",
    "usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md": b"public\n",
    **{
        f"usr/share/source/npa-gymnasium-robotics/{artifact['path']}": ANNEX
        for delivery in CORRESPONDING_LOCK["deliveries"]
        for artifact in delivery["artifacts"]
    },
    ASSET_PREFIX + "LICENSE.md": NOTICE,
    **{ASSET_PREFIX + name: raw for name, raw in {**XML, **MATERIAL}.items()},
}
ROOTFS_MANIFEST = {
    "schema": "npa.gymnasium-robotics.rootfs-manifest.v2",
    "entries": {
        **{
            path: {
                **_regular_record(raw, path),
                "class": "npa-runtime",
                "source": "synthetic-test-fixture",
            }
            for path, raw in {**BASE_FILES, **COMMON_FILES}.items()
        },
        **{
            path: {
                **{
                    key: value
                    for key, value in _regular_record(b"").items()
                    if key != "sha256"
                },
                "class": "npa-runtime",
                "source": "independently-pinned-trust-file",
            }
            for path in (
                "opt/npa/gymnasium-robotics/corresponding-source.lock.json",
                SCAN.ROOTFS_MANIFEST,
            )
        },
    },
}
ROOTFS_MANIFEST_BYTES = json.dumps(ROOTFS_MANIFEST).encode()
CORRESPONDING_LOCK["final_rootfs_manifest_sha256"] = hashlib.sha256(
    ROOTFS_MANIFEST_BYTES
).hexdigest()
REQUIRED = {
    **COMMON_FILES,
    "opt/npa/gymnasium-robotics/corresponding-source.lock.json": json.dumps(
        CORRESPONDING_LOCK
    ).encode(),
    SCAN.ROOTFS_MANIFEST: ROOTFS_MANIFEST_BYTES,
}


@pytest.fixture(autouse=True)
def _synthetic_asset_lock_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_ASSET_LOCK",
        hashlib.sha256(json.dumps(ASSET_LOCK).encode()).hexdigest(),
    )
    monkeypatch.setattr(SCAN, "EXPECTED_SOURCE_FIELDS", TEST_SOURCE_FIELDS)
    monkeypatch.setattr(SCAN, "EXPECTED_BASE", TEST_EXPECTED_BASE)
    monkeypatch.setattr(
        SCAN,
        "EXPECTED_COMPLETE_LOCK_SHA256",
        {
            name: hashlib.sha256(
                REQUIRED[f"opt/npa/gymnasium-robotics/{name}"]
            ).hexdigest()
            for name in (
                "source-lock.json",
                "apt-runtime.lock.json",
                "corresponding-source.lock.json",
                "requirements.lock",
            )
        },
    )


def test_production_scanner_binds_the_reviewed_asset_lock() -> None:
    path = ROOT / "npa/docker/workbench/gymnasium-robotics/asset-lock.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == PRODUCTION_EXPECTED_ASSET_LOCK
    )


def test_production_scanner_binds_the_reviewed_source_identities() -> None:
    path = ROOT / "npa/docker/workbench/gymnasium-robotics/source-lock.json"
    actual = json.loads(path.read_text(encoding="utf-8"))["components"]
    for name, expected in PRODUCTION_EXPECTED_SOURCE_FIELDS.items():
        assert all(actual[name][key] == value for key, value in expected.items())


def test_phase_a_has_no_approved_complete_lock_or_base_diff_id(tmp_path: Path) -> None:
    assert all(
        value is None for value in PRODUCTION_EXPECTED_COMPLETE_LOCK_SHA256.values()
    )
    assert PRODUCTION_EXPECTED_BASE["uncompressed_layer_digest"] is None
    assert all(
        value is None for value in VERIFIER.EXPECTED_COMPLETE_LOCK_SHA256.values()
    )
    with pytest.raises(ValueError, match="reviewed complete lock digest"):
        VERIFIER.verify(tmp_path)


def _docker_save(
    path: Path,
    files: dict[str, bytes],
    *,
    history: str = "clean",
    user: str = "ubuntu",
    env: list[str] | None = None,
    labels: dict[str, str] | None = None,
    file_modes: dict[str, int] | None = None,
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
    fifos: tuple[str, ...] = (),
    devices: tuple[str, ...] = (),
    app_suffix: bytes = b"",
    diff_ids: list[str] | None = None,
    base_layer: bytes = BASE_LAYER,
    config_name: str | None = None,
) -> None:
    layer = _tar_bytes(
        files,
        file_modes=file_modes,
        symlinks=symlinks,
        hardlinks=hardlinks,
        fifos=fifos,
        devices=devices,
        suffix=app_suffix,
    )
    runtime_config: dict[str, object] = {"User": user}
    if env is not None:
        runtime_config["Env"] = env
    if labels is not None:
        runtime_config["Labels"] = labels
    actual_diff_ids = [
        "sha256:" + hashlib.sha256(base_layer).hexdigest(),
        "sha256:" + hashlib.sha256(layer).hexdigest(),
    ]
    config = json.dumps(
        {
            "config": runtime_config,
            "history": [{"created_by": history}],
            "rootfs": {"type": "layers", "diff_ids": diff_ids or actual_diff_ids},
        }
    ).encode()
    actual_config_name = config_name or f"{hashlib.sha256(config).hexdigest()}.json"
    manifest = json.dumps(
        [{"Config": actual_config_name, "Layers": ["base.tar", "layer.tar"]}]
    ).encode()
    with tarfile.open(path, mode="w") as archive:
        for name, raw in (
            ("manifest.json", manifest),
            (actual_config_name, config),
            ("base.tar", base_layer),
            ("layer.tar", layer),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))


def test_complete_synthetic_graph_passes_without_authorizing_release(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "image.tar"
    _docker_save(archive, REQUIRED)
    result = SCAN.scan(archive)
    assert result["status"] == "passed"
    assert result["layer_count"] == 2
    assert result["whiteout_entry_count"] == 0
    assert result["unresolved_findings"] == 0
    assert result["accepted_manifest_present"] is False
    assert result["release_authorized"] is False


def test_root_user_and_unclassified_files_fail(tmp_path: Path) -> None:
    root_user = tmp_path / "root-user.tar"
    _docker_save(root_user, REQUIRED, user="root")
    with pytest.raises(ValueError, match="non-root ubuntu user"):
        SCAN.scan(root_user)

    unclassified = tmp_path / "unclassified.tar"
    _docker_save(unclassified, {**REQUIRED, "opt/unknown-clean-file": b"plain"})
    with pytest.raises(ValueError, match="unclassified entries"):
        SCAN.scan(unclassified)

    metadata = tmp_path / "changed-metadata.tar"
    _docker_save(
        metadata,
        REQUIRED,
        file_modes={"opt/npa/gymnasium-robotics/capability_smoke.py": 0o777},
    )
    with pytest.raises(ValueError, match="entry metadata changed"):
        SCAN.scan(metadata)


def test_incomplete_corresponding_source_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "image.tar"
    files = {
        **REQUIRED,
        "opt/npa/gymnasium-robotics/corresponding-source.lock.json": b'{"status":"phase-a-incomplete"}',
    }
    expected = dict(SCAN.EXPECTED_COMPLETE_LOCK_SHA256)
    expected["corresponding-source.lock.json"] = hashlib.sha256(
        files["opt/npa/gymnasium-robotics/corresponding-source.lock.json"]
    ).hexdigest()
    monkeypatch.setattr(SCAN, "EXPECTED_COMPLETE_LOCK_SHA256", expected)
    _docker_save(archive, files)
    try:
        SCAN.scan(archive)
    except ValueError as error:
        assert "incomplete evidence lock" in str(error)
    else:
        raise AssertionError("incomplete corresponding source was accepted")


def _scan_with_corresponding_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    lock: dict[str, object],
) -> None:
    raw = json.dumps(lock).encode()
    files = {
        **REQUIRED,
        "opt/npa/gymnasium-robotics/corresponding-source.lock.json": raw,
    }
    expected = dict(SCAN.EXPECTED_COMPLETE_LOCK_SHA256)
    expected["corresponding-source.lock.json"] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(SCAN, "EXPECTED_COMPLETE_LOCK_SHA256", expected)
    archive = tmp_path / f"corresponding-{name}.tar"
    _docker_save(archive, files)
    SCAN.scan(archive)


def test_corresponding_source_artifact_roles_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutations: dict[str, dict[str, object]] = {}

    missing = copy.deepcopy(CORRESPONDING_LOCK)
    missing["deliveries"][1]["artifacts"].pop()
    mutations["missing"] = missing

    duplicate = copy.deepcopy(CORRESPONDING_LOCK)
    duplicate_artifact = dict(duplicate["deliveries"][1]["artifacts"][0])
    duplicate_artifact["path"] = "shadow-hand-xml-mesh-texture-assets/duplicate.bin"
    duplicate["deliveries"][1]["artifacts"].append(duplicate_artifact)
    mutations["duplicate"] = duplicate

    substituted = copy.deepcopy(CORRESPONDING_LOCK)
    substituted["deliveries"][2]["artifacts"][0]["role"] = "unexpected_role"
    mutations["substituted"] = substituted

    mismatched = copy.deepcopy(CORRESPONDING_LOCK)
    mismatched["deliveries"][2]["binary_manifest_sha256"] = "c" * 64
    mutations["mismatched"] = mismatched

    for name, lock in mutations.items():
        with pytest.raises(
            ValueError,
            match="corresponding-source (?:artifact roles|role digest) changed",
        ):
            _scan_with_corresponding_lock(tmp_path, monkeypatch, name, lock)


def test_secret_in_raw_layer_and_eula_in_history_fail(tmp_path: Path) -> None:
    secret = tmp_path / "secret.tar"
    private_key_marker = b"BEGIN " + b"OPENSSH PRIVATE" + b" KEY"
    _docker_save(secret, {**REQUIRED, "tmp/key": private_key_marker})
    try:
        SCAN.scan(secret)
    except ValueError as error:
        assert "forbidden secret signature" in str(error)
    else:
        raise AssertionError("private key was accepted")
    history = tmp_path / "history.tar"
    _docker_save(history, REQUIRED, history="ACCEPT_EULA=YES")
    try:
        SCAN.scan(history)
    except ValueError as error:
        assert "forbidden vendor payload signature" in str(error)
    else:
        raise AssertionError("cached consent was accepted")


def test_model_and_cache_paths_fail(tmp_path: Path) -> None:
    for name in (
        "opt/model.safetensors",
        "opt/isaac/runtime",
        "opt/omniverse/kit",
        "opt/ngc/explicit-cache",
        "root/.cache/pip/wheel",
        "usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.1",
    ):
        archive = tmp_path / (name.replace("/", "-") + ".tar")
        _docker_save(archive, {**REQUIRED, name: b"payload"})
        try:
            SCAN.scan(archive)
        except ValueError as error:
            assert "forbidden image path" in str(
                error
            ) or "forbidden vendor payload signature" in str(error)
        else:
            raise AssertionError(f"forbidden path accepted: {name}")
    for path in ("opt/isaac/runtime", "opt/omniverse/kit", "opt/ngc/cache"):
        assert SCAN.FORBIDDEN_PATH.search(path)


def test_notice_files_do_not_bypass_vendor_signature_scan(tmp_path: Path) -> None:
    archive = tmp_path / "notice-vendor.tar"
    files = {
        **REQUIRED,
        "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md": (
            b"unexpected registry: " + b"nvcr.io/vendor/image"
        ),
    }
    _docker_save(archive, files)
    with pytest.raises(ValueError, match="forbidden vendor payload signature"):
        SCAN.scan(archive)


def test_nested_archive_members_are_scanned(tmp_path: Path) -> None:
    archive = tmp_path / "nested.tar"
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as nested:
        nested.writestr("source/private-key.txt", b"BEGIN " + b"RSA PRIVATE" + b" KEY")
    _docker_save(archive, {**REQUIRED, "opt/extra/source.whl": output.getvalue()})
    try:
        SCAN.scan(archive)
    except ValueError as error:
        assert "forbidden secret signature" in str(error)
    else:
        raise AssertionError("nested private key was accepted")


def test_nested_archive_vendor_signatures_are_scanned(tmp_path: Path) -> None:
    archive = tmp_path / "nested-vendor.tar"
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as nested:
        nested.writestr("source/runtime.txt", b"registry=" + b"nvcr.io/vendor/image")
    _docker_save(archive, {**REQUIRED, "opt/extra/source.whl": output.getvalue()})
    with pytest.raises(ValueError, match="forbidden vendor payload signature"):
        SCAN.scan(archive)


def test_misnamed_zip_and_tar_archives_are_scanned_by_bytes(tmp_path: Path) -> None:
    zip_output = io.BytesIO()
    with zipfile.ZipFile(
        zip_output, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as nested:
        nested.writestr("source/private-key.txt", b"BEGIN " + b"RSA PRIVATE" + b" KEY")
    zip_archive = tmp_path / "misnamed-zip.tar"
    _docker_save(
        zip_archive,
        {**REQUIRED, "opt/extra/innocent.bin": zip_output.getvalue()},
    )
    with pytest.raises(ValueError, match="forbidden secret signature"):
        SCAN.scan(zip_archive)

    tar_output = _tar_bytes({"usr/local/cuda/libcuda.so": b"otherwise-clean-binary"})
    tar_archive = tmp_path / "misnamed-tar.tar"
    _docker_save(
        tar_archive,
        {**REQUIRED, "opt/extra/innocent.data": tar_output},
    )
    with pytest.raises(ValueError, match="forbidden nested archive member"):
        SCAN.scan(tar_archive)


@pytest.mark.parametrize(
    "compressed",
    (
        gzip.compress(b"registry=" + b"nvcr.io/vendor/image"),
        bz2.compress(b"registry=" + b"nvcr.io/vendor/image"),
        lzma.compress(b"registry=" + b"nvcr.io/vendor/image"),
    ),
)
def test_misnamed_compressed_streams_are_expanded(
    tmp_path: Path, compressed: bytes
) -> None:
    archive = tmp_path / f"compressed-{hashlib.sha256(compressed).hexdigest()}.tar"
    _docker_save(archive, {**REQUIRED, "opt/extra/innocent.bin": compressed})
    with pytest.raises(ValueError, match="forbidden vendor payload signature"):
        SCAN.scan(archive)


@pytest.mark.parametrize(
    ("name", "content"),
    (
        ("payload.tar.gz", gzip.compress(b"not-a-tar")),
        ("payload.tar.bz2", bz2.compress(b"not-a-tar")),
        ("payload.tar.xz", lzma.compress(b"not-a-tar")),
        ("payload.gz", b"not-gzip"),
        ("payload.bz2", b"not-bzip2"),
        ("payload.xz", b"not-xz"),
    ),
)
def test_declared_archive_and_compression_types_must_match_bytes(
    tmp_path: Path, name: str, content: bytes
) -> None:
    archive = tmp_path / (name.replace(".", "-") + ".tar")
    _docker_save(archive, {**REQUIRED, f"opt/extra/{name}": content})
    with pytest.raises(
        ValueError,
        match="(?:declared compression does not match|compressed tar payload is not)",
    ):
        SCAN.scan(archive)


def test_appended_tar_and_zip_streams_are_rejected(tmp_path: Path) -> None:
    first_tar = _tar_bytes({"source/clean.txt": b"clean"})
    second_tar = _tar_bytes({"usr/local/cuda/libcuda.so": b"otherwise-clean-binary"})
    tar_archive = tmp_path / "concatenated-tar.tar"
    _docker_save(
        tar_archive,
        {**REQUIRED, "opt/extra/concatenated.bin": first_tar + second_tar},
    )
    with pytest.raises(ValueError, match="unaccounted tar bytes"):
        SCAN.scan(tar_archive)

    zip_output = io.BytesIO()
    with zipfile.ZipFile(zip_output, mode="w") as nested:
        nested.writestr("source/clean.txt", b"clean")
    zip_archive = tmp_path / "zip-with-suffix.tar"
    _docker_save(
        zip_archive,
        {**REQUIRED, "opt/extra/concatenated.bin": zip_output.getvalue() + second_tar},
    )
    with pytest.raises(ValueError, match="unaccounted zip bytes"):
        SCAN.scan(zip_archive)


def test_gzip_requires_one_exact_member_and_compatible_archive_name(
    tmp_path: Path,
) -> None:
    clean_gzip = gzip.compress(b"otherwise-clean-content")
    cases = {
        "concatenated.bin": clean_gzip + gzip.compress(b"second-member"),
        "trailing.bin": clean_gzip + b"\0" * 8,
        "mislabeled.zip": clean_gzip,
        "mislabeled.tar": clean_gzip,
    }
    for name, content in cases.items():
        archive = tmp_path / (name.replace(".", "-") + ".tar")
        _docker_save(archive, {**REQUIRED, f"opt/extra/{name}": content})
        with pytest.raises(
            ValueError,
            match="(?:ambiguous compressed stream|declared (?:ZIP|tar) does not match)",
        ):
            SCAN.scan(archive)


def test_second_level_nested_archive_signatures_are_scanned(tmp_path: Path) -> None:
    archive = tmp_path / "nested-twice.tar"
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, mode="w", compression=zipfile.ZIP_DEFLATED) as nested:
        nested.writestr("metadata.txt", b"registry=" + b"nvcr.io/vendor/image")
    outer = io.BytesIO()
    with tarfile.open(fileobj=outer, mode="w:gz") as nested:
        info = tarfile.TarInfo("source/dependency.whl")
        info.size = len(inner.getvalue())
        nested.addfile(info, io.BytesIO(inner.getvalue()))
    _docker_save(archive, {**REQUIRED, "opt/extra/sources.bin": outer.getvalue()})
    with pytest.raises(ValueError, match="forbidden vendor payload signature"):
        SCAN.scan(archive)


@pytest.mark.parametrize(
    ("env", "labels", "pattern"),
    [
        (["PASSWORD=not-a-real-test-secret"], None, "secret signature"),
        (None, {"vendor.registry": "nvcr.io/vendor/image"}, "vendor payload"),
    ],
)
def test_full_config_bytes_are_scanned(
    tmp_path: Path,
    env: list[str] | None,
    labels: dict[str, str] | None,
    pattern: str,
) -> None:
    archive = tmp_path / "config.tar"
    _docker_save(archive, REQUIRED, env=env, labels=labels)
    with pytest.raises(ValueError, match=pattern):
        SCAN.scan(archive)


def test_ordered_diff_ids_must_match_saved_layer_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "wrong-diff-id.tar"
    _docker_save(archive, REQUIRED, diff_ids=["sha256:" + "0" * 64] * 2)
    with pytest.raises(ValueError, match="rootfs diff IDs"):
        SCAN.scan(archive)

    config_archive = tmp_path / "wrong-config-descriptor.tar"
    _docker_save(config_archive, REQUIRED, config_name="0" * 64 + ".json")
    with pytest.raises(ValueError, match="config filename"):
        SCAN.scan(config_archive)


def test_reviewed_base_diff_id_must_match_first_saved_layer(tmp_path: Path) -> None:
    archive = tmp_path / "replacement-base.tar"
    replacement = _tar_bytes({"etc/npa-synthetic-base": b"replacement"})
    _docker_save(archive, REQUIRED, base_layer=replacement)
    with pytest.raises(ValueError, match="reviewed Ubuntu base layer"):
        SCAN.scan(archive)


def test_complete_lock_bytes_and_python_versions_are_exact(tmp_path: Path) -> None:
    archive = tmp_path / "replacement-lock.tar"
    changed = REQUIREMENTS.replace("mujoco==3.12.0", "mujoco==3.11.0").encode()
    files = {
        **REQUIRED,
        "opt/npa/gymnasium-robotics/requirements.lock": changed,
    }
    _docker_save(archive, files)
    with pytest.raises(ValueError, match="reviewed complete lock bytes changed"):
        SCAN.scan(archive)
    assert SCAN._locked_python_distributions(changed)["mujoco"] == "3.11.0"
    assert (
        SCAN._locked_python_distributions(changed) != SCAN.EXPECTED_PYTHON_DISTRIBUTIONS
    )


@pytest.mark.parametrize(
    ("symlinks", "hardlinks"),
    [
        ({"usr/bin/clean-link": "../../etc/npa-synthetic-base"}, None),
        (None, {"usr/bin/clean-hardlink": "etc/npa-synthetic-base"}),
    ],
)
def test_unclassified_links_fail(
    tmp_path: Path,
    symlinks: dict[str, str] | None,
    hardlinks: dict[str, str] | None,
) -> None:
    archive = tmp_path / "unclassified-link.tar"
    _docker_save(archive, REQUIRED, symlinks=symlinks, hardlinks=hardlinks)
    with pytest.raises(ValueError, match="unclassified entries"):
        SCAN.scan(archive)


def test_forbidden_link_target_and_unsupported_member_fail(tmp_path: Path) -> None:
    link_archive = tmp_path / "forbidden-link.tar"
    _docker_save(
        link_archive,
        REQUIRED,
        symlinks={"usr/lib/clean-link": "/usr/local/cuda/lib64/libcuda.so"},
    )
    with pytest.raises(ValueError, match="forbidden image link target"):
        SCAN.scan(link_archive)

    fifo_archive = tmp_path / "fifo.tar"
    _docker_save(fifo_archive, REQUIRED, fifos=("run/unclassified-fifo",))
    with pytest.raises(ValueError, match="unsupported image member type"):
        SCAN.scan(fifo_archive)


@pytest.mark.parametrize("kind", ("fifo", "device", "symlink", "hardlink"))
@pytest.mark.parametrize("name", (".wh.hidden", "opt/.wh..wh..opq"))
def test_typed_whiteout_entries_fail_before_semantics(
    tmp_path: Path, kind: str, name: str
) -> None:
    archive = tmp_path / f"{kind}-{name.replace('/', '-')}.tar"
    if kind == "fifo":
        _docker_save(archive, REQUIRED, fifos=(name,))
    elif kind == "device":
        _docker_save(archive, REQUIRED, devices=(name,))
    elif kind == "symlink":
        _docker_save(archive, REQUIRED, symlinks={name: "absent"})
    else:
        _docker_save(
            archive,
            REQUIRED,
            hardlinks={name: "etc/npa-synthetic-base"},
        )
    with pytest.raises(ValueError, match="invalid whiteout entry"):
        SCAN.scan(archive)


def test_nonempty_whiteout_fails_and_empty_regular_whiteout_is_hashed(
    tmp_path: Path,
) -> None:
    for name in (".wh.absent", "opt/.wh..wh..opq"):
        nonempty = tmp_path / ("nonempty-" + name.replace("/", "-") + ".tar")
        _docker_save(nonempty, {**REQUIRED, name: b"not-empty"})
        with pytest.raises(ValueError, match="invalid whiteout entry"):
            SCAN.scan(nonempty)

    valid = tmp_path / "valid-whiteout.tar"
    _docker_save(valid, {**REQUIRED, ".wh.absent": b""})
    result = SCAN.scan(valid)
    assert result["whiteout_entry_count"] == 1
    assert len(result["whiteout_metadata_sha256"]) == 64


def test_raw_layer_trailing_bytes_are_scanned(tmp_path: Path) -> None:
    archive = tmp_path / "raw-trailing-secret.tar"
    marker = b"BEGIN " + b"OPENSSH PRIVATE" + b" KEY"
    _docker_save(archive, REQUIRED, app_suffix=marker)
    with pytest.raises(ValueError, match="forbidden secret signature"):
        SCAN.scan(archive)

    benign_layer = tmp_path / "raw-trailing-benign.tar"
    _docker_save(benign_layer, REQUIRED, app_suffix=b"benign trailing bytes")
    with pytest.raises(ValueError, match="unaccounted tar bytes"):
        SCAN.scan(benign_layer)

    outer = tmp_path / "outer-trailing-benign.tar"
    _docker_save(outer, REQUIRED)
    with outer.open("ab") as stream:
        stream.write(b"benign trailing bytes")
    with pytest.raises(ValueError, match="unaccounted tar bytes"):
        SCAN.scan(outer)
