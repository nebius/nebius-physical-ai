from __future__ import annotations

import importlib.util
import hashlib
import io
import json
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
PRODUCTION_EXPECTED_ASSET_LOCK = SCAN.EXPECTED_ASSET_LOCK
PRODUCTION_EXPECTED_SOURCE_FIELDS = SCAN.EXPECTED_SOURCE_FIELDS

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
    "components": {
        name: dict(fields) for name, fields in TEST_SOURCE_FIELDS.items()
    },
}
SOURCE_LOCK["components"]["shadow_sr_common"].update(
    {
        "preferred_form_sha256": ANNEX_SHA256,
        "transformation_manifest_sha256": "b" * 64,
    }
)
APT_LOCK = {
    "status": "complete",
    "base": SCAN.EXPECTED_BASE,
    "resolved_binary_packages": [{"name": "libegl1", "version": "exact"}],
    "resolved_source_packages": [{"name": "libglvnd", "version": "exact"}],
}
CORRESPONDING_LOCK = {
    "status": "complete",
    "deliveries": [
        {
            "binary_component": component,
            **(
                {"archive_sha256": ANNEX_SHA256, "license": "MIT"}
                if component == "farama-gymnasium-robotics"
                else {
                    "build_instructions_sha256": "b" * 64,
                    "license": "GPL-2.0-only AND Apache-2.0",
                    "preferred_form_archive_sha256": ANNEX_SHA256,
                    "source_commit": SCAN.EXPECTED_SHADOW_COMMIT,
                    "transformation_manifest_sha256": "b" * 64,
                }
                if component == "shadow-hand-xml-mesh-texture-assets"
                else {
                    "binary_manifest_sha256": "b" * 64,
                    "build_materials_sha256": "b" * 64,
                    "source_manifest_sha256": "b" * 64,
                }
            ),
            "artifacts": [
                {
                    "path": f"{component}/source.bin",
                    "sha256": ANNEX_SHA256,
                }
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
REQUIREMENTS = "# status: complete\n" + "\n".join(
    f"{name}==1 --hash=sha256:" + "a" * 64
    for name in sorted(SCAN.EXPECTED_PYTHON_DISTRIBUTIONS)
) + "\n"
COMMON_FILES = {
    "opt/npa/gymnasium-robotics/source-lock.json": json.dumps(SOURCE_LOCK).encode(),
    "opt/npa/gymnasium-robotics/apt-runtime.lock.json": json.dumps(APT_LOCK).encode(),
    "opt/npa/gymnasium-robotics/asset-lock.json": json.dumps(ASSET_LOCK).encode(),
    "opt/npa/gymnasium-robotics/requirements.lock": REQUIREMENTS.encode(),
    "opt/npa/gymnasium-robotics/capability_smoke.py": b"print('smoke')\n",
    "usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md": b"notices\n",
    "usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md": b"public\n",
    **{
        f"usr/share/source/npa-gymnasium-robotics/{component}/source.bin": ANNEX
        for component in (
            "farama-gymnasium-robotics",
            "shadow-hand-xml-mesh-texture-assets",
            "ubuntu-runtime-closure",
        )
    },
    ASSET_PREFIX + "LICENSE.md": NOTICE,
    **{ASSET_PREFIX + name: raw for name, raw in {**XML, **MATERIAL}.items()},
}
ROOTFS_MANIFEST = {
    "schema": "npa.gymnasium-robotics.rootfs-manifest.v1",
    "files": {
        path: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "class": "npa-runtime",
            "source": "synthetic-test-fixture",
        }
        for path, raw in COMMON_FILES.items()
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


def test_production_scanner_binds_the_reviewed_asset_lock() -> None:
    path = ROOT / "npa/docker/workbench/gymnasium-robotics/asset-lock.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PRODUCTION_EXPECTED_ASSET_LOCK


def test_production_scanner_binds_the_reviewed_source_identities() -> None:
    path = ROOT / "npa/docker/workbench/gymnasium-robotics/source-lock.json"
    actual = json.loads(path.read_text(encoding="utf-8"))["components"]
    for name, expected in PRODUCTION_EXPECTED_SOURCE_FIELDS.items():
        assert all(actual[name][key] == value for key, value in expected.items())


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def _docker_save(
    path: Path,
    files: dict[str, bytes],
    *,
    history: str = "clean",
    user: str = "ubuntu",
) -> None:
    layer = _tar_bytes(files)
    config = json.dumps(
        {"config": {"User": user}, "history": [{"created_by": history}]}
    ).encode()
    manifest = json.dumps([{"Config": "config.json", "Layers": ["layer.tar"]}]).encode()
    with tarfile.open(path, mode="w") as archive:
        for name, raw in (
            ("manifest.json", manifest),
            ("config.json", config),
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
    assert result["layer_count"] == 1
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
    with pytest.raises(ValueError, match="unclassified regular files"):
        SCAN.scan(unclassified)


def test_incomplete_corresponding_source_fails(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    files = {
        **REQUIRED,
        "opt/npa/gymnasium-robotics/corresponding-source.lock.json": b'{"status":"phase-a-incomplete"}',
    }
    _docker_save(archive, files)
    try:
        SCAN.scan(archive)
    except ValueError as error:
        assert "incomplete evidence lock" in str(error)
    else:
        raise AssertionError("incomplete corresponding source was accepted")


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
        assert "image history" in str(error)
    else:
        raise AssertionError("cached consent was accepted")


def test_model_and_cache_paths_fail(tmp_path: Path) -> None:
    for name in (
        "opt/model.safetensors",
        "root/.cache/pip/wheel",
        "usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.1",
    ):
        archive = tmp_path / (name.replace("/", "-") + ".tar")
        _docker_save(archive, {**REQUIRED, name: b"payload"})
        try:
            SCAN.scan(archive)
        except ValueError as error:
            assert "forbidden image path" in str(error)
        else:
            raise AssertionError(f"forbidden path accepted: {name}")


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
        nested.writestr(
            "source/private-key.txt", b"BEGIN " + b"RSA PRIVATE" + b" KEY"
        )
    _docker_save(archive, {**REQUIRED, "opt/extra/source.whl": output.getvalue()})
    try:
        SCAN.scan(archive)
    except ValueError as error:
        assert "forbidden nested archive" in str(error)
    else:
        raise AssertionError("nested private key was accepted")
