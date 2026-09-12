from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
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


def _tar_bytes(
    files: dict[str, bytes],
    *,
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
    suffix: bytes = b"",
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
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
) -> None:
    base = _tar_bytes({"etc/neutral-base": b"base"})
    app = _tar_bytes(
        files,
        symlinks=symlinks,
        hardlinks=hardlinks,
        suffix=layer_suffix,
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
        }
    )
    path.write_bytes(outer)


@pytest.fixture
def structural_scan(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_production_trust_roots_remain_withheld() -> None:
    assert all(value is None for value in SCAN.EXPECTED_NEUTRAL_FILE_SHA256.values())
    assert SCAN.EXPECTED_BASE["uncompressed_layer_digest"] is None
    assert all(
        value is None for value in VERIFIER.EXPECTED_NEUTRAL_FILE_SHA256.values()
    )


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
    with pytest.raises(ValueError, match="forbidden"):
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
        "expected_python_distribution_count": 26,
        "resolved_python_artifact_count": 26,
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
        "resolved_binary_packages": [{"name": "python3.12", "version": "exact"}],
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
    diff_ids = ["sha256:" + "b" * 64]
    return rootfs, diff_ids


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
