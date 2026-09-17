from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/scan_image_antioch_payload.py"
SPEC = importlib.util.spec_from_file_location("scan_image_antioch_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def _image(
    tmp_path: Path,
    files: dict[str, bytes],
    *,
    history: str = "",
    config_env: list[str] | None = None,
) -> Path:
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    config = json.dumps(
        {
            "config": {"Env": config_env or ["PATH=/usr/bin"]},
            "history": [{"created_by": history}],
        }
    ).encode()
    manifest = json.dumps(
        [{"Config": "config.json", "RepoTags": ["fixture:latest"], "Layers": ["layer.tar"]}]
    ).encode()
    target = tmp_path / "image.tar"
    with tarfile.open(target, "w") as outer:
        for name, payload in {"manifest.json": manifest, "config.json": config, "layer.tar": layer.getvalue()}.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            outer.addfile(member, io.BytesIO(payload))
    return target


def test_clean_adapter_fixture_passes(tmp_path: Path) -> None:
    report = scanner.scan_tarball(_image(tmp_path, {"opt/npa/app.py": b"print('ok')\n"}))
    assert report["verdict"] == "clean"
    assert report["layers_scanned"] == 1
    assert report["entries_scanned"] == 1


def test_renamed_distribution_is_found_from_metadata(tmp_path: Path) -> None:
    image = _image(
        tmp_path,
        {"opt/renamed/harmless.dist-info/METADATA": b"Metadata-Version: 2.1\nName: antioch_sim\n"},
    )
    kinds = {item["kind"] for item in scanner.scan_tarball(image)["findings"]}
    assert "renamed_vendor_distribution" in kinds


def test_binary_weight_vendor_state_and_private_key_are_rejected(tmp_path: Path) -> None:
    image = _image(
        tmp_path,
        {
            "usr/lib/librenamed.so": b"\x7fELF\0antioch-sim proprietary runtime",
            "opt/data/model.safetensors": b"weights",
            "root/.antioch/session": b"state",
            "tmp/innocent": (
                b"-----BEGIN PRIVATE KEY-----\n"
                + b"A" * 96
                + b"\n-----END PRIVATE KEY-----\n"
            ),
        },
    )
    kinds = {item["kind"] for item in scanner.scan_tarball(image)["findings"]}
    assert {"proprietary_binary", "checkpoint_or_weight", "vendor_state", "credential_material"} <= kinds


def test_config_and_history_leakage_are_rejected(tmp_path: Path) -> None:
    image = _image(
        tmp_path,
        {},
        history="RUN pip install antioch_sim==0.3.63",
        config_env=["ANTIOCH_API_KEY=not-a-real-test-token"],
    )
    findings = scanner.scan_tarball(image)["findings"]
    assert any(item["kind"] == "vendor_install" for item in findings)
    assert any(item["kind"] == "credential_material" for item in findings)
