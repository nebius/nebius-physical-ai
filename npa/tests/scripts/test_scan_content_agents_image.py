from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import ModuleType
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/scan_content_agents_image.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_content_agents_image", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = _load()


def _config(**labels: str) -> dict:
    expected = {
        "npa.tool": "content-agents",
        "npa.redistribution": "public",
        "npa.driver_provisioning": "gpu-operator-host-mounted",
        "npa.driver_capabilities": "compute,utility,graphics,display",
        "npa.ovrtx.delivery": "runtime-fetch-from-nvidia",
        "npa.ovrtx.version": "0.3.0.312915",
        "npa.content_agents.version": "0.5.2",
        "org.opencontainers.image.revision": "36dbf3f274f8e256637230a05a085853f65cc175",
        "org.opencontainers.image.version": "0.5.2-npa2",
        "org.opencontainers.image.licenses": "Apache-2.0",
        "npa.source_revision": "3" * 40,
    }
    expected.update(labels)
    return {"config": {"User": "ubuntu", "Labels": expected}, "history": []}


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def test_public_metadata_contract_passes() -> None:
    result = scanner.audit_config(_config(), expected_npa_source_sha="3" * 40)
    assert result["npa_source_revision"] == "3" * 40


def test_metadata_refuses_restricted_or_proprietary_labels() -> None:
    with pytest.raises(scanner.ImageAuditError, match="npa.redistribution"):
        scanner.audit_config(_config(**{"npa.redistribution": "restricted"}))
    with pytest.raises(scanner.ImageAuditError, match="licenses"):
        scanner.audit_config(
            _config(**{"org.opencontainers.image.licenses": "NVIDIA Proprietary"})
        )


def test_clean_source_and_reviewed_lock_are_allowed(tmp_path: Path) -> None:
    image = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/content-agents/world_understanding/functions/graphics/"
            "pylock.ovrtx-runtime.toml": b'name = "ovrtx"\nversion = "0.3.0.312915"\n',
            "opt/npa/src/npa/workflows/content_agents_runtime.py": b"downloader only",
        },
    )
    assert scanner.scan(image, _config()) == []


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("opt/cache/lib/python3.12/site-packages/ovrtx/core.py", "ovrtx_runtime"),
        ("opt/cache/ovrtx-0.3.dist-info/METADATA", "ovrtx_runtime"),
        ("usr/lib/libGLX_nvidia.so.570", "nvidia_graphics_driver_userspace"),
        ("opt/content-agents/samples/customer.usdz", "sample_or_customer_payload"),
        ("workspace/customer.usd", "customer_workspace_data"),
        ("root/.aws/credentials", "credential_file"),
    ],
)
def test_payload_mutations_fail_closed(tmp_path: Path, path: str, kind: str) -> None:
    image = _tar(tmp_path / f"{kind}.tar", {path: b"mutated"})
    assert kind in {item.kind for item in scanner.scan(image, _config())}


def test_nested_ovrtx_wheel_mutation_is_detected(tmp_path: Path) -> None:
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as wheel:
        wheel.writestr("ovrtx/bin/libovrtx.so", b"native runtime")
    image = _tar(tmp_path / "rootfs.tar", {"tmp/runtime.whl": nested.getvalue()})
    assert "ovrtx_runtime" in {item.kind for item in scanner.scan(image, _config())}


def test_deleted_earlier_layer_still_fails_scan(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "rootfs.tar", {"opt/npa/app.py": b"clean"})
    old_layer = _tar(
        tmp_path / "old-layer.tar",
        {"opt/old/lib/python3.12/site-packages/ovrtx/core.py": b"deleted later"},
    )
    findings = scanner.scan_tars([rootfs, old_layer], _config())
    assert "ovrtx_runtime" in {item.kind for item in findings}


def test_build_time_bootstrap_and_acceptance_mutations_are_detected(
    tmp_path: Path,
) -> None:
    image = _tar(tmp_path / "rootfs.tar", {"opt/npa/app.py": b"clean"})
    config = _config()
    config["history"] = [
        {"created_by": "RUN python -m render_ovrtx --provision-only"},
        {"created_by": "ENV OMNI_KIT_ACCEPT_EULA=YES"},
    ]
    kinds = {item.kind for item in scanner.scan(image, config)}
    assert {"ovrtx_bootstrap_at_build", "content_agents_acceptance_gate"} <= kinds


@pytest.mark.parametrize(
    "secret",
    [
        b"AK" + b"IA" + b"ABCDEFGHIJKLMNOP",
        b"h" + b"f_" + b"0123456789abcdefghijklmnopqrstuvwx",
        b"nva" + b"pi-" + b"0123456789abcdefghijklmnopqrstuvwxyz",
        b"v" + b"1." + b"a" * 136 + b"." + b"b" * 96,
        b"AWS_SECRET_ACCESS_" + b"KEY=" + b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcd",
    ],
)
def test_secret_bytes_mutation_is_detected(tmp_path: Path, secret: bytes) -> None:
    image = _tar(
        tmp_path / "rootfs.tar",
        {"opt/npa/config.txt": secret},
    )
    assert "credential_content" in {
        item.kind for item in scanner.scan(image, _config())
    }


def test_sdk_symbol_and_placeholder_shapes_do_not_false_positive(
    tmp_path: Path,
) -> None:
    image = _tar(
        tmp_path / "rootfs.tar",
        {
            "usr/lib/sdk.so": b"hf_xet_internal_symbol nvapi_QueryInterface",
            "usr/bin/build-tool": b"internal version v1.0123456789abcdefghijklmnopqrstuvwxyz",
            "opt/npa/code.py": (
                b'os.environ.get("AWS_SECRET_ACCESS_KEY", "")\n'
                b'example = "NEBIUS_TOKEN_FACTORY_KEY=<token>"\n'
            ),
        },
    )
    assert scanner.scan(image, _config()) == []


def test_cli_offline_scan_reports_numeric_archive_count(tmp_path: Path, capsys) -> None:
    image = _tar(tmp_path / "rootfs.tar", {"opt/npa/app.py": b"clean"})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")

    assert scanner.main(["--rootfs-tar", str(image), "--config-json", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["archives_scanned"] == 1
