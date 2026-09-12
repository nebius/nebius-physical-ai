"""Mutation tests for the RoboTwin zero-vendor-payload image boundary."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("scan_image_wan_payload")
scanner = _load("scan_image_robotwin_payload")


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_source_vendor_runtime_and_cuda_are_all_refused(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/script/collect_data.py": b"official source",
            "usr/local/lib/python3.10/site-packages/curobo/__init__.py": b"",
            "usr/local/cuda/lib64/libcudart.so.12": b"authorized private runtime",
        },
    )
    config = {
        "history": [
            {"created_by": "FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"},
            {"created_by": "RUN pip install -r /tmp/robotwin-requirements.lock"},
        ]
    }
    kinds = _kinds(scanner.scan(rootfs, config))
    assert {
        "robotwin_source",
        "curobo_source_or_runtime",
        "cuda_or_cudnn_runtime",
        "nvidia_or_pytorch_base",
    } <= kinds


def test_asset_archives_and_extracted_assets_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/assets/objects.zip": b"asset bytes",
            "opt/robotwin/assets/embodiments/aloha-agilex/config.yml": b"x",
        },
    )
    kinds = _kinds(scanner.scan(rootfs, {}))
    assert {"robotwin_asset_archive", "robotwin_extracted_asset"} <= kinds


def test_renamed_nested_archive_is_traversed(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/opaque/payload.bin": _zip(
                {"objects/020_hammer/base0/model_data0.json": b"asset bytes"}
            )
        },
    )
    assert "robotwin_extracted_asset" in _kinds(scanner.scan(rootfs, {}))


def test_renamed_source_metadata_and_private_evidence_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opaque/source/script/collect_data.py": b"official source",
            "opaque/metadata/.git/config": b"repository metadata",
            "renamed/credential.bin": b"AKIAABCDEFGHIJKLMNOP",
            "renamed/evidence.bin": (
                b'{"solution":"robotwin",'
                b'"ownership_provenance":"manager-issued",'
                b'"project":"private-project-canary",'
                b'"output_root":"s3://private-bucket-canary/output"}'
            ),
        },
    )

    findings = scanner.scan(rootfs, {})
    kinds = _kinds(findings)
    assert {
        "robotwin_source",
        "source_control_metadata",
        "credential_content",
    } <= kinds
    credential_paths = {
        finding.path for finding in findings if finding.kind == "credential_content"
    }
    assert credential_paths == {
        "renamed/credential.bin",
        "renamed/evidence.bin",
    }


def test_cache_outputs_and_build_time_fetch_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "root/.cache/huggingface/hub/datasets--TianxingChen--RoboTwin2.0/ref": b"x",
            "opt/custom-cache/datasets--TianxingChen--RoboTwin2.0/snapshots/ref": b"x",
            "workspace/byof-runs/run/robotwin-native/episode_0/episode_0000000.hdf5": b"x",
        },
    )
    config = {
        "history": [
            {
                "created_by": (
                    "RUN huggingface-cli download TianxingChen/RoboTwin2.0 "
                    "objects.zip"
                )
            }
        ]
    }
    kinds = _kinds(scanner.scan(rootfs, config))
    assert {
        "robotwin_huggingface_cache",
        "robotwin_generated_output",
        "robotwin_asset_fetch_at_build",
    } <= kinds


def test_cli_report_records_a_clean_offline_scan(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {"opt/npa/robotwin/REDISTRIBUTION.md": b"NPA bootstrap notice"},
    )
    output = tmp_path / "report.json"
    assert scanner.main(["--rootfs-tar", str(rootfs), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {
        "format": "npa_robotwin_image_byte_scan_v1",
        "image": "offline-rootfs",
        "status": "pass",
        "archives_scanned": 1,
        "findings": [],
    }


def test_private_image_stdin_is_bounded_and_never_echoed_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = "registry.example/private/npa-robotwin@sha256:" + "a" * 64
    monkeypatch.setattr(
        scanner.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(private.encode()), encoding="utf-8"),
    )
    monkeypatch.setattr(
        scanner,
        "_private_remote_material",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(private)),
    )

    assert scanner.main(["--image-stdin"]) == 2
    output = capsys.readouterr().out
    assert "private image scan failed" in output
    assert private not in output

    oversized = b"x" * (scanner.MAX_IMAGE_REFERENCE_BYTES + 1)
    monkeypatch.setattr(
        scanner.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(oversized), encoding="utf-8"),
    )
    assert scanner.main(["--image-stdin"]) == 2
    assert "private image scan failed" in capsys.readouterr().out


class _RegistryResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_private_registry_transport_downloads_exact_oci_bytes_without_child_image_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layer = _tar_bytes({"opt/npa/robotwin/REDISTRIBUTION.md": b"NPA notice"})
    config = json.dumps({"history": []}, separators=(",", ":")).encode()
    layer_digest = _digest(layer)
    config_digest = _digest(config)
    platform_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": len(config)},
            "layers": [{"digest": layer_digest, "size": len(layer)}],
        },
        separators=(",", ":"),
    ).encode()
    platform_digest = _digest(platform_manifest)
    index = json.dumps(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    "digest": platform_digest,
                    "size": len(platform_manifest),
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    index_digest = _digest(index)
    image = f"registry.example/private/npa-robotwin@{index_digest}"
    payloads = {
        f"manifests/{index_digest}": index,
        f"manifests/{platform_digest}": platform_manifest,
        f"blobs/{config_digest}": config,
        f"blobs/{layer_digest}": layer,
    }

    class FakeOpener:
        def open(self, request, **_kwargs):
            suffix = request.full_url.split("/v2/private/npa-robotwin/", 1)[1]
            return _RegistryResponse(payloads[suffix])

    monkeypatch.setattr(scanner, "_URL_OPENER", FakeOpener())
    monkeypatch.setattr(scanner, "_docker_credentials", lambda _registry: {})

    def no_child_process(argv, **_kwargs):
        assert image not in argv
        raise AssertionError("private OCI materialization must not launch a child")

    monkeypatch.setattr(scanner.subprocess, "run", no_child_process)
    tars, parsed_config = scanner._private_remote_material(image, tmp_path)

    assert len(tars) == 2
    assert parsed_config == {"history": []}
    assert scanner.scan_tars(tars, parsed_config) == []


def test_docker_credential_helper_receives_registry_only_on_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        json.dumps({"credHelpers": {"registry.example": "unit"}}),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], str]] = []

    def helper(argv, **kwargs):
        calls.append((list(argv), kwargs["input"]))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"Username": "operator", "Secret": "credential"}),
            stderr="",
        )

    monkeypatch.setattr(scanner.subprocess, "run", helper)
    credentials = scanner._docker_credentials(
        "registry.example", {"DOCKER_CONFIG": str(docker_config), "HOME": str(tmp_path)}
    )

    assert credentials == {"username": "operator", "secret": "credential"}
    assert calls == [(["docker-credential-unit", "get"], "registry.example\n")]
