"""Complete-layer negative controls for the SeedVR2 runtime-payload scanner."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_seedvr2_payload.py"
)
SPEC = importlib.util.spec_from_file_location("scan_image_seedvr2_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _saved_image(
    tmp_path: Path,
    layers: list[dict[str, bytes]],
    *,
    config: bytes = b"{}",
    rootfs_diff_ids: list[str] | None = None,
) -> Path:
    layer_names = []
    payloads = {
        "manifest.json": b"",
    }
    diff_ids = []
    for index, members in enumerate(layers):
        name = f"layer-{index}.tar"
        payloads[name] = _tar(tmp_path / name, members).read_bytes()
        layer_names.append(name)
        diff_ids.append("sha256:" + hashlib.sha256(payloads[name]).hexdigest())
    config_document = json.loads(config)
    config_document["rootfs"] = {
        "type": "layers",
        "diff_ids": rootfs_diff_ids if rootfs_diff_ids is not None else diff_ids,
    }
    payloads["config.json"] = json.dumps(config_document).encode()
    payloads["manifest.json"] = json.dumps(
        [
            {
                "Config": "config.json",
                "RepoTags": ["seedvr2:test"],
                "Layers": layer_names,
            }
        ]
    ).encode()
    return _tar(tmp_path / "image.tar", payloads)


def test_clean_source_runtime_and_python_pth_file_pass(tmp_path: Path) -> None:
    saved = _saved_image(
        tmp_path,
        [
            {
                "opt/seedvr2/LICENSE": b"Apache License 2.0",
                "opt/seedvr2-venv/lib/python3.12/site-packages/runtime.pth": b"/opt",
            }
        ],
    )
    findings, layers = scanner.scan_saved_image(saved)
    assert layers == 1
    assert findings == []
    _, identity = scanner.inspect_saved_image(saved)
    assert identity["image_config_digest"].startswith("sha256:")
    assert identity["layers"][0]["diff_id"] == identity["rootfs_diff_ids"][0]


def test_saved_layer_must_match_config_diff_id(tmp_path: Path) -> None:
    saved = _saved_image(
        tmp_path,
        [{"opt/seedvr2/LICENSE": b"Apache License 2.0"}],
        rootfs_diff_ids=["sha256:" + "0" * 64],
    )
    with pytest.raises(RuntimeError, match="rootfs diff IDs"):
        scanner.inspect_saved_image(saved)


def test_cudnn_sdk_headers_and_static_archives_fail(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h": b"sdk",
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/include/unreviewed.hpp": b"sdk",
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_static.a": b"sdk",
                    "usr/include/cudnn_version.h": b"sdk",
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == [
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
    ]


def test_nested_wheel_cudnn_and_nvshmem_sdk_payloads_fail(tmp_path: Path) -> None:
    wheel = _zip(
        tmp_path / "cuda-runtime.whl",
        {
            "nvidia/cudnn/include/cudnn.h": b"sdk",
            "nvidia/nvshmem/include/device/nvshmem.cuh": b"sdk",
            "nvidia/nvshmem/lib/libnvshmem_device.a": b"sdk",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/cuda-runtime.whl": wheel.read_bytes()}],
        )
    )
    assert [finding.kind for finding in findings] == [
        "cudnn_sdk_payload",
        "nvshmem_sdk_payload",
        "nvshmem_sdk_payload",
    ]


def test_weights_cache_and_sensor_media_fail_in_lower_layer(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/model.pth": b"x",
                    "opt/seedvr2/test_videos/customer.mp4": b"x",
                },
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/.wh.model.pth": b"",
                    "opt/seedvr2/test_videos/.wh.customer.mp4": b"",
                },
            ],
        )
    )
    assert layers == 2
    kinds = {finding.kind for finding in findings}
    assert {"model_weight", "populated_hf_cache", "sensor_or_output_media"} <= kinds


def test_weight_hidden_in_nested_source_tar_fails(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "seedvr2-source.tar",
        {"SeedVR/pos_emb.pt": b"runtime-only embedding"},
    )
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/seedvr2.tar.gz": source_tar.read_bytes()}],
        )
    )
    assert layers == 1
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("model_weight", "tmp/seedvr2.tar.gz!/SeedVR/pos_emb.pt")
    ]


def test_unreadable_nested_source_tar_fails_closed(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/seedvr2.tar.gz": b"not a tar archive"}],
        )
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("unreadable_nested_archive", "tmp/seedvr2.tar.gz")
    ]


def test_weight_hidden_in_nested_source_zip_fails(tmp_path: Path) -> None:
    source_zip = _zip(
        tmp_path / "source.zip",
        {"SeedVR/weights/model.safetensors": b"runtime-only weight"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.zip": source_zip.read_bytes()}])
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("model_weight", "tmp/source.zip!/SeedVR/weights/model.safetensors")
    ]


def test_workspace_media_and_nested_egg_payload_fail(tmp_path: Path) -> None:
    egg = _zip(
        tmp_path / "dependency.egg",
        {"package/hidden/model.pth": b"runtime-only weight"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "workspace/output.mp4": b"generated media",
                    "tmp/dependency.egg": egg.read_bytes(),
                }
            ],
        )
    )
    assert {finding.kind for finding in findings} == {
        "model_weight",
        "sensor_or_output_media",
    }


def test_unsupported_nested_archive_fails_closed(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/source.tar.zst": b"opaque archive bytes"}],
        )
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("unsupported_nested_archive", "tmp/source.tar.zst")
    ]


def test_nested_dotfile_path_is_not_normalized_away(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {"./.aws/credentials": b"[default]\ncredential_process = example"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("credential_file", "tmp/source.tar!/.aws/credentials")
    ]


def test_nested_parent_or_backslash_path_fails_closed(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {
            "../escaped.txt": b"x",
            "folder\\disguised.txt": b"x",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert [finding.kind for finding in findings] == [
        "unsafe_archive_path",
        "unsafe_archive_path",
    ]


def test_nested_application_media_and_secret_content_fail(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {
            "opt/seedvr2/private/customer.mp4": b"video",
            "opt/npa-src/config.txt": b"hf_abcdefghijklmnopqrstuvwxyz",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert {finding.kind for finding in findings} == {
        "sensor_or_output_media",
        "credential_content",
    }


def test_secret_in_application_or_image_environment_fails(tmp_path: Path) -> None:
    token = b"hf_abcdefghijklmnopqrstuvwxyz"
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"opt/npa-src/config.json": token}],
            config=b'{"config":{"Env":["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"]}}',
        )
    )
    assert [finding.kind for finding in findings].count("credential_content") == 2


def test_dependency_secret_shaped_fixture_is_not_operator_payload(
    tmp_path: Path,
) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2-venv/lib/python3.12/site-packages/pkg/test.py": (
                        b"hf_abcdefghijklmnopqrstuvwxyz"
                    )
                }
            ],
        )
    )
    assert findings == []
