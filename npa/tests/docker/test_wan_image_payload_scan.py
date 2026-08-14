from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import importlib.util
import json
import lzma
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import zstandard as zstd


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_image_wan_payload.py"
_SPEC = importlib.util.spec_from_file_location("scan_image_wan_payload", _SCRIPT)
assert _SPEC and _SPEC.loader
scanner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scanner
_SPEC.loader.exec_module(scanner)


def _gzip(payload: bytes) -> bytes:
    """Stable gzip bytes so xdist workers collect identical parametrized IDs."""

    return gzip.compress(payload, mtime=0)


def _tar(path: Path, members: dict[str, bytes], *, mode: str = "w") -> Path:
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _tar_bytes(members: dict[str, bytes], *, mode: str = "w") -> bytes:
    payload = io.BytesIO()
    archive_mode = "w" if mode == "w:gz" else mode
    with tarfile.open(fileobj=payload, mode=archive_mode) as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    raw = payload.getvalue()
    return _gzip(raw) if mode == "w:gz" else raw


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, content)
    return payload.getvalue()


def _ar_bytes(members: dict[str, bytes]) -> bytes:
    payload = bytearray(b"!<arch>\n")
    for name, content in members.items():
        encoded_name = f"{name}/".encode("utf-8")
        assert len(encoded_name) <= 16
        header = (
            encoded_name.ljust(16, b" ")
            + b"0".ljust(12, b" ")
            + b"0".ljust(6, b" ")
            + b"0".ljust(6, b" ")
            + b"100644".ljust(8, b" ")
            + str(len(content)).encode("ascii").ljust(10, b" ")
            + b"`\n"
        )
        assert len(header) == 60
        payload.extend(header)
        payload.extend(content)
        if len(content) % 2:
            payload.extend(b"\n")
    return bytes(payload)


def _v7_tar_bytes(name: str, payload: bytes) -> bytes:
    """Build a checksum-valid legacy tar header with no ustar magic."""

    def octal(value: int, width: int) -> bytes:
        return f"{value:0{width - 1}o}\0".encode("ascii")

    encoded_name = name.encode("utf-8")
    assert len(encoded_name) <= 100
    header = bytearray(512)
    header[: len(encoded_name)] = encoded_name
    header[100:108] = octal(0o644, 8)
    header[108:116] = octal(0, 8)
    header[116:124] = octal(0, 8)
    header[124:136] = octal(len(payload), 12)
    header[136:148] = octal(0, 12)
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[148:156] = f"{sum(header):06o}\0 ".encode("ascii")
    padding = b"\0" * ((512 - len(payload) % 512) % 512)
    return bytes(header) + payload + padding + b"\0" * 1024


def test_clean_oss_rootfs_and_runtime_fetch_plumbing_pass(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "clean.tar",
        {
            "opt/byof/LICENSE.txt": b"Apache License 2.0",
            "opt/npa/wan2-2/runtime-requirements.txt": b"torch==2.13.0\n",
            "usr/local/bin/wan-runtime": b"NVIDIA terms; runtime fetch only\n",
        },
    )
    assert (
        scanner.scan(
            rootfs, {"history": [{"created_by": "COPY runtime-requirements.txt /opt/"}]}
        )
        == []
    )


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        (
            "opt/venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12",
            "nvidia_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/easydict-1.13.dist-info/METADATA",
            "historical_lgpl_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/soxr-1.1.0.dist-info/METADATA",
            "historical_lgpl_python_distribution",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
            "bundled_ffmpeg_executable",
        ),
        ("usr/local/lib/libcudnn.so.9", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvcuvid.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnvoptix.so.1", "cuda_library"),
        (
            "opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so",
            "cuda_library",
        ),
        (
            "opt/venv/lib/python3.10/site-packages/torch/lib/libc10_cuda.so",
            "cuda_library",
        ),
        ("usr/lib/x86_64-linux-gnu/libcufile.so.0", "cuda_library"),
        ("usr/local/cuda/bin/ptxas", "cuda_tool"),
        ("usr/local/cuda/include/cuda.h", "cuda_tool"),
        ("usr/local/cuda-12.8/include/cuda.h", "cuda_tool"),
        ("opt/cuda/include/nccl.h", "cuda_tool"),
        ("usr/include/cuda_runtime_api.h", "cuda_tool"),
        ("usr/bin/nvidia-smi", "cuda_tool"),
        ("usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1", "cuda_library"),
        ("usr/lib/x86_64-linux-gnu/libnppc.so.12", "cuda_library"),
        (
            "opt/venv/lib/python3.10/site-packages/nvidia_cuda_runtime_cu12-12.8.dist-info/METADATA",
            "nvidia_python_distribution",
        ),
        ("opt/byof/model.safetensors", "checkpoint_or_weight"),
        ("opt/byof/pytorch_model-00001-of-00002.bin", "checkpoint_or_weight"),
        ("root/.cache/huggingface/hub/model.bin", "checkpoint_or_weight"),
        ("root/.aws/credentials", "credential_file"),
        ("tmp/wheelhouse/download.whl", "package_cache"),
        ("etc/ssh/ssh_host_rsa_key", "credential_file"),
    ],
)
def test_forbidden_path_mutations_fail(tmp_path: Path, name: str, kind: str) -> None:
    findings = scanner.scan(_tar(tmp_path / "bad.tar", {name: b"payload"}), {})
    assert kind in {item.kind for item in findings}


@pytest.mark.parametrize(
    "created_by",
    [
        "RUN pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1",
        "RUN pip install nvidia-cublas-cu12==12.8.3.14",
        "RUN wan-runtime ensure",
        "FROM nvidia/cuda:12.8.1-runtime",
        "ENV NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS=YES",
    ],
)
def test_forbidden_history_mutations_fail(tmp_path: Path, created_by: str) -> None:
    rootfs = _tar(tmp_path / "history.tar", {"opt/byof/LICENSE.txt": b"Apache-2.0"})
    assert scanner.scan(rootfs, {"history": [{"created_by": created_by}]})


def test_negative_inventory_assertion_is_not_a_cuda_install(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "history.tar", {"opt/byof/LICENSE.txt": b"Apache-2.0"})
    created_by = (
        "RUN python -m pip install --index-url "
        "https://download.pytorch.org/whl/cpu torch==2.7.1+cpu "
        "&& ! grep -Eiq '^nvidia-|cu(da|dnn|blas)|nccl' inventory.txt"
    )
    assert scanner.scan(rootfs, {"history": [{"created_by": created_by}]}) == []


def test_secret_content_mutation_fails(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "secret.tar", {"tmp/config": b"AKIAABCDEFGHIJKLMNOP"})
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {"credential_content"}


@pytest.mark.parametrize(
    "config",
    [
        {"config": {"Labels": {"build.token": "hf_token=actual_secret_value"}}},
        {"config": {"Env": ["AWS_SECRET_ACCESS_KEY=actual_secret_value"]}},
        {"history": [{"created_by": "RUN use AKIAABCDEFGHIJKLMNOP"}]},
    ],
)
def test_secret_like_oci_config_metadata_fails(
    tmp_path: Path, config: dict[str, object]
) -> None:
    rootfs = _tar(tmp_path / "metadata-secret.tar", {"opt/byof/LICENSE": b"clean"})
    findings = scanner.scan(rootfs, config)
    assert "credential_metadata" in {item.kind for item in findings}


@pytest.mark.parametrize(
    ("path", "expected_kind"),
    [
        (".aws/credentials", "credential_file"),
        ("./.aws/credentials", "credential_file"),
        (".cache/huggingface/model.bin", "package_cache"),
    ],
)
def test_root_dot_paths_remain_forbidden(
    tmp_path: Path, path: str, expected_kind: str
) -> None:
    findings = scanner.scan(_tar(tmp_path / "dot-path.tar", {path: b"payload"}), {})
    assert expected_kind in {item.kind for item in findings}


def test_large_secret_content_and_chunk_boundary_mutations_fail(tmp_path: Path) -> None:
    prefix = b"x" * (2 * 1024 * 1024 - 8)
    rootfs = _tar(
        tmp_path / "large-secret.tar",
        {"opt/application/large.dat": prefix + b"AKIAABCDEFGHIJKLMNOP"},
    )
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {"credential_content"}


def test_deleted_forbidden_bytes_in_a_lower_layer_still_fail(tmp_path: Path) -> None:
    lower = _tar(
        tmp_path / "lower.tar",
        {
            "opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9": b"vendor"
        },
    )
    upper = _tar(
        tmp_path / "upper.tar",
        {
            "opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib/.wh.libcudnn.so.9": b""
        },
    )
    findings = scanner._scan_tars([lower, upper], {})
    assert "cuda_library" in {item.kind for item in findings}


def test_renamed_elf_with_cuda_dependency_fails(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "elf.tar",
        {"opt/application/libinnocent.so": b"\x7fELF\x00libcuda.so.1\x00"},
    )
    findings = scanner.scan(rootfs, {})
    assert "cuda_elf_dependency" in {item.kind for item in findings}


@pytest.mark.parametrize(
    ("nested_name", "nested_payload", "expected_kind"),
    [
        (
            "opt/application/vendor.whl",
            _zip_bytes({"site-packages/nvidia/cublas/lib/libcublas.so.12": b"vendor"}),
            "nvidia_python_distribution",
        ),
        (
            "opt/application/payload.tar.gz",
            _tar_bytes({"models/model.safetensors": b"weights"}, mode="w:gz"),
            "checkpoint_or_weight",
        ),
        (
            "opt/application/renamed.zip",
            _zip_bytes(
                {
                    "lib/innocent.so": b"\x7fELF\x00libnvencode.so.1\x00",
                }
            ),
            "cuda_elf_dependency",
        ),
    ],
)
def test_nested_archive_mutations_fail_closed(
    tmp_path: Path, nested_name: str, nested_payload: bytes, expected_kind: str
) -> None:
    findings = scanner.scan(
        _tar(tmp_path / "nested-rootfs.tar", {nested_name: nested_payload}), {}
    )
    assert expected_kind in {item.kind for item in findings}
    assert any("!/" in item.path for item in findings)


def test_legacy_v7_tar_mutation_is_not_opaque(tmp_path: Path) -> None:
    nested_payload = _v7_tar_bytes("models/model.safetensors", b"weights")
    findings = scanner.scan(
        _tar(
            tmp_path / "legacy-tar-rootfs.tar",
            {"opt/application/legacy.payload": nested_payload},
        ),
        {},
    )
    assert "checkpoint_or_weight" in {item.kind for item in findings}
    assert any(
        "legacy.payload!/models/model.safetensors" in item.path for item in findings
    )


def test_debian_ar_data_archive_mutation_is_not_opaque(tmp_path: Path) -> None:
    data_tar = _tar_bytes(
        {"usr/lib/x86_64-linux-gnu/libcudart.so.12": b"\x7fELF\x00libcudart.so.12\x00"},
        mode="w:xz",
    )
    deb = _ar_bytes({"debian-binary": b"2.0\n", "data.tar.xz": data_tar})
    findings = scanner.scan(
        _tar(tmp_path / "deb-rootfs.tar", {"opt/application/payload.deb": deb}), {}
    )
    kinds = {item.kind for item in findings}
    assert "cuda_library" in kinds
    assert "cuda_elf_dependency" in kinds
    assert any(
        "payload.deb!/data.tar.xz!/usr/lib/x86_64-linux-gnu/libcudart.so.12"
        in item.path
        for item in findings
    )


def test_debian_ar_zstd_data_archive_mutation_is_not_opaque(tmp_path: Path) -> None:
    data_tar = _tar_bytes(
        {"usr/lib/x86_64-linux-gnu/libcudart.so.12": b"\x7fELF\x00libcudart.so.12\x00"}
    )
    compressed = zstd.ZstdCompressor().compress(data_tar)
    deb = _ar_bytes({"debian-binary": b"2.0\n", "data.tar.zst": compressed})
    findings = scanner.scan(
        _tar(tmp_path / "zstd-deb-rootfs.tar", {"opt/application/payload.deb": deb}),
        {},
    )
    kinds = {item.kind for item in findings}
    assert "cuda_library" in kinds
    assert "cuda_elf_dependency" in kinds


def test_zstd_archive_fails_closed_when_decoder_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compressed = zstd.ZstdCompressor().compress(b"clean data")
    monkeypatch.setattr(scanner, "zstd", None)
    findings = scanner.scan(
        _tar(tmp_path / "zstd-rootfs.tar", {"opt/application/data.zst": compressed}),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}
    assert any(
        "zstd archive support is unavailable" in item.detail for item in findings
    )


def test_malformed_debian_ar_fails_closed(tmp_path: Path) -> None:
    malformed = b"!<arch>\n" + b"broken"
    findings = scanner.scan(
        _tar(
            tmp_path / "malformed-deb-rootfs.tar",
            {"opt/application/broken.deb": malformed},
        ),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}


def test_nested_siblings_share_cumulative_decompression_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scanner, "MAX_NESTED_UNCOMPRESSED_BYTES", 100)
    nested_payload = _zip_bytes(
        {
            "first.gz": _gzip(b"a" * 40),
            "second.gz": _gzip(b"b" * 40),
        }
    )
    findings = scanner.scan(
        _tar(
            tmp_path / "cumulative-budget.tar",
            {"opt/application/data.zip": nested_payload},
        ),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}
    assert any("cumulative uncompressed byte limit" in item.detail for item in findings)


def test_nested_siblings_share_cumulative_member_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scanner, "MAX_NESTED_ARCHIVE_MEMBERS", 1)
    nested_payload = _zip_bytes({"first.txt": b"a", "second.txt": b"b"})
    findings = scanner.scan(
        _tar(
            tmp_path / "member-budget.tar", {"opt/application/data.zip": nested_payload}
        ),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}
    assert any("cumulative member limit" in item.detail for item in findings)


def test_malformed_nested_archive_fails_closed(tmp_path: Path) -> None:
    findings = scanner.scan(
        _tar(
            tmp_path / "malformed-nested-rootfs.tar",
            {"opt/application/broken.zip": b"PK\x03\x04not-a-real-archive"},
        ),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("opt/application/service.json.gz", _gzip(b'{"service": "clean"}')),
        ("opt/application/graph.gpickle.bz2", bz2.compress(b"clean graph data")),
        ("opt/application/index.json.xz", lzma.compress(b'{"index": "clean"}')),
    ],
)
def test_compressed_non_tar_data_is_inspected_without_tar_false_positive(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    findings = scanner.scan(_tar(tmp_path / "compressed-data.tar", {name: payload}), {})
    assert findings == []


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("opt/application/service.json.gz", _gzip(b"AKIAABCDEFGHIJKLMNOP")),
        (
            "opt/application/graph.gpickle.bz2",
            bz2.compress(b"hf_token=actual_secret_value"),
        ),
        (
            "opt/application/index.json.xz",
            lzma.compress(b"-----BEGIN OPENSSH PRIVATE KEY-----"),
        ),
    ],
)
def test_compressed_non_tar_secret_mutations_fail_closed(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    findings = scanner.scan(
        _tar(tmp_path / "compressed-secret.tar", {name: payload}), {}
    )
    assert "credential_content" in {item.kind for item in findings}
    assert any("!/<decompressed>" in item.path for item in findings)


def test_compressed_audited_example_literal_requires_exact_decompressed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer_path = "opt/application/service.json.gz"
    virtual_path = outer_path + "!/<decompressed>"
    audited_payload = b'{"access_key": "AKIAIOSFODNN7EXAMPLE"}'
    monkeypatch.setattr(
        scanner,
        "AUDITED_SECRET_LITERAL_FILE_SHA256",
        {virtual_path: hashlib.sha256(audited_payload).hexdigest()},
    )
    clean = scanner.scan(
        _tar(
            tmp_path / "compressed-audited.tar",
            {outer_path: _gzip(audited_payload)},
        ),
        {},
    )
    assert clean == []

    mutated = scanner.scan(
        _tar(
            tmp_path / "compressed-audited-mutated.tar",
            {outer_path: _gzip(audited_payload + b" mutated")},
        ),
        {},
    )
    assert "audited_literal_byte_drift" in {item.kind for item in mutated}


def test_malformed_gzip_stream_fails_closed(tmp_path: Path) -> None:
    findings = scanner.scan(
        _tar(
            tmp_path / "malformed-gzip-rootfs.tar",
            {"opt/application/broken.json.gz": b"\x1f\x8bnot-a-real-gzip"},
        ),
        {},
    )
    assert "nested_archive_unreadable" in {item.kind for item in findings}


def test_member_content_is_extracted_once_for_elf_and_secret_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rootfs = _tar(
        tmp_path / "single-pass.tar.gz",
        {
            "opt/application/librenamed.so": (
                b"\x7fELF"
                + b"x" * (1024 * 1024)
                + b"libcuda.so.1\x00AKIAABCDEFGHIJKLMNOP"
            )
        },
        mode="w:gz",
    )
    original = scanner.tarfile.TarFile.extractfile
    calls = 0

    def counting_extractfile(archive, member):
        nonlocal calls
        calls += 1
        return original(archive, member)

    monkeypatch.setattr(scanner.tarfile.TarFile, "extractfile", counting_extractfile)
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {
        "credential_content",
        "cuda_elf_dependency",
    }
    assert calls == 1


def test_known_debian_parser_and_optional_acceleration_literals_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = {
        "usr/lib/x86_64-linux-gnu/libavcodec.so.59.37.100": (
            b"\x7fELF\x00libcuda.so.1\x00libnvcuvid.so.1\x00"
        ),
        "usr/lib/x86_64-linux-gnu/libavutil.so.57.28.100": (
            b"\x7fELF\x00libcuda.so.1\x00"
        ),
        "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3": (
            b"\x7fELF\x00-----BEGIN PRIVATE KEY-----\x00"
        ),
    }
    monkeypatch.setattr(
        scanner,
        "AUDITED_LITERAL_LIBRARY_SHA256",
        {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        },
    )
    rootfs = _tar(
        tmp_path / "debian-literals.tar",
        members,
    )
    assert scanner.scan(rootfs, {}) == []


def test_secret_literal_exception_requires_exact_audited_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = "opt/wan-base/lib/python3.10/site-packages/example.py"
    audited_payload = b"example = 'AKIAABCDEFGHIJKLMNOP'\n"
    monkeypatch.setattr(
        scanner,
        "AUDITED_SECRET_LITERAL_FILE_SHA256",
        {path: hashlib.sha256(audited_payload).hexdigest()},
    )

    assert (
        scanner.scan(_tar(tmp_path / "audited.tar", {path: audited_payload}), {}) == []
    )


def test_precompiled_secret_literal_allowlist_is_exact() -> None:
    expected = {
        "opt/wan-base/lib/python3.10/site-packages/PIL/__pycache__/ImageFont.cpython-310.pyc": (
            "59632aaf913b02078acc5d366bcef3e28a14194acaf7904c4c13ac4714c319a2"
        ),
        "opt/wan-base/lib/python3.10/site-packages/cryptography/hazmat/primitives/serialization/__pycache__/ssh.cpython-310.pyc": (
            "b269114f93539cfc4c55511c3ecb8e55e7a8f1f9dd8755deef8762f9938eec3e"
        ),
    }

    assert {
        path: scanner.AUDITED_SECRET_LITERAL_FILE_SHA256.get(path)
        for path in expected
    } == expected


@pytest.mark.parametrize(
    "mutation",
    [
        b"\nAKIAQRSTUVWXYZABCDEF",
        b"\n-----BEGIN OPENSSH PRIVATE KEY-----",
        b"\nhf_token=hf_actual_secret_value",
        b"\nbenign byte drift",
    ],
)
def test_secret_literal_exception_mutations_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: bytes
) -> None:
    path = "opt/wan-base/lib/python3.10/site-packages/example.py"
    audited_payload = b"example = 'AKIAABCDEFGHIJKLMNOP'\n"
    monkeypatch.setattr(
        scanner,
        "AUDITED_SECRET_LITERAL_FILE_SHA256",
        {path: hashlib.sha256(audited_payload).hexdigest()},
    )

    findings = scanner.scan(
        _tar(tmp_path / "mutated-audited.tar", {path: audited_payload + mutation}), {}
    )
    assert "audited_literal_byte_drift" in {item.kind for item in findings}


@pytest.mark.parametrize(
    ("path", "audited_payload", "mutation", "expected_kind"),
    [
        (
            "usr/lib/x86_64-linux-gnu/libavcodec.so.59.37.100",
            b"\x7fELF\x00libcuda.so.1\x00",
            b"modified",
            "cuda_elf_dependency",
        ),
        (
            "usr/lib/x86_64-linux-gnu/libavutil.so.57.28.100",
            b"\x7fELF\x00libcuda.so.1\x00",
            b"modified",
            "cuda_elf_dependency",
        ),
        (
            "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3",
            b"\x7fELF\x00-----BEGIN PRIVATE KEY-----\x00",
            b"-----BEGIN OPENSSH PRIVATE KEY-----",
            "credential_content",
        ),
        (
            "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3",
            b"\x7fELF\x00-----BEGIN PRIVATE KEY-----\x00",
            b"AKIAABCDEFGHIJKLMNOP",
            "credential_content",
        ),
        (
            "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3",
            b"\x7fELF\x00-----BEGIN PRIVATE KEY-----\x00",
            b"hf_token=hf_example_secret_value",
            "credential_content",
        ),
    ],
)
def test_audited_library_path_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    audited_payload: bytes,
    mutation: bytes,
    expected_kind: str,
) -> None:
    monkeypatch.setattr(
        scanner,
        "AUDITED_LITERAL_LIBRARY_SHA256",
        {path: hashlib.sha256(audited_payload).hexdigest()},
    )
    findings = scanner.scan(
        _tar(tmp_path / "mutated-library.tar", {path: audited_payload + mutation}), {}
    )
    assert expected_kind in {item.kind for item in findings}


@pytest.mark.parametrize(
    "replacement",
    [
        b"\x7fELF\x00libother.so.1\x00",
        b"XXXX\x00libcuda.so.1\x00",
        b"\x7fELF\x00benign replacement\x00",
    ],
)
def test_audited_library_replacement_or_header_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: bytes
) -> None:
    path = "usr/lib/x86_64-linux-gnu/libavcodec.so.59.37.100"
    audited_payload = b"\x7fELF\x00libcuda.so.1\x00"
    monkeypatch.setattr(
        scanner,
        "AUDITED_LITERAL_LIBRARY_SHA256",
        {path: hashlib.sha256(audited_payload).hexdigest()},
    )
    findings = scanner.scan(
        _tar(tmp_path / "replaced-library.tar", {path: replacement}), {}
    )
    assert "audited_literal_byte_drift" in {item.kind for item in findings}


def test_gzip_layer_detects_boundary_spanning_elf_dependency(tmp_path: Path) -> None:
    chunk_size = 1024 * 1024
    payload = b"\x7fELF" + b"x" * (chunk_size - 5) + b"libcu" + b"dart.so.12\x00"
    rootfs = _tar(
        tmp_path / "boundary.tar.gz",
        {"opt/application/librenamed.so": payload},
        mode="w:gz",
    )
    findings = scanner.scan(rootfs, {})
    assert "cuda_elf_dependency" in {item.kind for item in findings}


@pytest.mark.parametrize("secret_first", [False, True])
def test_gzip_layer_detects_both_hit_orders_across_chunks(
    tmp_path: Path, secret_first: bool
) -> None:
    dependency = b"libcuda.so.1\x00"
    secret = b"AKIAABCDEFGHIJKLMNOP"
    first, second = (secret, dependency) if secret_first else (dependency, secret)
    payload = b"\x7fELF" + first + b"x" * (1024 * 1024 + 32) + second
    rootfs = _tar(
        tmp_path / f"hit-order-{secret_first}.tar.gz",
        {"opt/application/librenamed.so": payload},
        mode="w:gz",
    )
    findings = scanner.scan(rootfs, {})
    assert {item.kind for item in findings} == {
        "credential_content",
        "cuda_elf_dependency",
    }


@pytest.mark.parametrize(
    "dependency",
    [b"libtorch_cuda.so", b"libc10_cuda.so", b"libnvToolsExt.so.1"],
)
def test_renamed_elf_with_embedded_cuda_dependency_fails(
    tmp_path: Path, dependency: bytes
) -> None:
    rootfs = _tar(
        tmp_path / "elf-dependency.tar",
        {"opt/application/librenamed.so": b"\x7fELF\x00" + dependency + b"\x00"},
    )
    findings = scanner.scan(rootfs, {})
    assert "cuda_elf_dependency" in {item.kind for item in findings}


def test_cli_emits_machine_readable_failure(tmp_path: Path, capsys) -> None:
    rootfs = _tar(tmp_path / "bad.tar", {"models/weights.ckpt": b"x"})
    assert scanner.main(["--rootfs-tar", str(rootfs)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
