#!/usr/bin/env python3
"""Fail closed when a built Wan image contains forbidden shipped bytes/history."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import lzma
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

try:
    import zstandard as zstd
except ModuleNotFoundError:  # The zstd signature still fails closed at scan time.
    zstd = None  # type: ignore[assignment]

ZSTD_ERROR: type[Exception] = zstd.ZstdError if zstd is not None else RuntimeError


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bundled_ffmpeg_executable",
        re.compile(r"(?:^|/)site-packages/imageio_ffmpeg/binaries/ffmpeg[^/]*$", re.I),
    ),
    (
        "historical_lgpl_python_distribution",
        re.compile(
            r"(?:^|/)site-packages/(?:easydict|soxr|librosa)-[^/]*\.dist-info/",
            re.I,
        ),
    ),
    (
        "nvidia_python_distribution",
        re.compile(
            r"(?:^|/)site-packages/(?:nvidia(?:/|_)|nvidia_[^/]*\.dist-info/)", re.I
        ),
    ),
    (
        "cuda_library",
        re.compile(
            r"(?:^|/)(?:lib)?(?:(?:[a-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|"
            r"nccl|nvrtc|nvjitlink|nvtx|nvtoolsext|nvfatbin|nvptxcompiler|cupti|"
            r"cufile|cusparse|cusolver|curand|cufft|npp[a-z]*)"
            r"(?:[a-z0-9_-]*)|nvidia(?:-[a-z0-9_-]+)?|(?:glx|egl)_nvidia|"
            r"nv(?:cuvid|optix|encode|decode))(?:[^/]*)"
            r"\.(?:so(?:\.|$)|a$)",
            re.I,
        ),
    ),
    (
        "cuda_tool",
        re.compile(
            r"(?:^|/)(?:(?:usr/local|opt)/cuda(?:-[0-9.]+)?(?:/|$)|"
            r"usr/include/(?:cuda|cublas|cudnn|nccl|nvrtc|nvToolsExt|npp|cupti|"
            r"cufft|cusparse|cusolver|curand)[^/]*\.h$|(?:nvcc|ptxas|cuobjdump|"
            r"compute-sanitizer|nvidia-smi)(?:$|\.))",
            re.I,
        ),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|bin\.index\.json)$|"
            r"(?:^|/)(?:models?|weights?|checkpoints?)(?:/|[^/]*)[^/]*\.(?:bin|pt|pth)$|"
            r"(?:^|/)(?:pytorch_model|diffusion_pytorch_model|model|weights?|checkpoint)"
            r"[^/]*\.(?:bin|pt|pth)$)",
            re.I,
        ),
    ),
    (
        "package_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:pip|huggingface)|pip-cache|wheelhouse)(?:/|$)",
            re.I,
        ),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|kubeconfig|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cuda_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
    (
        "cuda_install_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:download\.pytorch\.org/whl/cu|"
            r"pip\s+install(?:(?!&&|\|\||;)[^\n])*(?:nvidia-|torch==[^\s;&|]*\+cu))",
            re.I,
        ),
    ),
    (
        "runtime_bootstrap_at_build",
        re.compile(r"\bRUN\b.*\bwan-runtime\s+(?:ensure|warm|exec)\b", re.I | re.S),
    ),
)

SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)(?:aws_secret_access_key|hf_token)\s*[=:]\s*[^$<\s][^\s]{7,}"),
)

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"lib(?:[A-Za-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|"
    rb"nvtx|nvtoolsext|nvfatbin|nvptxcompiler|cupti|cufile|cusparse|cusolver|"
    rb"curand|cufft|npp[a-z]*|nvidia|nvcuvid|nvoptix|nvencode|nvdecode)"
    rb"[A-Za-z0-9_.-]*\.so",
    re.I,
)

MAX_NESTED_ARCHIVE_DEPTH = 8
MAX_NESTED_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_NESTED_ARCHIVE_MEMBERS = 100_000
MAX_AR_NAME_TABLE_BYTES = 8 * 1024**2


@dataclass
class _NestedArchiveBudget:
    """Cumulative resource budget shared by one image archive's nested tree."""

    byte_limit: int
    member_limit: int
    bytes_consumed: int = 0
    members_consumed: int = 0

    def consume_bytes(self, size: int) -> None:
        if size < 0:
            raise ValueError("nested archive reports a negative member size")
        self.bytes_consumed += size
        if self.bytes_consumed > self.byte_limit:
            raise ValueError(
                "nested archive tree exceeds cumulative uncompressed byte limit"
            )

    def consume_member(self, size: int) -> None:
        self.members_consumed += 1
        if self.members_consumed > self.member_limit:
            raise ValueError("nested archive tree exceeds cumulative member limit")
        self.consume_bytes(size)


# Exact OSS files whose examples, parser literals, or command-line help include
# secret-shaped text.  This is deliberately a path+byte allowlist rather than a
# path exclusion: inserting a real credential into any of these locations must
# fail the publication scan.  A dependency/base refresh fails closed until the
# replacement bytes have been independently audited.
AUDITED_SECRET_LITERAL_FILE_SHA256: dict[str, str] = {
    "opt/wan-base/lib/python3.10/site-packages/PIL/ImageFont.py": (
        "24fa5feeb91b4bf63eaad0ebba08a8161e9c889d9fd056a37c928134097b9649"
    ),
    "opt/wan-base/lib/python3.10/site-packages/accelerate/commands/config/sagemaker.py": (
        "4912eea7d5eb57f67edb703777c8196e4a9ba270dcb1c3030e66e99b2b42cdb6"
    ),
    "opt/wan-base/lib/python3.10/site-packages/boto3/examples/cloudfront.rst": (
        "2beb01599c682e30010991eba8066ce7b71879fc0f9838abd37a66e8f13f9a1e"
    ),
    "opt/wan-base/lib/python3.10/site-packages/boto3/session.py": (
        "98493dcacb64905e94418b46250f4427938bf0f35badb02b5436049c5d5b5fcc"
    ),
    "opt/wan-base/lib/python3.10/site-packages/boto3-1.39.11.dist-info/METADATA": (
        "5524ef6a8ea9110d3b1c2b0b4d034393a5408e6c9b80c9168379ce37fd4ea374"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore/credentials.py": (
        "9b14c632b0a025c02655170cc6b42654e8cbb258e78e3c84f17025129fd7dee3"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore/data/iam/2010-05-08/examples-1.json": (
        "4f912aac51590625652fd76c37e4f90e78a05355273125df555c012b4d005ab5"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore/data/sts/2011-06-15/examples-1.json": (
        "c83fc27073767fdb7d3e5190e4dcce25a09871c7b118fa289db056d93e0e31c9"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore/data/rds/2014-10-31/service-2.json.gz!/<decompressed>": (
        "c5c1c925c3e977c10a0d77464e243eafc77a7a04364912a45965912421bad817"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore/data/sts/2011-06-15/service-2.json.gz!/<decompressed>": (
        "5bec182c6a1f8a96993c78b95ba10087b2eaf5c617484ba755f38b03e2104901"
    ),
    "opt/wan-base/lib/python3.10/site-packages/botocore-1.39.17.dist-info/METADATA": (
        "d1ce694b50d009bb3dab11331a281323d70c46ff4724e692dab3c2481ed62074"
    ),
    "opt/wan-base/lib/python3.10/site-packages/cryptography/hazmat/primitives/serialization/ssh.py": (
        "162b177bf9d429d3c67ea10d5612a99a86b399a23ca87f067be5466dcd1dca4c"
    ),
    "opt/wan-base/lib/python3.10/site-packages/diffusers/loaders/textual_inversion.py": (
        "9cc89fb4b0ac9762e4434723f8447d8833d26b03844804ee11493ce4b5f7512a"
    ),
    "usr/bin/ssh": "04f2ff5f506a3f332e7adeb1478a4c551ae74acdd328e6fb5c2495664d4064e6",
    "usr/bin/ssh-add": (
        "1e02d3fca3c8d72570c11ded9681dcacd4775a33560786a27c489b9e4388ec06"
    ),
    "usr/bin/ssh-agent": (
        "165e70f42cf6a147ae2101fae05a4503791d95ac9de7cca7974577316d963829"
    ),
    "usr/bin/ssh-keygen": (
        "2843cd46c617cf771c32e6f8a2a11585d83510ec528e3aeff8f7a7f0445420ba"
    ),
    "usr/bin/ssh-keyscan": (
        "9475d0851a26a4f494dc7d40240b68b04874543cbd30f54195ff9af055aaf848"
    ),
    "usr/lib/openssh/ssh-keysign": (
        "c4f5b14604acf3cd0111ca969a0975194e4219f818565df12a7f3ad4663ca93a"
    ),
    "usr/lib/openssh/ssh-pkcs11-helper": (
        "9951c90e5173921f2c3ae553d6b2ce5967862944c12eb63d635586130728bd07"
    ),
    "usr/lib/openssh/ssh-sk-helper": (
        "9caaf480bbd3bfb916195c8ac868e3a1e78968408500f090f037b4a186aac1df"
    ),
    "usr/lib/x86_64-linux-gnu/libmbedcrypto.so.2.28.3": (
        "c04f91fdb172e17ddb21c9e0b75c04cb4f802bdfc40bb65484550746cd0019a8"
    ),
    "usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.9.6": (
        "6733636aeb1c5d541aa06c578a95d12c2253c4e99fc85623595341905d3d6221"
    ),
    "usr/lib/x86_64-linux-gnu/libssh2.so.1.0.1": (
        "66f751ec9d3d5bff254a498e020d37f9bbde8e0193ba11d1b78f29153ffe694a"
    ),
    "usr/sbin/sshd": "9f6cdc787a2d5144f3189e850fc104aa7d8ab12593a3d4e902c692a38794716e",
    "usr/lib/x86_64-linux-gnu/libunistring.so.2.2.0": (
        "bc5951aa3d6eaba20ff9688efa3420dc95785aae3709ec48ff6df46d6f409ee5"
    ),
}

# Exact Debian base-library bytes whose parser/dlopen literals were independently
# audited. A base-image update fails closed until its new bytes are re-audited;
# path-only exclusions would let renamed CUDA libraries or credentials pass.
AUDITED_LITERAL_LIBRARY_SHA256: dict[str, str] = {
    "usr/lib/x86_64-linux-gnu/libavcodec.so.59.37.100": (
        "4af5d9cffe2721f5c2cabf35d63c5c6a039b400df5721aaedf69995e37bd2a0d"
    ),
    "usr/lib/x86_64-linux-gnu/libavutil.so.57.28.100": (
        "c46ee8987cacb9f9af711f676cc98bb6d465a340566dad9c678d21baa26e2c9d"
    ),
    "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3": (
        "779b25d20249988bea2c1aa6bbeb218f5ae7ea8a9d30ce4f54ea37372965cc4b"
    ),
}


def _is_audited_literal_library(path: str, sha256: str | None) -> bool:
    """Whether path and bytes equal one independently audited Debian library."""

    return sha256 is not None and AUDITED_LITERAL_LIBRARY_SHA256.get(path) == sha256


def _contains_secret(stream: IO[bytes]) -> bool:
    """Stream arbitrary-size files while retaining boundary-spanning matches."""

    _, secret, _ = _inspect_content(stream, scan_secret=True)
    return secret


def _contains_forbidden_elf_dependency(stream: IO[bytes]) -> bool:
    """Detect a renamed ELF that still dynamically depends on NVIDIA/CUDA."""

    elf_dependency, _, _ = _inspect_content(stream, scan_secret=False)
    return elf_dependency


def _inspect_content(
    stream: IO[bytes],
    *,
    scan_secret: bool,
    calculate_sha256: bool = False,
    copy_to: IO[bytes] | None = None,
) -> tuple[bool, bool, str | None]:
    """Inspect one member once, avoiding rewinds of compressed image layers."""

    head = stream.read(4)
    is_elf = head == b"\x7fELF"
    elf_dependency = False
    secret = False
    elf_overlap = b""
    secret_overlap = b""
    digest = hashlib.sha256() if calculate_sha256 else None
    chunk = head
    while chunk:
        if copy_to is not None:
            copy_to.write(chunk)
        if digest is not None:
            digest.update(chunk)
        if is_elf and not elf_dependency:
            elf_payload = elf_overlap + chunk
            elf_dependency = bool(FORBIDDEN_ELF_DEPENDENCY.search(elf_payload))
            elf_overlap = elf_payload[-256:]
        if scan_secret and not secret:
            secret_payload = secret_overlap + chunk
            secret = any(pattern.search(secret_payload) for pattern in SECRET_CONTENT)
            secret_overlap = secret_payload[-1024:]
        if copy_to is None and (
            digest is None
            and (not is_elf or elf_dependency)
            and (not scan_secret or secret)
        ):
            break
        chunk = stream.read(1024 * 1024)
    return elf_dependency, secret, digest.hexdigest() if digest is not None else None


def _add_forbidden_path_findings(
    path: str, source: str, findings: list[Finding]
) -> None:
    for kind, pattern in FORBIDDEN_PATHS:
        if pattern.search(path):
            findings.append(Finding(kind, path, f"forbidden bytes in {source}"))


def _nested_archive_kind(stream: IO[bytes]) -> str | None:
    stream.seek(0)
    head = stream.read(512)
    stream.seek(0)
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"BZh"):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if head.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    # Debian packages are Unix ar archives whose data.tar member may contain
    # runtime libraries, weights, caches, or credentials.  Treat the outer ar
    # as a first-class archive instead of only content-scanning its opaque
    # bytes.
    if head.startswith(b"!<arch>\n"):
        return "ar"
    # POSIX ustar is only one valid tar dialect. V7/legacy tars have no magic,
    # so ask the standard-library reader to validate the header/checksum rather
    # than leaving those archives opaque to the payload scanner.
    if len(head) == 512:
        try:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                next(iter(archive), None)
            return "tar"
        except (EOFError, OSError, tarfile.TarError, ValueError):
            pass
        finally:
            stream.seek(0)
    return None


def _decompress_nested_stream(
    *,
    kind: str,
    stream: IO[bytes],
    parent_path: str,
    source: str,
    findings: list[Finding],
    depth: int,
    budget: _NestedArchiveBudget,
) -> None:
    """Inspect a gzip/bzip2/xz member without assuming it contains a tar.

    Python's ``tarfile`` can transparently decompress a tar archive, but it
    rejects ordinary compressed data such as botocore's ``service-2.json.gz``
    or NetworkX's ``*.gpickle.bz2``.  Those are still byte containers and must
    be inspected.  Decompress once to a disk-backed temporary file, then route
    an actual tar back through the archive walker and all other streams through
    the ordinary content scanner.
    """

    stream.seek(0)
    openers = {
        "gzip": gzip.GzipFile,
        "bzip2": bz2.BZ2File,
        "xz": lzma.LZMAFile,
    }
    compressed: Any
    if kind == "zstd":
        if zstd is None:
            raise ValueError("zstd archive support is unavailable")
        compressed = zstd.ZstdDecompressor().stream_reader(stream)
    else:
        opener = openers[kind]
        compressed = (
            gzip.GzipFile(fileobj=stream, mode="rb")
            if kind == "gzip"
            else opener(stream, mode="rb")
        )
    with compressed, tempfile.TemporaryFile() as expanded:
        while True:
            chunk = compressed.read(1024 * 1024)
            if not chunk:
                break
            budget.consume_bytes(len(chunk))
            expanded.write(chunk)
        expanded.seek(0)
        expanded_kind = _nested_archive_kind(expanded)
        expanded.seek(0)
        if expanded_kind in {
            "tar",
            "zip",
            "gzip",
            "bzip2",
            "xz",
            "zstd",
            "ar",
        }:
            _scan_nested_archive(
                stream=expanded,
                parent_path=parent_path,
                source=source,
                findings=findings,
                depth=depth + 1,
                budget=budget,
            )
        else:
            _scan_file_stream(
                path=f"{parent_path}!/<decompressed>",
                stream=expanded,
                source=source,
                findings=findings,
                depth=depth + 1,
                budget=budget,
            )


def _read_ar_member(stream: IO[bytes], size: int, output: IO[bytes]) -> None:
    """Copy exactly one ar member to disk and consume its alignment byte."""

    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("truncated ar member payload")
        output.write(chunk)
        remaining -= len(chunk)
    if size % 2 and stream.read(1) != b"\n":
        raise ValueError("invalid ar member alignment byte")
    output.seek(0)


def _gnu_ar_name(name_table: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(name_table):
        raise ValueError("ar GNU filename offset is outside the name table")
    tail = name_table[offset:]
    end = len(tail)
    for terminator in (b"/\n", b"\n", b"\x00"):
        position = tail.find(terminator)
        if position >= 0:
            end = min(end, position)
    name = tail[:end].decode("utf-8")
    if not name:
        raise ValueError("ar GNU filename is empty")
    return name


def _scan_ar_archive(
    *,
    stream: IO[bytes],
    parent_path: str,
    source: str,
    findings: list[Finding],
    depth: int,
    budget: _NestedArchiveBudget,
) -> None:
    """Walk a System V/GNU/BSD ar archive, including Debian packages."""

    stream.seek(0)
    if stream.read(8) != b"!<arch>\n":
        raise ValueError("invalid ar global header")
    gnu_name_table = b""
    while True:
        header = stream.read(60)
        if not header:
            break
        if len(header) != 60 or header[58:60] != b"`\n":
            raise ValueError("invalid or truncated ar member header")
        raw_size = header[48:58].decode("ascii").strip()
        if not raw_size.isdecimal():
            raise ValueError("invalid ar member size")
        stored_size = int(raw_size)
        budget.consume_member(stored_size)
        raw_name = header[:16].decode("utf-8").rstrip()

        with tempfile.TemporaryFile() as member:
            _read_ar_member(stream, stored_size, member)
            payload_offset = 0
            if raw_name == "//":
                if stored_size > MAX_AR_NAME_TABLE_BYTES:
                    raise ValueError("ar GNU filename table exceeds the size limit")
                gnu_name_table = member.read()
                member.seek(0)
                member_name = "<gnu-filename-table>"
            elif raw_name.startswith("#1/"):
                encoded_length = raw_name[3:]
                if not encoded_length.isdecimal():
                    raise ValueError("invalid BSD ar extended filename length")
                payload_offset = int(encoded_length)
                if payload_offset <= 0 or payload_offset > stored_size:
                    raise ValueError("BSD ar extended filename exceeds member size")
                member_name = (
                    member.read(payload_offset).rstrip(b"\x00").decode("utf-8")
                )
                if not member_name:
                    raise ValueError("BSD ar extended filename is empty")
            elif raw_name.startswith("/") and raw_name[1:].isdecimal():
                if not gnu_name_table:
                    raise ValueError(
                        "ar GNU filename reference precedes its name table"
                    )
                member_name = _gnu_ar_name(gnu_name_table, int(raw_name[1:]))
            elif raw_name in {"/", "/SYM64/", "__.SYMDEF", "__.SYMDEF SORTED"}:
                member_name = "<symbol-table>"
            else:
                member_name = raw_name.removesuffix("/")
                if not member_name:
                    raise ValueError("ar member filename is empty")

            nested_path = f"{parent_path}!/{member_name.lstrip('/')}"
            member.seek(payload_offset)
            _scan_file_stream(
                path=nested_path,
                stream=member,
                source=source,
                findings=findings,
                depth=depth + 1,
                budget=budget,
            )


def _scan_file_stream(
    *,
    path: str,
    stream: IO[bytes],
    source: str,
    findings: list[Finding],
    depth: int,
    budget: _NestedArchiveBudget,
) -> None:
    _add_forbidden_path_findings(path, source, findings)
    audited_secret_sha256 = AUDITED_SECRET_LITERAL_FILE_SHA256.get(path)
    audited_library_sha256 = AUDITED_LITERAL_LIBRARY_SHA256.get(path)
    # zipfile requires a fully seekable object; Python 3.10's
    # SpooledTemporaryFile wrapper does not expose seekable(). Keep the captured
    # bytes on disk so arbitrarily large image members do not consume RAM.
    with tempfile.TemporaryFile() as captured:
        elf_dependency, secret, sha256 = _inspect_content(
            stream,
            scan_secret=True,
            calculate_sha256=(
                audited_secret_sha256 is not None or audited_library_sha256 is not None
            ),
            copy_to=captured,
        )
        audited_literal_library = _is_audited_literal_library(path, sha256)
        audited_secret_literal = (
            audited_secret_sha256 is not None and sha256 == audited_secret_sha256
        )
        expected_audited_sha256 = audited_secret_sha256 or audited_library_sha256
        if expected_audited_sha256 is not None and sha256 != expected_audited_sha256:
            findings.append(
                Finding(
                    "audited_literal_byte_drift",
                    path,
                    f"audited literal bytes changed in {source}",
                )
            )
        if elf_dependency and not audited_literal_library:
            findings.append(
                Finding(
                    "cuda_elf_dependency",
                    path,
                    f"ELF links to forbidden runtime in {source}",
                )
            )
        if secret and not (audited_literal_library or audited_secret_literal):
            findings.append(
                Finding(
                    "credential_content",
                    path,
                    f"secret-like bytes in {source}",
                )
            )
        captured.seek(0)
        _scan_nested_archive(
            stream=captured,
            parent_path=path,
            source=source,
            findings=findings,
            depth=depth,
            budget=budget,
        )


def _scan_nested_archive(
    *,
    stream: IO[bytes],
    parent_path: str,
    source: str,
    findings: list[Finding],
    depth: int,
    budget: _NestedArchiveBudget,
) -> None:
    kind = _nested_archive_kind(stream)
    if kind is None:
        return
    if depth >= MAX_NESTED_ARCHIVE_DEPTH:
        findings.append(
            Finding(
                "nested_archive_depth",
                parent_path,
                f"nested archive exceeds depth {MAX_NESTED_ARCHIVE_DEPTH} in {source}",
            )
        )
        return
    try:
        if kind in {"gzip", "bzip2", "xz", "zstd"}:
            _decompress_nested_stream(
                kind=kind,
                stream=stream,
                parent_path=parent_path,
                source=source,
                findings=findings,
                depth=depth,
                budget=budget,
            )
        elif kind == "zip":
            with zipfile.ZipFile(stream) as archive:
                for zip_member in archive.infolist():
                    nested_path = f"{parent_path}!/{zip_member.filename.lstrip('/')}"
                    if zip_member.is_dir():
                        _add_forbidden_path_findings(nested_path, source, findings)
                        continue
                    budget.consume_member(zip_member.file_size)
                    with archive.open(zip_member) as nested_zip_stream:
                        _scan_file_stream(
                            path=nested_path,
                            stream=nested_zip_stream,
                            source=source,
                            findings=findings,
                            depth=depth + 1,
                            budget=budget,
                        )
        elif kind == "ar":
            _scan_ar_archive(
                stream=stream,
                parent_path=parent_path,
                source=source,
                findings=findings,
                depth=depth,
                budget=budget,
            )
        else:
            with tarfile.open(fileobj=stream, mode="r:*") as archive:
                for tar_member in archive:
                    nested_path = f"{parent_path}!/{tar_member.name.lstrip('/')}"
                    if not tar_member.isfile():
                        _add_forbidden_path_findings(nested_path, source, findings)
                        continue
                    budget.consume_member(tar_member.size)
                    nested_tar_stream = archive.extractfile(tar_member)
                    if nested_tar_stream is not None:
                        with nested_tar_stream:
                            _scan_file_stream(
                                path=nested_path,
                                stream=nested_tar_stream,
                                source=source,
                                findings=findings,
                                depth=depth + 1,
                                budget=budget,
                            )
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        ZSTD_ERROR,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        findings.append(
            Finding(
                "nested_archive_unreadable",
                parent_path,
                f"cannot inspect nested archive in {source}: {exc}",
            )
        )


@contextmanager
def payload_policy(
    *,
    forbidden_paths: tuple[tuple[str, re.Pattern[str]], ...] | None = None,
    forbidden_history: tuple[tuple[str, re.Pattern[str]], ...] | None = None,
    audited_secret_files: dict[str, str] | None = None,
    audited_libraries: dict[str, str] | None = None,
    secret_content: tuple[re.Pattern[bytes], ...] | None = None,
    forbidden_elf_dependency: re.Pattern[bytes] | None = None,
) -> Iterator[None]:
    """Scan under a different payload policy, reusing this archive walker.

    The nested-archive traversal here (tar/zip/ar/gzip/bzip2/xz/zstd, ELF
    dependency inspection, budgets) is generic; only the pattern tables are
    Wan-specific. Sibling scanners for other images declare their own tables and
    borrow the walker rather than forking 800 lines of it, which is how the
    traversal fixes and hardening stay in one place.

    The tables are module globals because the walker reads them at every level of
    recursion, so this swaps and restores them. It is therefore not re-entrant or
    thread-safe — fine for a CLI scanner, and the restore is in a ``finally`` so a
    failed scan cannot leave the Wan policy replaced.
    """

    global FORBIDDEN_PATHS, FORBIDDEN_HISTORY
    global AUDITED_SECRET_LITERAL_FILE_SHA256, AUDITED_LITERAL_LIBRARY_SHA256
    global SECRET_CONTENT, FORBIDDEN_ELF_DEPENDENCY
    previous = (
        FORBIDDEN_PATHS,
        FORBIDDEN_HISTORY,
        AUDITED_SECRET_LITERAL_FILE_SHA256,
        AUDITED_LITERAL_LIBRARY_SHA256,
        SECRET_CONTENT,
        FORBIDDEN_ELF_DEPENDENCY,
    )
    if forbidden_paths is not None:
        FORBIDDEN_PATHS = forbidden_paths
    if forbidden_history is not None:
        FORBIDDEN_HISTORY = forbidden_history
    if audited_secret_files is not None:
        AUDITED_SECRET_LITERAL_FILE_SHA256 = audited_secret_files
    if audited_libraries is not None:
        AUDITED_LITERAL_LIBRARY_SHA256 = audited_libraries
    if secret_content is not None:
        SECRET_CONTENT = secret_content
    if forbidden_elf_dependency is not None:
        FORBIDDEN_ELF_DEPENDENCY = forbidden_elf_dependency
    try:
        yield
    finally:
        (
            FORBIDDEN_PATHS,
            FORBIDDEN_HISTORY,
            AUDITED_SECRET_LITERAL_FILE_SHA256,
            AUDITED_LITERAL_LIBRARY_SHA256,
            SECRET_CONTENT,
            FORBIDDEN_ELF_DEPENDENCY,
        ) = previous


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[Finding]:
    """Scan several layer/rootfs tars plus image history under the active policy."""

    return _scan_tars(tars, config)


def remote_material(image: str, temp_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Export a built image's rootfs, layers, and OCI config with crane."""

    return _remote_material(image, temp_dir)


def docker_save_material(
    archive_path: Path, temp_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Read every layer and the OCI config from a local ``docker save`` archive.

    A merged rootfs is insufficient before public push because a later whiteout
    can hide a credential, CUDA library, or model payload from the final tree.
    """

    with tarfile.open(archive_path, "r:*") as archive:
        manifest_stream = archive.extractfile(archive.getmember("manifest.json"))
        if manifest_stream is None:
            raise RuntimeError("docker save archive has no readable manifest.json")
        manifests = json.load(io.TextIOWrapper(manifest_stream, encoding="utf-8"))
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError("docker save archive must contain exactly one image")
        manifest = manifests[0]
        config_name = str(manifest.get("Config") or "")
        layer_names = manifest.get("Layers") or []
        if not config_name or not layer_names:
            raise RuntimeError("docker save manifest has no config or layers")

        config_stream = archive.extractfile(archive.getmember(config_name))
        if config_stream is None:
            raise RuntimeError("docker save archive has no readable image config")
        config = json.load(io.TextIOWrapper(config_stream, encoding="utf-8"))

        layers: list[Path] = []
        for index, layer_name in enumerate(layer_names):
            layer_stream = archive.extractfile(archive.getmember(str(layer_name)))
            if layer_stream is None:
                raise RuntimeError(f"docker save layer {index} is not readable")
            layer_path = temp_dir / f"layer-{index:03d}.tar"
            with layer_path.open("wb") as output:
                shutil.copyfileobj(layer_stream, output)
            layers.append(layer_path)
    return layers, config


def _config_text(config: dict[str, Any]) -> str:
    """Serialize every shipped OCI config/history field for policy scanning."""

    return json.dumps(config, sort_keys=True, default=str)


def _normalize_archive_path(path: str) -> str:
    """Remove archive-root syntax without stripping a real leading dotfile."""

    normalized = path
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[Finding]:
    """Scan one tar plus image history (used directly by mutation tests)."""

    return _scan_tars([rootfs_tar], config)


def _scan_tars(tars: list[Path], config: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for tar_path in tars:
        budget = _NestedArchiveBudget(
            byte_limit=MAX_NESTED_UNCOMPRESSED_BYTES,
            member_limit=MAX_NESTED_ARCHIVE_MEMBERS,
        )
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                path = _normalize_archive_path(member.name)
                if not path:
                    continue
                if member.isfile():
                    fileobj = archive.extractfile(member)
                    if fileobj is not None:
                        with fileobj:
                            _scan_file_stream(
                                path=path,
                                stream=fileobj,
                                source=tar_path.name,
                                findings=findings,
                                depth=0,
                                budget=budget,
                            )
                else:
                    _add_forbidden_path_findings(path, tar_path.name, findings)

    history = _config_text(config)
    for kind, pattern in FORBIDDEN_HISTORY:
        if pattern.search(history):
            findings.append(
                Finding(
                    kind, "<image-history>", "forbidden build history or environment"
                )
            )
    metadata = history.encode("utf-8")
    if any(pattern.search(metadata) for pattern in SECRET_CONTENT):
        findings.append(
            Finding(
                "credential_metadata",
                "<oci-config>",
                "secret-like bytes in image configuration or history",
            )
        )
    return findings


def _repository(image: str) -> str:
    ref = image.removeprefix("docker:").split("@", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    return ref[:colon] if colon > slash else ref


def _remote_material(image: str, temp_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    if not shutil.which("crane"):
        raise RuntimeError("crane is required to scan a built image")
    rootfs = temp_dir / "rootfs.tar"
    with rootfs.open("wb") as stream:
        proc = subprocess.run(
            ["crane", "export", "--platform", "linux/amd64", image, "-"],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode:
        raise RuntimeError(
            f"crane export failed: {proc.stderr.decode(errors='replace')}"
        )
    config_proc = subprocess.run(
        ["crane", "config", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if config_proc.returncode:
        raise RuntimeError(f"crane config failed: {config_proc.stderr}")
    manifest_proc = subprocess.run(
        ["crane", "manifest", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if manifest_proc.returncode:
        raise RuntimeError(f"crane manifest failed: {manifest_proc.stderr}")
    manifest = json.loads(manifest_proc.stdout)
    tars = [rootfs]
    repository = _repository(image)
    for index, layer in enumerate(manifest.get("layers") or []):
        digest = str(layer.get("digest") or "")
        if not digest:
            raise RuntimeError("image manifest layer has no digest")
        layer_path = temp_dir / f"layer-{index:03d}.tar.gz"
        with layer_path.open("wb") as stream:
            layer_proc = subprocess.run(
                ["crane", "blob", f"{repository}@{digest}"],
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
            )
        if layer_proc.returncode:
            raise RuntimeError(
                f"crane blob failed for {digest}: {layer_proc.stderr.decode(errors='replace')}"
            )
        tars.append(layer_path)
    return tars, json.loads(config_proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.image, args.rootfs_tar, args.docker_save)) != 1:
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-wan-byte-scan-") as tmp:
            if args.image:
                tars, config = _remote_material(args.image, Path(tmp))
            elif args.docker_save:
                tars, config = docker_save_material(args.docker_save, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = _scan_tars(tars, config)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_wan_image_byte_scan_v1",
        "image": args.image or ("docker-save" if args.docker_save else "offline-rootfs"),
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
