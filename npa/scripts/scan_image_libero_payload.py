#!/usr/bin/env python3
"""Fail closed unless an image is the zero-payload LIBERO neutral bootstrap."""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import io
import json
import lzma
import mmap
import re
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402

BASE_MANIFEST = (
    "sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496"
)
BASE_ROOTFS_MATERIAL = (
    "sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867"
)
BOOTSTRAP_CONTRACT = "skypilot-0.12.2-v1"
BASE_PROVENANCE_SHA256 = (
    "0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b"
)
BASE_SOURCE_REVISION = "688a0b86bb44289df16a363e9f41d90514c1a5f9"
RUNTIME_MANIFEST_SHA256 = (
    "d999dd97e8f9b324b68ec2a7f19b6360f5599868cd873cd752779106b8ea4f02"
)
RUNTIME_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "workbench"
    / "libero"
    / "runtime-manifest.json"
)
MAX_UNCOMPRESSED_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024

FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "libero_source_outside_neutral_bootstrap",
        re.compile(
            r"(?:^|/)(?:opt/byof/.+|opt/(?:libero|robomimic|robosuite)|"
            r"usr/src/(?:libero|robomimic|robosuite)|"
            r"workspace/(?:libero|robomimic|robosuite))(?:/|$)",
            re.I,
        ),
    ),
    (
        "torch_runtime_outside_python_distribution",
        re.compile(
            r"(?:^|/)(?:opt|usr/local|workspace)/(?:pytorch|torch)(?:/|$)", re.I
        ),
    ),
    (
        "libero_or_simulation_distribution",
        re.compile(
            r"(?:^|/)(?:site|dist)-packages/(?:libero|robomimic|robosuite|mujoco)(?:/|-)",
            re.I,
        ),
    ),
    (
        "torch_distribution",
        re.compile(
            r"(?:^|/)(?:site|dist)-packages/(?:torch(?:/|-)|torchvision(?:/|-)|triton(?:/|-))",
            re.I,
        ),
    ),
    (
        "nvidia_python_distribution",
        re.compile(
            r"(?:^|/)(?:site|dist)-packages/(?:nvidia(?:/|_)|nvidia_[^/]*\.dist-info/)",
            re.I,
        ),
    ),
    (
        "cuda_or_nvidia_payload",
        re.compile(
            r"(?:^|/)(?:(?:usr/local|opt)/cuda(?:-[0-9.]+)?(?:/|$)|"
            r"usr/include/(?:cuda|cublas|cudnn|nccl|nvrtc|cupti)[^/]*\.h$|"
            r"(?:nvcc|ptxas|cuobjdump|compute-sanitizer|nvidia-smi)(?:$|\.)|"
            # ICU's Unicode data library contains "cuda" across its name.
            # Exclude only that library basename; content/ELF scans still apply.
            r"(?!libicudata\.so(?:\.[0-9]+)*$)"
            r"[^/]*(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|cupti|cufile|"
            r"cusparse|cusolver|curand|cufft|nvidia)[^/]*\.so(?:\.[0-9]+)*)",
            re.I,
        ),
    ),
    (
        "model_weight_checkpoint_or_dataset",
        re.compile(
            r"(?:^(?!usr/local/lib/python[0-9.]+/(?:site-packages/|"
            r"ensurepip/_bundled/setuptools-[^/]+\.whl!/)"
            r"distutils-precedence\.pth$)(?i:.*\.(?:safetensors|ckpt|pth|hdf5|h5|onnx))$|"
            r"(?i:(?:^|/)(?:pytorch_model|model|weights?|checkpoint)[^/]*"
            r"\.(?:bin|pth|pt)$))"
        ),
    ),
    (
        "libero_task_or_render_asset",
        re.compile(
            r"(?i:libero/libero/(?:assets|bddl_files|init_files)(?:/|$))|"
            r"^(?!etc/security/namespace\.init$)(?i:.*\.(?:bddl|init))$"
        ),
    ),
    (
        "populated_runtime_cache",
        re.compile(r"(?:^|/)workspace/\.cache/npa/libero/.+", re.I),
    ),
    (
        "workflow_output",
        re.compile(r"(?:^|/)workspace/byof-runs/.+", re.I),
    ),
    (
        "package_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:pip|uv|huggingface)|pip-cache|wheelhouse)(?:/|$)",
            re.I,
        ),
    ),
    (
        "credential_or_private_configuration",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|"
            r"kubeconfig|runtime-context\.json|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key(?:\.pub)?)$",
            re.I,
        ),
    ),
    (
        "git_objects",
        re.compile(r"(?:^|/)\.git/(?:objects|config|credentials)(?:/|$)", re.I),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gpu_runtime_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
    (
        "runtime_payload_install_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:pip\s+install[^\n]*(?:torch|nvidia-|mujoco|robomimic|robosuite|libero)|"
            r"runtime-bootstrap\.py\s+ensure)",
            re.I | re.S,
        ),
    ),
    (
        "runtime_payload_fetch_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:Lifelong-Robot-Learning/LIBERO|LIBERO-datasets|bert-base-cased)",
            re.I | re.S,
        ),
    ),
    (
        "baked_acceptance_proxy",
        re.compile(r"\b(?:ENV|ARG)\s+[^\n]*\bACCEPT_[A-Z0-9_]*\s*=\s*\S", re.I),
    ),
)

SECRET_CONTENT: tuple[re.Pattern[bytes], ...] = (
    re.compile(
        rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}"
        rb"-----END (?:RSA |OPENSSH )?PRIVATE KEY-----"
    ),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)(?:aws_secret_access_key|hf_token)\s*[=:]\s*[^$<\s][^\s]{7,}"),
)

# These signatures detect renamed pure-Python payloads independently of their
# archive path. The exact NPA smoke driver legitimately imports these projects,
# so its path is exempt only when its bytes retain the reviewed digest.
FORBIDDEN_PAYLOAD_CONTENT: tuple[re.Pattern[bytes], ...] = (
    re.compile(
        rb"(?m)^(?:from|import)\s+libero(?:\.|\s)|^class\s+(?:BCRNNPolicy|SequenceVLDataset)\b"
    ),
    re.compile(rb"(?m)^(?:from|import)\s+robomimic(?:\.|\s)|^class\s+RolloutPolicy\b"),
    re.compile(
        rb"(?m)^(?:from|import)\s+(?:torch|torchvision|mujoco|robosuite)(?:\.|\s)|"
        rb"^class\s+Tensor\b|^__all__\s*=.*['\"]Tensor['\"]"
    ),
)
NEUTRAL_PAYLOAD_CONTENT_ALLOWLIST = {
    "opt/npa/libero/libero_smoke.py": (
        "4e81a5f2ec9892c9a441a7482da53694660575dcbe2496935d9ff9937735790b"
    )
}
NEVER_MATCH_ELF = re.compile(rb"(?!)")
RUNTIME_INJECTED_EXPORT_PATHS = frozenset(
    {
        ".dockerenv",
        "dev/console",
        "etc/hostname",
        "etc/hosts",
        "etc/mtab",
        "etc/resolv.conf",
    }
)
IMAGE_INVENTORY_SCHEMA = "npa.libero.neutral-image-inventory.v1"
REQUIRED_BUILD_ATTESTATION_PREDICATE_TYPES = frozenset(
    {"https://slsa.dev/provenance/v1", "https://spdx.dev/Document"}
)

# The pinned neutral base contains two libraries with secret-shaped binary
# substrings.  These exact path+byte identities are already independently
# audited by the shared complete-byte scanner.  Reusing only those immutable
# identities keeps binary scanning enabled and turns any base-byte drift into a
# refusal instead of adding a path exclusion.
AUDITED_SECRET_LITERAL_FILE_SHA256 = {
    "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3": (
        "779b25d20249988bea2c1aa6bbeb218f5ae7ea8a9d30ce4f54ea37372965cc4b"
    ),
    "usr/lib/x86_64-linux-gnu/libunistring.so.2.2.0": (
        "bc5951aa3d6eaba20ff9688efa3420dc95785aae3709ec48ff6df46d6f409ee5"
    ),
}

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"lib(?:[A-Za-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|"
    rb"cupti|cufile|cusparse|cusolver|curand|cufft|nvidia)[A-Za-z0-9_.-]*\.so",
    re.I,
)

_ARCHIVE_METADATA_FILES = frozenset(
    {"manifest.json", "repositories", "index.json", "oci-layout"}
)


@dataclass(frozen=True)
class _ArchiveIdentity:
    """Identity of every byte in one uncompressed tar stream."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ImageInventory:
    """Canonical ordered-layer and flattened-rootfs identity."""

    sha256: str
    entry_count: int
    layer_count: int
    payload: bytes


def _config_findings(config: dict[str, Any]) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    runtime = config.get("config") or {}
    labels = runtime.get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.licenses": (
            "Apache-2.0 AND LicenseRef-NPA-LIBERO-Neutral-Third-Party"
        ),
        "org.nebius.npa.redistribution": "public-neutral-bootstrap",
        "org.nebius.npa.validation-status": "quarantined-unvalidated",
        "org.nebius.npa.base-manifest": BASE_MANIFEST,
        "org.nebius.npa.base-rootfs-material": BASE_ROOTFS_MATERIAL,
        "org.nebius.npa.skypilot-bootstrap-contract": BOOTSTRAP_CONTRACT,
        "org.nebius.npa.third-party-notices": (
            "/opt/npa/libero/THIRD_PARTY_NOTICES.md"
        ),
    }
    for key, expected in expected_labels.items():
        if labels.get(key) != expected:
            findings.append(
                walker.Finding(
                    "config_contract", key, "required OCI label is absent or changed"
                )
            )
    revision = str(labels.get("org.opencontainers.image.revision") or "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        findings.append(
            walker.Finding("config_contract", "revision", "full source SHA is required")
        )
    if runtime.get("User") != "ubuntu":
        findings.append(
            walker.Finding("config_contract", "user", "runtime user must be ubuntu")
        )
    if runtime.get("Entrypoint") != ["/usr/local/bin/npa-libero-entrypoint"]:
        findings.append(
            walker.Finding(
                "config_contract", "entrypoint", "neutral entrypoint differs"
            )
        )
    if runtime.get("ExposedPorts"):
        findings.append(
            walker.Finding(
                "config_contract", "ports", "neutral job image exposes a service"
            )
        )
    return findings


def _metadata_findings(
    metadata: dict[str, Any],
    observed_config_digest: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    provenance = metadata.get("buildx.build.provenance")
    materials = provenance.get("materials") if isinstance(provenance, dict) else None
    base_sha256 = BASE_MANIFEST.removeprefix("sha256:")
    base_observed = False
    if isinstance(materials, list):
        for material in materials:
            if not isinstance(material, dict):
                continue
            uri = str(material.get("uri") or "")
            digests = material.get("digest") or {}
            if isinstance(digests, dict) and (
                digests.get("sha256") == base_sha256
                or uri.endswith(f"@sha256:{base_sha256}")
            ):
                base_observed = True
                break
    if not base_observed:
        findings.append(
            walker.Finding(
                "independent_build_lineage",
                "<buildx-metadata>",
                "structured provenance materials do not bind the base manifest",
            )
        )
    config_digest = str(metadata.get("containerimage.config.digest") or "")
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", observed_config_digest) is None
        or config_digest != observed_config_digest
    ):
        findings.append(
            walker.Finding(
                "independent_build_lineage",
                "<buildx-metadata>",
                "metadata config digest does not equal the scanned OCI config",
            )
        )
    return findings


def canonical_build_metadata_bytes(metadata: dict[str, Any]) -> bytes:
    """Return the reproducible Buildx identity used by image qualification.

    Buildx emits runner/session fields whose raw JSON is intentionally unstable.
    The security-relevant identity is the scanned config plus the complete,
    canonically ordered provenance material set.
    """

    provenance = metadata.get("buildx.build.provenance")
    materials = provenance.get("materials") if isinstance(provenance, dict) else None
    if not isinstance(materials, list) or not materials:
        raise RuntimeError("Buildx metadata has no structured provenance materials")
    canonical_materials = []
    for material in materials:
        if not isinstance(material, dict) or set(material) != {"uri", "digest"}:
            raise RuntimeError("Buildx provenance material schema is not closed")
        uri = material.get("uri")
        digests = material.get("digest")
        if (
            not isinstance(uri, str)
            or not uri
            or not isinstance(digests, dict)
            or not digests
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in digests.items()
            )
        ):
            raise RuntimeError("Buildx provenance material identity is invalid")
        canonical_materials.append(
            {"uri": uri, "digest": dict(sorted(digests.items()))}
        )
    canonical_materials.sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    config_digest = metadata.get("containerimage.config.digest")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(config_digest or "")) is None:
        raise RuntimeError("Buildx metadata has no immutable config digest")
    return json.dumps(
        {
            "schema": "npa.libero.canonical-build-metadata.v1",
            "config_digest": config_digest,
            "materials": canonical_materials,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _runtime_payload_hashes() -> frozenset[str]:
    payload = RUNTIME_MANIFEST.read_bytes()
    if hashlib.sha256(payload).hexdigest() != RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("scanner runtime manifest differs from its reviewed bytes")
    manifest = json.loads(payload)
    source = manifest["source"]
    task = manifest["task"]
    hashes = {
        source["license_sha256"],
        task["bddl"]["sha256"],
        task["initial_states"]["sha256"],
        task["embedding_source"]["sha256"],
        manifest["demonstration"]["sha256"],
        *(item["sha256"] for item in manifest["language_model"]["files"]),
        *(item["sha256"] for item in manifest["runtime_artifacts"]),
    }
    return frozenset(hashes)


def _renamed_payload_content_findings(
    tars: list[Path], config: dict[str, Any]
) -> list[walker.Finding]:
    original_secret_content = walker.SECRET_CONTENT
    original_elf_dependency = walker.FORBIDDEN_ELF_DEPENDENCY
    try:
        walker.SECRET_CONTENT = FORBIDDEN_PAYLOAD_CONTENT
        walker.FORBIDDEN_ELF_DEPENDENCY = NEVER_MATCH_ELF
        with walker.payload_policy(
            forbidden_paths=(),
            forbidden_history=(),
            audited_secret_files={},
            audited_libraries={},
        ):
            raw = walker.scan_tars(tars, config)
    finally:
        walker.SECRET_CONTENT = original_secret_content
        walker.FORBIDDEN_ELF_DEPENDENCY = original_elf_dependency
    return [
        walker.Finding(
            "renamed_runtime_payload_content",
            finding.path,
            "forbidden LIBERO, robomimic, or PyTorch source signature",
        )
        for finding in raw
        if finding.kind == "credential_content"
        and finding.path not in NEUTRAL_PAYLOAD_CONTENT_ALLOWLIST
    ]


def _base_provenance_findings(
    provenance_bytes: bytes, provenance: dict[str, Any]
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    observed_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    predicate = provenance.get("predicate") or {}
    subjects = provenance.get("subject") or []
    materials = predicate.get("materials") or []
    subject_digests = {
        str((item.get("digest") or {}).get("sha256") or "")
        for item in subjects
        if isinstance(item, dict)
    }
    material_sha256 = {
        str((item.get("digest") or {}).get("sha256") or "")
        for item in materials
        if isinstance(item, dict)
    }
    material_sha1 = {
        str((item.get("digest") or {}).get("sha1") or "")
        for item in materials
        if isinstance(item, dict)
    }
    expected = {
        "blob_sha256": observed_sha256 == BASE_PROVENANCE_SHA256,
        "statement_type": provenance.get("_type")
        == "https://in-toto.io/Statement/v0.1",
        "predicate_type": provenance.get("predicateType")
        == "https://slsa.dev/provenance/v0.2",
        "builder": (predicate.get("builder") or {}).get("id")
        == "https://github.com/docker-library",
        "base_subject": BASE_MANIFEST.removeprefix("sha256:") in subject_digests,
        "rootfs_material": BASE_ROOTFS_MATERIAL.removeprefix("sha256:")
        in material_sha256,
        "base_source_revision": BASE_SOURCE_REVISION in material_sha1,
    }
    for name, valid in expected.items():
        if not valid:
            findings.append(
                walker.Finding(
                    "independent_base_provenance",
                    "<published-base-provenance>",
                    f"published base provenance failed {name}",
                )
            )
    return findings


def _normalized_tar_path(name: str) -> str:
    candidate = name
    if candidate in {".", "./"}:
        return "."
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or candidate.startswith("/"):
        raise ValueError("archive member path is empty or absolute")
    parts = candidate.rstrip("/").split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("archive member path is not canonical and confined")
    return "/".join(parts)


@contextmanager
def _uncompressed_tar_file(path: Path) -> Any:
    """Yield a seekable bounded tar stream without retaining archive bytes in RAM."""

    if path.stat().st_size > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
        raise RuntimeError("compressed archive exceeds the scanner size budget")
    source = path.open("rb")
    temporary: Any | None = None
    decompressor: Any | None = None
    try:
        magic = source.read(6)
        source.seek(0)
        if magic.startswith(b"\x1f\x8b"):
            decompressor = gzip.GzipFile(fileobj=source)
        elif magic.startswith(b"BZh"):
            decompressor = bz2.BZ2File(source)
        elif magic.startswith(b"\xfd7zXZ\x00"):
            decompressor = lzma.LZMAFile(source)
        elif magic.startswith(b"\x28\xb5\x2f\xfd"):
            if walker.zstd is None:
                raise RuntimeError("zstd archive cannot be accounted without zstandard")
            decompressor = walker.zstd.ZstdDecompressor().stream_reader(source)
        stream = decompressor or source
        output = source
        if decompressor is not None:
            temporary = tempfile.TemporaryFile(mode="w+b")
            output = temporary
        digest = hashlib.sha256()
        total = 0
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UNCOMPRESSED_ARCHIVE_BYTES:
                raise RuntimeError(
                    "uncompressed archive exceeds the scanner size budget"
                )
            digest.update(chunk)
            if temporary is not None:
                temporary.write(chunk)
        output.flush()
        output.seek(0)
        yield output, total, digest.hexdigest()
    finally:
        if decompressor is not None:
            decompressor.close()
        if temporary is not None:
            temporary.close()
        source.close()


def _archive_records(
    path: Path,
    *,
    payload_hashes: frozenset[str] = frozenset(),
    scan_payload_content: bool = False,
) -> tuple[
    list[tuple[tarfile.TarInfo, str, tuple[object, ...]]],
    list[walker.Finding],
    _ArchiveIdentity,
]:
    findings: list[walker.Finding] = []
    records: list[tuple[tarfile.TarInfo, str, tuple[object, ...]]] = []
    cursor = 0
    with _uncompressed_tar_file(path) as (stream, size_bytes, sha256):
        identity = _ArchiveIdentity(sha256=sha256, size_bytes=size_bytes)
        if size_bytes % tarfile.BLOCKSIZE:
            findings.append(
                walker.Finding(
                    "unaccounted_archive_bytes",
                    path.name,
                    "decompressed tar length is not block aligned",
                )
            )
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for member in archive:
                    try:
                        normalized = _normalized_tar_path(member.name)
                    except ValueError as exc:
                        findings.append(
                            walker.Finding(
                                "unsafe_archive_member", member.name, str(exc)
                            )
                        )
                        continue
                    if member.offset < cursor or member.offset_data < member.offset:
                        findings.append(
                            walker.Finding(
                                "overlapping_archive_member",
                                normalized,
                                f"invalid tar offsets in {path.name}",
                            )
                        )
                        continue
                    if any(memoryview(mapped)[cursor : member.offset]):
                        findings.append(
                            walker.Finding(
                                "unaccounted_archive_bytes",
                                normalized,
                                f"nonzero bytes precede a member in {path.name}",
                            )
                        )
                    data_end = member.offset_data + member.size
                    padded_end = (data_end + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
                    padded_end *= tarfile.BLOCKSIZE
                    if data_end > size_bytes:
                        findings.append(
                            walker.Finding(
                                "truncated_archive_member",
                                normalized,
                                f"member exceeds {path.name}",
                            )
                        )
                        continue
                    if any(memoryview(mapped)[data_end:padded_end]):
                        findings.append(
                            walker.Finding(
                                "nonzero_archive_padding",
                                normalized,
                                f"member padding contains bytes in {path.name}",
                            )
                        )
                    descriptor: tuple[object, ...]
                    if member.isfile():
                        content = memoryview(mapped)[member.offset_data : data_end]
                        content_sha256 = hashlib.sha256(content).hexdigest()
                        descriptor = ("file", member.size, content_sha256)
                        forbidden = scan_payload_content and any(
                            pattern.search(content)
                            for pattern in FORBIDDEN_PAYLOAD_CONTENT
                        )
                        if forbidden and (
                            NEUTRAL_PAYLOAD_CONTENT_ALLOWLIST.get(normalized)
                            != content_sha256
                        ):
                            findings.append(
                                walker.Finding(
                                    "renamed_runtime_payload_content",
                                    normalized,
                                    "forbidden source signature is not the reviewed neutral smoke driver",
                                )
                            )
                        if content_sha256 in payload_hashes:
                            findings.append(
                                walker.Finding(
                                    "exact_runtime_payload_bytes",
                                    normalized,
                                    "file bytes equal a runtime-only source, package, model, or data artifact",
                                )
                            )
                        content.release()
                    elif member.isdir():
                        descriptor = ("directory",)
                    elif member.issym():
                        descriptor = ("symlink", member.linkname)
                    elif member.islnk():
                        try:
                            target = _normalized_tar_path(member.linkname)
                        except ValueError as exc:
                            findings.append(
                                walker.Finding(
                                    "unsafe_archive_hardlink", normalized, str(exc)
                                )
                            )
                            target = ""
                        descriptor = ("hardlink", target)
                    elif (
                        member.ischr() and member.devmajor == 0 and member.devminor == 0
                    ):
                        descriptor = ("whiteout-device", member.size)
                    else:
                        descriptor = (
                            "unsupported",
                            member.type.decode(errors="replace"),
                        )
                        findings.append(
                            walker.Finding(
                                "unsupported_archive_member",
                                normalized,
                                f"unsupported tar member type in {path.name}",
                            )
                        )
                    records.append((member, normalized, descriptor))
                    cursor = padded_end
            if any(memoryview(mapped)[cursor:]):
                findings.append(
                    walker.Finding(
                        "unaccounted_archive_bytes",
                        path.name,
                        "nonzero bytes remain after the final accounted member",
                    )
                )
        finally:
            mapped.close()
    return records, findings, identity


def _apply_layer_records(
    records: list[tuple[tarfile.TarInfo, str, tuple[object, ...]]],
    state: dict[str, tuple[object, ...]],
    *,
    source: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    seen: set[str] = set()
    for member, path, descriptor in records:
        if path == ".":
            if descriptor == ("directory",):
                continue
            findings.append(
                walker.Finding(
                    "unsafe_archive_member",
                    path,
                    f"archive root record is not a directory in {source}",
                )
            )
            continue
        if path in seen:
            findings.append(
                walker.Finding(
                    "duplicate_archive_member", path, f"duplicate path in {source}"
                )
            )
        seen.add(path)
        parent, _, name = path.rpartition("/")
        if name.startswith(".wh."):
            valid_whiteout = (member.isfile() and member.size == 0) or descriptor == (
                "whiteout-device",
                0,
            )
            if not valid_whiteout:
                findings.append(
                    walker.Finding(
                        "malformed_whiteout",
                        path,
                        f"whiteout is not an empty file or 0:0 device in {source}",
                    )
                )
                continue
            if name == ".wh..wh..opq":
                prefix = f"{parent}/" if parent else ""
                for existing in tuple(state):
                    if existing.startswith(prefix) and existing != parent:
                        state.pop(existing, None)
            else:
                target_name = name.removeprefix(".wh.")
                if not target_name:
                    findings.append(
                        walker.Finding(
                            "malformed_whiteout", path, "whiteout target is empty"
                        )
                    )
                    continue
                target = f"{parent}/{target_name}" if parent else target_name
                for existing in tuple(state):
                    if existing == target or existing.startswith(f"{target}/"):
                        state.pop(existing, None)
            continue
        if descriptor[0] == "whiteout-device":
            findings.append(
                walker.Finding(
                    "unsupported_archive_member",
                    path,
                    f"0:0 device is not named as an OCI whiteout in {source}",
                )
            )
            continue
        state[path] = descriptor
    return findings


def _resolve_hardlinks(
    state: dict[str, tuple[object, ...]],
) -> tuple[dict[str, tuple[object, ...]], list[walker.Finding]]:
    resolved: dict[str, tuple[object, ...]] = {}
    findings: list[walker.Finding] = []
    for path, descriptor in state.items():
        current = descriptor
        visited = {path}
        while current[0] == "hardlink":
            target = str(current[1])
            if target in visited or target not in state:
                findings.append(
                    walker.Finding(
                        "invalid_archive_hardlink",
                        path,
                        "hardlink target is absent or cyclic",
                    )
                )
                break
            visited.add(target)
            current = state[target]
        resolved[path] = current
    return resolved, findings


def _image_inventory(
    layer_identities: list[_ArchiveIdentity],
    flattened: dict[str, tuple[object, ...]],
) -> _ImageInventory:
    entries = [
        {"path": path, "descriptor": list(flattened[path])}
        for path in sorted(flattened)
    ]
    layers = [
        {"position": position, **asdict(identity)}
        for position, identity in enumerate(layer_identities)
    ]
    document = {
        "schema": IMAGE_INVENTORY_SCHEMA,
        "layers": layers,
        "rootfs_entries": entries,
    }
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return _ImageInventory(
        sha256=hashlib.sha256(payload).hexdigest(),
        entry_count=len(entries),
        layer_count=len(layers),
        payload=payload,
    )


def _layer_graph_findings(
    layer_tars: list[Path], exported_rootfs: Path | None = None
) -> tuple[list[walker.Finding], _ImageInventory]:
    findings: list[walker.Finding] = []
    state: dict[str, tuple[object, ...]] = {}
    layer_identities: list[_ArchiveIdentity] = []
    payload_hashes = _runtime_payload_hashes()
    for layer in layer_tars:
        records, archive_findings, identity = _archive_records(
            layer, payload_hashes=payload_hashes, scan_payload_content=True
        )
        layer_identities.append(identity)
        findings.extend(archive_findings)
        findings.extend(_apply_layer_records(records, state, source=layer.name))
    flattened, hardlink_findings = _resolve_hardlinks(state)
    findings.extend(hardlink_findings)
    byof = flattened.get("opt/byof")
    if byof != (
        "symlink",
        "/workspace/.cache/npa/libero/current/source",
    ):
        findings.append(
            walker.Finding(
                "neutral_bootstrap_link",
                "opt/byof",
                "neutral source path is not the exact empty-cache symlink",
            )
        )
    for path in flattened:
        for kind, pattern in FORBIDDEN_PATHS:
            if pattern.search(path):
                findings.append(
                    walker.Finding(
                        kind,
                        path,
                        "forbidden path exists in flattened filesystem",
                    )
                )
    if exported_rootfs is not None:
        records, archive_findings, _identity = _archive_records(
            exported_rootfs,
            payload_hashes=payload_hashes,
            scan_payload_content=True,
        )
        findings.extend(archive_findings)
        exported: dict[str, tuple[object, ...]] = {}
        findings.extend(
            _apply_layer_records(records, exported, source=exported_rootfs.name)
        )
        exported, hardlink_findings = _resolve_hardlinks(exported)
        findings.extend(hardlink_findings)
        comparable = {
            path: value
            for path, value in flattened.items()
            if value[0] != "directory" and path not in RUNTIME_INJECTED_EXPORT_PATHS
        }
        exported_comparable = {
            path: value
            for path, value in exported.items()
            if value[0] != "directory" and path not in RUNTIME_INJECTED_EXPORT_PATHS
        }
        if comparable != exported_comparable:
            findings.append(
                walker.Finding(
                    "flattened_rootfs_mismatch",
                    "<exported-rootfs>",
                    "exported filesystem differs from the ordered OCI layers",
                )
            )
    return findings, _image_inventory(layer_identities, flattened)


def _private_stage_identity_findings(
    inventory: _ImageInventory,
    observed_config_digest: str,
    expected_image_inventory_sha256: str,
    expected_config_digest: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_image_inventory_sha256) is None
        or inventory.sha256 != expected_image_inventory_sha256
    ):
        findings.append(
            walker.Finding(
                "accepted_private_stage_identity",
                "<complete-image-inventory>",
                "ordered layers and rootfs do not match qualified private-stage bytes",
            )
        )
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", expected_config_digest) is None
        or observed_config_digest != expected_config_digest
    ):
        findings.append(
            walker.Finding(
                "accepted_private_stage_identity",
                "<oci-config>",
                "OCI config does not match qualified private-stage bytes",
            )
        )
    return findings


def _accepted_lineage_findings(
    build_metadata: dict[str, Any],
    base_provenance_bytes: bytes,
    expected_canonical_build_metadata_sha256: str,
    expected_base_provenance_sha256: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    try:
        canonical_metadata = canonical_build_metadata_bytes(build_metadata)
    except RuntimeError as exc:
        findings.append(
            walker.Finding("accepted_build_lineage", "<buildx-metadata>", str(exc))
        )
        canonical_metadata = b""
    for kind, label, payload, expected in (
        (
            "accepted_build_lineage",
            "<buildx-metadata>",
            canonical_metadata,
            expected_canonical_build_metadata_sha256,
        ),
        (
            "accepted_base_lineage",
            "<published-base-provenance>",
            base_provenance_bytes,
            expected_base_provenance_sha256,
        ),
    ):
        observed = hashlib.sha256(payload).hexdigest()
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected or "") is None
            or observed != expected
        ):
            findings.append(
                walker.Finding(
                    kind, label, "bytes differ from checked-in qualification"
                )
            )
    return findings


def _docker_save_outer_findings(path: Path) -> list[walker.Finding]:
    records, findings, _identity = _archive_records(path)
    with tarfile.open(path, "r:*") as archive:
        manifest_stream = archive.extractfile(archive.getmember("manifest.json"))
        if manifest_stream is None:
            raise RuntimeError("docker save archive has no readable manifest.json")
        manifests = json.load(io.TextIOWrapper(manifest_stream, encoding="utf-8"))
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError("docker save archive must contain exactly one image")
        manifest = manifests[0]
        legacy_layers = frozenset(str(item) for item in (manifest.get("Layers") or []))
        runtime_payload_hashes = _runtime_payload_hashes()
        for metadata_name in _ARCHIVE_METADATA_FILES.intersection(archive.getnames()):
            member = archive.getmember(metadata_name)
            if not member.isfile():
                continue
            metadata_stream = archive.extractfile(member)
            if metadata_stream is None:
                raise RuntimeError("docker-save metadata member is unreadable")
            findings.extend(
                _opaque_content_findings(
                    member_name=metadata_name,
                    blob=metadata_stream.read(),
                    runtime_payload_hashes=runtime_payload_hashes,
                    description="docker-save metadata",
                )
            )
        descriptor_members, descriptor_findings = _oci_descriptor_members(
            archive, legacy_layers, runtime_payload_hashes
        )
    allowed = {
        *_ARCHIVE_METADATA_FILES,
        str(manifest.get("Config") or ""),
        *(str(item) for item in (manifest.get("Layers") or [])),
    }
    allowed.update(descriptor_members)
    findings.extend(descriptor_findings)
    seen: set[str] = set()
    for member, normalized, descriptor in records:
        if normalized in seen:
            findings.append(
                walker.Finding(
                    "duplicate_outer_archive_member",
                    normalized,
                    "docker-save archive contains a duplicate normalized path",
                )
            )
        seen.add(normalized)
        if descriptor[0] != "directory" and normalized not in allowed:
            findings.append(
                walker.Finding(
                    "unaccounted_outer_archive_member",
                    normalized,
                    "docker-save member is not bound by its manifest",
                )
            )
        if descriptor[0] not in {"file", "directory"}:
            findings.append(
                walker.Finding(
                    "unsupported_outer_archive_member",
                    normalized,
                    "docker-save archive member is not a regular file or directory",
                )
            )
    return findings


OCI_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
OCI_MANIFEST_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
)
OCI_CONFIG_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.oci.empty.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }
)
OCI_LAYER_MEDIA_TYPES = frozenset(
    {
        "application/vnd.in-toto+json",
        "application/vnd.oci.image.layer.v1.tar",
        "application/vnd.oci.image.layer.v1.tar+gzip",
        "application/vnd.oci.image.layer.v1.tar+zstd",
        "application/vnd.oci.image.layer.nondistributable.v1.tar",
        "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip",
        "application/vnd.oci.image.layer.nondistributable.v1.tar+zstd",
        "application/vnd.docker.image.rootfs.diff.tar.gzip",
        "application/vnd.docker.image.rootfs.diff.tar",
    }
)
OCI_DESCRIPTOR_MEDIA_TYPES = {
    "manifest": OCI_INDEX_MEDIA_TYPES | OCI_MANIFEST_MEDIA_TYPES,
    "config": OCI_CONFIG_MEDIA_TYPES,
    "layer": OCI_LAYER_MEDIA_TYPES,
    "subject": OCI_INDEX_MEDIA_TYPES | OCI_MANIFEST_MEDIA_TYPES,
}


def _verified_oci_descriptor(
    archive: tarfile.TarFile, descriptor: object, role: str
) -> tuple[str, bytes | None, list[walker.Finding]]:
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"OCI {role} descriptor is not an object")
    media_type = str(descriptor.get("mediaType") or "")
    if media_type not in OCI_DESCRIPTOR_MEDIA_TYPES[role]:
        raise RuntimeError(f"OCI {role} descriptor has unsupported media type")
    digest = str(descriptor.get("digest") or "")
    size = descriptor.get("size")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("OCI descriptor has no immutable SHA-256")
    if not isinstance(size, int) or size < 0:
        raise RuntimeError("OCI descriptor has no valid byte size")
    member_name = "blobs/sha256/" + digest.removeprefix("sha256:")
    try:
        member = archive.getmember(member_name)
    except KeyError:
        return (
            member_name,
            None,
            [
                walker.Finding(
                    "missing_oci_descriptor_member",
                    member_name,
                    "OCI descriptor target is absent from docker-save archive",
                )
            ],
        )
    if not member.isfile():
        return (
            member_name,
            None,
            [
                walker.Finding(
                    "invalid_oci_descriptor_member",
                    member_name,
                    "OCI descriptor target is not a regular file",
                )
            ],
        )
    blob_stream = archive.extractfile(member)
    if blob_stream is None:
        raise RuntimeError("OCI descriptor target is unreadable")
    blob = blob_stream.read()
    if len(blob) != size or hashlib.sha256(blob).hexdigest() != digest.removeprefix(
        "sha256:"
    ):
        return (
            member_name,
            None,
            [
                walker.Finding(
                    "oci_descriptor_identity_mismatch",
                    member_name,
                    "OCI descriptor digest or size does not match stored bytes",
                )
            ],
        )
    return member_name, blob, []


def _oci_descriptor_children(blob: bytes, media_type: str) -> list[tuple[str, object]]:
    if media_type not in OCI_INDEX_MEDIA_TYPES | OCI_MANIFEST_MEDIA_TYPES:
        return []
    document = json.loads(blob)
    if not isinstance(document, dict) or document.get("schemaVersion") != 2:
        raise RuntimeError("OCI descriptor document is malformed")
    declared_media_type = document.get("mediaType")
    if declared_media_type is not None and declared_media_type != media_type:
        raise RuntimeError("OCI descriptor document media type differs from descriptor")
    if media_type in OCI_INDEX_MEDIA_TYPES:
        manifests = document.get("manifests")
        if not isinstance(manifests, list):
            raise RuntimeError("OCI manifests descriptor collection is malformed")
        return [("manifest", item) for item in manifests]
    config = document.get("config")
    layers = document.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list):
        raise RuntimeError("OCI image manifest descriptor collection is malformed")
    children = [("config", config)]
    children.extend(("layer", item) for item in layers)
    subject = document.get("subject")
    if subject is not None:
        children.append(("subject", subject))
    return children


def _opaque_content_findings(
    *,
    member_name: str,
    blob: bytes,
    runtime_payload_hashes: frozenset[str],
    description: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    if any(pattern.search(blob) for pattern in SECRET_CONTENT):
        findings.append(
            walker.Finding(
                "credential_content",
                member_name,
                f"secret-like bytes in {description}",
            )
        )
    if any(pattern.search(blob) for pattern in FORBIDDEN_PAYLOAD_CONTENT):
        findings.append(
            walker.Finding(
                "renamed_runtime_payload_content",
                member_name,
                f"forbidden source signature in {description}",
            )
        )
    if hashlib.sha256(blob).hexdigest() in runtime_payload_hashes:
        findings.append(
            walker.Finding(
                "exact_runtime_payload_bytes",
                member_name,
                f"{description} equals a runtime-only artifact",
            )
        )
    return findings


def _oci_descriptor_content_findings(
    *,
    member_name: str,
    blob: bytes,
    media_type: str,
    role: str,
    legacy_layers: frozenset[str],
    runtime_payload_hashes: frozenset[str],
) -> list[walker.Finding]:
    findings = _opaque_content_findings(
        member_name=member_name,
        blob=blob,
        runtime_payload_hashes=runtime_payload_hashes,
        description="an OCI descriptor target",
    )
    if (
        role == "layer"
        and media_type
        in {
            "application/vnd.oci.image.layer.v1.tar",
            "application/vnd.docker.image.rootfs.diff.tar",
        }
        and member_name in legacy_layers
        and any(item.kind == "credential_content" for item in findings)
    ):
        # Apply the same existing path+hash audits without exempting tar headers,
        # padding, or other files from the raw descriptor credential scan.
        audited_ranges: list[tuple[int, int]] = []
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as layer:
            for member in layer:
                expected = AUDITED_SECRET_LITERAL_FILE_SHA256.get(member.name)
                if expected and member.isfile():
                    start = member.offset_data
                    end = start + member.size
                    if hashlib.sha256(blob[start:end]).hexdigest() == expected:
                        audited_ranges.append((start, end))
        if all(
            any(
                start <= match.start() and match.end() <= end
                for start, end in audited_ranges
            )
            for pattern in SECRET_CONTENT
            for match in pattern.finditer(blob)
        ):
            findings = [item for item in findings if item.kind != "credential_content"]
    if role == "config" or media_type == "application/vnd.in-toto+json":
        document = json.loads(blob)
        if not isinstance(document, dict):
            raise RuntimeError(f"OCI {role} JSON descriptor target is not an object")
    if role == "layer" and media_type != "application/vnd.in-toto+json":
        if member_name not in legacy_layers:
            findings.append(
                walker.Finding(
                    "auxiliary_oci_runtime_layer",
                    member_name,
                    "OCI runtime layer is absent from the Docker image manifest",
                )
            )
    return findings


def _oci_descriptor_members(
    archive: tarfile.TarFile,
    legacy_layers: frozenset[str],
    runtime_payload_hashes: frozenset[str],
) -> tuple[set[str], list[walker.Finding]]:
    """Bind every OCI-index descendant by descriptor digest and size."""

    if "index.json" not in archive.getnames():
        return set(), []
    stream = archive.extractfile(archive.getmember("index.json"))
    if stream is None:
        raise RuntimeError("docker-save OCI index is unreadable")
    index = json.load(io.TextIOWrapper(stream, encoding="utf-8"))
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 2
        or not isinstance(index.get("manifests"), list)
    ):
        raise RuntimeError("docker-save OCI index is malformed")
    pending = [("manifest", item) for item in index["manifests"]]
    allowed: set[str] = set()
    findings: list[walker.Finding] = []
    expanded: set[str] = set()
    while pending:
        role, descriptor = pending.pop()
        member_name, blob, descriptor_findings = _verified_oci_descriptor(
            archive, descriptor, role
        )
        findings.extend(descriptor_findings)
        if blob is None:
            continue
        allowed.add(member_name)
        media_type = str(descriptor.get("mediaType"))
        findings.extend(
            _oci_descriptor_content_findings(
                member_name=member_name,
                blob=blob,
                media_type=media_type,
                role=role,
                legacy_layers=legacy_layers,
                runtime_payload_hashes=runtime_payload_hashes,
            )
        )
        if member_name in expanded:
            continue
        expanded.add(member_name)
        pending.extend(_oci_descriptor_children(blob, media_type))
    return allowed, findings


def _build_oci_attestation_findings(
    path: Path, build_metadata: dict[str, Any]
) -> tuple[list[walker.Finding], dict[str, object]]:
    """Verify one BuildKit OCI export binds attestations to its runtime image."""

    records, findings, identity = _archive_records(path)
    expected_config_digest = str(
        build_metadata.get("containerimage.config.digest") or ""
    )
    derive_config = "containerimage.config.digest" not in build_metadata
    if (
        not derive_config
        and re.fullmatch(r"sha256:[0-9a-f]{64}", expected_config_digest) is None
    ):
        raise RuntimeError("build metadata has no immutable config digest")
    build_metadata = dict(build_metadata)
    allowed = {"index.json", "oci-layout"}
    seen_outer: set[str] = set()
    with tarfile.open(path, "r:*") as archive:
        names = archive.getnames()
        for required in ("index.json", "oci-layout"):
            if required not in names or not archive.getmember(required).isfile():
                raise RuntimeError(f"build OCI archive has no regular {required}")
        layout_stream = archive.extractfile(archive.getmember("oci-layout"))
        index_stream = archive.extractfile(archive.getmember("index.json"))
        if layout_stream is None or index_stream is None:
            raise RuntimeError("build OCI archive metadata is unreadable")
        layout = json.load(io.TextIOWrapper(layout_stream, encoding="utf-8"))
        index = json.load(io.TextIOWrapper(index_stream, encoding="utf-8"))
        if layout != {"imageLayoutVersion": "1.0.0"}:
            raise RuntimeError("build OCI archive layout is invalid")
        if (
            not isinstance(index, dict)
            or index.get("schemaVersion") != 2
            or not isinstance(index.get("manifests"), list)
        ):
            raise RuntimeError("build OCI archive index is malformed")

        if derive_config:
            # Attested BuildKit exports report an index digest, not a config
            # digest. Bind the complete descriptor walk to that immutable root
            # before deriving the one runtime config for downstream byte gates.
            root_digest = str(build_metadata.get("containerimage.digest") or "")
            roots = index["manifests"]
            if (
                re.fullmatch(r"sha256:[0-9a-f]{64}", root_digest) is None
                or len(roots) != 1
                or not isinstance(roots[0], dict)
                or roots[0].get("digest") != root_digest
            ):
                raise RuntimeError("build OCI root does not match Buildx metadata")

        pending = list(index["manifests"])
        expanded: set[str] = set()
        manifests: list[tuple[dict[str, Any], dict[str, Any]]] = []
        while pending:
            descriptor = pending.pop()
            member_name, blob, descriptor_findings = _verified_oci_descriptor(
                archive, descriptor, "manifest"
            )
            findings.extend(descriptor_findings)
            if blob is None:
                continue
            allowed.add(member_name)
            if member_name in expanded:
                continue
            expanded.add(member_name)
            media_type = str(descriptor.get("mediaType") or "")
            document = json.loads(blob)
            if not isinstance(document, dict):
                raise RuntimeError("build OCI manifest is not an object")
            if media_type in OCI_INDEX_MEDIA_TYPES:
                children = document.get("manifests")
                if not isinstance(children, list):
                    raise RuntimeError("build OCI nested index is malformed")
                pending.extend(children)
            elif media_type in OCI_MANIFEST_MEDIA_TYPES:
                manifests.append((descriptor, document))
            else:
                raise RuntimeError("build OCI root has unsupported media type")

        runtime = [
            item
            for item in manifests
            if item[0].get("platform") == {"architecture": "amd64", "os": "linux"}
            and str((item[1].get("config") or {}).get("mediaType") or "")
            == "application/vnd.oci.image.config.v1+json"
        ]
        attestations = [
            item
            for item in manifests
            if (item[0].get("annotations") or {}).get("vnd.docker.reference.type")
            == "attestation-manifest"
        ]
        if len(runtime) != 1 or len(attestations) != 1 or len(manifests) != 2:
            findings.append(
                walker.Finding(
                    "build_attestation_graph",
                    "<build-oci-index>",
                    "OCI export must contain exactly one runtime and one "
                    "attestation manifest",
                )
            )
        else:
            runtime_descriptor, runtime_manifest = runtime[0]
            attestation_descriptor, attestation_manifest = attestations[0]
            runtime_digest = str(runtime_descriptor.get("digest") or "")
            annotations = attestation_descriptor.get("annotations") or {}
            if annotations.get("vnd.docker.reference.digest") != runtime_digest:
                findings.append(
                    walker.Finding(
                        "build_attestation_graph",
                        "<build-oci-index>",
                        "attestation descriptor does not bind the runtime manifest",
                    )
                )

            runtime_config = runtime_manifest.get("config")
            config_name, config_blob, config_findings = _verified_oci_descriptor(
                archive, runtime_config, "config"
            )
            findings.extend(config_findings)
            if config_blob is not None:
                allowed.add(config_name)
                if derive_config:
                    expected_config_digest = str(runtime_config["digest"])
                    build_metadata["containerimage.config.digest"] = (
                        expected_config_digest
                    )
            if (
                str((runtime_config or {}).get("digest") or "")
                != expected_config_digest
            ):
                findings.append(
                    walker.Finding(
                        "build_attestation_graph",
                        "<build-oci-runtime-config>",
                        "OCI export config does not match Buildx metadata",
                    )
                )
            runtime_layers = runtime_manifest.get("layers")
            if not isinstance(runtime_layers, list) or not runtime_layers:
                raise RuntimeError("build OCI runtime layer set is incomplete")
            for layer in runtime_layers:
                member_name, blob, descriptor_findings = _verified_oci_descriptor(
                    archive, layer, "layer"
                )
                findings.extend(descriptor_findings)
                if blob is not None:
                    allowed.add(member_name)

            attestation_config = attestation_manifest.get("config")
            config_name, config_blob, config_findings = _verified_oci_descriptor(
                archive, attestation_config, "config"
            )
            findings.extend(config_findings)
            if config_blob is not None:
                allowed.add(config_name)
                if json.loads(config_blob) != {}:
                    findings.append(
                        walker.Finding(
                            "build_attestation_graph",
                            config_name,
                            "attestation config is not the empty OCI config",
                        )
                    )
            if str((attestation_config or {}).get("mediaType") or "") != (
                "application/vnd.oci.empty.v1+json"
            ):
                findings.append(
                    walker.Finding(
                        "build_attestation_graph",
                        "<build-oci-attestation-config>",
                        "attestation config media type is invalid",
                    )
                )
            layers = attestation_manifest.get("layers")
            if not isinstance(layers, list):
                raise RuntimeError("build OCI attestation layer set is malformed")
            predicate_types: set[str] = set()
            for layer in layers:
                member_name, blob, descriptor_findings = _verified_oci_descriptor(
                    archive, layer, "layer"
                )
                findings.extend(descriptor_findings)
                if blob is None:
                    continue
                allowed.add(member_name)
                predicate_type = str(
                    (layer.get("annotations") or {}).get("in-toto.io/predicate-type")
                    or ""
                )
                statement = json.loads(blob)
                subjects = (
                    statement.get("subject") if isinstance(statement, dict) else None
                )
                bound = isinstance(subjects, list) and any(
                    isinstance(subject, dict)
                    and (subject.get("digest") or {}).get("sha256")
                    == runtime_digest.removeprefix("sha256:")
                    for subject in subjects
                )
                if (
                    str(layer.get("mediaType") or "") != "application/vnd.in-toto+json"
                    or not isinstance(statement, dict)
                    or statement.get("_type")
                    not in {
                        "https://in-toto.io/Statement/v0.1",
                        "https://in-toto.io/Statement/v1",
                    }
                    or statement.get("predicateType") != predicate_type
                    or not bound
                ):
                    findings.append(
                        walker.Finding(
                            "build_attestation_graph",
                            member_name,
                            "attestation statement does not bind the runtime manifest",
                        )
                    )
                predicate_types.add(predicate_type)
                findings.extend(
                    _opaque_content_findings(
                        member_name=member_name,
                        blob=blob,
                        runtime_payload_hashes=_runtime_payload_hashes(),
                        description="a build attestation",
                    )
                )
            if predicate_types != REQUIRED_BUILD_ATTESTATION_PREDICATE_TYPES:
                findings.append(
                    walker.Finding(
                        "build_attestation_graph",
                        "<build-oci-attestations>",
                        "required provenance and SPDX attestations are not exact",
                    )
                )

    for _member, normalized, descriptor in records:
        if normalized in seen_outer:
            findings.append(
                walker.Finding(
                    "duplicate_outer_archive_member",
                    normalized,
                    "build OCI archive contains a duplicate normalized path",
                )
            )
        seen_outer.add(normalized)
        if descriptor[0] != "directory" and normalized not in allowed:
            findings.append(
                walker.Finding(
                    "unaccounted_outer_archive_member",
                    normalized,
                    "build OCI archive member is not bound by its index",
                )
            )
    findings.extend(_metadata_findings(build_metadata, expected_config_digest))
    canonical_metadata = canonical_build_metadata_bytes(build_metadata)
    return findings, {
        "archive_sha256": identity.sha256,
        "archive_size_bytes": identity.size_bytes,
        "config_digest": expected_config_digest,
        "canonical_build_metadata_sha256": hashlib.sha256(
            canonical_metadata
        ).hexdigest(),
        "predicate_types": sorted(REQUIRED_BUILD_ATTESTATION_PREDICATE_TYPES),
    }


def _docker_save_config_digest(path: Path) -> str:
    with tarfile.open(path, "r:*") as archive:
        manifest_stream = archive.extractfile(archive.getmember("manifest.json"))
        if manifest_stream is None:
            raise RuntimeError("docker save archive has no readable manifest.json")
        manifests = json.load(io.TextIOWrapper(manifest_stream, encoding="utf-8"))
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError("docker save archive must contain exactly one image")
        config_name = str(manifests[0].get("Config") or "")
        config_stream = archive.extractfile(archive.getmember(config_name))
        if config_stream is None:
            raise RuntimeError("docker save archive has no readable image config")
        return "sha256:" + hashlib.sha256(config_stream.read()).hexdigest()


def _remote_config_digest(image: str) -> str:
    completed = subprocess.run(
        ["crane", "manifest", "--platform", "linux/amd64", image],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError("cannot read remote manifest for config identity")
    digest = str((json.loads(completed.stdout).get("config") or {}).get("digest") or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("remote manifest has no immutable config digest")
    return digest


def scan_tars(
    tars: list[Path],
    config: dict[str, Any],
    build_metadata: dict[str, Any],
    base_provenance_bytes: bytes,
    base_provenance: dict[str, Any],
    *,
    exported_rootfs: Path | None = None,
    docker_save: Path | None = None,
    observed_config_digest: str = "",
    expected_image_inventory_sha256: str,
    expected_config_digest: str,
    expected_canonical_build_metadata_sha256: str | None = None,
    expected_base_provenance_sha256: str | None = None,
    inventory_output: Path | None = None,
    evidence: dict[str, object] | None = None,
) -> list[walker.Finding]:
    original_secret_content = walker.SECRET_CONTENT
    original_elf_dependency = walker.FORBIDDEN_ELF_DEPENDENCY
    try:
        walker.SECRET_CONTENT = SECRET_CONTENT
        walker.FORBIDDEN_ELF_DEPENDENCY = FORBIDDEN_ELF_DEPENDENCY
        with walker.payload_policy(
            forbidden_paths=FORBIDDEN_PATHS,
            forbidden_history=FORBIDDEN_HISTORY,
            audited_secret_files=AUDITED_SECRET_LITERAL_FILE_SHA256,
            audited_libraries={},
        ):
            findings = walker.scan_tars(tars, config)
    finally:
        walker.SECRET_CONTENT = original_secret_content
        walker.FORBIDDEN_ELF_DEPENDENCY = original_elf_dependency
    layer_tars = (
        [item for item in tars if item != exported_rootfs]
        if exported_rootfs is not None
        else tars
    )
    graph_findings, inventory = _layer_graph_findings(layer_tars, exported_rootfs)
    if inventory_output is not None:
        inventory_output.write_bytes(inventory.payload)
        inventory_output.chmod(0o600)
    if evidence is not None:
        evidence.update(
            image_inventory_sha256=inventory.sha256,
            image_inventory_layer_count=inventory.layer_count,
            image_inventory_rootfs_entry_count=inventory.entry_count,
            observed_config_digest=observed_config_digest,
        )
    if expected_canonical_build_metadata_sha256 is None:
        try:
            canonical_metadata = canonical_build_metadata_bytes(build_metadata)
        except RuntimeError:
            expected_canonical_build_metadata_sha256 = ""
        else:
            expected_canonical_build_metadata_sha256 = hashlib.sha256(
                canonical_metadata
            ).hexdigest()
    if expected_base_provenance_sha256 is None:
        expected_base_provenance_sha256 = hashlib.sha256(
            base_provenance_bytes
        ).hexdigest()
    return [
        *findings,
        *_renamed_payload_content_findings(layer_tars, config),
        *graph_findings,
        *(_docker_save_outer_findings(docker_save) if docker_save else []),
        *_config_findings(config),
        *_metadata_findings(build_metadata, observed_config_digest),
        *_base_provenance_findings(base_provenance_bytes, base_provenance),
        *_accepted_lineage_findings(
            build_metadata,
            base_provenance_bytes,
            expected_canonical_build_metadata_sha256,
            expected_base_provenance_sha256,
        ),
        *_private_stage_identity_findings(
            inventory,
            observed_config_digest,
            expected_image_inventory_sha256,
            expected_config_digest,
        ),
    ]


def scan(
    rootfs_tar: Path,
    config: dict[str, Any],
    build_metadata: dict[str, Any],
    base_provenance_bytes: bytes,
    base_provenance: dict[str, Any],
    *,
    observed_config_digest: str,
    expected_image_inventory_sha256: str,
    expected_config_digest: str,
    expected_canonical_build_metadata_sha256: str | None = None,
    expected_base_provenance_sha256: str | None = None,
) -> list[walker.Finding]:
    return scan_tars(
        [rootfs_tar],
        config,
        build_metadata,
        base_provenance_bytes,
        base_provenance,
        observed_config_digest=observed_config_digest,
        expected_image_inventory_sha256=expected_image_inventory_sha256,
        expected_config_digest=expected_config_digest,
        expected_canonical_build_metadata_sha256=(
            expected_canonical_build_metadata_sha256
        ),
        expected_base_provenance_sha256=expected_base_provenance_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--verify-build-oci", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--exported-rootfs", type=Path)
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--build-metadata", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path)
    parser.add_argument("--expected-image-inventory-sha256")
    parser.add_argument("--expected-config-digest")
    parser.add_argument("--expected-canonical-build-metadata-sha256")
    parser.add_argument("--expected-base-provenance-sha256")
    parser.add_argument(
        "--record-image-inventory",
        action="store_true",
        help="Establish the first locally built image identity while running all byte gates",
    )
    parser.add_argument("--image-inventory-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.verify_build_oci:
        if any(
            item
            for item in (
                args.image,
                args.docker_save,
                args.exported_rootfs,
                args.rootfs_tar,
                args.config_json,
                args.base_provenance,
                args.expected_image_inventory_sha256,
                args.expected_config_digest,
                args.expected_canonical_build_metadata_sha256,
                args.expected_base_provenance_sha256,
                args.image_inventory_output,
                args.record_image_inventory,
            )
        ):
            parser.error("--verify-build-oci is a separate validation mode")
        try:
            metadata = json.loads(args.build_metadata.read_bytes())
            findings, evidence = _build_oci_attestation_findings(
                args.verify_build_oci,
                metadata,
            )
        except Exception as exc:  # noqa: BLE001 - unreadable evidence fails closed
            print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
            return 2
        if not findings and "containerimage.config.digest" not in metadata:
            # Persist only after the entire hash-rooted graph and attestations
            # pass; never replace a supplied config identity.
            metadata["containerimage.config.digest"] = evidence["config_digest"]
            args.build_metadata.write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
        result = {
            "format": "npa_libero_build_oci_attestation_scan_v1",
            "status": "pass" if not findings else "fail",
            **evidence,
            "findings": [asdict(item) for item in findings],
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if not findings else 1
    if sum(bool(item) for item in (args.image, args.docker_save, args.rootfs_tar)) != 1:
        parser.error("provide exactly one IMAGE, --docker-save, or --rootfs-tar")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")
    if bool(args.exported_rootfs) != bool(args.docker_save):
        parser.error("--docker-save and --exported-rootfs are required together")
    if args.record_image_inventory and (
        not args.docker_save
        or not args.image_inventory_output
        or any(
            (
                args.expected_image_inventory_sha256,
                args.expected_config_digest,
                args.expected_canonical_build_metadata_sha256,
                args.expected_base_provenance_sha256,
            )
        )
    ):
        parser.error("inventory recording requires a local docker-save and output, without prior identities")
    if any(
        item is None
        for item in (
            args.base_provenance,
            *(() if args.record_image_inventory else (
                args.expected_image_inventory_sha256,
                args.expected_config_digest,
                args.expected_canonical_build_metadata_sha256,
                args.expected_base_provenance_sha256,
            )),
        )
    ):
        parser.error("complete lineage inputs are required for an image scan")
    try:
        metadata = json.loads(args.build_metadata.read_bytes())
        base_provenance_bytes = args.base_provenance.read_bytes()
        base_provenance = json.loads(base_provenance_bytes)
        with tempfile.TemporaryDirectory(prefix="npa-libero-byte-scan-") as temporary:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(temporary))
                exported_rootfs = tars[0]
                docker_save = None
                observed_config_digest = _remote_config_digest(args.image)
            elif args.docker_save:
                tars, config = walker.docker_save_material(
                    args.docker_save, Path(temporary)
                )
                exported_rootfs = args.exported_rootfs
                docker_save = args.docker_save
                observed_config_digest = _docker_save_config_digest(args.docker_save)
            else:
                tars = [args.rootfs_tar]
                exported_rootfs = None
                docker_save = None
                config = (
                    json.loads(args.config_json.read_text(encoding="utf-8"))
                    if args.config_json
                    else {}
                )
                observed_config_digest = (
                    "sha256:"
                    + hashlib.sha256(args.config_json.read_bytes()).hexdigest()
                    if args.config_json
                    else ""
                )
            evidence: dict[str, object] = {}
            if args.record_image_inventory:
                # The initial local build has no previous registry identity. Bind
                # its complete ordered layers and independent rootfs export now;
                # scan_tars below still rejects every payload/lineage finding.
                _, inventory = _layer_graph_findings(tars, exported_rootfs)
                args.expected_image_inventory_sha256 = inventory.sha256
                args.expected_config_digest = metadata["containerimage.config.digest"]
                args.expected_canonical_build_metadata_sha256 = hashlib.sha256(
                    canonical_build_metadata_bytes(metadata)
                ).hexdigest()
                args.expected_base_provenance_sha256 = BASE_PROVENANCE_SHA256
            findings = scan_tars(
                tars,
                config,
                metadata,
                base_provenance_bytes,
                base_provenance,
                exported_rootfs=exported_rootfs,
                docker_save=docker_save,
                observed_config_digest=observed_config_digest,
                expected_image_inventory_sha256=(args.expected_image_inventory_sha256),
                expected_config_digest=args.expected_config_digest,
                expected_canonical_build_metadata_sha256=(
                    args.expected_canonical_build_metadata_sha256
                ),
                expected_base_provenance_sha256=args.expected_base_provenance_sha256,
                inventory_output=args.image_inventory_output,
                evidence=evidence,
            )
    except Exception as exc:  # noqa: BLE001 - every unreadable byte fails closed
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    result = {
        "format": "npa_libero_neutral_image_byte_scan_v1",
        "image": args.image
        or ("docker-save" if args.docker_save else "offline-rootfs"),
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "base_manifest": BASE_MANIFEST,
        "base_rootfs_material": BASE_ROOTFS_MATERIAL,
        "canonical_build_metadata_sha256": args.expected_canonical_build_metadata_sha256,
        "base_provenance_sha256": args.expected_base_provenance_sha256,
        **evidence,
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
