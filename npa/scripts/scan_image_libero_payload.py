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
import re
import sys
import tarfile
import tempfile
from dataclasses import asdict
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

FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "libero_source_outside_neutral_bootstrap",
        re.compile(
            r"(?:^|/)(?:opt/byof|opt/(?:libero|robomimic|robosuite)|"
            r"usr/src/(?:libero|robomimic|robosuite)|"
            r"workspace/(?:libero|robomimic|robosuite))(?:/|$)",
            re.I,
        ),
    ),
    (
        "torch_runtime_outside_python_distribution",
        re.compile(r"(?:^|/)(?:opt|usr/local|workspace)/(?:pytorch|torch)(?:/|$)", re.I),
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
            r"[^/]*(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|cupti|cufile|"
            r"cusparse|cusolver|curand|cufft|nvidia)[^/]*\.so(?:\.[0-9]+)*)",
            re.I,
        ),
    ),
    (
        "model_weight_checkpoint_or_dataset",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|pth|hdf5)$|(?:^|/)(?:pytorch_model|model|weights?|checkpoint)[^/]*\.bin$)",
            re.I,
        ),
    ),
    (
        "libero_task_or_render_asset",
        re.compile(
            r"(?:^|/)(?:libero/libero/(?:assets|bddl_files|init_files)|[^/]+\.(?:bddl|init))(?=/|$)",
            re.I,
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
            r"kubeconfig|runtime-context\.json|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
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

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"lib(?:[A-Za-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|"
    rb"cupti|cufile|cusparse|cusolver|curand|cufft|nvidia)[A-Za-z0-9_.-]*\.so",
    re.I,
)

_ARCHIVE_METADATA_FILES = frozenset(
    {"manifest.json", "repositories", "index.json", "oci-layout"}
)


def _config_findings(config: dict[str, Any]) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    runtime = config.get("config") or {}
    labels = runtime.get("Labels") or {}
    expected_labels = {
        "org.nebius.npa.redistribution": "public-neutral-bootstrap",
        "org.nebius.npa.validation-status": "quarantined-unvalidated",
        "org.nebius.npa.base-manifest": BASE_MANIFEST,
        "org.nebius.npa.base-rootfs-material": BASE_ROOTFS_MATERIAL,
        "org.nebius.npa.skypilot-bootstrap-contract": BOOTSTRAP_CONTRACT,
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
    metadata: dict[str, Any], config: dict[str, Any]
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    provenance = metadata.get("buildx.build.provenance")
    serialized = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    labels = (config.get("config") or {}).get("Labels") or {}
    revision = labels.get("org.opencontainers.image.revision")
    for label, expected in (
        ("base manifest", BASE_MANIFEST),
        ("source revision", revision),
    ):
        if not expected or str(expected) not in serialized:
            findings.append(
                walker.Finding(
                    "independent_build_lineage",
                    "<buildx-metadata>",
                    f"independently observed {label} is absent",
                )
            )
    config_digest = str(metadata.get("containerimage.config.digest") or "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", config_digest) is None:
        findings.append(
            walker.Finding(
                "independent_build_lineage",
                "<buildx-metadata>",
                "container config digest is absent",
            )
        )
    return findings


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
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or candidate.startswith("/"):
        raise ValueError("archive member path is empty or absolute")
    parts = candidate.rstrip("/").split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("archive member path is not canonical and confined")
    return "/".join(parts)


def _uncompressed_tar_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if payload.startswith(b"\x1f\x8b"):
        return gzip.decompress(payload)
    if payload.startswith(b"BZh"):
        return bz2.decompress(payload)
    if payload.startswith(b"\xfd7zXZ\x00"):
        return lzma.decompress(payload)
    if payload.startswith(b"\x28\xb5\x2f\xfd"):
        if walker.zstd is None:
            raise RuntimeError("zstd archive cannot be accounted without zstandard")
        with walker.zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
            return reader.read()
    return payload


def _archive_records(
    path: Path,
) -> tuple[list[tuple[tarfile.TarInfo, str, tuple[object, ...]]], list[walker.Finding]]:
    payload = _uncompressed_tar_bytes(path)
    findings: list[walker.Finding] = []
    if len(payload) % tarfile.BLOCKSIZE:
        findings.append(
            walker.Finding(
                "unaccounted_archive_bytes",
                path.name,
                "decompressed tar length is not block aligned",
            )
        )
    records: list[tuple[tarfile.TarInfo, str, tuple[object, ...]]] = []
    cursor = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive:
            try:
                normalized = _normalized_tar_path(member.name)
            except ValueError as exc:
                findings.append(
                    walker.Finding("unsafe_archive_member", member.name, str(exc))
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
            if any(payload[cursor : member.offset]):
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
            if data_end > len(payload):
                findings.append(
                    walker.Finding(
                        "truncated_archive_member",
                        normalized,
                        f"member exceeds {path.name}",
                    )
                )
                continue
            if any(payload[data_end:padded_end]):
                findings.append(
                    walker.Finding(
                        "nonzero_archive_padding",
                        normalized,
                        f"member padding contains bytes in {path.name}",
                    )
                )
            descriptor: tuple[object, ...]
            if member.isfile():
                content = payload[member.offset_data:data_end]
                descriptor = ("file", member.size, hashlib.sha256(content).hexdigest())
            elif member.isdir():
                descriptor = ("directory",)
            elif member.issym():
                descriptor = ("symlink", member.linkname)
            elif member.islnk():
                try:
                    target = _normalized_tar_path(member.linkname)
                except ValueError as exc:
                    findings.append(
                        walker.Finding("unsafe_archive_hardlink", normalized, str(exc))
                    )
                    target = ""
                descriptor = ("hardlink", target)
            elif member.ischr() and member.devmajor == 0 and member.devminor == 0:
                descriptor = ("whiteout-device", member.size)
            else:
                descriptor = ("unsupported", member.type.decode(errors="replace"))
                findings.append(
                    walker.Finding(
                        "unsupported_archive_member",
                        normalized,
                        f"unsupported tar member type in {path.name}",
                    )
                )
            records.append((member, normalized, descriptor))
            cursor = padded_end
    if any(payload[cursor:]):
        findings.append(
            walker.Finding(
                "unaccounted_archive_bytes",
                path.name,
                "nonzero bytes remain after the final accounted member",
            )
        )
    return records, findings


def _apply_layer_records(
    records: list[tuple[tarfile.TarInfo, str, tuple[object, ...]]],
    state: dict[str, tuple[object, ...]],
    *,
    source: str,
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    seen: set[str] = set()
    for member, path, descriptor in records:
        if path in seen:
            findings.append(
                walker.Finding(
                    "duplicate_archive_member", path, f"duplicate path in {source}"
                )
            )
        seen.add(path)
        parent, _, name = path.rpartition("/")
        if name.startswith(".wh."):
            valid_whiteout = (
                (member.isfile() and member.size == 0)
                or descriptor == ("whiteout-device", 0)
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


def _layer_graph_findings(
    layer_tars: list[Path], exported_rootfs: Path | None = None
) -> list[walker.Finding]:
    findings: list[walker.Finding] = []
    state: dict[str, tuple[object, ...]] = {}
    for layer in layer_tars:
        records, archive_findings = _archive_records(layer)
        findings.extend(archive_findings)
        findings.extend(_apply_layer_records(records, state, source=layer.name))
    flattened, hardlink_findings = _resolve_hardlinks(state)
    findings.extend(hardlink_findings)
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
        records, archive_findings = _archive_records(exported_rootfs)
        findings.extend(archive_findings)
        exported: dict[str, tuple[object, ...]] = {}
        findings.extend(
            _apply_layer_records(records, exported, source=exported_rootfs.name)
        )
        exported, hardlink_findings = _resolve_hardlinks(exported)
        findings.extend(hardlink_findings)
        comparable = {
            path: value for path, value in flattened.items() if value[0] != "directory"
        }
        exported_comparable = {
            path: value for path, value in exported.items() if value[0] != "directory"
        }
        if comparable != exported_comparable:
            findings.append(
                walker.Finding(
                    "flattened_rootfs_mismatch",
                    "<exported-rootfs>",
                    "exported filesystem differs from the ordered OCI layers",
                )
            )
    return findings


def _docker_save_outer_findings(path: Path) -> list[walker.Finding]:
    records, findings = _archive_records(path)
    with tarfile.open(path, "r:*") as archive:
        manifest_stream = archive.extractfile(archive.getmember("manifest.json"))
        if manifest_stream is None:
            raise RuntimeError("docker save archive has no readable manifest.json")
        manifests = json.load(io.TextIOWrapper(manifest_stream, encoding="utf-8"))
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise RuntimeError("docker save archive must contain exactly one image")
    manifest = manifests[0]
    allowed = {
        *_ARCHIVE_METADATA_FILES,
        str(manifest.get("Config") or ""),
        *(str(item) for item in (manifest.get("Layers") or [])),
    }
    for member, normalized, descriptor in records:
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


def scan_tars(
    tars: list[Path],
    config: dict[str, Any],
    build_metadata: dict[str, Any],
    base_provenance_bytes: bytes,
    base_provenance: dict[str, Any],
    *,
    exported_rootfs: Path | None = None,
    docker_save: Path | None = None,
) -> list[walker.Finding]:
    original_secret_content = walker.SECRET_CONTENT
    original_elf_dependency = walker.FORBIDDEN_ELF_DEPENDENCY
    try:
        walker.SECRET_CONTENT = SECRET_CONTENT
        walker.FORBIDDEN_ELF_DEPENDENCY = FORBIDDEN_ELF_DEPENDENCY
        with walker.payload_policy(
            forbidden_paths=FORBIDDEN_PATHS,
            forbidden_history=FORBIDDEN_HISTORY,
            audited_secret_files={},
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
    return [
        *findings,
        *_layer_graph_findings(layer_tars, exported_rootfs),
        *(_docker_save_outer_findings(docker_save) if docker_save else []),
        *_config_findings(config),
        *_metadata_findings(build_metadata, config),
        *_base_provenance_findings(base_provenance_bytes, base_provenance),
    ]


def scan(
    rootfs_tar: Path,
    config: dict[str, Any],
    build_metadata: dict[str, Any],
    base_provenance_bytes: bytes,
    base_provenance: dict[str, Any],
) -> list[walker.Finding]:
    return scan_tars(
        [rootfs_tar],
        config,
        build_metadata,
        base_provenance_bytes,
        base_provenance,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--build-metadata", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if sum(bool(item) for item in (args.image, args.docker_save, args.rootfs_tar)) != 1:
        parser.error("provide exactly one IMAGE, --docker-save, or --rootfs-tar")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")
    try:
        metadata = json.loads(args.build_metadata.read_text(encoding="utf-8"))
        base_provenance_bytes = args.base_provenance.read_bytes()
        base_provenance = json.loads(base_provenance_bytes)
        with tempfile.TemporaryDirectory(prefix="npa-libero-byte-scan-") as temporary:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(temporary))
                exported_rootfs = tars[0]
                docker_save = None
            elif args.docker_save:
                tars, config = walker.docker_save_material(
                    args.docker_save, Path(temporary)
                )
                exported_rootfs = None
                docker_save = args.docker_save
            else:
                tars = [args.rootfs_tar]
                exported_rootfs = None
                docker_save = None
                config = (
                    json.loads(args.config_json.read_text(encoding="utf-8"))
                    if args.config_json
                    else {}
                )
            findings = scan_tars(
                tars,
                config,
                metadata,
                base_provenance_bytes,
                base_provenance,
                exported_rootfs=exported_rootfs,
                docker_save=docker_save,
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
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
