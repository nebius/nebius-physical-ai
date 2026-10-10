#!/usr/bin/env python3
"""Fail closed when a built neutral HY-World image contains runtime payloads."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_ltx_payload as base  # noqa: E402

# HY-World deliberately builds the upstream native extensions at operator runtime,
# so its neutral bootstrap needs the *devel* CUDA base.  This is deliberately not
# the LTX policy: LTX is CPU-only and must reject CUDA in every layer.  The
# ordered, uncompressed diff IDs below are the linux/amd64 rootfs closure of the
# exact digest in HY-World's Dockerfile.  CUDA is tolerated only in those exact
# inherited layers; a CUDA byte introduced by a later layer still fails.
DECLARED_CUDA_BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"
DECLARED_CUDA_BASE_DIGEST = (
    "sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed"
)
DECLARED_CUDA_BASE_DIFF_IDS = (
    "sha256:4b7c01ed0534d4f9be9cf97d068da1598c6c20b26cb6134fad066defdb6d541d",
    "sha256:5d874d849ec64ad89b8e1ff34cefc562238dfaaa6540905f0a8bac73777d4f25",
    "sha256:5ef6645238f3599acf1a7b450ae0a8e9306bea4e342a7af89d3992b804676999",
    "sha256:723554db4ba2a109baacf03b25cff50db586ce45370dfdbd21557171201992f5",
    "sha256:6020d84069cf575521faca653d2817e2195b8233b8c4e2d7bc65f3296296c2bb",
    "sha256:09d8fc1b386fa3eb8491421932e290ce3c33a55265e1d6da8cc9d720a08d3306",
    "sha256:d9ab1aaed362889c6b77ea27e8df132379e24e2a0aacdfaed8fadf3cddb3131d",
    "sha256:9004b01e4a91e9612a7374cce04c721e1e3a8f9e99619de547fb333425c79637",
    "sha256:a2dd2dc8d2b5463fe695696cc9b429d90af7bb8b1ad911859146a38b720f4963",
    "sha256:ac5865c2298fb29741c8243c732ce3eda6c4bf96f3f571cfc0fd4fb800c0a2ea",
    "sha256:ed18f36cfec73e644207a28dc46967159c8307e8afcf2c3bc34ab2e27e2db271",
    "sha256:f52fadb0c4e495831e119100a2ad788c0624e21fd91fddd3a30962fd7aa70363",
)

# The LTX walker is shared traversal code, but its policy table includes
# LTX-specific and CPU-only constraints.  HY retains the generic restricted
# byte checks and supplies a separate CUDA-base-aware policy below.
_GENERIC_PATH_KINDS = {
    "checkpoint_or_weight",
    "nvidia_python_distribution",
    "cuda_library",
    "cuda_tool",
    "torch_distribution",
    "package_cache",
    "credential_file",
}
_GENERIC_FORBIDDEN_PATHS = tuple(
    item for item in base.FORBIDDEN_PATHS if item[0] in _GENERIC_PATH_KINDS
)

FORBIDDEN_PATHS = _GENERIC_FORBIDDEN_PATHS + (
    (
        "hy_world_source_tree",
        re.compile(
            r"(?:^|/)(?:HY-World-2\.0/(?:hyworld2|assets|examples)/|"
            r"hyworld2/(?:worldgen|worldrecon|panogen)/)",
            re.I,
        ),
    ),
    (
        "hy_world_python_distribution",
        re.compile(
            r"(?:^|/)(?:site-packages|dist-packages)/(?:hyworld2(?:/|-)|"
            r"hy[-_]?world[^/]*\.(?:dist-info|egg-info)/)",
            re.I,
        ),
    ),
    (
        "hy_world_or_worldstereo_weight",
        re.compile(
            r"(?:^|/)(?:HY-(?:Pano|WorldMirror)-2\.0|worldstereo-(?:memory|camera)|"
            r"worldstereo-memory-dmd)[^/]*",
            re.I,
        ),
    ),
)

_CUDA_PATH_KINDS = {"cuda_library", "cuda_tool"}
_CUDA_BASE_FORBIDDEN_PATHS = tuple(
    item for item in FORBIDDEN_PATHS if item[0] not in _CUDA_PATH_KINDS
)

FORBIDDEN_HISTORY = (
    (
        "runtime_bootstrap_at_build",
        re.compile(
            r"\bRUN\b[^\n]*\bhy-world-runtime\s+(?:ensure|fetch-models|run-image-to-world)\b",
            re.I | re.S,
        ),
    ),
    (
        "hy_world_fetch_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:"
            r"\b(?:git\s+clone|curl|wget|aria2c)\b[^\n]*"
            r"Tencent-Hunyuan/HY-World-2\.0|"
            r"\b(?:pip|uv)\s+(?:pip\s+)?install\b[^\n]*"
            r"Tencent-Hunyuan/HY-World-2\.0|"
            r"\b(?:hf|huggingface-cli)\s+download\s+(?:tencent/HY-World-2\.0|"
            r"hanshanxue/WorldStereo|Qwen/Qwen-Image-Edit-2509))",
            re.I | re.S,
        ),
    ),
)

# These are exact Ubuntu 24.04 / NVIDIA CUDA-base bytes that contain
# credential-shaped parser/test literals.  They are not path exceptions: a byte
# change fails closed.  The base layer prefix above binds the first group to the
# declared CUDA digest; the second group was independently package-provenanced
# after the bootstrap installs ffmpeg/OpenSSH for decoding and SkyPilot.
CUDA_BASE_AUDITED_SECRET_LITERAL_FILE_SHA256 = {
    "usr/lib/x86_64-linux-gnu/libgnutls.so.30.37.1": (
        "2e644667e5e61567648ef912ea5b54931fee3c3cfc85979dc6b6a575fb306621"
    ),
    "usr/lib/x86_64-linux-gnu/libunistring.so.5.0.0": (
        "e08e4cf4c4d57cbdccea128d1b9de58cfa24be54c091e5c70b02ba44076c6585"
    ),
    "opt/nvidia/nsight-compute/2025.1.1/host/linux-desktop-glibc_2_11_3-x64/Plugins/tls/libqopensslbackend.so": (
        "7a32df2020b882b75e6597f802fc4680089b7525680d85628f388d0807dde8eb"
    ),
    "opt/nvidia/nsight-compute/2025.1.1/host/linux-desktop-glibc_2_11_3-x64/libQt6Network.so.6": (
        "91760f60542f5e8397103a9466b8695320e7ddc854fd6ed977777c13d6809252"
    ),
    "opt/nvidia/nsight-compute/2025.1.1/host/linux-desktop-glibc_2_11_3-x64/libssh.so": (
        "38df6e0e116bed94762673d967682b373da61eb8f6fec90d0561761c282138ef"
    ),
}

RUNTIME_AUDITED_SECRET_LITERAL_FILE_SHA256 = {
    "usr/bin/ssh": "31430165113e51a046f3b9fa2a60ec98e073cc6c0411a081b3b1d76ed6a4eb6a",
    "usr/bin/ssh-add": (
        "d8f8bc11af29c8f61b545191c981b1ea9d1c6c2bfdf61f4841ad5588af783067"
    ),
    "usr/bin/ssh-keygen": (
        "5175ddce2146fc8a03ab8e1ef25a1b0382dd3cb209484f7ae11ac782171c0d04"
    ),
    "usr/lib/openssh/ssh-keysign": (
        "7a7bf85b1dbe297da7508fc389d1ff27c54678d502e83214a7f9040d86ac4022"
    ),
    "usr/sbin/sshd": "f30333b455e6fe10536d544992ea3dfa87cc1d83b8af7945a815c86f8b4f6f6d",
    "usr/lib/x86_64-linux-gnu/libmbedcrypto.so.2.28.8": (
        "6a8310e17b6825987ba3b95fbb57b1c23f8098d5887c62ed813bc723e2b1d55b"
    ),
    "usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.9.6": (
        "079b5af8caf1af1c667d55a005948fcff92aeb1ed905392488f61375af9bf7fc"
    ),
    "usr/lib/x86_64-linux-gnu/libssh.so.4.9.6": (
        "b82bd54edb908605b8142be1a6d242df65501af0e9bb7581cc36ebf4d8a5ecfd"
    ),
}

RUNTIME_AUDITED_LITERAL_LIBRARY_SHA256 = {
    "usr/lib/x86_64-linux-gnu/libavcodec.so.60.31.102": (
        "8231c9aecba7d02c132de0e16195ccb3e8002a235a5f364b3635df09e7f513e0"
    ),
    "usr/lib/x86_64-linux-gnu/libavutil.so.58.29.100": (
        "11b0836441b899e1b2183fc10eda5a5f637c03547fdfe5b97774e2cbd5aeb72c"
    ),
}

_NO_CUDA_ELF_DEPENDENCY = re.compile(rb"(?!)")


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[base.walker.Finding]:
    """Scan one root filesystem tar under the HY-World no-payload policy."""

    return scan_tars([rootfs_tar], config)


def _layer_diff_id(path: Path) -> str:
    """Return an OCI diff ID over a raw or gzip-compressed Docker-save layer.

    Docker-save retains the compressed OCI blob.  OCI config ``diff_ids`` bind
    its uncompressed tar stream, so hashing the saved gzip member itself would
    reject every genuine compressed layer.  ``gzip.GzipFile`` also validates the
    compressed stream checksum while draining it; malformed compression fails
    closed instead of becoming an identity bypass.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        magic = stream.read(3)
        stream.seek(0)
        if magic == b"\x1f\x8b\x08":
            with gzip.GzipFile(fileobj=stream, mode="rb") as expanded:
                for block in iter(lambda: expanded.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _config_diff_ids(config: dict[str, Any]) -> tuple[str, ...] | None:
    """Read a complete ordered rootfs record, or retain rootfs-free unit mode."""

    rootfs = config.get("rootfs")
    if rootfs is None:
        return None
    if not isinstance(rootfs, dict) or rootfs.get("type") != "layers":
        return ()
    diff_ids = rootfs.get("diff_ids")
    if not isinstance(diff_ids, list) or not all(
        isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        for value in diff_ids
    ):
        return ()
    return tuple(diff_ids)


def _scan_layers(
    tars: list[Path],
    *,
    forbidden_paths: tuple[tuple[str, re.Pattern[str]], ...],
    audited_secret_files: dict[str, str],
    audited_libraries: dict[str, str],
    forbidden_elf_dependency: re.Pattern[bytes] | None = None,
) -> list[base.walker.Finding]:
    """Scan layers without re-running OCI history for every policy segment."""

    policy: dict[str, Any] = {
        "forbidden_paths": forbidden_paths,
        "forbidden_history": FORBIDDEN_HISTORY,
        "audited_secret_files": audited_secret_files,
        "audited_libraries": audited_libraries,
    }
    if forbidden_elf_dependency is not None:
        policy["forbidden_elf_dependency"] = forbidden_elf_dependency
    with base.walker.payload_policy(**policy):
        return base.walker.scan_tars(tars, {})


def _scan_history(config: dict[str, Any]) -> list[base.walker.Finding]:
    """Scan OCI metadata once under the HY policy after every layer is checked."""

    with base.walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files=RUNTIME_AUDITED_SECRET_LITERAL_FILE_SHA256,
        audited_libraries=RUNTIME_AUDITED_LITERAL_LIBRARY_SHA256,
    ):
        return base.walker.scan_tars([], config)


def _verified_layer_diff_ids(
    tars: list[Path],
    configured_diff_ids: tuple[str, ...],
) -> tuple[str, ...] | list[base.walker.Finding]:
    """Bind every compressed-or-raw layer to the OCI config, or fail closed."""

    actual_diff_ids = tuple(_layer_diff_id(tar_path) for tar_path in tars)
    if not configured_diff_ids or len(configured_diff_ids) != len(actual_diff_ids):
        return [
            base.walker.Finding(
                "invalid_layer_binding",
                "<oci-config>",
                "rootfs.diff_ids is missing, malformed, or does not cover every layer",
            )
        ]
    if actual_diff_ids != configured_diff_ids:
        return [
            base.walker.Finding(
                "layer_diff_id_mismatch",
                "<oci-config>",
                "saved layer bytes do not match OCI rootfs.diff_ids",
            )
        ]
    return actual_diff_ids


def _scan_bound_layers(tars: list[Path], base_count: int) -> list[base.walker.Finding]:
    """Scan the pinned CUDA base and later layers under their separate rules."""

    findings = _scan_layers(
        tars[:base_count],
        forbidden_paths=_CUDA_BASE_FORBIDDEN_PATHS,
        audited_secret_files=CUDA_BASE_AUDITED_SECRET_LITERAL_FILE_SHA256,
        audited_libraries={},
        forbidden_elf_dependency=_NO_CUDA_ELF_DEPENDENCY,
    )
    findings.extend(
        _scan_layers(
            tars[base_count:],
            forbidden_paths=FORBIDDEN_PATHS,
            audited_secret_files=RUNTIME_AUDITED_SECRET_LITERAL_FILE_SHA256,
            audited_libraries=RUNTIME_AUDITED_LITERAL_LIBRARY_SHA256,
        )
    )
    return findings


def scan_tars(
    tars: list[Path],
    config: dict[str, Any],
) -> list[base.walker.Finding]:
    """Scan every layer and history, binding CUDA acceptance to the pinned base.

    Unit callers may provide a rootfs tar without OCI rootfs metadata.  Real OCI
    material must instead bind every saved layer's uncompressed tar stream to
    ``rootfs.diff_ids`` and bind its config prefix to the exact declared base.
    This keeps deleted
    ancestor payload visible while preventing a later Dockerfile instruction
    from smuggling CUDA, weights, source, or credentials behind the base rule.
    """

    configured_diff_ids = _config_diff_ids(config)
    if configured_diff_ids is None:
        return _scan_layers(
            tars,
            forbidden_paths=FORBIDDEN_PATHS,
            audited_secret_files=RUNTIME_AUDITED_SECRET_LITERAL_FILE_SHA256,
            audited_libraries=RUNTIME_AUDITED_LITERAL_LIBRARY_SHA256,
        ) + _scan_history(config)

    actual_diff_ids = _verified_layer_diff_ids(tars, configured_diff_ids)
    if isinstance(actual_diff_ids, list):
        return actual_diff_ids
    base_count = len(DECLARED_CUDA_BASE_DIFF_IDS)
    if actual_diff_ids[:base_count] != DECLARED_CUDA_BASE_DIFF_IDS:
        return [
            base.walker.Finding(
                "declared_cuda_base_mismatch",
                "<oci-config>",
                "CUDA bytes are not bound to the declared pinned CUDA base",
            )
        ]

    # CUDA is allowed only for exact inherited base layers.  All base files,
    # including deleted Nsight ancestor bytes, still receive path, secret and
    # nested-archive inspection.  CUDA ELF linkage is the one base-specific
    # semantic that is expected from this compiler/runtime closure.
    findings = _scan_bound_layers(tars, base_count)
    findings.extend(_scan_history(config))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if sum(bool(item) for item in (args.image, args.rootfs_tar, args.docker_save)) != 1:
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")
    try:
        with tempfile.TemporaryDirectory(prefix="npa-hy-world-byte-scan-") as temporary:
            if args.image:
                tars, config = base.walker.remote_material(args.image, Path(temporary))
                # The shared remote material includes a merged export followed
                # by raw layers.  HY's base proof requires the latter only.
                if len(tars) < 2:
                    raise RuntimeError("remote image did not return raw OCI layers")
                tars = tars[1:]
            elif args.docker_save:
                tars, config = base.docker_save_material(
                    args.docker_save, Path(temporary)
                )
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - an incomplete scan must fail closed
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    result = {
        "format": "npa_hy_world_image_byte_scan_v1",
        "image": args.image
        or ("docker-save" if args.docker_save else "offline-rootfs"),
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
    raise SystemExit(main())
