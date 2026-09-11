#!/usr/bin/env python3
"""Fail closed when a RoboTwin bootstrap contains any runtime/vendor payload."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "robotwin_source",
        re.compile(
            r"(?:^|/)(?:opt|workspace|src)/(?:RoboTwin|robotwin-source)(?:/|$)",
            re.I,
        ),
    ),
    (
        "curobo_source_or_runtime",
        re.compile(r"(?:^|/)(?:opt/)?curobo(?:/|$)", re.I),
    ),
    (
        "vendor_python_runtime",
        re.compile(
            r"(?:^|/)site-packages/(?:curobo|sapien|mplib|warp|torch|torchvision|"
            r"nvidia|cv2|h5py)(?:[./_-]|/|$)",
            re.I,
        ),
    ),
    (
        "cuda_or_cudnn_runtime",
        re.compile(
            r"(?:^|/)(?:usr/local/cuda(?:/|$)|[^/]*(?:lib)?cu(?:da|dnn|blas|fft|"
            r"rand|solver|sparse|pti|rtc)[^/]*\.so(?:[./0-9]*|$))",
            re.I,
        ),
    ),
    (
        "robotwin_asset_archive",
        re.compile(r"(?:^|/)(?:embodiments|objects|background_texture)\.zip$", re.I),
    ),
    (
        "robotwin_extracted_asset",
        re.compile(
            r"(?:^|/)(?:assets/)?(?:embodiments|objects)/[^/]+(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_huggingface_cache",
        re.compile(
            r"(?:^|/)datasets--TianxingChen--RoboTwin2\.0(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_runtime_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:huggingface|torch|warp)|robotwin-runtime-cache|"
            r"runtime-ready\.json)(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_generated_output",
        re.compile(
            r"(?:^|/)(?:robotwin-native(?:/|$)|robotwin-smoke\.json$|"
            r"episode_[0-9]+\.(?:hdf5|mp4)$|[^/]+\.(?:hdf5|mp4)$|frames?(?:/|$))",
            re.I,
        ),
    ),
    (
        "credential_or_manager_context",
        re.compile(
            r"(?:^|/)(?:runtime-context\.json|kubeconfig(?:\.ya?ml)?|"
            r"docker/config\.json|\.aws/credentials)$",
            re.I,
        ),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "robotwin_asset_fetch_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:hf_hub_download|(?:hf|huggingface-cli)\s+download|"
            r"(?:curl|wget)\b)[^\n]*(?:TianxingChen/RoboTwin2\.0|"
            r"embodiments\.zip|objects\.zip|background_texture\.zip)",
            re.I | re.S,
        ),
    ),
    (
        "vendor_runtime_installed_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:pip|uv)\s+(?:install|sync)[^\n]*"
            r"(?:curobo|sapien|mplib|warp-lang|torch|nvidia-cuda|nvidia-cudnn)",
            re.I | re.S,
        ),
    ),
    (
        "vendor_source_fetched_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:git\s+clone|curl|wget)[^\n]*"
            r"(?:RoboTwin-Platform/RoboTwin|NVlabs/curobo)",
            re.I | re.S,
        ),
    ),
    (
        "nvidia_or_pytorch_base",
        re.compile(r"\bFROM\s+(?:nvidia/cuda|nvcr\.io/|pytorch/)", re.I),
    ),
)

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"(?:libcuda|libcudnn|libcublas|libcudart|libnvrtc|libtorch)[^\x00]*\.so",
    re.I,
)


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one tar plus image history under the RoboTwin boundary policy."""

    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan layer/rootfs tars plus image history under the RoboTwin policy."""

    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files={},
        audited_libraries={},
        secret_content=(),
        forbidden_elf_dependency=FORBIDDEN_ELF_DEPENDENCY,
    ):
        return walker.scan_tars(tars, config)


docker_save_material = walker.docker_save_material


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    choices = (args.image, args.rootfs_tar, args.docker_save)
    if sum(bool(value) for value in choices) != 1:
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-robotwin-byte-scan-") as tmp:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(tmp))
            elif args.docker_save:
                tars, config = docker_save_material(args.docker_save, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - every scan failure is fatal
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_robotwin_image_byte_scan_v1",
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
