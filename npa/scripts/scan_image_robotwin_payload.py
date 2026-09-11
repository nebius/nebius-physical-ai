#!/usr/bin/env python3
"""Fail closed when a built RoboTwin image contains runtime-only bytes.

RoboTwin source and the authorized CUDA/cuDNN/CuRobo runtime are deliberately
baked into an operator-private image. The official RoboTwin asset archives,
their extracted trees, download caches, and generated episodes are not. This
scanner verifies that narrower byte-boundary claim against the flattened
rootfs, every image layer, and OCI build history.
"""

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
            r"(?:^|/)(?:\.cache/huggingface|huggingface/hub)/"
            r"datasets--TianxingChen--RoboTwin2\.0(?:/|$)",
            re.I,
        ),
    ),
    (
        "robotwin_generated_output",
        re.compile(
            r"(?:^|/)(?:robotwin-native(?:/|$)|robotwin-smoke\.json$|"
            r"episode_[0-9]+\.(?:hdf5|mp4)$)",
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
        forbidden_elf_dependency=re.compile(rb"(?!)"),
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
