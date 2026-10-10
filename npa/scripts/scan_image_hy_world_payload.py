#!/usr/bin/env python3
"""Fail closed when a built neutral HY-World image contains runtime payloads."""

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

import scan_image_ltx_payload as base  # noqa: E402

FORBIDDEN_PATHS = base.FORBIDDEN_PATHS + (
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

FORBIDDEN_HISTORY = base.FORBIDDEN_HISTORY + (
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
            r"\bRUN\b[^\n]*(?:Tencent-Hunyuan/HY-World-2\.0|"
            r"\b(?:hf|huggingface-cli)\s+download\s+(?:tencent/HY-World-2\.0|"
            r"hanshanxue/WorldStereo|Qwen/Qwen-Image-Edit-2509))",
            re.I | re.S,
        ),
    ),
)


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[base.walker.Finding]:
    """Scan one root filesystem tar under the HY-World no-payload policy."""

    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[base.walker.Finding]:
    """Scan every layer and image history; deleted later-layer payload still fails."""

    with base.walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files=base.AUDITED_SECRET_LITERAL_FILE_SHA256,
        audited_libraries=base.AUDITED_LITERAL_LIBRARY_SHA256,
    ):
        return base.walker.scan_tars(tars, config)


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
