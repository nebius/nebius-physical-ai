#!/usr/bin/env python3
"""Verify every SAM 3.1 image layer excludes upstream source, weights and CUDA."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_image_ltx_payload as neutral  # noqa: E402
import scan_image_wan_payload as walker  # noqa: E402

SAM_PATHS = (
    (
        "sam_source",
        re.compile(
            r"(?:^|/)sam3/(?:model/|sam/|train/|perflib/|assets/|__init__\.py$)|"
            r"(?:site|dist)-packages/sam3(?:/|[^/]*\.(?:dist|egg)-info/)",
            re.I,
        ),
    ),
    ("sam_checkpoint", re.compile(r"(?:^|/)sam3[^/]*\.(?:pt|pth|safetensors)$", re.I)),
    ("source_checkout", re.compile(r"(?:^|/)\.git/(?:objects/|index$)", re.I)),
)
SAM_HISTORY = (
    (
        "sam_fetch_at_build",
        re.compile(
            r"\bRUN\b.*(?:sam3-runtime\s+(?:ensure|segment|exec)|"
            r"git\s+(?:clone|fetch).*facebookresearch/sam3|"
            r"(?:pip|uv)\s+(?:pip\s+)?install[^\n]*facebookresearch/sam3)",
            re.I,
        ),
    ),
)


def _scan(tars: list[Path], config: dict) -> list:
    with walker.payload_policy(
        forbidden_paths=neutral.FORBIDDEN_PATHS + SAM_PATHS,
        forbidden_history=neutral.FORBIDDEN_HISTORY + SAM_HISTORY,
        audited_secret_files=neutral.AUDITED_SECRET_LITERAL_FILE_SHA256,
        audited_libraries=neutral.AUDITED_LITERAL_LIBRARY_SHA256,
    ):
        return walker.scan_tars(tars, config)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="npa-sam3-byte-scan-") as temporary:
        layers, config = walker.docker_save_material(args.docker_save, Path(temporary))
        findings = _scan(layers, config)
    result = {
        "format": "npa_sam3_image_byte_scan_v1",
        "archives_scanned": len(layers),
        "status": "fail" if findings else "pass",
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(_main())
