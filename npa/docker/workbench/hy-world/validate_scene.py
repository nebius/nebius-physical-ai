#!/usr/bin/env python3
"""Validate a completed HY-World image-to-world run and write evidence JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asset_contract import HyWorldEvidenceError, write_evidence  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--runtime-metadata", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metadata = json.loads(args.runtime_metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise HyWorldEvidenceError("runtime metadata must be a JSON object")
        evidence = write_evidence(
            args.artifact,
            scene_dir=args.scene_dir,
            result_dir=args.result_dir,
            input_image=args.input_image,
            runtime_metadata=metadata,
        )
    except (OSError, json.JSONDecodeError, HyWorldEvidenceError) as exc:
        print(f"npa-hy-world: {exc}", file=sys.stderr)
        return 70
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
