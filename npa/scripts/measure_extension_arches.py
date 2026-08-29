#!/usr/bin/env python3
"""measure_extension_arches.py - read the CUDA architectures a compiled
extension actually contains, offline.

WHY THIS EXISTS

validate_blackwell_image.sh reports torch._C._cuda_getArchFlags(), which is the
*torch* wheel's fat-binary arch set. It says nothing about separately installed
CUDA extensions - natten, flash-attn, transformer-engine, custom ops - and those
carry their own, often narrower, arch sets. npa-cosmos is the worked example: its
L40S / sm_120 / B300 cells were recorded as blocked by an upstream compute-
capability allowlist, and the open question was whether the pinned NATTEN wheel
even had kernels for those parts. That question is answerable from the wheel
bytes, with no GPU and no Docker.

WHAT IT MEASURES

nvcc emits fat-binary containers (magic 0xBA55ED50) into each object. This
parses their entry headers directly: kind at +0 (1 = PTX, 2 = ELF/SASS), header
size at +4, payload size at +8, SM version at +28. Reading the headers rather
than scanning for ELF magic also counts compressed payloads, which an ELF scan
silently misses.

HOW TO READ THE RESULT

SASS is per-architecture and does not cross a CUDA major, but within a major it
is forward compatible: sm_86 runs on sm_89 (L40S), and sm_100 runs on sm_103
(B300). PTX crosses majors because the driver JITs it. So an extension with
sm_100 SASS and no PTX reaches B200 natively and B300 by forward compatibility,
while an extension whose highest entry is sm_90 reaches no Blackwell part at all.

Coverage is a necessary condition, never a sufficient one. A kernel can be
present and still fail - flash-attn-4 ships sm_120 SASS and its CuTe forward
pass raises on sm_120 because the epilogue needs TMA. Only a real capability run
on the part decides a cell.

USAGE
  measure_extension_arches.py <wheel|.so|directory> [...] [--require sm_100]
                             [--min-size-mb N] [--json]

EXAMPLES
  # The pinned Cosmos Predict2 dependency wheels, straight from the release:
  curl -sLO https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/natten-0.21.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl
  measure_extension_arches.py natten-0.21.0+cu128.torch27-cp310-cp310-linux_x86_64.whl

  # Gate a build: fail unless every extension can reach B200.
  measure_extension_arches.py /opt/cosmos/venv/lib/python3.10/site-packages \\
    --require sm_100
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

FATBIN_MAGIC = struct.pack("<I", 0xBA55ED50)
FATBIN_HEADER_SIZE = 16
ENTRY_KIND_PTX = 1
# A container header larger than this is a false positive on the magic bytes
# rather than a real entry, so stop walking instead of trusting the offsets.
MAX_ENTRY_HEADER = 4096


def scan(blob: bytes) -> tuple[Counter, Counter]:
    """Return (sass, ptx) counters keyed by SM version for one binary."""

    sass: Counter = Counter()
    ptx: Counter = Counter()
    offset = 0
    while True:
        offset = blob.find(FATBIN_MAGIC, offset)
        if offset < 0:
            return sass, ptx
        header_size, = struct.unpack_from("<H", blob, offset + 6)
        fat_size, = struct.unpack_from("<Q", blob, offset + 8)
        if header_size != FATBIN_HEADER_SIZE or fat_size <= 0:
            offset += 4
            continue
        if offset + FATBIN_HEADER_SIZE + fat_size > len(blob):
            offset += 4
            continue
        cursor = offset + header_size
        end = cursor + fat_size
        while cursor + 64 <= end:
            kind, = struct.unpack_from("<H", blob, cursor)
            entry_header, = struct.unpack_from("<I", blob, cursor + 4)
            payload, = struct.unpack_from("<Q", blob, cursor + 8)
            arch, = struct.unpack_from("<I", blob, cursor + 28)
            if entry_header == 0 or entry_header > MAX_ENTRY_HEADER:
                break
            (ptx if kind == ENTRY_KIND_PTX else sass)[arch] += 1
            cursor += entry_header + payload
        offset = end


def _binaries(target: Path, min_size: int):
    """Yield (label, bytes) for every native binary reachable from target."""

    if target.is_dir():
        for path in sorted(target.rglob("*.so*")):
            if path.is_file() and path.stat().st_size >= min_size:
                yield str(path), path.read_bytes()
        return
    if target.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(target) as archive:
            members = [
                info
                for info in archive.infolist()
                if ".so" in info.filename and info.file_size >= min_size
            ]
            for info in sorted(members, key=lambda item: -item.file_size):
                with archive.open(info) as handle:
                    yield f"{target.name}:{info.filename}", handle.read()
        return
    yield str(target), target.read_bytes()


def _fmt(counter: Counter) -> list[str]:
    return [f"sm_{arch}" for arch in sorted(counter)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="sm_NNN",
        help="fail unless every measured binary carries this SASS architecture",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=1.0,
        help="skip binaries smaller than this (default: 1 MB)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    min_size = int(args.min_size_mb * 1_000_000)
    required = {name if name.startswith("sm_") else f"sm_{name}" for name in args.require}
    report: dict[str, dict] = {}
    failures: list[str] = []

    for target in args.targets:
        if not target.exists():
            print(f"ERROR: no such path: {target}", file=sys.stderr)
            return 2
        for label, blob in _binaries(target, min_size):
            sass, ptx = scan(blob)
            entry = {
                "bytes": len(blob),
                "sass": _fmt(sass),
                "ptx": [f"compute_{arch}" for arch in sorted(ptx)],
            }
            report[label] = entry
            missing = sorted(required - set(entry["sass"]))
            if missing:
                entry["missing"] = missing
                failures.append(f"{label} lacks {', '.join(missing)}")
            if not args.json:
                print(f"{label} ({len(blob) / 1e6:.0f} MB)")
                print(f"  SASS: {' '.join(entry['sass']) or 'none'}")
                print(f"  PTX:  {' '.join(entry['ptx']) or 'none'}")

    if not report:
        print("ERROR: no native binaries found to measure", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if failures:
        print("\nMISSING REQUIRED ARCHITECTURES", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
