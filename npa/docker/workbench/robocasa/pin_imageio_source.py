"""Restrict uv's imageio-ffmpeg lock entry to its reviewed source-only artifact."""

import argparse
from pathlib import Path
import re

_REQUIREMENT = "imageio-ffmpeg==0.6.0"
_SOURCE_SHA256 = "e2556bed8e005564a9f925bb7afa4002d82770d6b08825078b7697ab88ba1755"
_BLOCK = re.compile(r"^imageio-ffmpeg[^\n]*(?:\n[ \t]+[^\n]*)*", re.MULTILINE)
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def _single_block(text):
    matches = list(_BLOCK.finditer(text))
    if len(matches) != 1:
        raise ValueError("Require exactly one imageio-ffmpeg declaration")
    block = matches[0]
    declaration = block.group().replace("\\\n", "").split("--hash", 1)[0].strip()
    if declaration != _REQUIREMENT:
        raise ValueError("Require the reviewed imageio-ffmpeg source version")
    return block


def _restrict_source(requirements, lock):
    source = _single_block(requirements)
    if _HASH.findall(source.group()) != [_SOURCE_SHA256]:
        raise ValueError(
            "Input must contain only the reviewed imageio-ffmpeg sdist hash"
        )
    block = _single_block(lock)
    if _SOURCE_SHA256 not in _HASH.findall(block.group()):
        raise ValueError("Resolved imageio-ffmpeg lock omitted the reviewed sdist hash")
    comments = [
        line for line in block.group().splitlines() if line.lstrip().startswith("#")
    ]
    replacement = _REQUIREMENT + " \\\n    --hash=sha256:" + _SOURCE_SHA256
    if comments:
        replacement += "\n" + "\n".join(comments)
    return lock[: block.start()] + replacement + lock[block.end() :]


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = _restrict_source(args.input.read_text(), args.lock.read_text())
    except ValueError as error:
        parser.error(str(error))
    args.lock.write_text(result)


if __name__ == "__main__":
    _main()
