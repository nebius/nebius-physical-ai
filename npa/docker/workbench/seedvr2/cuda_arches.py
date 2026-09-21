"""Validate matching FlashAttention/Apex CUDA targets and emit scanner arguments."""

from __future__ import annotations

import argparse
import re


def exact_arches(flash_arches: str, torch_arches: str) -> list[str]:
    """Reject ambiguous inputs before building and return shared SASS targets."""
    if not re.fullmatch(r"[1-9][0-9]*(;[1-9][0-9]*)*", flash_arches):
        raise ValueError("FlashAttention targets must be positive numeric SM values")
    if not re.fullmatch(r"[1-9][0-9]*\.[0-9](;[1-9][0-9]*\.[0-9])*", torch_arches):
        raise ValueError("Torch targets must be explicit major.minor values")
    flash = flash_arches.split(";")
    torch = [value.replace(".", "") for value in torch_arches.split(";")]
    if len(set(flash)) != len(flash) or len(set(torch)) != len(torch):
        raise ValueError("CUDA targets must not contain duplicates")
    if set(flash) != set(torch):
        raise ValueError("FlashAttention and Apex must target the same architectures")
    return [f"sm_{value}" for value in sorted(flash, key=int)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-arches", required=True)
    parser.add_argument("--torch-arches", required=True)
    args = parser.parse_args(argv)
    try:
        arches = exact_arches(args.flash_arches, args.torch_arches)
    except ValueError as error:
        parser.error(str(error))
    print(" ".join(f"--exact {arch}" for arch in arches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
