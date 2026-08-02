#!/usr/bin/env python
"""Assert that a torch build (and optionally the live GPU) targets an architecture.

This runs *inside* a workbench container image, so it must stay dependency-free
apart from ``torch``. It answers the two questions that decide whether an image
is deployable on a given GPU:

1. Does the prebuilt torch wheel ship SASS for the architecture?
   The wheel's arch set is fixed at wheel-build time and cannot be changed by
   ``TORCH_CUDA_ARCH_LIST``; only picking a different wheel index changes it.
   cu128/cu130 carry ``sm_100`` and ``sm_120``, cu124/cu126 stop at ``sm_90``.
   This check reads it without needing a driver, so it also works at image
   build time.
2. Is the device we actually landed on the one we meant to validate?
   ``torch.cuda.get_device_capability()`` is the only trustworthy answer -
   a CUDA probe that merely imports torch proves nothing.

SASS has no cross-major compatibility: ``sm_120`` (major 12) binaries do not run
on ``sm_100``/``sm_103`` (major 10) and vice versa. Within a major, forward
compatibility holds, so ``sm_100`` SASS runs on a ``sm_103`` device (B300) but
not the reverse. ``--require-arch`` therefore accepts a lower minor of the same
major as satisfying the requirement unless ``--exact-arch`` is passed.

Usage:
  # Wheel arch set only (no GPU needed - works at image build time)
  python check_torch_gpu_arch.py --require-arch sm_100 --require-arch sm_120

  # Full check on a real B200/B300 node
  python check_torch_gpu_arch.py --require-capability 10.0 --require-capability 10.3

  # Machine-readable, for harnesses
  python check_torch_gpu_arch.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

_ARCH_RE = re.compile(r"^sm_(\d+)$")

# Compute capability -> the GPUs an operator would recognize it as.
KNOWN_CAPABILITIES = {
    (8, 0): "A100",
    (8, 6): "A10/A40",
    (8, 9): "L40S / L4 (Ada)",
    (9, 0): "H100 / H200 (Hopper)",
    (10, 0): "B200 (Blackwell datacenter)",
    (10, 3): "B300 / Blackwell Ultra",
    (12, 0): "RTX PRO 6000 Blackwell (workstation)",
}


def parse_arch(token: str) -> tuple[int, int]:
    """Parse ``sm_100`` / ``10.0`` / ``100`` into a ``(major, minor)`` pair."""

    cleaned = token.strip().lower()
    match = _ARCH_RE.match(cleaned)
    if match:
        digits = match.group(1)
    elif "." in cleaned:
        major_text, _, minor_text = cleaned.partition(".")
        return int(major_text), int(minor_text)
    elif cleaned.isdigit():
        digits = cleaned
    else:
        raise ValueError(f"cannot parse architecture {token!r}; use sm_100, 10.0, or 100")
    # The bare form is the sm_ number without its prefix, so the last digit is
    # the minor version and at least one digit must precede it. A single digit
    # would mean sm_9, which is not a real architecture.
    if len(digits) < 2:
        raise ValueError(f"cannot parse architecture {token!r}; use sm_100, 10.0, or 100")
    # sm_90 -> (9, 0); sm_100 -> (10, 0); sm_103 -> (10, 3).
    return int(digits[:-1]), int(digits[-1])


def format_arch(arch: tuple[int, int]) -> str:
    return f"sm_{arch[0]}{arch[1]}"


def wheel_arch_list(torch_module) -> list[str]:
    """Return the SASS/PTX architectures baked into the torch wheel.

    ``torch.cuda.get_arch_list()`` short-circuits to ``[]`` when no CUDA device
    is visible, which is exactly the situation on a driverless build host - a
    gate written against it would be silently unenforceable there. The
    underlying ``_cuda_getArchFlags`` reads the compiled binary and does not
    need a driver, so prefer it and fall back to the public API.
    """

    getter = getattr(torch_module._C, "_cuda_getArchFlags", None)
    if getter is not None:
        try:
            flags = getter()
        except Exception:  # pragma: no cover - depends on the torch build
            flags = None
        if flags:
            return flags.split()
    return list(torch_module.cuda.get_arch_list())


def wheel_arch_set(arch_list: list[str]) -> set[tuple[int, int]]:
    """Parse ``get_arch_list()`` into capability pairs, ignoring PTX-only entries."""

    parsed: set[tuple[int, int]] = set()
    for entry in arch_list:
        # Entries look like "sm_90", "sm_100", or "compute_120" (PTX).
        if not entry.startswith("sm_"):
            continue
        try:
            parsed.add(parse_arch(entry))
        except ValueError:
            continue
    return parsed


def arch_is_covered(
    required: tuple[int, int], available: set[tuple[int, int]], *, exact: bool
) -> bool:
    """Is ``required`` satisfied by the SASS architectures in ``available``?"""

    if required in available:
        return True
    if exact:
        return False
    # Minor-version forward compatibility inside one major: sm_100 SASS runs on
    # a sm_103 device. Never across majors.
    return any(
        major == required[0] and minor <= required[1] for major, minor in available
    )


def build_report(args: argparse.Namespace) -> tuple[dict, list[str]]:
    import torch

    arch_list = wheel_arch_list(torch)
    available = wheel_arch_set(arch_list)
    failures: list[str] = []

    report: dict = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "arch_list": arch_list,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": [],
    }

    for token in args.require_arch:
        required = parse_arch(token)
        if not arch_is_covered(required, available, exact=args.exact_arch):
            failures.append(
                f"torch wheel does not cover {format_arch(required)}: "
                f"get_arch_list()={arch_list}. The wheel's fat-binary arch set is "
                "fixed at build time - switch to a wheel index that ships it "
                "(cu128/cu130 carry sm_100 and sm_120; cu124/cu126 stop at sm_90)."
            )

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            report["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(capability),
                    "arch": format_arch(capability),
                    "known_as": KNOWN_CAPABILITIES.get(capability, "unknown"),
                    "sass_covered": arch_is_covered(capability, available, exact=False),
                }
            )
    elif args.require_capability:
        failures.append(
            "no CUDA device is visible, so --require-capability cannot be checked. "
            "Run this on the target GPU node (or drop --require-capability to check "
            "the wheel arch set only)."
        )

    if args.require_capability and report["devices"]:
        wanted = {parse_arch(token) for token in args.require_capability}
        for device in report["devices"]:
            capability = (device["capability"][0], device["capability"][1])
            if capability not in wanted:
                failures.append(
                    f"device {device['index']} ({device['name']}) has capability "
                    f"{capability} ({device['known_as']}), expected one of "
                    f"{sorted(wanted)}. Validating on the wrong GPU family proves "
                    "nothing: sm_120 and sm_100/sm_103 are different CUDA majors."
                )

    if args.require_sass_coverage:
        for device in report["devices"]:
            if not device["sass_covered"]:
                failures.append(
                    f"device {device['index']} ({device['name']}, {device['arch']}) has "
                    f"no matching SASS in the wheel ({arch_list}); kernels will fall back "
                    "to PTX JIT or fail with 'no kernel image is available for execution "
                    "on the device'."
                )

    report["ok"] = not failures
    report["failures"] = failures
    return report, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-arch",
        action="append",
        default=[],
        metavar="SM",
        help="architecture the torch wheel must ship SASS for (repeatable), e.g. sm_100",
    )
    parser.add_argument(
        "--require-capability",
        action="append",
        default=[],
        metavar="CC",
        help="compute capability every visible device must report (repeatable), e.g. 10.0",
    )
    parser.add_argument(
        "--exact-arch",
        action="store_true",
        help="require an exact SASS match instead of accepting same-major forward compat",
    )
    parser.add_argument(
        "--require-sass-coverage",
        action="store_true",
        help="fail when a visible device has no matching SASS (PTX JIT fallback only)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    try:
        report, failures = build_report(args)
    except ImportError as exc:
        print(f"FAIL: torch is not importable in this image: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"torch {report['torch_version']} (cuda {report['torch_cuda_version']})")
        print(f"arch_list {report['arch_list']}")
        if report["devices"]:
            for device in report["devices"]:
                print(
                    f"device {device['index']}: {device['name']} "
                    f"capability={tuple(device['capability'])} arch={device['arch']} "
                    f"known_as={device['known_as']} sass_covered={device['sass_covered']}"
                )
        else:
            print("device: none visible (wheel-arch check only)")
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print("OK" if report["ok"] else "FAILED")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
