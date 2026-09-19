"""Report a group-writable checkout once instead of 428 times.

``image_byte_scan.core.authorized_roots`` refuses to trust a root that is
group- or other-writable, and ``private_path`` applies the same rule to every
input file it opens. That refusal is correct: a writable checkout could be
modified between the scan and the use of its result. But the scanner raises a
bare ``ScanError`` code by design, so a checkout created under ``umask 002``
turns into hundreds of ``root_permissions`` and ``input_permissions`` errors
with nothing pointing at the one-line remedy.

This hook leaves the scanner and its error surface untouched, and neither
skips a test nor changes permissions itself.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
WRITABLE = stat.S_IWGRP | stat.S_IWOTH

REMEDY = f"""\
image_byte_scan cannot run: this checkout is group- or other-writable.

{{offenders}}

The scanner refuses a writable authorized root because its bytes could change
between the scan and the use of its result, so every test that opens a scan
root fails at setup. Nothing is wrong with the code under test.

Fix the checkout, then rerun:

    chmod -R g-w,o-w {CHECKOUT}

Files created afterwards are group-writable again under umask 002, so run
validation with `umask 022` if this keeps coming back.\
"""


def _writable_bits(path: Path) -> int:
    try:
        return path.stat().st_mode & WRITABLE
    except OSError:
        return 0


def _offending_roots() -> list[str]:
    """The checkout root and the scanner's own source directory, if writable."""

    candidates = (CHECKOUT, CHECKOUT / "npa", CHECKOUT / "npa/scripts/image_byte_scan")
    return [
        f"  {path.relative_to(CHECKOUT.parent)} is mode {oct(path.stat().st_mode & 0o777)}"
        for path in candidates
        if _writable_bits(path)
    ]


def _needs_scan_roots(item: pytest.Item) -> bool:
    module = getattr(item, "module", None)
    if module is None:
        return False
    for value in vars(module).values():
        name = getattr(value, "__name__", "")
        if isinstance(name, str) and name.split(".")[0] == "image_byte_scan":
            return True
    return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    offenders = _offending_roots()
    if not offenders:
        return
    if not any(_needs_scan_roots(item) for item in items):
        return
    raise pytest.UsageError(REMEDY.format(offenders="\n".join(offenders)))
