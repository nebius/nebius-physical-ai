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

The scanner refuses a writable authorized root or input file because its bytes
could change between the scan and the use of its result, so every test that
opens a scan root fails. Nothing is wrong with the code under test.

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


def _scan_inputs() -> list[Path]:
    """The roots and files ``core.source_bindings`` opens, which must not be writable.

    Directory modes alone are not enough: a single `0664` file inside the
    scanner package is enough to raise `input_permissions`, which is the state
    a new file lands in under `umask 002` after the directories are fixed. The
    closure below mirrors `core.source_bindings`; keep it in step with that
    function, including its suffix filter, which is what keeps `__pycache__`
    out of the check.
    """

    folder = CHECKOUT / "npa/scripts/image_byte_scan"
    suffixes = {".py", ".go", ".mod", ".sum", ".json", ".md"}
    inputs = [
        CHECKOUT,
        CHECKOUT / "npa",
        folder,
        CHECKOUT / ".gitleaks.toml",
        CHECKOUT / "npa/scripts/scan_image_bytes.py",
        CHECKOUT / "npa/tests/docker/test_image_byte_go_build.py",
    ]
    inputs.extend(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (path.suffix in suffixes or path.name.startswith("LICENSE"))
    )
    return inputs


def _offenders() -> list[str]:
    found = [
        f"  {path.relative_to(CHECKOUT.parent)} is mode {oct(path.stat().st_mode & 0o777)}"
        for path in _scan_inputs()
        if _writable_bits(path)
    ]
    if len(found) > 6:
        return [*found[:6], f"  ...and {len(found) - 6} more"]
    return found


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
    offenders = _offenders()
    if not offenders:
        return
    if not any(_needs_scan_roots(item) for item in items):
        return
    raise pytest.UsageError(REMEDY.format(offenders="\n".join(offenders)))
