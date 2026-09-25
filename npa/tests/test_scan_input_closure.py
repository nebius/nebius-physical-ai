"""Keep the collection-time permission diagnostic aligned with the real scanner.

The diagnostic in ``conftest`` must report exactly the paths whose mode
``image_byte_scan`` rejects. Listing fewer only degrades the message, but
listing more converts a diagnostic into a stricter gate that fails a checkout
the production scanner accepts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import conftest as suite_conftest

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT / "npa/scripts"))
from image_byte_scan import core as W  # noqa: E402


def test_scan_input_closure_matches_production(tmp_path: Path) -> None:
    """Fail when the hardcoded closure drifts from ``core.source_bindings``.

    Args:
        tmp_path: Private analysis root, which the scanner requires to be
            separate from the checkout and not group-readable.
    Returns:
        None.
    """

    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        produced = {Path(binding["path"]) for binding in W.source_bindings().values()}

    declared = {path for path in suite_conftest._scan_inputs() if path.is_file()}

    assert declared == produced, (
        "conftest._scan_inputs() drifted from core.source_bindings():\n"
        f"  only in conftest: {sorted(str(p) for p in declared - produced)}\n"
        f"  only in scanner : {sorted(str(p) for p in produced - declared)}"
    )


def test_intermediate_directories_are_not_permission_checked() -> None:
    """Hold the diagnostic to the modes production actually rejects.

    ``authorized_roots`` stats only the trusted root and ``private_path`` stats
    only each source file; the directories between them are walked for symlinks
    alone. A group-writable ``npa/`` or scanner package directory therefore
    binds successfully in production, so the diagnostic must not abort on it.

    Returns:
        None.
    """

    inputs = set(suite_conftest._scan_inputs())

    assert CHECKOUT in inputs, "the trusted root's mode is checked by authorized_roots"
    for intermediate in (CHECKOUT / "npa", CHECKOUT / "npa/scripts/image_byte_scan"):
        assert intermediate not in inputs, (
            f"{intermediate} is an intermediate directory whose mode production "
            "never inspects; listing it rejects a checkout that binds cleanly"
        )
