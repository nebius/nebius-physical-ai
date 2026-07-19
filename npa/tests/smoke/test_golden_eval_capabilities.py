"""Golden-eval capability chart stays complete and aligned with the manifest."""

from __future__ import annotations

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.smoke.manifest import load_manifest


def test_every_manifest_container_has_capability_probes() -> None:
    specs = load_manifest()
    missing = set(specs) - set(GOLDEN_EVAL_CAPABILITIES)
    assert not missing, f"missing capability entries: {sorted(missing)}"


def test_every_tool_container_has_capability_probes() -> None:
    missing = set(CONTAINER_IMAGE_NAMES) - set(GOLDEN_EVAL_CAPABILITIES)
    assert not missing, f"tools missing capability entries: {sorted(missing)}"


def test_capability_entries_are_non_empty() -> None:
    for name, checks in GOLDEN_EVAL_CAPABILITIES.items():
        assert checks, f"{name} has empty capability list"
        assert all(check.strip() for check in checks)


def test_audit_workbench_image_tags_passes() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = root / "npa" / "scripts" / "audit_workbench_image_tags.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
