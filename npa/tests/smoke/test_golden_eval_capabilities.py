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


def test_cosmos3_stale_tag_audit_distinguishes_rollback_from_runtime_refs(
    tmp_path, monkeypatch
) -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = root / "npa" / "scripts" / "audit_workbench_image_tags.py"
    spec = importlib.util.spec_from_file_location("tag_audit", script)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    stale = tmp_path / "npa/workflows/cosmos3.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("image_id: npa-cosmos3:1.2.2-cu130\n", encoding="utf-8")
    assert audit._scan_file(stale) == [
        "npa/workflows/cosmos3.yaml: npa-cosmos3:1.2.2-cu130 "
        "(use npa-cosmos3:1.2.2-cu130-r6)"
    ]

    rollback = tmp_path / "docs/workbench/cosmos3-generate.md"
    rollback.parent.mkdir(parents=True)
    rollback.write_text("Rollback: `npa-cosmos3:1.2.2-cu130`.\n", encoding="utf-8")
    assert audit._scan_file(rollback) == []


def test_cosmos3_stale_tag_audit_rejects_previous_release_in_current_docs(
    tmp_path, monkeypatch
) -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = root / "npa" / "scripts" / "audit_workbench_image_tags.py"
    spec = importlib.util.spec_from_file_location("tag_audit_current_docs", script)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    current_doc = tmp_path / "docs/security/container-golden-evals.md"
    current_doc.parent.mkdir(parents=True)
    current_doc.write_text("Required tag: `1.2.2-cu130-r2`.\n", encoding="utf-8")

    assert audit._scan_file(current_doc) == [
        "docs/security/container-golden-evals.md: npa-cosmos3:1.2.2-cu130-r2 "
        "(use npa-cosmos3:1.2.2-cu130-r6)"
    ]


def test_cosmos3_stale_tag_audit_requires_explicit_allowlisted_history(
    tmp_path, monkeypatch
) -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = root / "npa" / "scripts" / "audit_workbench_image_tags.py"
    spec = importlib.util.spec_from_file_location("tag_audit_history", script)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    matrix = tmp_path / "docs/workbench/image-gpu-compatibility-matrix.md"
    matrix.parent.mkdir(parents=True)
    matrix.write_text("Historical measurement: `1.2.2-cu130-r2`.\n", encoding="utf-8")
    assert audit._scan_file(matrix) == []

    matrix.write_text("Measured tag: `1.2.2-cu130-r2`.\n", encoding="utf-8")
    assert audit._scan_file(matrix) == [
        "docs/workbench/image-gpu-compatibility-matrix.md: "
        "npa-cosmos3:1.2.2-cu130-r2 (use npa-cosmos3:1.2.2-cu130-r6)"
    ]
