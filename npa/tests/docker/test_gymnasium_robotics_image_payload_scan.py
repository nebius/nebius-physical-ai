from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCANNER = ROOT / "npa/scripts/scan_image_gymnasium_robotics_payload.py"
WORKFLOW = ROOT / ".github/workflows/publish-public-images.yml"


def test_scanner_covers_config_all_layers_whiteouts_and_rootfs_entries() -> None:
    text = SCANNER.read_text(encoding="utf-8")
    for token in (
        '"manifest.json"',
        '_scan_policy_bytes("exact image config", config_raw)',
        'config_name != f"{config_digest}.json"',
        'entry.get("Layers", [])',
        'config_rootfs.get("diff_ids") != layer_diff_ids',
        '_scan_policy_bytes(f"raw layer bytes: {layer_name}", raw)',
        'leaf == ".wh..wh..opq"',
        'leaf.startswith(".wh.")',
        "rootfs[path] = content",
        'kind = "symlink" if item.issym() else "hardlink"',
        "_nested_archive_members(path, content)",
        '"requirements.lock": None',
        "corresponding-source annex is empty",
        "final rootfs contains missing or unclassified entries",
        "final image must declare the non-root ubuntu user",
        '"unresolved_findings": 0',
        '"release_authorized": False',
    ):
        assert token in text


def test_product_scan_is_staged_before_push_and_after_exact_pull() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("scan_image_gymnasium_robotics_payload.py") == 2
    before_push = text.index("scan_image_gymnasium_robotics_payload.py")
    push = text.index('docker push "$IMAGE"')
    after_pull = text.rindex("scan_image_gymnasium_robotics_payload.py")
    assert before_push < push < after_pull
    for gate in (
        "Generate pre-publication SBOM",
        "Attest exact pushed digest provenance",
        "Attest exact pushed digest SBOM",
        "--scanners vuln,secret,license",
        'anonymous_config="$(mktemp -d)"',
    ):
        assert gate in text


def test_trusted_workflow_refuses_phase_a_selection_before_build() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    refusal = text.index("Phase A pre-registration candidate has no build authority")
    build = text.index("docker buildx build")
    assert refusal < build
