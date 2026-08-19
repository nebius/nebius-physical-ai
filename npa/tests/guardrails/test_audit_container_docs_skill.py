from __future__ import annotations

from pathlib import Path
import re

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    public_mirror_tag_for_tool,
    publicly_publishable_tools,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "docs/workbench/container-image-catalog.md"
SKILL = REPO_ROOT / "skills/atomic/audit-container-docs/SKILL.md"
ONBOARDING_SKILLS = (
    REPO_ROOT / "skills/workflows/contribute-workbench-image/SKILL.md",
    REPO_ROOT / "skills/workflows/add-workbench-tool/SKILL.md",
    REPO_ROOT / "skills/workflows/oss-solution-registry-onboard/SKILL.md",
)


def _catalog_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in CATALOG.read_text().splitlines():
        if not line.startswith("| ") or "`npa-" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        match = re.fullmatch(r"`(npa-[^`]+)`", cells[1])
        assert match, line
        image = match.group(1)
        assert image not in rows, f"duplicate public catalog row for {image}"
        rows[image] = cells[2]
    return rows


def test_public_catalog_matches_the_repository_publish_inventory() -> None:
    rows = _catalog_rows()
    tools = publicly_publishable_tools()
    expected_images = {CONTAINER_IMAGE_NAMES[tool] for tool in tools}

    assert set(rows) == expected_images
    for tool in tools:
        image = CONTAINER_IMAGE_NAMES[tool]
        assert f"`{public_mirror_tag_for_tool(tool)}`" in rows[image], (
            f"{image} catalog row does not contain the current public-mirror pin"
        )


def test_skill_names_each_authoritative_inventory_layer() -> None:
    text = SKILL.read_text()
    for required in (
        "npa/docker/workbench/packaging-contract.yaml",
        "npa/src/npa/deploy/images.py",
        "npa/pyproject.toml",
        "npa/src/npa/deploy/*_image_manifest.json",
        "npa/src/npa/deploy/publish_public.py",
        "publicly_publishable_tools()",
        "docker buildx imagetools inspect",
    ):
        assert required in text


def test_image_and_solution_onboarding_requires_catalog_reconciliation() -> None:
    for onboarding_skill in ONBOARDING_SKILLS:
        text = onboarding_skill.read_text()
        assert "skills/atomic/audit-container-docs/SKILL.md" in text, onboarding_skill
        assert "docs/workbench/container-image-catalog.md" in text, onboarding_skill
