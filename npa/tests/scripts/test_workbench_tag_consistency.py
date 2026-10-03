"""Keep CUDA family validation strict while accepting declared migration aliases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


WORKBENCH = Path(__file__).resolve().parents[2] / "docker/workbench"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location(
        "workbench_tag_consistency", WORKBENCH / "check_tag_consistency.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_and_legacy_families_are_accepted(checker, tmp_path):
    families = checker._load_valid_tags(WORKBENCH / "tags.yaml")
    assert families == {"cuda12", "cuda13-blackwell", "cuda13-b300"}
    path = tmp_path / "images.md"
    path.write_text(
        "\n".join(f"`npa-base:{family}-dev-abc`" for family in sorted(families))
    )
    assert checker._line_violations(path, families, tmp_path) == []


def test_unknown_family_is_still_rejected(checker, tmp_path):
    families = checker._load_valid_tags(WORKBENCH / "tags.yaml")
    path = tmp_path / "images.md"
    path.write_text("`npa-base:cuda13-blackwel-dev-abc`")
    violations = checker._line_violations(path, families, tmp_path)
    assert len(violations) == 1
    assert "cuda13-blackwel-dev-abc" in violations[0]


@pytest.mark.parametrize(
    "aliases",
    [
        ["cuda13-b300"],
        {"cuda13-b300": "unknown"},
        {"cuda13-b300": ["cuda13-blackwell"]},
        {"": "cuda13-blackwell"},
        {123: "cuda13-blackwell"},
        {"cuda12": "cuda13-blackwell"},
    ],
)
def test_invalid_alias_mapping_fails_closed(checker, tmp_path, aliases):
    path = tmp_path / "tags.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tag_families": {"cuda12": {}, "cuda13-blackwell": {}},
                "legacy_tag_families": aliases,
            }
        )
    )
    with pytest.raises(RuntimeError, match="legacy"):
        checker._load_valid_tags(path)


def test_legacy_family_cannot_replace_required_canonical_family(checker, tmp_path):
    path = tmp_path / "tags.yaml"
    path.write_text("tag_families:\n  cuda12: {}\n  cuda13-b300: {}\n")
    with pytest.raises(RuntimeError, match="missing required.*cuda13-blackwell"):
        checker._load_valid_tags(path)


def test_scan_inventories_canonical_recipe_once(checker):
    files = checker._iter_scan_files(WORKBENCH.parents[2])
    assert WORKBENCH / "base/cuda13-blackwell/Dockerfile" in files
    assert WORKBENCH / "base/cuda13-b300/Dockerfile" not in files
