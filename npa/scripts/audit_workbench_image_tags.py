#!/usr/bin/env python
"""Fail when tracked files reference known-stale npa-* image tags.

Canonical versions live in ``pyproject.toml`` → ``[tool.npa.supported-tools]``.
This gate targets golden-eval packaging bumps (not SONIC variant tags or test
fixtures like ``:smoke``).

Usage:
  npa/.venv/bin/python npa/scripts/audit_workbench_image_tags.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from npa.deploy.images import CONTAINER_IMAGE_NAMES, supported_tool_version

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (
    REPO_ROOT / "npa" / "workflows",
    REPO_ROOT / "npa" / "src",
    REPO_ROOT / "ops",
    REPO_ROOT / "docs" / "workbench",
    REPO_ROOT / "docs" / "security",
)
SKIP_DIR_NAMES = {".venv", "__pycache__", ".git", ".tmp-paste-verify-demo"}
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md"}

# Wrapper Dockerfile intentionally references the upstream transfer base.
ALLOWLIST_SUBSTRINGS = (
    "npa-cosmos2-transfer:2.5.0",
    "COSMOS2_TRANSFER_BASE_IMAGE",
    "published `2.5.0`",
    "published 2.5.0",
)

# Tags that must not appear on these tools anywhere outside allowlists.
STALE_TOOL_TAGS: dict[str, set[str]] = {
    "cosmos2-transfer": {"2.5.0"},
    "envgen": {"0.1.1"},
    "reference-policy": {"0.1.1"},
    "loop-eval": {"0.1.1-genuine-sm120"},
    "lerobot-vlm-rl": {"0.1.0"},
    "lerobot-policy": {"0.1.0"},
    "lancedb": {"0.30.2"},
    "retargeting": {"0.1.0"},
    "detection-training": {
        "bdd100k-real-labelmap-eval-w9-registry-fix-20260519T214847Z",
    },
}

IMAGE_REF_RE = re.compile(r"(npa-[a-z0-9-]+):([a-z0-9._-]+(?:T[0-9]+Z)?)", re.IGNORECASE)

SKIP_TAG_SUFFIXES = (":smoke", ":test", ":local", ":latest", "-k8s-runtime", "-sm80-sm90-sm120")


def _tool_for_image(image: str) -> str | None:
    for tool, name in CONTAINER_IMAGE_NAMES.items():
        if name.lower() == image.lower():
            return tool
    return None


def _scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if any(token in text for token in ALLOWLIST_SUBSTRINGS) and "cosmos2-transfer" in str(
        path
    ):
        return []
    rel = path.relative_to(REPO_ROOT)
    issues: list[str] = []
    for image, tag in IMAGE_REF_RE.findall(text):
        if any(image.endswith(suffix) or f":{tag}".endswith(suffix) for suffix in SKIP_TAG_SUFFIXES):
            continue
        tool = _tool_for_image(image)
        if tool is None:
            continue
        stale = STALE_TOOL_TAGS.get(tool, set())
        if tag in stale:
            canonical = supported_tool_version(tool)
            issues.append(
                f"{rel}: {image}:{tag} (use {CONTAINER_IMAGE_NAMES[tool]}:{canonical})"
            )
    return issues


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.suffix not in TEXT_SUFFIXES:
                continue
            if path.name == "audit_workbench_image_tags.py":
                continue
            files.append(path)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    violations: list[str] = []
    for path in _iter_files():
        violations.extend(_scan_file(path))

    if not violations:
        print(f"OK: no known-stale npa-* tags in {len(_iter_files())} tracked files")
        return 0

    print("Known-stale workbench image tag references:")
    for line in sorted(set(violations)):
        print(f"  - {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
