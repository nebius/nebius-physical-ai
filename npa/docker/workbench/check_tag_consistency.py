"""Static check for npa workbench image tag-family consistency.

The runtime deploy code currently uses version tags. This check covers the
image build, CI, and security documentation surfaces owned by image hardening,
and rejects drift when CUDA tag-family references appear there.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in bare CI failures.
    yaml = None


REQUIRED_TAGS = {"cuda12", "cuda13-b300"}
TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml"}
IMAGE_REF_RE = re.compile(
    r"(?:^|[\\s\"'`])(?:[a-z0-9.-]+(?::[0-9]+)?/)?"
    r"(?:[a-z0-9._-]+/)*npa-[a-z0-9-]+:([a-z0-9_.-]+)",
    re.IGNORECASE,
)
CUDA_TOKEN_RE = re.compile(
    r"\\b(cuda[0-9][a-z0-9_.-]*|cuda-[0-9][a-z0-9_.-]*|[a-z0-9_.-]*b300[a-z0-9_.-]*)\\b",
    re.IGNORECASE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_valid_tags(tags_yaml: Path) -> set[str]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: python -m pip install PyYAML")
    config = yaml.safe_load(tags_yaml.read_text()) or {}
    tag_families = config.get("tag_families")
    if not isinstance(tag_families, dict):
        raise RuntimeError(f"{tags_yaml} must define a tag_families mapping")
    valid_tags = set(tag_families)
    missing = REQUIRED_TAGS - valid_tags
    if missing:
        raise RuntimeError(f"{tags_yaml} missing required tag families: {sorted(missing)}")
    return valid_tags


def _iter_scan_files(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "npa" / "docker" / "workbench",
        repo_root / ".github" / "workflows",
        repo_root / "docs" / "security",
        repo_root / "docs" / "images",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("Dockerfile") or path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def _line_violations(path: Path, valid_tags: set[str], repo_root: Path) -> list[str]:
    violations: list[str] = []
    rel = path.relative_to(repo_root)
    for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        for match in IMAGE_REF_RE.finditer(line):
            tag = match.group(1)
            if _looks_like_gpu_tag(tag) and not _uses_valid_tag_family(tag, valid_tags):
                violations.append(
                    f"{rel}:{line_no}: image tag '{tag}' is not one of {sorted(valid_tags)}"
                )
        for match in CUDA_TOKEN_RE.finditer(line):
            token = match.group(1)
            normalized = token.lower()
            if not _uses_valid_tag_family(normalized, valid_tags) and _looks_like_gpu_tag(
                normalized
            ):
                violations.append(
                    f"{rel}:{line_no}: CUDA tag token '{token}' is not one of {sorted(valid_tags)}"
                )
    return violations


def _looks_like_gpu_tag(tag: str) -> bool:
    lowered = tag.lower()
    return lowered.startswith("cuda") or "b300" in lowered or lowered.startswith("sm_")


def _uses_valid_tag_family(tag: str, valid_tags: set[str]) -> bool:
    lowered = tag.lower()
    return any(
        lowered == valid_tag
        or any(lowered.startswith(f"{valid_tag}{separator}") for separator in ("-", ".", "_"))
        for valid_tag in valid_tags
    )


def main() -> int:
    repo_root = _repo_root()
    tags_yaml = repo_root / "npa" / "docker" / "workbench" / "tags.yaml"
    if not tags_yaml.exists():
        print(f"ERROR: {tags_yaml} not found. Two-tag strategy needs a canonical source.")
        return 1

    try:
        valid_tags = _load_valid_tags(tags_yaml)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    violations: list[str] = []
    for path in _iter_scan_files(repo_root):
        violations.extend(_line_violations(path, valid_tags, repo_root))

    if violations:
        print("Two-tag strategy violations:")
        for violation in violations:
            print(f"  {violation}")
        return 1

    print(f"Two-tag strategy: all scanned references use canonical tags {sorted(valid_tags)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
