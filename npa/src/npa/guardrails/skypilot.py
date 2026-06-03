"""Static guardrails for SkyPilot workflows and helper scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


FORBIDDEN_TEARDOWN_RE = re.compile(r"(^|\s)(--down)(\s|$)|\bautodown\b")
SKYPILOT_LAUNCH_RE = re.compile(r"\bsubmit_workflow\b|\bsky\s+(jobs\s+launch|launch)\b")
ENV_REF_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-[^}]*)?\}")
PY_ENV_REF_RE = re.compile(
    r"os\.environ(?:\.get)?\(\s*[\"']([A-Z0-9_]+)[\"']|"
    r"os\.environ\[\s*[\"']([A-Z0-9_]+)[\"']"
)


@dataclass(frozen=True)
class TextHit:
    """A source-location hit for a static guard."""

    path: Path
    line_number: int
    line: str


def load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    """Load SkyPilot YAML documents as mappings."""

    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError(f"SkyPilot YAML documents must be mappings: {path}")
    return docs


def env_names_for_yaml(path: Path) -> set[str]:
    """Return all env keys declared by a SkyPilot YAML file."""

    names: set[str] = set()
    for doc in load_yaml_documents(path):
        envs = doc.get("envs")
        if isinstance(envs, dict):
            names.update(str(key) for key in envs)
    return names


def env_refs_for_yaml(path: Path) -> set[str]:
    """Return all shell-style env references in setup/run blocks."""

    refs: set[str] = set()
    for doc in load_yaml_documents(path):
        for key in ("setup", "run"):
            value = doc.get(key)
            if isinstance(value, str):
                refs.update(ENV_REF_RE.findall(value))
                for match in PY_ENV_REF_RE.findall(value):
                    refs.update(name for name in match if name)
    return refs


def scan_for_forbidden_teardown(paths: list[Path]) -> list[TextHit]:
    """Find unsupported SkyPilot teardown flags in workflow files and scripts."""

    hits: list[TextHit] = []
    for path in paths:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FORBIDDEN_TEARDOWN_RE.search(line):
                hits.append(TextHit(path=path, line_number=line_number, line=line.strip()))
    return hits


def skypilot_launching_scripts_missing_sigterm(paths: list[Path]) -> list[Path]:
    """Return SkyPilot-launching scripts without an explicit SIGTERM hook."""

    missing: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not SKYPILOT_LAUNCH_RE.search(text):
            continue
        if "SIGTERM" not in text and "trap " not in text:
            missing.append(path)
    return missing


def image_refs_for_workflows(paths: list[Path]) -> list[str]:
    """Extract workflow image references."""

    refs: list[str] = []
    for path in paths:
        for doc in load_yaml_documents(path):
            resources = doc.get("resources")
            if not isinstance(resources, dict):
                continue
            image_id = resources.get("image_id")
            if isinstance(image_id, str) and image_id.strip():
                image = image_id.removeprefix("docker:").strip()
                refs.append(image)
    return refs


def unresolved_image_placeholders(image: str) -> bool:
    """Return true when a workflow image still needs operator substitution."""

    return "<" in image or ">" in image or "${" in image


def resolve_workflow_image(image: str, *, registry_id: str) -> str:
    """Resolve the standard registry-id placeholder for local registry checks."""

    return image.replace("<your-registry-id>", registry_id)


def inspect_image_exists(image: str, *, timeout_s: int = 30) -> bool:
    """Check a remote image with a locally installed registry inspection tool."""

    commands = [
        ["crane", "digest", image],
        ["skopeo", "inspect", f"docker://{image}"],
        ["docker", "manifest", "inspect", image],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
        return result.returncode == 0
    raise RuntimeError("install crane, skopeo, or docker to inspect registry images")
