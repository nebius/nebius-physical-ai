"""Detect workflow YAML format for ``npa workbench workflow submit``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from npa.orchestration.npa_workflow.spec import SUPPORTED_API_VERSIONS

SubmitFormat = Literal["sim2real_runbook", "npa.workflow", "skypilot"]


def peek_workflow_document(path: Path) -> dict[str, Any]:
    """Return the first YAML mapping document, or ``{}`` if none."""

    import yaml

    text = path.read_text(encoding="utf-8")
    for doc in yaml.safe_load_all(text):
        if isinstance(doc, dict):
            return doc
    return {}


def is_npa_workflow_spec(path: Path) -> bool:
    """True when the first document declares a supported ``npa.workflow`` apiVersion."""

    if not path.is_file():
        return False
    try:
        doc = peek_workflow_document(path)
    except Exception:
        return False
    api_version = str(doc.get("apiVersion") or "").strip()
    return api_version in SUPPORTED_API_VERSIONS


def detect_submit_format(path: Path) -> SubmitFormat:
    """Classify a submit target.

    Order: Sim2Real runbook path → ``npa.workflow`` apiVersion → SkyPilot default.
    """

    from npa.workflows.sim2real.k8s_submit import is_sim2real_runbook

    if is_sim2real_runbook(path):
        return "sim2real_runbook"
    if is_npa_workflow_spec(path):
        return "npa.workflow"
    return "skypilot"
