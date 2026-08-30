"""Grounded, resumable HF/NGC approval interaction for the NPA agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping


def classify_followup(text: str, *, has_pending_plan: bool) -> str:
    value = str(text or "").strip().lower()
    starts_plan = bool(
        re.search(r"\b(?:prepare|check|audit|review|open|start|help|approve)\b", value)
    )
    explicit_provider = bool(re.search(r"\b(?:hugging\s*face|hf|ngc)\b", value))
    explicit_access = bool(
        re.search(
            r"\b(?:gated(?:[-\s]+(?:access|catalog|model|dataset|artifact))?|"
            r"approval(?:[-\s]+pages?)?|access[-\s]+approval|catalog[-\s]+access)\b",
            value,
        )
    )
    if starts_plan and (explicit_provider or explicit_access):
        return "plan"
    if not has_pending_plan:
        return ""
    if re.fullmatch(r"(?:yes|y|open(?: them| the pages)?|do it|go ahead)[.! ]*", value):
        return "open"
    if re.search(r"\b(?:done|completed|accepted|approved|recheck|check again|continue|resume)\b", value):
        return "recheck"
    if re.fullmatch(r"(?:no|n|later|not now|decline)[.! ]*", value):
        return "later"
    return ""


def build_plan(
    *,
    capabilities: Iterable[str] | None,
    resume_command: str,
    state_path: Path,
    force: bool = False,
) -> dict[str, object]:
    from npa.clients.credentials import load_credentials
    from npa.clients.huggingface import validate_hf_access
    from npa.workbench.access_approval import (
        approval_plan,
        exact_requirements,
        probe_requirements,
    )
    from npa.workbench.nurec.nurec import check_ngc_image_access

    credentials = load_credentials()
    evidence = probe_requirements(
        exact_requirements(capabilities),
        hf_token=credentials.hf_token,
        ngc_key=credentials.ngc_api_key,
        hf_validator=validate_hf_access,
        ngc_validator=check_ngc_image_access,
        state_path=state_path,
        force=force,
    )
    return approval_plan(evidence, resume_command=resume_command)


def format_plan_reply(plan: Mapping[str, object]) -> str:
    counts = plan.get("counts") if isinstance(plan.get("counts"), Mapping) else {}
    hf = int(counts.get("hf") or 0)
    ngc = int(counts.get("ngc") or 0)
    if str(plan.get("status") or "") == "ready":
        return "All exact gated Hugging Face and NVIDIA NGC artifacts are **Ready**."
    return (
        f"This capability needs approval for **{hf} HF resource(s)** and "
        f"**{ngc} NGC artifact(s)**. Open the official pages now?\n\n"
        "NPA will not click an acceptance control or submit legal assent. "
        "Reply **yes** to open only these pages, **later** to preserve the handoff, "
        "or **done** after completing approval to re-check and resume."
    )


def format_open_reply(plan: Mapping[str, object]) -> str:
    urls = [str(item) for item in (plan.get("official_urls") or []) if str(item)]
    links = "\n".join(f"- <{url}>" for url in urls)
    return (
        "Opening the exact official approval pages in your browser. NPA has not "
        "clicked or submitted anything. If your browser blocked a tab, use these "
        f"official links:\n\n{links}\n\n"
        "Complete the user-bound steps, then reply **done** so I can re-check "
        "access and resume safely."
    )


def format_later_reply(plan: Mapping[str, object]) -> str:
    return (
        "Approval preparation is saved for later; unrelated Workbench capabilities "
        "remain usable. Resume safely with: `"
        + str(plan.get("resume_command") or "npa configure --prepare-catalog-access")
        + "`."
    )


__all__ = [
    "build_plan",
    "classify_followup",
    "format_later_reply",
    "format_open_reply",
    "format_plan_reply",
]
