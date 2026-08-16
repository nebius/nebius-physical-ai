"""OpenPI BYOF request detection and upstream terms references."""

from __future__ import annotations

GEMMA_TERMS_URL = "https://ai.google.dev/gemma/terms"
GEMMA_PROHIBITED_USE_URL = "https://ai.google.dev/gemma/prohibited_use_policy"
OPENPI_REPO_URL = "https://github.com/Physical-Intelligence/openpi.git"


def is_openpi_request(
    *, solution_name: str = "", repo_url: str = "", smoke_command: str = ""
) -> bool:
    """Return whether a BYOF request selects OpenPI even if its label is omitted."""

    normalized_repo = repo_url.strip().lower().removesuffix(".git")
    return (
        solution_name.strip().lower() == "openpi"
        or normalized_repo.endswith("physical-intelligence/openpi")
        or "pi05_droid_jointpos_polaris" in smoke_command
    )
