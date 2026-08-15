"""OpenPI BYOF runtime-access contract.

OpenPI's pi0.5 checkpoints contain Gemma-derived weights.  Acceptance is an
operator decision scoped to a run: it must never be rendered into workflow
YAML, embedded in an image, or persisted in project credentials.
"""

from __future__ import annotations

import os

OPENPI_TERMS_ENV = "NPA_OPENPI_ACCEPT_GEMMA_TERMS"
OPENPI_TERMS_ACCEPTED_VALUE = "YES"
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


def require_openpi_terms(env: dict[str, str] | None = None) -> None:
    """Fail closed unless the operator explicitly accepted both named policies."""

    runtime_env = os.environ if env is None else env
    if runtime_env.get(OPENPI_TERMS_ENV) != OPENPI_TERMS_ACCEPTED_VALUE:
        raise ValueError(
            "OpenPI pi0.5 requires scoped operator acceptance before image build "
            "or checkpoint download. Review the Gemma Terms of Use "
            f"({GEMMA_TERMS_URL}) and Gemma Prohibited Use Policy "
            f"({GEMMA_PROHIBITED_USE_URL}), then set "
            f"{OPENPI_TERMS_ENV}={OPENPI_TERMS_ACCEPTED_VALUE} for this run only."
        )
