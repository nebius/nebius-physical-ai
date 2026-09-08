"""OpenPI BYOF runtime-access contract.

OpenPI's public pi0.5 checkpoint contains Gemma-derived weights. The product
defaults to runtime use under the named terms, which bind by conduct. An explicit
opt-out stops access; no acceptance is embedded in images or saved credentials.
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
    """Apply OpenPI's runtime default and reject an explicit opt-out.

    Args:
        env: Runtime environment, or the current process environment when omitted.

    Returns:
        None when the default or an affirmative value permits runtime use.

    Raises:
        ValueError: The operator opted out or supplied an invalid value.
    """

    runtime_env = os.environ if env is None else env
    value = runtime_env.get(OPENPI_TERMS_ENV, OPENPI_TERMS_ACCEPTED_VALUE).strip().upper()
    if value in {"Y", "YES", "1", "TRUE"}:
        return
    reason = "was explicitly opted out" if value in {"", "N", "NO", "0", "FALSE"} else "has an invalid runtime acceptance value"
    raise ValueError(
        f"OpenPI pi0.5 {reason}; checkpoint access has not started. "
        f"Gemma Terms of Use ({GEMMA_TERMS_URL}) and Gemma Prohibited Use Policy "
        f"({GEMMA_PROHIBITED_USE_URL}) apply to runtime use. To resume, unset "
        f"{OPENPI_TERMS_ENV} or set {OPENPI_TERMS_ENV}={OPENPI_TERMS_ACCEPTED_VALUE} "
        "for this run only. This does not grant redistribution rights."
    )
