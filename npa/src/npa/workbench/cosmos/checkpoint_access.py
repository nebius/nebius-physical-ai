"""Fail-fast access preflight for exact Cosmos Transfer ControlNet files."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from npa.clients.huggingface import HFAccessResult, validate_hf_file_access
from npa.workbench.cosmos.control_contract import COSMOS_TRANSFER_CHECKPOINTS


class CosmosCheckpointAccessError(RuntimeError):
    """The selected exact checkpoint cannot be proven accessible."""


def preflight_control_checkpoint_access(
    *,
    modality: str,
    token: str,
    validator: Callable[[str, str, str, str], Any] = validate_hf_file_access,
) -> dict[str, str | int]:
    """Verify caller-owned gated access; never infer license acceptance."""

    selected = str(modality or "").strip().lower()
    checkpoint = COSMOS_TRANSFER_CHECKPOINTS.get(selected)
    if checkpoint is None:
        raise CosmosCheckpointAccessError(
            f"unsupported Cosmos Transfer control modality {selected!r}; expected "
            + ", ".join(COSMOS_TRANSFER_CHECKPOINTS)
        )
    if not str(token or "").strip():
        raise CosmosCheckpointAccessError(
            f"HF_TOKEN is required to verify gated access to {checkpoint.repo} "
            f"for {checkpoint.modality!r} control before provisioning or GPU work"
        )
    result: HFAccessResult = validator(
        token, checkpoint.repo, checkpoint.revision, checkpoint.filename
    )
    if not getattr(result, "ok", False):
        status = getattr(result, "status_code", None)
        if status in {401, 403}:
            detail = (
                f"the caller-owned token was denied (HTTP {status}); request access "
                f"at https://huggingface.co/{checkpoint.repo}"
            )
        elif status == 404:
            detail = "the pinned revision/file was not found (HTTP 404)"
        else:
            detail = "the exact access probe did not return authoritative success"
        raise CosmosCheckpointAccessError(
            f"Cosmos Transfer {checkpoint.modality!r} checkpoint access denied or "
            f"unverified: {detail}. No infrastructure was provisioned."
        )
    return {
        "modality": checkpoint.modality,
        "repo": checkpoint.repo,
        "revision": checkpoint.revision,
        "filename": checkpoint.filename,
        "status_code": int(getattr(result, "status_code", 0) or 0),
    }


__all__ = ["CosmosCheckpointAccessError", "preflight_control_checkpoint_access"]
