"""SDK helpers for Cosmos2 transfer workflow contracts."""

from __future__ import annotations

from npa.workflows.cosmos_split import Cosmos2TransferConfig, build_cosmos2_transfer_manifest


def transfer(**kwargs: str) -> dict[str, object]:
    """Return a Cosmos2 transfer manifest."""

    return build_cosmos2_transfer_manifest(Cosmos2TransferConfig(**kwargs))


__all__ = ["Cosmos2TransferConfig", "transfer"]
