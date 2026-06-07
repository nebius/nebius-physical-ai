"""SDK helpers for Cosmos3 reason workflow contracts."""

from __future__ import annotations

from npa.workflows.cosmos_split import Cosmos3ReasonConfig, build_cosmos3_reason_manifest


def reason(**kwargs: str) -> dict[str, object]:
    """Return a Cosmos3 reason manifest."""

    return build_cosmos3_reason_manifest(Cosmos3ReasonConfig(**kwargs))


__all__ = ["Cosmos3ReasonConfig", "reason"]
