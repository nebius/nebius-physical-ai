"""SDK helpers for Cosmos2 transfer workflow contracts."""

from __future__ import annotations

from npa.workflows.cosmos_split import Cosmos2TransferConfig, build_cosmos2_transfer_manifest


def transfer(
    *,
    input_uri: str,
    output_uri: str,
    assets_uri: str = "",
    scene_spec_uri: str = "",
    image: str = "",
    run_id: str = "",
) -> dict[str, object]:
    """Return a Cosmos2 transfer manifest."""

    return build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            assets_uri=assets_uri,
            scene_spec_uri=scene_spec_uri,
            image=image,
            run_id=run_id,
        )
    )


__all__ = ["Cosmos2TransferConfig", "transfer"]
