"""Direct Cosmos Transfer adapter shared by workflow and legacy callers."""

from __future__ import annotations

from typing import Any


def run_real_cosmos_transfer(
    client: Any,
    input_uri: str,
    augment_prefix: str,
    frames_root: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Delegate to the real, input-conditioned Transfer implementation."""

    from npa.workflows.sim2real.cosmos_transfer_stage import (
        run_real_cosmos_transfer as real_runner,
    )

    return real_runner(client, input_uri, augment_prefix, frames_root, run_id)


def run_cosmos2_transfer_component_from_s3(
    *,
    input_uri: str,
    output_uri: str,
    augmented_frames_uri: str,
    assets_uri: str = "",
    scene_spec_uri: str = "",
    image: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Execute real task-conditioned Transfer against explicit S3 boundaries."""

    from npa.workflows.sim2real.cosmos_transfer_stage import (
        run_cosmos_transfer_component,
    )

    return run_cosmos_transfer_component(
        input_uri=input_uri,
        output_uri=output_uri,
        augmented_frames_uri=augmented_frames_uri,
        assets_uri=assets_uri,
        scene_spec_uri=scene_spec_uri,
        image=image,
        run_id=run_id,
        real_runner=run_real_cosmos_transfer,
    )
