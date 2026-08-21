"""SDK for Alpamayo 2 Super inference."""

from typing import Any

from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperRequest,
    run_inference,
)


def infer(
    *,
    output_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    manifest: str = DEFAULT_MANIFEST,
    sample_index: int = 0,
    diffusion_steps: int = 10,
    seed: int = 42,
    figure_style: str = "blog",
    require_camera_projection: bool = True,
    run_id: str = "",
    runtime_image: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the same inference implementation used by CLI/API/workflow."""

    return run_inference(
        Alpamayo2SuperRequest(
            output_path=output_path,
            model_id=model_id,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            manifest=manifest,
            sample_index=sample_index,
            diffusion_steps=diffusion_steps,
            seed=seed,
            figure_style=figure_style,
            require_camera_projection=require_camera_projection,
            run_id=run_id,
            runtime_image=runtime_image,
            dry_run=dry_run,
        )
    )


__all__ = ["Alpamayo2SuperRequest", "infer"]
