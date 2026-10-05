"""Call the shared FLUX Action fine-tuning implementation from Python."""

from npa.workbench.flux_action.schemas import FinetuneRequest


def finetune(
    *,
    input_path: str,
    recipe_uri: str,
    output_path: str,
    processes: int = 8,
    dry_run: bool = False,
) -> dict:
    """Fine-tune a declared embodiment using S3 dataset and artifact handoffs.

    Args: Dataset/recipe/output S3 paths, local GPU processes, and validation mode.
    Returns: A planned or completed run receipt.
    Raises: ValueError for invalid requests; FluxActionError for failed runs.
    """
    from npa.workbench.flux_action.runner import finetune as run

    return run(
        FinetuneRequest(
            input_path=input_path,
            recipe_uri=recipe_uri,
            output_path=output_path,
            processes=processes,
        ),
        dry_run=dry_run,
    )
