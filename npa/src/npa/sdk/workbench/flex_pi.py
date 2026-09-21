"""SDK for Flex-Pi policy inference and pinned public training."""

from typing import Any

from npa.workbench.flex_pi.runtime import (
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_CHECKPOINT_REVISION,
    DEFAULT_INPUT_MANIFEST,
    FlexPiRequest,
    run_inference,
)


def infer(
    *,
    input_path: str = DEFAULT_INPUT_MANIFEST,
    output_path: str,
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION,
    num_inference_steps: int = 4,
    seed: int = 42,
    torch_compile: bool = False,
    expected_gpu: str = "",
    run_id: str = "",
    runtime_image: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the same real inference implementation as CLI/API/workflow.

    Args:
        input_path: Local or S3 public-observation manifest.
        output_path: Local directory or S3 artifact prefix.
        checkpoint_id: Hugging Face checkpoint repository.
        checkpoint_revision: Immutable checkpoint commit.
        num_inference_steps: Flow-matching denoising steps.
        seed: Action-noise seed.
        torch_compile: Compile the denoising step after model load.
        expected_gpu: Optional normalized GPU-name substring.
        run_id: Workflow provenance identifier.
        runtime_image: Runtime image provenance override.
        dry_run: Resolve without downloading or executing the model.
    Returns:
        Complete inference provenance and published artifact map.
    """
    return run_inference(
        FlexPiRequest(
            input_path=input_path,
            output_path=output_path,
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            num_inference_steps=num_inference_steps,
            seed=seed,
            torch_compile=torch_compile,
            expected_gpu=expected_gpu,
            run_id=run_id,
            runtime_image=runtime_image,
            dry_run=dry_run,
        )
    )


def train(
    *,
    output_path: str,
    mode: str = "train",
    normalization_path: str = "",
    normalization_sha256: str = "",
    num_workers: int = 4,
    prefetch_factor: int = 4,
    optimizer: str = "default",
    memory_fill: str = "on",
    run_id: str = "",
    runtime_image: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the fixed public four-GPU training and checkpoint-resume contract.

    Args:
        output_path: Authorized run-scoped S3 destination.
        mode: Train with full validation/resume, or profile.
        normalization_path: Original run's statistics as an exact S3 object.
        normalization_sha256: Required content hash when reusing statistics.
        num_workers: Data loader workers per GPU.
        prefetch_factor: Prefetched batches per worker.
        optimizer: Default, foreach or fused AdamW execution.
        memory_fill: Deterministic allocation fills; off requires exact parity.
        run_id: Workflow provenance identifier.
        runtime_image: Exact runtime image provenance.
        dry_run: Resolve without fetching data or executing a model.
    Returns:
        Measured training acceptance result or the dry-run contract.
    Raises:
        FlexPiError: An input, execution, or acceptance gate fails.
    """
    return _run_training_options(locals())


def _run_training_options(options):
    from npa.workbench.flex_pi.training import TrainingRequest, run_training

    return run_training(TrainingRequest(**options))


__all__ = ["FlexPiRequest", "infer", "train"]
