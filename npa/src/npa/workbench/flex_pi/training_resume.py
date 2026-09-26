"""Prepare the exact next anchor window for early and full-epoch resume proof."""

from npa.workbench.flex_pi.training import GLOBAL_BATCH, TRAIN_FRAMES
from npa.workbench.flex_pi.training_metrics import PROFILE_UPDATES


def prepare_continuation(trainer, *, profile=False):
    """Restore sampler offsets and independently derive the next rank's anchors.

    Args:
        trainer: Trainer at the complete profile or first-epoch checkpoint.
        profile: Accept only the diagnostic profile cursor when true.
    Returns:
        The exact next ordered local anchor indices.
    Raises:
        RuntimeError: The checkpoint cursor is outside the two proof contracts.
    """
    cursor = (trainer.epoch, trainer.batch_in_epoch, trainer.global_step)
    full = (0, TRAIN_FRAMES // (4 * trainer.batch_size), 1205)
    partial = (
        0,
        PROFILE_UPDATES * trainer.gradient_accumulation_steps,
        PROFILE_UPDATES,
    )
    if cursor != (partial if profile else full):
        raise RuntimeError("checkpoint cursor is outside the accepted resume contract")
    if not profile:
        trainer.epoch, trainer.batch_in_epoch = 1, 0
    trainer.train_sampler.set_epoch_offset(trainer.epoch)
    trainer.train_sampler.set_resume_batch_offset(trainer.batch_in_epoch)
    trainer.train_loader.set_epoch(0)
    return _expected_indices(trainer)


def _expected_indices(trainer):
    import torch

    generator = torch.Generator().manual_seed(trainer.seed + trainer.epoch)
    permutation = torch.randperm(TRAIN_FRAMES, generator=generator).tolist()
    width = trainer.batch_size
    offset = trainer.batch_in_epoch * width * 4
    window = permutation[offset : offset + GLOBAL_BATCH]
    rank_start = trainer.accelerator.process_index * width
    return [
        index
        for start in range(rank_start, GLOBAL_BATCH, 4 * width)
        for index in window[start : start + width]
    ]
