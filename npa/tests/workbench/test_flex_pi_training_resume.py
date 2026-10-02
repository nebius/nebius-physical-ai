"""A resume proof must continue the exact next anchors, not repeat the profile."""

from types import SimpleNamespace

import pytest

from npa.workbench.flex_pi.training import TRAIN_FRAMES
from npa.workbench.flex_pi.training_resume import prepare_continuation


class Sampler:
    def set_epoch_offset(self, value):
        self.epoch_offset = value

    def set_resume_batch_offset(self, value):
        self.batch_offset = value


def _trainer(width, rank, complete):
    loader = SimpleNamespace(set_epoch=lambda value: None)
    return SimpleNamespace(
        epoch=0,
        batch_size=width,
        seed=42,
        train_sampler=Sampler(),
        train_loader=loader,
        global_step=1205 if complete else 30,
        batch_in_epoch=TRAIN_FRAMES // (4 * width) if complete else 30 * (24 // width),
        gradient_accumulation_steps=24 // width,
        accelerator=SimpleNamespace(process_index=rank),
    )


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("complete", [False, True])
def test_resume_continues_exact_ordered_global_window(width, complete):
    torch = pytest.importorskip("torch")
    trainers = [_trainer(width, rank, complete) for rank in range(4)]
    ranks = [
        prepare_continuation(trainer, profile=not complete) for trainer in trainers
    ]
    reconstructed = [
        index
        for start in range(0, 24, width)
        for rank in ranks
        for index in rank[start : start + width]
    ]
    generator = torch.Generator().manual_seed(43 if complete else 42)
    permutation = torch.randperm(TRAIN_FRAMES, generator=generator).tolist()
    offset = 0 if complete else 2880
    assert reconstructed == permutation[offset : offset + 96]
    assert len(set(reconstructed)) == 96
    for trainer in trainers:
        assert trainer.train_sampler.epoch_offset == int(complete)
        assert trainer.train_sampler.batch_offset == (
            0 if complete else 30 * (24 // width)
        )


def test_unknown_checkpoint_cursor_cannot_claim_resume_proof():
    trainer = _trainer(1, 0, False)
    trainer.batch_in_epoch -= 1
    with pytest.raises(RuntimeError, match="outside the accepted resume contract"):
        prepare_continuation(trainer, profile=True)


def test_profile_checkpoint_cannot_claim_full_epoch_resume():
    with pytest.raises(RuntimeError, match="outside the accepted resume contract"):
        prepare_continuation(_trainer(1, 0, False))
