"""Verify the committed Comet training shuffle cursor."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from npa.workflows.behavior_challenge.comet_training_sampler import (
    CommittedCursor,
    NativeCursorSampler,
    native_epoch_indices,
    permutation_identity,
)


def test_cursor_validates_partial_and_epoch_boundary() -> None:
    cursor = CommittedCursor(epoch=0, batch_offset=1, committed_global_batch=1)
    cursor.validate(dataset_size=770, batch_size=256)

    next_cursor = cursor.advance_after_update(
        dataset_size=770, batch_size=256, update_completed=True
    )
    boundary = next_cursor.advance_after_update(
        dataset_size=770, batch_size=256, update_completed=True
    )

    assert next_cursor == CommittedCursor(0, 2, 2)
    assert boundary == CommittedCursor(1, 0, 3)


@pytest.mark.parametrize(
    ("cursor", "dataset_size", "batch_size"),
    [
        (CommittedCursor(0, 0, 0), 1, 2),
        (CommittedCursor(0, 0, 0), 8, 0),
        (CommittedCursor(0, 0, 0), 8, True),
        (CommittedCursor(-1, 0, 0), 8, 2),
        (CommittedCursor(0, 4, 4), 8, 2),
        (CommittedCursor(1, 0, 3), 8, 2),
        (CommittedCursor(True, 0, 0), 8, 2),
        (CommittedCursor(0, 0, False), 8, 2),
    ],
)
def test_cursor_rejects_invalid_state(
    cursor: CommittedCursor, dataset_size: int, batch_size: int
) -> None:
    with pytest.raises(ValueError):
        cursor.validate(dataset_size=dataset_size, batch_size=batch_size)


@pytest.mark.parametrize("completed", [False, 0, 1, None])
def test_cursor_does_not_advance_without_exact_completion(completed: object) -> None:
    cursor = CommittedCursor(0, 0, 0)

    with pytest.raises(ValueError, match="completed update"):
        cursor.advance_after_update(
            dataset_size=8,
            batch_size=2,
            update_completed=completed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("dataset_size", "seed", "epoch"),
    [(0, 1, 0), (1, True, 0), (1, 2**64, 0), (1, 1, -1)],
)
def test_native_order_rejects_inputs_before_torch_import(
    dataset_size: int, seed: int, epoch: int
) -> None:
    torch_before = sys.modules.get("torch")

    with pytest.raises(ValueError):
        native_epoch_indices(dataset_size, seed, epoch)

    assert sys.modules.get("torch") is torch_before


def test_sampler_rejects_seed_without_loading_torch() -> None:
    torch_before = sys.modules.get("torch")

    with pytest.raises(ValueError, match="seed"):
        NativeCursorSampler(
            range(8), seed=True, batch_size=2, cursor=CommittedCursor(0, 0, 0)
        )

    assert sys.modules.get("torch") is torch_before


@pytest.mark.parametrize("indices", [[0, 0], [0, True], [-1, 0]])
def test_permutation_identity_rejects_noncanonical_indices(indices: list[int]) -> None:
    with pytest.raises(ValueError):
        permutation_identity(indices)


def test_cursor_module_does_not_eagerly_import_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import npa.workflows.behavior_challenge.comet_training_sampler; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(
    os.environ.get("NPA_REQUIRE_PINNED_TORCH_SAMPLER_PROOF") != "1",
    reason="requires the private pinned Torch 2.7.1 CPU environment",
)
def test_pinned_torch_loader_matches_partial_resume_and_boundary() -> None:
    import torch

    from npa.workflows.behavior_challenge.comet_training_sampler import (
        NativeCursorSampler,
    )

    dataset_size, batch_size, seed = 770, 256, 2026
    generator = torch.Generator().manual_seed(seed)
    native = torch.utils.data.DataLoader(
        range(dataset_size),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=8,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    epoch_zero = [batch.tolist() for batch in native]
    epoch_one = [batch.tolist() for batch in native]

    cursor = CommittedCursor(0, 1, 1)
    sampler = NativeCursorSampler(
        range(dataset_size), seed=seed, batch_size=batch_size, cursor=cursor
    )
    resumed = torch.utils.data.DataLoader(
        range(dataset_size),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=8,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )

    assert [batch.tolist() for batch in resumed] == epoch_zero[1:]
    assert [batch.tolist() for batch in resumed] == epoch_one
    assert cursor.advance_after_update(
        dataset_size=dataset_size, batch_size=batch_size, update_completed=True
    ) == CommittedCursor(0, 2, 2)
