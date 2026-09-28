"""Track committed Comet training batches independently of Torch prefetch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sized
from dataclasses import asdict, dataclass
from typing import Any

_PINNED_TORCH_VERSION = "2.7.1"
_TORCH_SEED_MIN = -(2**63)
_TORCH_SEED_MAX = 2**64 - 1


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _batch_count(dataset_size: int, batch_size: int) -> int:
    _require_int("dataset_size", dataset_size, minimum=1)
    _require_int("batch_size", batch_size, minimum=1)
    batches = dataset_size // batch_size
    if batches < 1:
        raise ValueError("drop-last training requires at least one complete batch")
    return batches


def _require_seed(seed: int) -> None:
    if type(seed) is not int or not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
        raise ValueError("seed is outside Torch's supported integer range")


@dataclass(frozen=True)
class CommittedCursor:
    """Identify the next batch whose optimizer update has not completed.

    Args:
        epoch: Zero-based shuffle epoch containing the next uncommitted batch.
        batch_offset: Zero-based complete-batch offset within ``epoch``.
        committed_global_batch: Number of optimizer updates already completed.

    Returns:
        None.

    Raises:
        None.
    """

    epoch: int
    batch_offset: int
    committed_global_batch: int

    def validate(self, *, dataset_size: int, batch_size: int) -> None:
        """Validate this cursor against one drop-last dataset.

        Args:
            dataset_size: Number of examples in the dataset.
            batch_size: Number of examples in each committed update.

        Returns:
            None.

        Raises:
            ValueError: If the geometry or cursor is noncanonical.
        """
        batches = _batch_count(dataset_size, batch_size)
        _require_int("epoch", self.epoch, minimum=0)
        _require_int("batch_offset", self.batch_offset, minimum=0)
        _require_int("committed_global_batch", self.committed_global_batch, minimum=0)
        expected = self.epoch * batches + self.batch_offset
        if self.batch_offset >= batches or self.committed_global_batch != expected:
            raise ValueError("committed sampler cursor differs from batch geometry")

    def advance_after_update(
        self, *, dataset_size: int, batch_size: int, update_completed: bool
    ) -> CommittedCursor:
        """Advance after, and only after, one optimizer update completed.

        Args:
            dataset_size: Number of examples in the dataset.
            batch_size: Number of examples in each committed update.
            update_completed: Must be exactly ``True`` after successful completion.

        Returns:
            The cursor for the next uncommitted batch.

        Raises:
            ValueError: If completion is unproven or the cursor is invalid.
        """
        if update_completed is not True:
            raise ValueError("cursor advancement requires a completed update")
        self.validate(dataset_size=dataset_size, batch_size=batch_size)
        batches = dataset_size // batch_size
        offset = self.batch_offset + 1
        epoch = self.epoch
        if offset == batches:
            epoch, offset = epoch + 1, 0
        return CommittedCursor(epoch, offset, self.committed_global_batch + 1)


def native_epoch_indices(dataset_size: int, seed: int, epoch: int) -> list[int]:
    """Replay Torch 2.7.1 RandomSampler's shared-generator ordering.

    Args:
        dataset_size: Number of examples included in the permutation.
        seed: Seed passed to the native Torch DataLoader generator.
        epoch: Zero-based persistent-worker epoch to replay.

    Returns:
        The ordered example indices for the requested epoch.

    Raises:
        ImportError: If Torch is unavailable when replay is requested.
        ValueError: If inputs or the installed Torch version differ.
    """
    _require_int("dataset_size", dataset_size, minimum=1)
    _require_seed(seed)
    _require_int("epoch", epoch, minimum=0)
    import torch

    if torch.__version__.split("+")[0] != _PINNED_TORCH_VERSION:
        raise ValueError("pinned Torch version differs")
    generator = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    permutation: list[int] = []
    for _ in range(epoch + 1):
        sampler = torch.utils.data.RandomSampler(
            range(dataset_size), replacement=False, generator=generator
        )
        permutation = list(sampler)
    if len(permutation) != dataset_size or len(set(permutation)) != dataset_size:
        raise ValueError("native shuffle permutation differs")
    return permutation


def permutation_identity(indices: list[int]) -> dict[str, Any]:
    """Return an exact ordered identity for one epoch permutation.

    Args:
        indices: Ordered, unique, nonnegative integer example indices.

    Returns:
        The element count, SHA-256, and encoding name.

    Raises:
        ValueError: If ``indices`` is not a canonical permutation.
    """
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("permutation indices must be nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("permutation indices must be unique")
    raw = json.dumps(indices, separators=(",", ":")).encode()
    return {
        "count": len(indices),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "canonical_json_integer_array",
    }


class NativeCursorSampler:
    """Yield native shuffled indices from a committed logical batch cursor.

    Args:
        dataset: Sized map-style dataset consumed by Torch DataLoader.
        seed: Seed passed to the original native DataLoader generator.
        batch_size: Drop-last training batch size.
        cursor: Next batch whose update has not completed.

    Returns:
        None.

    Raises:
        ValueError: If the dataset geometry or cursor is invalid.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        seed: int,
        batch_size: int,
        cursor: CommittedCursor,
    ) -> None:
        _require_seed(seed)
        cursor.validate(dataset_size=len(dataset), batch_size=batch_size)
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.initial_cursor = cursor
        self._physical_epoch = cursor.epoch
        self._first_offset = cursor.batch_offset

    def __iter__(self) -> Iterator[int]:
        """Yield the next physical epoch, skipping committed batches once.

        Args:
            None.

        Returns:
            An iterator over example indices.

        Raises:
            ImportError: If Torch is unavailable.
            ValueError: If the installed Torch version differs.
        """
        epoch = self._physical_epoch
        offset = self._first_offset if epoch == self.initial_cursor.epoch else 0
        self._physical_epoch += 1
        self._first_offset = 0
        indices = native_epoch_indices(len(self.dataset), self.seed, epoch)
        return iter(indices[offset * self.batch_size :])

    def __len__(self) -> int:
        """Return examples remaining in the next sampler iteration.

        Args:
            None.

        Returns:
            The example count, including a tail DataLoader will drop.

        Raises:
            None.
        """
        remaining = len(self.dataset) - self._first_offset * self.batch_size
        return max(0, remaining)

    def contract(self) -> dict[str, Any]:
        """Describe the logical cursor independently of worker prefetch.

        Args:
            None.

        Returns:
            A JSON-compatible cursor and permutation identity.

        Raises:
            ImportError: If Torch is unavailable.
            ValueError: If the installed Torch version differs.
        """
        indices = native_epoch_indices(
            len(self.dataset), self.seed, self.initial_cursor.epoch
        )
        return {
            "schema": "npa.behavior.comet-native-loader-cursor.v1",
            "algorithm": "torch-2.7.1-random-sampler-shared-seed-replay",
            "dataset_size": len(self.dataset),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "drop_last": True,
            "cursor": asdict(self.initial_cursor),
            "epoch_permutation": permutation_identity(indices),
            "prefetch_changes_committed_cursor": False,
        }
