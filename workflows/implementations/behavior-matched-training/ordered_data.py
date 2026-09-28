"""Decode fixed evaluation batches concurrently without changing their order."""

from __future__ import annotations

import os


def _keep_samples(samples):
    return samples


def _cpu_worker(_worker_id: int) -> None:
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    torch.set_num_threads(1)


class _IndexedSamples:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class OrderedBatches:
    """Reuse CPU decoder workers across finite passes of explicit index batches."""

    def __init__(self, dataset, batches, *, workers: int):
        import torch

        self.batches = tuple(tuple(batch) for batch in batches)
        if workers < 0 or any(not batch for batch in self.batches):
            raise ValueError("workers must be nonnegative and batches nonempty")
        if any(
            not isinstance(index, int) or not 0 <= index < len(dataset)
            for batch in self.batches
            for index in batch
        ):
            raise ValueError("an ordered batch contains an invalid dataset index")
        self.loader = torch.utils.data.DataLoader(
            _IndexedSamples(dataset),
            batch_sampler=self.batches,
            num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
            persistent_workers=workers > 0,
            collate_fn=_keep_samples,
            worker_init_fn=_cpu_worker,
            generator=torch.Generator().manual_seed(0),
        )

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        iterator = self.loader._iterator
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator = None

    def __iter__(self):
        for expected, samples in zip(self.batches, self.loader, strict=True):
            actual = tuple(index for index, _sample in samples)
            if actual != expected:
                raise ValueError("decoder workers changed the frozen sample order")
            yield expected, [sample for _index, sample in samples]
