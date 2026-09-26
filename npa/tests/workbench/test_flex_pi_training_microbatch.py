"""Check complete anchor coverage and actual-tail loss scaling for larger batches."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from npa.sdk.workbench.flex_pi import train
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.schemas import TrainingBody


@pytest.mark.parametrize("width", [0, 2, 4, 6, 24, True, 3.0, "3"])
def test_unsupported_microbatch_fails_before_vendor_execution(width, monkeypatch):
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training._execute_training",
        lambda *args: pytest.fail("invalid batch must not launch a worker"),
    )
    with pytest.raises(FlexPiError, match="microbatch-per-rank"):
        train(output_path="s3://example-bucket/run", microbatch_per_rank=width)


@pytest.mark.parametrize("width", [2, 4, 6])
def test_http_rejects_widths_that_cannot_split_the_true_tail(width):
    with pytest.raises(ValidationError):
        TrainingBody(microbatch_per_rank=width)


@pytest.mark.parametrize("width", [1, 3])
def test_qualification_preserves_all_anchors_and_weights_the_true_tail(width):
    torch = pytest.importorskip("torch")
    from npa.workbench.flex_pi.training_fixture import (
        _fixture_updates,
        _rank_fixture_indices,
        fixture_indices,
    )

    global_indices = fixture_indices()
    ranks = []
    for rank in range(4):
        trainer = _ToyTrainer(width, rank)
        indices = _rank_fixture_indices(global_indices, trainer)
        ranks.append(indices)
        loader = [
            {
                "npa_sample_index": torch.tensor(indices[start : start + width]),
                "values": torch.tensor(
                    indices[start : start + width], dtype=torch.float64
                ),
            }
            for start in range(0, 33, width)
        ]
        observed, _ = _fixture_updates(trainer, loader, trainer.recorder)
        assert observed == indices and len(observed) == 33
        assert trainer.means == pytest.approx(
            [sum(indices[:24]) / 24, sum(indices[24:]) / 9]
        )
    reconstructed = [
        index
        for start in range(0, 33, width)
        for rank in ranks
        for index in rank[start : start + width]
    ]
    assert reconstructed == global_indices
    assert len(set(reconstructed)) == 132


class _ToyTrainer:
    def __init__(self, width, rank):
        self.batch_size = width
        self.accelerator = SimpleNamespace(process_index=rank, sync_gradients=False)
        self.global_step = 0
        self.position = 0
        self.total = 0.0
        self.means = []
        self.local_samples = 0
        self.recorder = SimpleNamespace(updates=[])

    def _backward(self, sample, divisor):
        self.position += self.batch_size
        self.local_samples += self.batch_size
        self.total += float(sample["values"].mean()) / divisor
        self.accelerator.sync_gradients = self.position in {24, 33}
        if self.accelerator.sync_gradients:
            self.global_step += 1
            self.means.append(self.total)
            self.recorder.updates.append({"local_samples": self.local_samples})
            self.total = 0.0
            self.local_samples = 0
