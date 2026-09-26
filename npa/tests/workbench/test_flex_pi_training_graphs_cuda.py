"""Exercise native CUDA capture and accumulated gradients on the pinned runtime."""

from copy import deepcopy
import os

import pytest

from npa.workbench.flex_pi import training_graphs as graphs


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("NPA_FLEX_PI_CUDA_REGRESSION") != "1",
    reason="set NPA_FLEX_PI_CUDA_REGRESSION=1 inside the pinned GPU runtime",
)
def test_native_cuda_replay_matches_eager_accumulated_updates():
    import torch

    graphs._check_runtime()
    torch.manual_seed(42)
    model = torch.nn.Module()
    model.mot = torch.nn.Linear(2, 1, device="cuda", dtype=torch.float64)
    eager = deepcopy(model)
    graphs.configure_training_graphs(
        model, {"npa_cuda_graphs": "mot", "npa_activation_checkpointing": "off"}
    )
    actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=0.1)
    for _ in range(2):
        actual_optimizer.zero_grad(set_to_none=True)
        eager_optimizer.zero_grad(set_to_none=True)
        for value in (0.2, 0.5, 0.9):
            sample = torch.tensor([[value, -value]], device="cuda", dtype=torch.float64)
            actual, expected = model.mot(sample), eager.mot(sample)
            assert torch.equal(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            for left, right in zip(model.parameters(), eager.parameters(), strict=True):
                assert torch.equal(left.grad, right.grad)
        actual_optimizer.step()
        eager_optimizer.step()
        for key, value in model.state_dict().items():
            assert torch.equal(value, eager.state_dict()[key])
    assert model._npa_training_graphs.receipt()["native_replays"] == {
        "forward": 6,
        "backward": 6,
    }
