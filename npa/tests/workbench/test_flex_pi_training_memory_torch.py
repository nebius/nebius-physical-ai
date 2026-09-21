"""Exercise deterministic allocation scopes when the optional Torch is installed."""

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")

fixture_indices = import_module("npa.workbench.flex_pi.training_fixture").fixture_indices
memory = import_module("npa.workbench.flex_pi.training_memory")
execution_without_memory_fill = memory.execution_without_memory_fill
memory_fill_receipt = memory.memory_fill_receipt
observation_with_memory_fill = memory.observation_with_memory_fill


@pytest.fixture
def strict_determinism(monkeypatch):
    monkeypatch.setattr("npa.workbench.flex_pi.training_memory._disabled_scopes", 0)
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.utils.deterministic.fill_uninitialized_memory = original[2]
        torch.backends.cudnn.benchmark = original[3]
        torch.backends.cudnn.deterministic = original[4]


def test_execution_scope_preserves_rng_algorithms_and_restores_after_failure(
    strict_determinism,
):
    original = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model failed"):
        with execution_without_memory_fill():
            assert not torch.utils.deterministic.fill_uninitialized_memory
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            raise ValueError("model failed")
    assert torch.equal(original, torch.get_rng_state())
    receipt = memory_fill_receipt({"npa_memory_fill": "off"})
    assert receipt["model_execution_fill"] is False
    assert receipt["restored_fill"] is True


@pytest.mark.parametrize("warn_only", [False, True])
def test_scope_cannot_silently_relax_algorithm_determinism(
    strict_determinism, warn_only
):
    torch.use_deterministic_algorithms(warn_only, warn_only=warn_only)
    with pytest.raises(RuntimeError, match="strict determinism"):
        with execution_without_memory_fill():
            pytest.fail("must reject before model execution")
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_scope_detects_policy_mutation_and_still_restores(strict_determinism):
    with pytest.raises(RuntimeError, match="changed its allocation policy"):
        with execution_without_memory_fill():
            torch.utils.deterministic.fill_uninitialized_memory = True
    assert torch.utils.deterministic.fill_uninitialized_memory


def test_off_policy_requires_observed_execution_scopes(strict_determinism):
    with pytest.raises(RuntimeError, match="observed allocation scopes"):
        memory_fill_receipt({"npa_memory_fill": "off"})


def test_fixture_selection_uses_true_epoch_tail_without_changing_rng():
    state = torch.get_rng_state().clone()
    permutation = torch.randperm(115620, generator=torch.Generator().manual_seed(42))
    indices = fixture_indices()
    assert indices == permutation[:96].tolist() + permutation[-36:].tolist()
    assert len(set(indices)) == 132
    assert torch.equal(state, torch.get_rng_state())


def test_observation_restores_enclosing_candidate_scope(strict_determinism):
    with execution_without_memory_fill():
        with pytest.raises(ValueError):
            with observation_with_memory_fill():
                assert torch.utils.deterministic.fill_uninitialized_memory
                raise ValueError("failed diagnostic")
        assert not torch.utils.deterministic.fill_uninitialized_memory
    assert torch.utils.deterministic.fill_uninitialized_memory
